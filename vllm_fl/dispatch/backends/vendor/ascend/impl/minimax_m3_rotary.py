# Copyright (c) 2026 BAAI. All rights reserved.
"""Ascend ``ApplyRotaryEmb`` override (partial-rotary correctness).

Why this override is required
-----------------------------
Upstream ``ApplyRotaryEmb.forward_static`` rotates the *entire* head dimension:

```
x1, x2 = torch.chunk(x, 2, dim=-1)      # splits the whole head_dim
o1 = x1 * cos - x2 * sin                 # cos: [seq, head_dim // 2]
```

On CUDA that path is never used for MiniMax-M3's vision tower: ``CustomOp``
dispatches to ``forward_cuda``, which calls the FlashAttention rotary kernel that
takes an explicit ``rotary_dim`` and supports *partial* rotation. OOT platforms
have no ``forward_cuda``, so dispatch falls through ``forward_oot`` to
``forward_native`` — and that native path is mathematically wrong when
``rotary_dim < head_dim`` and ``rotary_dim`` is odd.

Measured on A3 with MiniMax-M3 vision (head_dim 80, rotary_dim 78):

```
RuntimeError: The size of tensor a (40) must match the size of tensor b (39)
              at non-singleton dimension 3
```

``chunk(80, 2)`` yields 40 while the correctly-sized cos holds ``78 / 2 = 39``.

The fix
-------
Rotate only the leading ``rotary_dim`` lanes and pass the rest through, using the
platform's own fused operator ``torch.ops.npu.npu_rotary_mul`` (no FlagGems).

The ``is_neox_style`` (NeoX/"half") layout is the one every model in scope uses
and is served by ``npu_rotary_mul``; the GPT-J/"interleave" layout is computed
explicitly in torch, because that is not the layout the fused op consumes.
"""

from __future__ import annotations

import logging

import torch

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

logger = logging.getLogger(__name__)


def _interleave_rotary(
    x_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """GPT-J style (pairs) rotation, used only when ``is_neox_style`` is False.

    ``x_rot`` is ``[..., rotary_dim]``; ``cos``/``sin`` are ``[seq, rotary_dim // 2]``
    and are broadcast over the head axis.
    """
    cos = cos.unsqueeze(-2).to(x_rot.dtype)
    sin = sin.unsqueeze(-2).to(x_rot.dtype)
    x1 = x_rot[..., ::2]
    x2 = x_rot[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class AscendApplyRotaryEmb(ApplyRotaryEmb):
    """Partial-rotation-correct ``ApplyRotaryEmb`` backed by ``npu_rotary_mul``."""

    def forward_oot(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)

        head_dim = x.shape[-1]
        rotary_dim = cos.shape[-1] * 2
        if rotary_dim > head_dim:
            raise ValueError(
                f"rotary_dim ({rotary_dim}) must not exceed head_dim ({head_dim})"
            )

        if not self.is_neox_style:
            rotated = _interleave_rotary(x[..., :rotary_dim], cos, sin)
        else:
            # Elementwise NeoX/"half" rotation; see ops_qknorm_rope.py for why
            # ATB's npu_rotary_mul cannot be used once compilation is enabled.
            # Verified bitwise identical to npu_rotary_mul("half") on A3.
            rotated = _half_rotary(x[..., :rotary_dim], cos, sin)

        if rotary_dim == head_dim:
            output = rotated
        else:
            output = torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

        return self._post_process(output, origin_shape, origin_dtype)


def _half_rotary(
    x_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """NeoX ("half") rotation over the leading ``2 * cos.shape[-1]`` lanes.

    ``cos``/``sin`` are ``[seq, rotary_dim // 2]`` and broadcast over the head
    axis. Bitwise identical to ATB ``npu_rotary_mul`` in "half" mode, but
    expressible inside a compiled region.
    """
    dim = x_rot.shape[-1]
    half = dim // 2
    cos_h = cos.unsqueeze(-2).to(torch.float32)
    sin_h = sin.unsqueeze(-2).to(torch.float32)
    xf = x_rot.to(torch.float32)
    x1, x2 = xf[..., :half], xf[..., half:]
    o1 = x1 * cos_h - x2 * sin_h
    o2 = x2 * cos_h + x1 * sin_h
    return torch.cat((o1, o2), dim=-1).to(x_rot.dtype)


def install_apply_rotary_emb_override() -> bool:
    """Register the Ascend ``ApplyRotaryEmb``. Idempotent."""
    from vllm.model_executor.custom_op import CustomOp, op_registry_oot

    existing = op_registry_oot.get("ApplyRotaryEmb")
    if existing is AscendApplyRotaryEmb:
        return True
    if existing is not None:
        logger.warning(
            "ApplyRotaryEmb is already overridden by %r; leaving it alone", existing
        )
        return False

    AscendApplyRotaryEmb.__name__ = "AscendApplyRotaryEmb"
    CustomOp.register_oot(
        _decorated_op_cls=AscendApplyRotaryEmb, name="ApplyRotaryEmb"
    )
    logger.info(
        "MiniMax-M3: registered Ascend ApplyRotaryEmb "
        "(partial-rotary fix; upstream forward_native mis-handles rotary_dim < head_dim)"
    )
    return True


__all__ = ["AscendApplyRotaryEmb", "install_apply_rotary_emb_override"]
