# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""clipped_swiglu -- torch_npu.npu_clipped_swiglu -> Triton

Source:
    ``npu_clipped_swiglu`` is a torch_npu / CANN built-in operator, not an operator of
    this tree: torch_npu is not vendored and ``csrc/ascend`` holds no clipped-swiglu
    kernel. The formula is pinned by the AscendC kernel that implements the same
    clamped + alpha + bias normalisation and by its two Python oracles:

    1. AscendC, ``SwiGluGate`` of the DequantSwigluQuant family -- the only in-tree
       kernel that carries all three of clamp / alpha / bias:
           csrc/ascend/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h
               L627-671            clamp L647-660, bias L654, alpha L661, silu L663-667,
                                   multiply L669
           csrc/ascend/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_proto.h
               L46-50              attr documentation for swiglu_mode / clamp_limit /
                                   glu_alpha / glu_bias
               L89-92              attr defaults 0 / 0.0 / 1.702 / 1.0
       and the independent clamp site of the second swiglu kernel:
           csrc/ascend/mc2/dispatch_ffn_combine/op_kernel/utils/
               block_epilogue_pertoken_swiglu.hpp L227-246
    2. The eager oracle this operator replaces in vllm-ascend, which spells the exact
       same expression:
           vllm_ascend/models/minimax_m3/minimax_m3.py L570-572
    3. The clamp-then-activate oracles:
           vllm_ascend/ops/activation.py L41-48
           vllm_ascend/lora/quant_moe.py L133-138

Call sites (this repository):
    vllm_fl/dispatch/backends/vendor/ascend/impl/moe/moe_mlp.py L307-313, L425-431,
    L439-445 and L529-535 -- four sites, all of the form::

        torch_npu.npu_clipped_swiglu(
            hidden_states, interleaved=False, alpha=..., limit=..., bias=...)

    vllm_fl/dispatch/backends/vendor/ascend/impl/minimax_m3_injections.py L257-263
    calls the same operator through ``torch.ops.npu.npu_clipped_swiglu`` with the same
    four keywords. ``dim`` is added to the signature because vllm-ascend passes it
    explicitly (``vllm_ascend/models/minimax_m3/minimax_m3.py`` L584, ``dim=-1``); every
    site in this repository leaves it at the default, i.e. the last dimension.

Formula:
    ``x`` is split along ``dim=-1`` into two contiguous halves of width ``H = d // 2``::

        g = clamp(x[..., :H], max=limit)                    # one-sided
        u = clamp(x[..., H:], min=-limit, max=limit)        # two-sided
        out = g / (1 + exp(-alpha * g)) * (u + bias)

    Order of operations, from ``SwiGluGate``: the two-sided clamp (L649/L651) and the
    bias (L654) act on the multiplicand half, with the bias added **after** the clamp;
    the one-sided clamp (L658) acts on the SiLU half; ``glu_alpha`` enters only inside
    the exponential (L661-667: ``Muls(-alpha) -> Exp -> Adds(1) -> Div``); L669 multiplies
    the two. The eager oracle agrees line for line::

        gate = torch.clamp(x[..., :d], max=self.limit)                  # minimax_m3.py L570
        up   = torch.clamp(x[..., d:], min=-self.limit, max=self.limit)  # L571
        activated = gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)   # L572

    ``g / (1 + exp(-alpha * g))`` is used instead of ``g * sigmoid(alpha * g)`` because
    that is the sequence the AscendC kernel executes; the two can differ by one rounding
    step, and matching the native's sequence keeps the comparison direct. Note that
    ``alpha`` multiplies inside the exponential only -- it does not scale ``g`` a second
    time -- for both the AscendC kernel and the eager oracle.

Clamp guard -- ``limit <= 0`` disables the clamp:
    Both AscendC kernels skip the whole clamp under ``if (swigluLimit > 0.0f)``
    (``block_epilogue_pertoken_swiglu.hpp`` L228, ``dequant_swiglu_quant.h`` L647 and
    L656), and the sibling ``npu_swiglu_group_quant`` gates its clamp on
    ``clamp_value != 0`` as well (``impl/swiglu_group_quant.py`` L47). The same guard is
    used here. UNVERIFIED for ``torch_npu.npu_clipped_swiglu`` itself: no call site in
    this tree reaches the clipped path with ``limit <= 0`` (``moe_mlp.py`` L435 selects it
    only for ``swiglu_limit > 0``; the ``swigluoai_uninterleave`` branches pass the model
    value, which is 10.0 for DeepSeek-V4, ``vllm_fl/models/hy4/config.py`` L84), so the
    native's zero-limit behaviour could not be observed offline.

Attribute defaults:
    Every call site passes ``interleaved`` / ``alpha`` / ``limit`` / ``bias`` explicitly,
    so the native default values are NOT observable offline and are not claimed here.
    The defaults below (``dim=-1``, ``alpha=1.0``, ``limit=0.0``, ``bias=0.0``) are this
    repository's MoE attribute defaults (``impl/moe/moe_mlp.py`` L122-124,
    ``impl/moe/moe_runtime_args.py`` L148-151) and make ``clipped_swiglu(x)`` identical
    to ``swiglu(x)``. They are a compatibility choice, not a verified copy of the
    torch_npu schema.

Coverage:
    Implemented       ``interleaved=False``; float16 / bfloat16 / float32; arbitrary
                      leading dimensions; arbitrary H; empty input; any ``alpha``,
                      ``limit``, ``bias``.
    NotImplementedError
                      ``interleaved=True``: the odd-even layout of this operator is not
                      documented anywhere in the tree. ``dequant_swiglu_quant_proto.h``
                      L47 documents an odd-even variant for that *sibling* operator only,
                      and the native ``npu_clipped_swiglu`` interleaved path could not be
                      run offline to settle its lane order, so it is rejected rather than
                      guessed. Every call site passes ``interleaved=False``.
                      ``dim`` other than -1 (or ``x.dim() - 1``).

Tiling:
    One program covers ``BLK_M`` rows x ``BLK_N`` columns. ``BLK_N`` is capped at 1024 and
    ``BLK_M`` is chosen so that a tile stays at 2048 elements, i.e. about
    ``2048 * 4 B * 3`` = 24 KiB of float32 UB for the g / u / result triple, well under
    the 192 KiB UB of the 910B/910C family. Every element is both loaded and stored under
    a mask, so the tail of a non-power-of-two H never touches an out-of-range address.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["clipped_swiglu", "npu_clipped_swiglu"]

#: Elements handled by one program; keeps the float32 tile triple at ~24 KiB of UB.
_TILE_ELEMS = 2048
#: Upper bound on the row width handled by a single program (UB and mask-length bound).
_BLK_N_MAX = 1024


@triton.jit
def _clipped_swiglu_kernel(
    X, Y,
    R, H,
    ALPHA, LIMIT, BIAS,
    CLAMP: tl.constexpr,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
):
    """out[r, c] = silu_alpha(clamp(x[r, c])) * (clamp(x[r, H + c]) + bias).

    ``X`` is the contiguous ``[R, 2H]`` input (row stride ``2 * H``), ``Y`` the ``[R, H]``
    output. The SiLU operand is the first half and is clamped on the max side only; the
    multiplicand is the second half, is clamped on both sides and carries the bias
    (``dequant_swiglu_quant.h`` L647-660). ``CLAMP`` is False when ``limit <= 0``, which
    the AscendC kernels treat as "no clamp at all" (L647 / L656).
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BLK_M + tl.arange(0, BLK_M)
    cols = pn * BLK_N + tl.arange(0, BLK_N)
    mask = (rows < R)[:, None] & (cols < H)[None, :]

    in_off = rows[:, None] * (2 * H) + cols[None, :]
    # Masked loads: lanes past H must not read the next row's data (or past the buffer).
    g = tl.load(X + in_off, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(X + in_off + H, mask=mask, other=0.0).to(tl.float32)

    if CLAMP:
        # One-sided on the SiLU half, two-sided on the multiplicand half (L649-660).
        g = tl.minimum(g, LIMIT)
        u = tl.maximum(tl.minimum(u, LIMIT), 0.0 - LIMIT)
    # Bias after the clamp (L654), alpha inside the exponential only (L661-667).
    y = g / (1.0 + tl.exp(0.0 - ALPHA * g)) * (u + BIAS)

    out_off = rows[:, None] * H + cols[None, :]
    tl.store(Y + out_off, y.to(Y.dtype.element_ty), mask=mask)


def clipped_swiglu(
    x: torch.Tensor,
    dim: int = -1,
    alpha: float = 1.702,
    limit: float = 7.0,
    bias: float = 1.0,
    interleaved: bool = False,
) -> torch.Tensor:
    """Triton equivalent of ``torch_npu.npu_clipped_swiglu``.

    Args:
        x: ``[..., 2H]`` with ``x.shape[-1]`` even; float16, bfloat16 or float32. Made
            contiguous internally (the kernel assumes a row stride of exactly ``2 * H``).
        dim: the dimension that is split in half. Only the last dimension is
            implemented, which is the value every call site uses.
        alpha: SiLU coefficient, applied inside the exponential only:
            ``g / (1 + exp(-alpha * g))`` (``dequant_swiglu_quant.h`` L661).
        limit: clamp bound. ``limit <= 0`` disables the clamp, matching the AscendC
            ``if (clampLimit > 0.0f)`` guard (L647 / L656).
        bias: added to the multiplicand half **after** its clamp (L654), i.e. the result
            is ``silu(alpha * g) * (u + bias)``.
        interleaved: only ``False`` is implemented; ``True`` describes an even/odd lane
            layout that is not documented anywhere in this tree, so it raises instead of
            guessing.

    Returns:
        ``[..., H]`` tensor with ``x``'s dtype and device.
    """
    if interleaved:
        raise NotImplementedError(
            "interleaved=True is not implemented: the odd/even lane order of "
            "npu_clipped_swiglu is not documented in this tree "
            "(dequant_swiglu_quant_proto.h L47 describes an odd-even variant for that "
            "sibling operator only) and could not be settled offline. Every call site "
            "passes interleaved=False (moe_mlp.py L309/L427/L441/L531, "
            "minimax_m3_injections.py L262)"
        )
    if x.dim() < 1:
        raise ValueError(
            f"clipped_swiglu expects at least 1-D input, got shape {tuple(x.shape)}")
    if dim not in (-1, x.dim() - 1):
        raise NotImplementedError(
            f"dim={dim} is not implemented: only the last dimension is supported, which "
            f"is the value every call site uses (vllm_ascend/models/minimax_m3/"
            f"minimax_m3.py L584 passes dim=-1; moe_mlp.py leaves the default)"
        )

    d = int(x.shape[-1])
    if d % 2 != 0:
        raise ValueError(f"the last dimension of x must be even, got {d}")
    H = d // 2
    rows = int(x.numel()) // d if d else 0

    xc = x.contiguous()
    y = torch.empty(x.shape[:-1] + (H,), dtype=x.dtype, device=x.device)
    limit_f = float(limit)
    if rows > 0 and H > 0:
        blk_n = min(_next_pow2(H), _BLK_N_MAX)
        blk_m = max(1, _TILE_ELEMS // blk_n)
        grid = (_ceil_div(rows, blk_m), _ceil_div(H, blk_n))
        _clipped_swiglu_kernel[grid](
            xc.view(rows, 2 * H),
            y.view(rows, H),
            rows,
            H,
            float(alpha),
            limit_f,
            float(bias),
            CLAMP=limit_f > 0.0,
            BLK_M=blk_m,
            BLK_N=blk_n,
        )
    return y


#: The call sites spell the operator ``torch_npu.npu_clipped_swiglu`` (``torch.ops.npu``
#: in minimax_m3_injections.py); the deployment shim imports the module-level name, so
#: both spellings are kept.
npu_clipped_swiglu = clipped_swiglu
