# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""swiglu -- torch_npu.npu_swiglu -> Triton

Source:
    This tree carries no AscendC kernel for ``npu_swiglu``: torch_npu is not vendored and
    ``csrc/ascend`` contains no swiglu operator, so the semantics are pinned by the
    sources below rather than by a single kernel file.

    1. The AscendC SwiGLU epilogue whose formula this operator reproduces, including the
       Muls(-1) / Exp / Adds(1) / Div / Mul sequence used for the SiLU:
           csrc/ascend/mc2/dispatch_ffn_combine/op_kernel/utils/
               block_epilogue_pertoken_swiglu.hpp L227-246
           csrc/ascend/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h
               L627-671            SwiGluGate -- same normalisation with
                                   glu_alpha (L661) and glu_bias (L654)
    2. The Python oracles vllm-ascend uses for the same activation:
           vllm_ascend/ops/activation.py L35-48
           vllm_ascend/lora/quant_moe.py L133-138
    3. An independent Triton kernel with the same layout and formula:
           FlagGems-vllm/src/flaggems_vllm/ops/swiglu.py L52-66

Call sites (this repository):
    vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py L1328
        out = torch_npu.npu_swiglu(hidden_states)
    vllm_fl/dispatch/backends/vendor/ascend/impl/moe/moe_mlp.py L315 and L549
        hidden_states = torch_npu.npu_swiglu(hidden_states)
    All three pass exactly one positional argument and no keyword, so the only attribute
    the repository exercises is the default ``dim=-1``. ``dim`` is accepted here for
    signature compatibility and validated; only the last dimension is implemented (see
    Coverage).

Formula:
    ``x`` is split along ``dim=-1`` into two **contiguous** halves of width
    ``H = d // 2`` -- not into even/odd lanes::

        a   = x[..., :H]                 # SiLU operand (the "gate" half)
        b   = x[..., H:]                 # multiplicand (the "up" half)
        out = a / (1 + exp(-a)) * b

    Layout evidence, three independent sources:
      * ``block_epilogue_pertoken_swiglu.hpp`` L244-246 divides the first tile
        ``ubCFp32`` by ``1 + exp(ubCFp32)`` and multiplies by
        ``ubCFp32[ChunkTileLen]``, i.e. by the tile that physically follows it;
      * ``FlagGems-vllm ops/swiglu.py`` L52-66 loads ``x_a = x[..., :H]`` and
        ``x_b = x[..., H:]``;
      * ``vllm_ascend/ops/activation.py`` L43-48 clamps ``x[..., :d]`` as the gate half
        and ``x[..., d:]`` as the up half before calling ``npu_swiglu``.
    The same conclusion is recorded for the sibling operator in
    ``impl/swiglu_group_quant.py`` ("the two halves are contiguous, not interleaved").

    The exponential form (``a / (1 + exp(-a))``) is used instead of
    ``a * sigmoid(a)`` because that is the sequence both AscendC kernels execute
    (``Muls(-1) -> Exp -> Adds(1) -> Div``); the two forms can differ by one rounding
    step and staying with the native's sequence keeps the comparison direct.

Coverage:
    Implemented       float16 / bfloat16 / float32, arbitrary leading dimensions,
                      arbitrary H (including H that is not a power of two), empty input.
    NotImplementedError
                      ``dim`` other than -1 (or the equivalent ``x.dim() - 1``): no call
                      site splits anywhere else and both AscendC analogues are hard-wired
                      to the last dimension.

Tiling:
    One program covers ``BLK_M`` rows x ``BLK_N`` columns. ``BLK_N`` is capped at 1024 and
    ``BLK_M`` is chosen so that a tile stays at 2048 elements, i.e. about
    ``2048 * 4 B * 3`` = 24 KiB of float32 UB for the a / b / result triple, well under
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



__all__ = ["swiglu", "npu_swiglu"]

#: Elements handled by one program; keeps the float32 tile triple at ~24 KiB of UB.
_TILE_ELEMS = 2048
#: Upper bound on the row width handled by a single program (UB and mask-length bound).
_BLK_N_MAX = 1024


@triton.jit
def _swiglu_kernel(
    X, Y,
    R, H,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
):
    """out[r, c] = silu(x[r, c]) * x[r, H + c] for the flat [R, 2H] view of x.

    ``X`` is the contiguous ``[R, 2H]`` input (row stride ``2 * H``), ``Y`` the ``[R, H]``
    output. The SiLU operand is the first half, the multiplicand the second half.
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BLK_M + tl.arange(0, BLK_M)
    cols = pn * BLK_N + tl.arange(0, BLK_N)
    mask = (rows < R)[:, None] & (cols < H)[None, :]

    in_off = rows[:, None] * (2 * H) + cols[None, :]
    # Masked loads: lanes past H must not read the next row's data (or past the buffer).
    a = tl.load(X + in_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(X + in_off + H, mask=mask, other=0.0).to(tl.float32)

    y = a / (1.0 + tl.exp(0.0 - a)) * b

    out_off = rows[:, None] * H + cols[None, :]
    tl.store(Y + out_off, y.to(Y.dtype.element_ty), mask=mask)


def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Triton equivalent of ``torch_npu.npu_swiglu``.

    Args:
        x: ``[..., 2H]`` with ``x.shape[-1]`` even; float16, bfloat16 or float32. Made
            contiguous internally (the kernel assumes a row stride of exactly ``2 * H``).
        dim: the dimension that is split in half. Only the last dimension is
            implemented, which is the default ``torch_npu`` uses and the only value the
            call sites pass (``device_op.py`` L1328, ``moe_mlp.py`` L315 / L549).

    Returns:
        ``[..., H]`` tensor with ``x``'s dtype and device, where
        ``out[..., c] = silu(x[..., c]) * x[..., H + c]``.
    """
    if x.dim() < 1:
        raise ValueError(f"swiglu expects at least 1-D input, got shape {tuple(x.shape)}")
    if dim not in (-1, x.dim() - 1):
        raise NotImplementedError(
            f"dim={dim} is not implemented: npu_swiglu is only ever called on the last "
            f"dimension (device_op.py L1328, moe_mlp.py L315/L549) and the AscendC "
            f"analogues split the last dimension as well "
            f"(block_epilogue_pertoken_swiglu.hpp L244-246)"
        )

    d = int(x.shape[-1])
    if d % 2 != 0:
        raise ValueError(f"the last dimension of x must be even, got {d}")
    H = d // 2
    rows = int(x.numel()) // d if d else 0

    xc = x.contiguous()
    y = torch.empty(x.shape[:-1] + (H,), dtype=x.dtype, device=x.device)
    if rows > 0 and H > 0:
        blk_n = min(_next_pow2(H), _BLK_N_MAX)
        blk_m = max(1, _TILE_ELEMS // blk_n)
        grid = (_ceil_div(rows, blk_m), _ceil_div(H, blk_n))
        _swiglu_kernel[grid](
            xc.view(rows, 2 * H),
            y.view(rows, H),
            rows,
            H,
            BLK_M=blk_m,
            BLK_N=blk_n,
        )
    return y


#: The call site spells the operator ``torch_npu.npu_swiglu``; the deployment shim
#: imports the module-level name, so both spellings are kept.
npu_swiglu = swiglu
