# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""hc_pre_inv_rms -- AscendC -> Triton

Source: csrc/ascend/moe/hc_pre_inv_rms/
Interface (op_host/hc_pre_inv_rms_def.cpp L24-37 + infershape L31-54):
    Input : x   [b, s, hc, d] (fp16/fp32/bf16)  or  [b*s, hc, d]
    Output: y   [b, s, 1] / [b*s, 1]   fp32
    Attr  : epsilon = 1e-6 (OPTIONAL, float)

Shape rules (op_host/hc_pre_inv_rms_tiling.cpp L103-123 / L326-347):
    x 4D: A = dim0 * dim1,  R = dim2 * dim3
    x 3D: A = dim0,         R = dim1 * dim2
    R == 28672 takes the large_d branch (same math)

Formula (op_kernel/hc_pre_inv_rms_full_load.h L152-195 ComputeB16 / L198-236 ComputeB32):
    s   = sum_r x[a, r]^2            # fp32
    mean = s * (1.0f / R)            # L181-183 Muls(y, y, meanCof)
    mean = mean + epsilon            # L185 Adds
    y[a] = 1.0 / sqrt(mean)          # L187 Duplicate(1.0) + L189 Sqrt + L191 Div
    i.e.  y[a] = 1 / sqrt(mean_r(x[a,:]^2) + eps)

Note: epsilon is added after the mean, not to each squared term.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["hc_pre_inv_rms"]


@triton.jit
def _hc_pre_inv_rms_kernel(
    X, Y, A, R, EPS,
    BLK_A: tl.constexpr,
    BLK_R: tl.constexpr,
    NCH: tl.constexpr,
    NCH_P2: tl.constexpr,
):
    """Row-wise 1-D accumulation with a **constexpr loop count**.

    Two rules that must be kept (measured on device):
    1. **The loop count must be constexpr and** a power of two (`NCH_P2 = next_pow2(NCH)`);
       iterations beyond the real block count are skipped by `if c < NCH:`.
       Reason: the Ascend backend miscompiles static loops whose **unroll count is not a
       power of two** -- the mixes kernel of `hc_pre` shows an error of 5.39 at `KCH=28`,
       while `KCH=20/24/32` all stay at 4e-7 (same family of evidence).
    2. Do not pair a `while r < R` (runtime bound) with a `[BLK_A, BLK_R]` 2-D mask --
       same family as BACKEND_NOTES issue 1. This file uses **row-wise 1-D** instead.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLK_R)
    for i in range(BLK_A):
        row = pid * BLK_A + i
        if row < A:
            acc = tl.zeros((BLK_R,), dtype=tl.float32)
            for c in range(NCH_P2):
                if c < NCH:
                    col = c * BLK_R + cols
                    x = tl.load(X + row * R + col, mask=col < R, other=0.0).to(tl.float32)
                    acc += x * x
            s = tl.sum(acc, axis=0)
            y = 1.0 / tl.sqrt(s / R + EPS)
            tl.store(Y + row, y)


def hc_pre_inv_rms(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Triton equivalent of the AscendC HcPreInvRms.

    Args:
        x: 3D [b*s, hc, d] or 4D [b, s, hc, d], dtype in {fp16, bf16, fp32}
        epsilon: attribute, defaults to 1e-6

    Returns:
        y: fp32, shape = x.shape[:-2] + (1,) (4D -> [b,s,1]; 3D -> [b*s,1])
    """
    assert x.dim() in (3, 4), f"x must be 3D or 4D, got {tuple(x.shape)}"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), \
        f"x dtype {x.dtype} is not supported"

    xc = x.contiguous()
    A = 1
    for d in xc.shape[:-2]:
        A *= int(d)
    R = int(xc.shape[-2]) * int(xc.shape[-1])
    assert A > 0 and R > 0, f"invalid A={A} R={R}"

    x2 = xc.view(A, R)
    y = torch.empty((A,), dtype=torch.float32, device=xc.device)

    # Ascend UB = 192KB: keep the tile small (several in-block iterations), otherwise
    # bishengir reports ub overflow.
    BLK_A = 4
    BLK_R = min(next_pow2(R), 512)
    nch = ceil_div(R, BLK_R)
    grid = (ceil_div(A, BLK_A),)
    _hc_pre_inv_rms_kernel[grid](
        x2, y, A, R, float(epsilon),
        BLK_A=BLK_A, BLK_R=BLK_R,
        NCH=nch, NCH_P2=next_pow2(nch),
    )
    return y.view(*xc.shape[:-2], 1)
