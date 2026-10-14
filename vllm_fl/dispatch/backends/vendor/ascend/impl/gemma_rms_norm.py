# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""gemma_rms_norm -- AscendC/native -> Triton

Source: `vllm-ascend/csrc/torch_binding.cpp` L2881-2887 (C++ layer implementation,
no dedicated op_kernel).
Call-site schema:
    npu_gemma_rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> (Tensor y, Tensor rstd)
Formula reference (`FlagGems/src/flag_gems/ops/gemma_rms_norm.py` L40-62, line by line):
    cdtype = fp32 (when x is fp16/bf16)
    x    = load(x).to(cdtype)                                   # L53
    var  = sum(x^2) / N                                         # L55
    rrms = 1 / sqrt(var + eps)                                  # L56
    w    = load(gamma).to(fp32)                                 # L60
    y    = (x * rrms * (1.0 + w)).to(x.dtype)                   # L61 <- Gemma-specific: weight + 1
    rstd = rrms                                                 # L62 (fp32)

Key points:
  * gamma is applied as **1.0 + gamma** (Gemma's unit-offset convention) and the addition is
    performed in fp32;
  * the second output is **rrms**, not a reciprocal rstd, i.e. 1/sqrt(mean(x^2) + eps);
  * for fp16/bf16 inputs every intermediate stays in fp32 and the result is cast back to the
    input dtype.

Ascend backend workarounds (a pitfall already observed in hc_pre, see the hc_pre.py docstring):
  * a dynamic `while` loop combined with multi-line 2-D offsets is miscomputed -> this file
    uses one program per row (the row index stays a scalar) and drives the chunked loop with
    `tl.static_range` (the chunk count is computed on the host and passed as a constexpr);
  * no `tl.dot`; no 2-D masks (only 1-D masks).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["gemma_rms_norm"]


@triton.jit
def _gemma_rms_norm_kernel(
    X, W, Y, RSTD,
    ROWS, N,
    EPS,
    BLK_N: tl.constexpr,
    NCH: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= ROWS:
        return
    base = row * N

    # Pass 1: sum(x^2) / N
    ss = tl.zeros([BLK_N], dtype=tl.float32)
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        xv = tl.load(X + base + cc, mask=cc < N, other=0.0).to(tl.float32)
        ss += xv * xv
    var = tl.sum(ss, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + EPS)
    tl.store(RSTD + row, rrms)

    # Pass 2: y = x * rrms * (1 + gamma)
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        m = cc < N
        xv = tl.load(X + base + cc, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + cc, mask=m, other=0.0).to(tl.float32)
        y = xv * rrms * (1.0 + w)
        if OUT_FP32:
            tl.store(Y + base + cc, y, mask=m)
        else:
            tl.store(Y + base + cc, y.to(Y.dtype.element_ty), mask=m)


def gemma_rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
):
    """Equivalent of `npu_gemma_rms_norm` from vllm-ascend.

    Args:
        x: [..., N] (internally flattened to [rows, N] over the trailing dimension)
        gamma: [N] (Gemma convention: the effective weight is 1 + gamma)
        epsilon: attribute, defaults to 1e-6

    Returns:
        (y with the same shape and dtype as x, rstd fp32 with shape [.., 1])
    """
    assert x.dim() >= 2, f"x must be at least 2D, got {tuple(x.shape)}"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), f"unsupported x dtype {x.dtype}"
    N = int(x.shape[-1])
    rows = 1
    for v in x.shape[:-1]:
        rows *= int(v)
    assert gamma.numel() == N, f"gamma length {gamma.numel()} != N={N}"
    dev = x.device
    xc = x.contiguous()
    g = gamma.contiguous()
    y = torch.empty_like(xc)
    rstd = torch.empty(tuple(x.shape[:-1]) + (1,), dtype=torch.float32, device=dev)

    blk_n = min(next_pow2(N), 1024)
    nch = ceil_div(N, blk_n)
    _gemma_rms_norm_kernel[(rows,)](
        xc.view(rows, N), g, y.view(rows, N), rstd.view(rows),
        rows, N, float(epsilon),
        BLK_N=blk_n, NCH=nch,
        OUT_FP32=(xc.dtype == torch.float32),
    )
    return y, rstd
