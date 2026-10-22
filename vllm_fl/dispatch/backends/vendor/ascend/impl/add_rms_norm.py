# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""npu_add_rms_norm -- torch_npu / CANN built-in -> Triton

Source and contract
-------------------
`npu_add_rms_norm` is a torch_npu built-in, so this checkout's `csrc/ascend`
has no AscendC source for it. The identical three-output arithmetic (minus the
bias) is implemented by the AscendC operator `npu_add_rms_norm_bias` in the
sibling vllm-ascend checkout, `_repos/vllm-ascend/csrc/moe/add_rms_norm_bias/`:

  * `op_kernel/add_rms_norm_bias_merge_n.h` L225-280 (`ComputeY`):
        xLocal = x1 + x2                       # the third output
        ComputeRstd(xLocal, rstd, ...)         # L195-223
        y = x * rstd                           # L241-252 (Brcb broadcasts rstd)
        y = gamma * y                          # L255 / L265 / L273
  * `add_rms_norm_bias_merge_n.h` L195-223 (`ComputeRstd`):
        sqx = x * x -> * (1/N) -> ReduceSum -> + eps -> Sqrt -> 1 / Sqrt
        i.e. rstd = 1 / sqrt(sum_j x[i,j]^2 / N + epsilon)

Measured semantics of the built-in (recorded in this tree, hardware-verified):
`vllm_fl/dispatch/backends/vendor/ascend/impl/minimax_m3_injections.py`
L109-124 documents

    npu_add_rms_norm(x1, x2, gamma, eps) -> (normed, rstd, x1 + x2)

and that it mutates **neither** input ("with x1=1, x2=2 the third output is 3
while both inputs stay unchanged"). This port keeps that out-of-place contract;
call sites that need in-place behaviour copy the results back themselves
(`layernorm.py:94-95`, `minimax_m3_injections.py:124-126`).

Call-site schema (every site in `vllm_fl`, verified by grep):
    npu_add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon) ->
        (Tensor y, Tensor rstd, Tensor x_sum)
      vllm_fl/compilation/passes/norm_quant_fusion_pass.py:302,401,500,547
          `output[0]` = y, `output[2]` = x1 + x2, `output[1]` unused
      vllm_fl/dispatch/backends/vendor/ascend/impl/layernorm.py:94,156
      .../impl/minimax_m3_injections.py:124
      .../impl/normalization.py:37

  * x1, x2, gamma: [..., N] / [N], one shared dtype (fp16 / bf16 / fp32);
  * y: x1 shape/dtype; rstd: fp32, shape x1.shape[:-1] + (1,); x_sum: x1 dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["npu_add_rms_norm"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _add_rms_norm_kernel(
    X1, X2, W, Y, RSTD, XOUT,
    ROWS, N, EPS,
    OUT_FP32: tl.constexpr,
    BLK_N: tl.constexpr,
    NCH: tl.constexpr,
):
    # One program per row -> scalar row base and plain 1-D masked accesses.
    row = tl.program_id(0)
    if row >= ROWS:
        return
    base = row * N

    # Pass 1: x = x1 + x2 -> third output, accumulating sum(x^2) in the same pass
    ss = tl.zeros([BLK_N], dtype=tl.float32)
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        m = cc < N
        a = tl.load(X1 + base + cc, mask=m, other=0.0).to(tl.float32)
        b = tl.load(X2 + base + cc, mask=m, other=0.0).to(tl.float32)
        xv = a + b
        if OUT_FP32:
            tl.store(XOUT + base + cc, xv, mask=m)
        else:
            tl.store(XOUT + base + cc, xv.to(XOUT.dtype.element_ty), mask=m)
        ss += xv * xv
    rstd = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / N + EPS)
    tl.store(RSTD + row, rstd)

    # Pass 2: y = (x1 + x2) * rstd * gamma
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        m = cc < N
        a = tl.load(X1 + base + cc, mask=m, other=0.0).to(tl.float32)
        b = tl.load(X2 + base + cc, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + cc, mask=m, other=0.0).to(tl.float32)
        y = (a + b) * rstd * w
        if OUT_FP32:
            tl.store(Y + base + cc, y, mask=m)
        else:
            tl.store(Y + base + cc, y.to(Y.dtype.element_ty), mask=m)


def npu_add_rms_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
):
    """Triton equivalent of `torch_npu.npu_add_rms_norm`.

    Args:
        x1, x2: [..., N], same shape and dtype; neither is modified.
        gamma: [N], same dtype.
        epsilon: attribute, default 1e-6.

    Returns:
        (y, rstd, x_sum):
          y     = (x1 + x2) * rstd * gamma, x1's shape/dtype;
          rstd  = 1 / sqrt(mean((x1+x2)^2) + epsilon), fp32, [.., 1];
          x_sum = x1 + x2, x1's shape/dtype (the residual written back by the
                  call sites).
    """
    if x1.shape != x2.shape:
        raise ValueError(f"x1/x2 shape mismatch: {tuple(x1.shape)} vs {tuple(x2.shape)}")
    if x1.dtype != x2.dtype or x1.dtype != gamma.dtype:
        raise ValueError("x1/x2/gamma must share one dtype")
    if x1.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {x1.dtype}")
    if x1.dim() < 1:
        raise ValueError(f"x1 must have at least 1 dim, got {tuple(x1.shape)}")
    n = int(x1.shape[-1])
    if n <= 0:
        raise ValueError(f"the normalised dim must be positive, got {n}")
    if gamma.numel() != n:
        raise ValueError(f"gamma numel {gamma.numel()} != N={n}")
    rows = 1
    for v in x1.shape[:-1]:
        rows *= int(v)

    a = x1.contiguous()
    b = x2.contiguous()
    g = gamma.contiguous()
    y = torch.empty_like(a)
    x_sum = torch.empty_like(a)
    rstd = torch.empty(tuple(x1.shape[:-1]) + (1,), dtype=torch.float32, device=x1.device)
    if rows == 0:
        return y, rstd, x_sum

    blk_n = min(next_pow2(n), 1024)
    _add_rms_norm_kernel[(rows,)](
        a.view(rows, n), b.view(rows, n), g, y.view(rows, n), rstd.view(rows),
        x_sum.view(rows, n),
        rows, n, float(epsilon),
        OUT_FP32=(a.dtype == torch.float32),
        BLK_N=blk_n,
        NCH=ceil_div(n, blk_n),
    )
    return y, rstd, x_sum
