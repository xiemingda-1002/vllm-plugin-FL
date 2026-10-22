# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""npu_rms_norm -- torch_npu / CANN built-in -> Triton

Source and contract
-------------------
`npu_rms_norm` is a torch_npu built-in, so `csrc/ascend` carries no AscendC
source for it. The arithmetic is pinned by three independent local references:

  * AscendC `csrc/ascend/attention/compressor/op_kernel/arch32/rms_norm.h`
    L36-85 (`Compressor::RmsNorm`):
        temp = x * x -> RowSum -> * (1/N) -> + eps -> Sqrt
              -> dst = src / Sqrt -> dst = dst * gamma
    i.e. the reciprocal form used below is algebraically the same sequence.
  * `_repos/vllm-ascend/csrc/moe/add_rms_norm_bias/op_kernel/
    add_rms_norm_bias_merge_n.h` L195-223 (`ComputeRstd`, sibling repo)
    documents rstd = 1 / sqrt(sum(x^2)/N + eps) for the add variant.
  * FlagGems `src/flag_gems/ops/rms_norm.py` (the upstream Triton port this
    family is modelled on) and `FlagGems-vllm/.../_ascend/ops/gemma_rms_norm.py`.

Call-site schema (every site in `vllm_fl`, verified by grep):
    npu_rms_norm(Tensor x, Tensor gamma, float epsilon) -> (Tensor y, Tensor rstd)
      vllm_fl/attention/ascend/sfa_v1.py:2005            `k_nope, _ = ...(rms_in, gamma, epsilon=eps)`
      vllm_fl/compilation/passes/norm_quant_fusion_pass.py:593,633
      vllm_fl/compilation/passes/qknorm_rope_fusion_pass.py:63,66,142,146
      vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py:1973  (gamma = 1.0 + weight, gemma)
      vllm_fl/dispatch/backends/vendor/ascend/impl/layernorm.py:98,165
      .../impl/minimax_m3_injections.py:101
      .../impl/normalization.py:40
      .../impl/linearnorm/minimax_m3_qknorm_rope.py:67

  * `x`    : [..., N], fp16 / bf16 / fp32 (the last dim is normalised).
  * `gamma`: [N], same dtype as `x` at every call site.
  * `y`    : same shape and dtype as `x`.
  * `rstd` : fp32, shape `x.shape[:-1] + (1,)`; it is the *reciprocal* RMS
             (1/sqrt(mean(x^2)+eps)), never a variance.
  * `epsilon` is positional-or-keyword at position 3.

Deviations from the C implementation, stated explicitly:
  * the AscendC kernel divides by sqrt; this port multiplies by
    1/sqrt (same rstd value, one fewer pass over the row);
  * no in-place write-back: the built-in returns fresh tensors and every call
    site binds the return value.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["npu_rms_norm"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _rms_norm_kernel(
    X, W, Y, RSTD,
    ROWS, N, EPS,
    HAS_GAMMA: tl.constexpr,
    OUT_FP32: tl.constexpr,
    BLK_N: tl.constexpr,
    NCH: tl.constexpr,
):
    # One program per row: the row base stays a scalar, so every access is a
    # plain 1-D masked vector (the Ascend backend miscompiles some 2-D masked
    # forms; see gemma_rms_norm.py).
    row = tl.program_id(0)
    if row >= ROWS:
        return
    base = row * N

    # Pass 1: sum(x^2)
    ss = tl.zeros([BLK_N], dtype=tl.float32)
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        m = cc < N
        xv = tl.load(X + base + cc, mask=m, other=0.0).to(tl.float32)
        ss += xv * xv
    rstd = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / N + EPS)
    tl.store(RSTD + row, rstd)

    # Pass 2: y = x * rstd * gamma
    for i in tl.static_range(NCH):
        cc = i * BLK_N + tl.arange(0, BLK_N)
        m = cc < N
        xv = tl.load(X + base + cc, mask=m, other=0.0).to(tl.float32)
        y = xv * rstd
        if HAS_GAMMA:
            w = tl.load(W + cc, mask=m, other=0.0).to(tl.float32)
            y = y * w
        if OUT_FP32:
            tl.store(Y + base + cc, y, mask=m)
        else:
            tl.store(Y + base + cc, y.to(Y.dtype.element_ty), mask=m)


def npu_rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor | None,
    epsilon: float = 1e-6,
):
    """Triton equivalent of `torch_npu.npu_rms_norm`.

    Args:
        x: [..., N] fp16 / bf16 / fp32; normalised over the last dimension.
        gamma: [N] weight, same dtype as x. `None` means "no weight" (the
            torch_npu schema declares the weight optional; no call site in this
            tree passes None, so that branch is unverified).
        epsilon: attribute, default 1e-6.

    Returns:
        (y, rstd): y has x's shape/dtype, rstd is fp32 with shape
        `x.shape[:-1] + (1,)` and holds 1/sqrt(mean(x^2) + epsilon).
    """
    if x.dim() < 1:
        raise ValueError(f"x must have at least 1 dim, got {tuple(x.shape)}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"x dtype must be fp16/bf16/fp32, got {x.dtype}")
    n = int(x.shape[-1])
    if n <= 0:
        raise ValueError(f"the normalised dim must be positive, got {n}")
    rows = 1
    for v in x.shape[:-1]:
        rows *= int(v)
    if gamma is not None:
        if gamma.numel() != n:
            raise ValueError(f"gamma numel {gamma.numel()} != N={n}")
        if gamma.dtype != x.dtype:
            raise ValueError(f"gamma dtype {gamma.dtype} must equal x dtype {x.dtype}")
        gamma = gamma.contiguous()

    xc = x.contiguous()
    y = torch.empty_like(xc)
    rstd = torch.empty(tuple(x.shape[:-1]) + (1,), dtype=torch.float32, device=x.device)
    if rows == 0:
        return y, rstd

    blk_n = min(_next_pow2(n), 1024)
    _rms_norm_kernel[(rows,)](
        xc.view(rows, n), gamma, y.view(rows, n), rstd.view(rows),
        rows, n, float(epsilon),
        HAS_GAMMA=gamma is not None,
        OUT_FP32=(xc.dtype == torch.float32),
        BLK_N=blk_n,
        NCH=_ceil_div(n, blk_n),
    )
    return y, rstd
