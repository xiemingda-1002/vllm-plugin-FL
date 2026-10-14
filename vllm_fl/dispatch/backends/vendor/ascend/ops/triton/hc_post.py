# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""hc_post -- AscendC to Triton.

Source: csrc/ascend/moe/hc_post/
Interface (op_host/hc_post_def.cpp L24-48 + hc_post_proto.cpp L29-59 +
hc_post_tiling.h L69-79):
    Input : x        [B, D]        fp16/bf16/fp32          (T1)
            residual [B, H, D]    fp16/bf16/fp32          (T1, same dtype as x)
            post     [B, H]       fp16/bf16/fp32          (T2, may differ from T1)
            comb     [B, H, H]    fp16/bf16/fp32          (T2, same dtype as post)
    Output: y        [B, H, D]    dtype = dtype of x (InferDtype L52-59)
    Attr  : none

    (in op_host/hc_post_tiling.cpp: hcParam = H, dParam = D, batchSize = B)

Formula (op_kernel/hc_post_d_split.h L299-359 DoCompute, 910B path;
         op_kernel/hc_post_float32.h L204-278 DoMulAndAdd, 950 regbase path;
         the two paths agree):

    910B: out[h,d] = post[h] * x[d]
          for k in range(H):  sumTemp[h,d] = comb[k,h] * residual[k,d]
                              out[h,d]    += sumTemp[h,d]
    950 : sumReg = residual[0]*comb[0,h] ; MulAddDst(..., residual[3]*comb[3,h])
                  += residual[1]*comb[1,h] ; += residual[2]*comb[2,h] ; += x*post[h]
    ==> y[b,h,d] = post[b,h] * x[b,d] + sum_k comb[b,k,h] * residual[b,k,d]
        (i.e. y = post[:,:,None]*x[:,None,:] + comb^T @ residual; note that comb is
        used transposed, indexed as [k,h])

Precision: accumulation is performed in fp32 (the kernel casts to fp32 before Mul/Add)
and the result is cast back to T1 with CAST_RINT.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["hc_post"]


@triton.jit
def _hc_post_kernel(
    X, RESIDUAL, POST, COMB, Y,
    D, H,
    BLK_H: tl.constexpr,
    BLK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    h = tl.arange(0, BLK_H)
    hmask = h < H
    d = pid_d * BLK_D + tl.arange(0, BLK_D)
    dmask = d < D

    post = tl.load(POST + pid_b * H + h, mask=hmask, other=0.0).to(tl.float32)      # [BLK_H]
    x = tl.load(X + pid_b * D + d, mask=dmask, other=0.0).to(tl.float32)            # [BLK_D]

    acc = post[:, None] * x[None, :]                                                # [BLK_H, BLK_D]

    base = pid_b * H * H
    rbase = pid_b * H * D
    for k in range(0, H):
        comb_k = tl.load(COMB + base + k * H + h, mask=hmask, other=0.0).to(tl.float32)
        res_k = tl.load(RESIDUAL + rbase + k * D + d, mask=dmask, other=0.0).to(tl.float32)
        acc += comb_k[:, None] * res_k[None, :]

    m = hmask[:, None] & dmask[None, :]
    tl.store(Y + rbase + h[:, None] * D + d[None, :],
             acc.to(Y.dtype.element_ty), mask=m)


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    dim: int | None = None,
) -> torch.Tensor:
    """Equivalent to the AscendC HcPost operator (hyper-connection post combination).

    Two equivalent layouts are supported (any number of leading dims; they are
    flattened to B internally):
        L0/op_host layout : x [B, D]        residual [B, H, D]       post [B, H]       comb [B, H, H]
        real call site    : x [b, s, d]     residual [b, s, hc, d]   post [b, s, hc]   comb [b, s, hc, hc]
        (the call sites vllm_ascend/models/deepseek_v41/model.py L866 /
          glm5next/model.py L523 / vllm_fl/models/deepseek_v4_ascend.py L975 all use
          `npu_hc_post(x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))`;
          the torch binding check_hc_post_shape_and_dtype (vllm-ascend csrc/torch_binding.cpp
          L1402-1428) requires x/post to be 3D and residual/comb to be 4D, and the output
          keeps the shape of residual, L1396)

    Args:
        x:        [..., D]      T1
        residual: [..., H, D]   T1
        post:     [..., H]      T2
        comb:     [..., H, H]   T2   (comb[..., k, h])
        dim:      **this operator has no `dim` parameter**. The `dim` reported by some
                  tooling is a call-site scanner false positive: the call site reads
                  `npu_hc_post(x.unsqueeze(dim=0), ...)`, so `dim=0` belongs to
                  `unsqueeze`. Both the binding schema (L3425-3432) and the op_host
                  definition carry only the four tensors. The keyword is accepted here
                  only to raise a clear error (dim=0 is a no-op, consistent with the
                  "all leading dims are flattened" semantics).

    Returns:
        y: shape of residual, [..., H, D], dtype = x.dtype
    """
    if dim is not None:
        assert int(dim) == 0, (
            f"dim={dim} is not supported: hc_post has no dim parameter (the binding schema "
            "only carries the four tensors x/residual/post/comb). The `dim` reported by "
            "tooling comes from the call site `x.unsqueeze(dim=0)` and is a scanner false "
            "positive. Only dim=0 (leading dim as batch) is a no-op and passes.")

    assert x.dim() >= 2, f"x must have at least 2 dims [..., D], got {tuple(x.shape)}"
    lead = tuple(int(v) for v in x.shape[:-1])
    D = int(x.shape[-1])
    assert residual.dim() == x.dim() + 1, \
        f"residual must be the leading dims of x plus (H,D), got x{tuple(x.shape)} residual{tuple(residual.shape)}"
    H = int(residual.shape[-2])
    assert tuple(int(v) for v in residual.shape) == lead + (H, D), \
        f"residual shape {tuple(residual.shape)} != {lead + (H, D)}"
    assert tuple(int(v) for v in post.shape) == lead + (H,), \
        f"post shape {tuple(post.shape)} != {lead + (H,)}"
    assert tuple(int(v) for v in comb.shape) == lead + (H, H), \
        f"comb shape {tuple(comb.shape)} != {lead + (H, H)}"

    B = 1
    for v in lead:
        B *= v
    xc = x.contiguous().view(B, D)
    rc = residual.contiguous().view(B, H, D)
    pc = post.contiguous().view(B, H)
    cc = comb.contiguous().view(B, H, H)
    y = torch.empty((B, H, D), dtype=xc.dtype, device=xc.device)

    BLK_H = max(2, next_pow2(H))
    # Bound the tile element count to avoid a bishengir ub overflow (UB = 192KB)
    BLK_D = min(next_pow2(D), max(16, 4096 // BLK_H))
    grid = (B, ceil_div(D, BLK_D))
    _hc_post_kernel[grid](
        xc, rc, pc, cc, y,
        D, H,
        BLK_H=BLK_H, BLK_D=BLK_D,
    )
    return y.view(*lead, H, D)
