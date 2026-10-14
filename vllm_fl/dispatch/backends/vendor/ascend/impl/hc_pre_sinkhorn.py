# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""hc_pre_sinkhorn -- AscendC to Triton

Source: csrc/ascend/moe/hc_pre_sinkhorn/
Interface (op_host/hc_pre_sinkhorn_def.cpp L24-67 + _proto.cpp L22-90 + _tiling.cpp L81-112):
    Input : mixes    [B, S, hc_mix] or [B*S, hc_mix]  fp32
            rsqrt    [B*S]                            fp32
            hc_scale [3]                              fp32
            hc_base  [hc_mix]                         fp32 (tiling L106-109: len == hc_mix)
            x        [B, S, hc_mult, d] or [B*S, hc_mult, d]  bf16
    Output: y        [B, S, d] / [B*S, d]            same dtype as x
            post     [B, S, hc_mult] / [B*S, hc_mult]        fp32
            comb_frag[B, S, hc_mult, hc_mult]                fp32
    Attr  : hc_mult = 4, hc_sinkhorn_iters = 20, hc_eps = 1e-6

    Layout: one row of mixes = [pre segment hc_mult | post segment hc_mult |
             comb segment hc_mult*hc_mult]
          => hc_mix = 2*hc_mult + hc_mult^2 (hc_base has length hc_mix as well)
          (the perf kernel reads the comb segment row by row with CopyInWithOuterFor and both
           interpretations agree for hm=1; here the row-major [hc_mult, hc_mult] block is used,
           which is consistent with infershape and with the hc_base length constraint)

Formula (op_kernel/hc_pre_sinkhorn_perf.h L95-202 Process + hc_pre_sinkhorn_base.h):
    hm = hc_mult, eps = hc_eps
    # 1) pre (the first hm entries of mixes):
    #    L120-121 ProcessPre -> L387-399: sigmoid(mix*rsqrt*scale[0] + base[0:hm]) + eps
    pre[i,j] = sigmoid(mixes[i, j] * rsqrt[i] * hc_scale[0] + hc_base[j]) + eps
    # 2) y (L132-141 ProcessY -> L462-472):
    y[i,:]   = sum_j pre[i,j] * x[i,j,:]                # fp32 accumulation, CAST_RINT back to bf16
    # 3) post (second segment of mixes; L145-154 -> L475-488 ProcessPost):
    post[i,j] = 2 * sigmoid(mixes[i, hm+j] * rsqrt[i] * hc_scale[1] + hc_base[hm+j])
    # 4) comb_frag (third segment of mixes, [hm,hm]; L158-192):
    A[i,j,k] = mixes[i, 2hm + j*hm + k] * rsqrt[i] * hc_scale[2] + hc_base[2hm + j*hm + k]
    A = softmax_over_k(A) + eps                          # L174 SoftmaxFP32Perf (last-dim softmax)
    A = A / (sum_j A[i,j,k] + eps)                       # L176-180 normalize over dim1
    for it in range(hc_sinkhorn_iters - 1):              # L181-192
        A = A / (sum_k A[i,j,k] + eps)                   #   L182-186 normalize over dim2
        A = A / (sum_j A[i,j,k] + eps)                   #   L187-191 normalize over dim1
    comb_frag = A

    - All intermediates are fp32; the y output is rounded back to the dtype of x
    - ReduceSumARAPerf (base.h L402-441) = sum over dim1; LastDimReduceSumPerf (L499-505) = sum
      over the last dim
    - Padding lanes do not take part in the sums (dim1/dim2 use the real hm); this implementation
      keeps that behaviour through masks
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["hc_pre_sinkhorn"]


@triton.jit
def _hc_pre_sinkhorn_kernel(
    MIX, RSQRT, SCALE, BASE, X, Y, POST, COMB,
    ROWS, HMIX, HM, D, EPS,
    ITERS: tl.constexpr,
    ITERS_P2: tl.constexpr,
    BLK_R: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_HM: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLK_R + tl.arange(0, BLK_R)
    rmask = rows < ROWS

    rs = tl.load(RSQRT + rows, mask=rmask, other=1.0)
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)

    jj = tl.arange(0, BLK_HM)
    jmask = jj < HM
    m2d = rmask[:, None] & jmask[None, :]

    # ---------- 1) pre + 2) y = sum_j pre[i,j] * x[i,j,:] ----------
    # pre[i,j] = sigmoid(mixes[i,j]*rsqrt[i]*scale[0] + base[j]) + eps
    dcur = 0
    while dcur < D:
        doff = dcur + tl.arange(0, BLK_D)
        dmask = doff < D
        acc = tl.zeros((BLK_R, BLK_D), dtype=tl.float32)
        j = 0
        while j < HM:
            mj = tl.load(MIX + rows * HMIX + j, mask=rmask, other=0.0)
            bj = tl.load(BASE + j)
            pj = 1.0 / (1.0 + tl.exp(-(mj * rs * s0 + bj))) + EPS
            xv = tl.load(
                X + (rows[:, None] * HM + j) * D + doff[None, :],
                mask=rmask[:, None] & dmask[None, :], other=0.0,
            ).to(tl.float32)
            acc += pj[:, None] * xv
            j += 1
        tl.store(Y + rows[:, None] * D + doff[None, :], acc.to(Y.dtype.element_ty),
                 mask=rmask[:, None] & dmask[None, :])
        dcur += BLK_D

    # ---------- 3) post ----------
    b1 = tl.load(BASE + HM + jj, mask=jmask, other=0.0)
    m1 = tl.load(MIX + rows[:, None] * HMIX + (HM + jj)[None, :], mask=m2d, other=0.0)
    postv = 2.0 / (1.0 + tl.exp(-(m1 * rs[:, None] * s1 + b1[None, :])))
    tl.store(POST + rows[:, None] * HM + jj[None, :], postv, mask=m2d)

    # ---------- 4) comb_frag (sinkhorn) ----------
    m3 = rmask[:, None, None] & jmask[None, :, None] & jmask[None, None, :]
    off3 = (2 * HM + jj[None, :, None] * HM + jj[None, None, :])
    m2 = tl.load(MIX + rows[:, None, None] * HMIX + off3, mask=m3, other=0.0)
    b2 = tl.load(BASE + 2 * HM + jj[:, None] * HM + jj[None, :],
                 mask=(jj[:, None] < HM) & (jj[None, :] < HM), other=0.0)
    a = m2 * rs[:, None, None] * s2 + b2[None, :, :]
    a = tl.where(m3, a, -1.0e30)

    mx = tl.max(a, axis=2)                                   # [BLK_R, BLK_HM]
    e = tl.exp(a - mx[:, :, None])
    e = tl.where(m3, e, 0.0)
    ssum = tl.sum(e, axis=2)                                 # softmax denominator (last dim)
    A = e / ssum[:, :, None] + EPS

    A = A / (tl.sum(tl.where(jmask[None, :, None], A, 0.0), axis=1)[:, None, :] + EPS)

    # Static unrolling: the Ascend backend fails to compile a "dynamic loop carrying a 3-D
    # tensor plus a masked reduce".  The unroll count must be a **power of two**
    # (`ITERS_P2 = next_pow2(ITERS-1)`); surplus iterations are skipped by
    # `if _it < ITERS - 1:`.  Same-family evidence: the `hc_pre` mixes kernel shows an error of
    # 5.388 when the unroll count is 28 (not a power of two), whereas 20/24/32 all give 4e-7.
    # sinkhorn defaults to ITERS=20, i.e. 19 iterations (not a power of two) originally.
    for _it in range(ITERS_P2):
        if _it < ITERS - 1:
            A = A / (tl.sum(tl.where(jmask[None, None, :], A, 0.0), axis=2)[:, :, None] + EPS)
            A = A / (tl.sum(tl.where(jmask[None, :, None], A, 0.0), axis=1)[:, None, :] + EPS)

    tl.store(COMB + rows[:, None, None] * (HM * HM) + jj[None, :, None] * HM + jj[None, None, :],
             A, mask=m3)


def hc_pre_sinkhorn(
    mixes: torch.Tensor,
    rsqrt: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    x: torch.Tensor,
    hc_mult: int = 4,
    hc_sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equivalent to the AscendC HcPreSinkhorn.

    Args:
        mixes: [B, S, 3*hc_mult] or [B*S, 3*hc_mult] fp32
        rsqrt: [B*S] fp32
        hc_scale: [3] fp32
        hc_base: [3*hc_mult + hc_mult*hc_mult] fp32
        x: [B, S, hc_mult, d] or [B*S, hc_mult, d] bf16
        attributes: hc_mult / hc_sinkhorn_iters / hc_eps

    Returns:
        (y with the dtype of x [B,S,d], post fp32 [B,S,hc_mult],
         comb_frag fp32 [B,S,hc_mult,hc_mult])
    """
    assert mixes.dtype == torch.float32, f"mixes must be fp32, got {mixes.dtype}"
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"

    hm = int(hc_mult)
    lead = tuple(int(v) for v in mixes.shape[:-1])
    hmix = int(mixes.shape[-1])
    want_mix = 2 * hm + hm * hm
    assert hmix == want_mix, f"mixes last dim {hmix} != 2*hc_mult + hc_mult^2 = {want_mix}"
    assert x.dim() == len(lead) + 2, "x dim does not match mixes"
    assert tuple(int(v) for v in x.shape[:-2]) == lead, \
        f"x leading dims {tuple(x.shape[:-2])} != mixes leading dims {lead}"
    assert int(x.shape[-2]) == hm, f"x hc dim {x.shape[-2]} != hc_mult={hm}"
    d = int(x.shape[-1])
    rows = 1
    for v in lead:
        rows *= v

    assert int(rsqrt.numel()) == rows, f"rsqrt length {rsqrt.numel()} != rows={rows}"
    assert int(hc_scale.numel()) == 3, f"hc_scale length {hc_scale.numel()} != 3"
    assert int(hc_base.numel()) == hmix, \
        f"hc_base length {hc_base.numel()} != hc_mix={hmix} (tiling L106-109 requires equality)"

    mix = mixes.contiguous().view(rows, hmix)
    rq = rsqrt.contiguous().view(rows).to(torch.float32)
    sc = hc_scale.contiguous().view(3).to(torch.float32)
    bs = hc_base.contiguous().view(-1).to(torch.float32)
    xv = x.contiguous().view(rows, hm, d)

    y = torch.empty((rows, d), dtype=x.dtype, device=x.device)
    post = torch.empty((rows, hm), dtype=torch.float32, device=x.device)
    comb = torch.empty((rows, hm, hm), dtype=torch.float32, device=x.device)

    BLK_R = 4
    BLK_D = min(next_pow2(d), 256)
    BLK_HM = max(2, next_pow2(hm))
    iters = int(hc_sinkhorn_iters)
    grid = (ceil_div(rows, BLK_R),)
    _hc_pre_sinkhorn_kernel[grid](
        mix, rq, sc, bs, xv, y, post, comb,
        rows, hmix, hm, d, float(hc_eps),
        ITERS=iters,
        ITERS_P2=next_pow2(max(1, iters - 1)),
        BLK_R=BLK_R, BLK_D=BLK_D, BLK_HM=BLK_HM,
    )
    return (y.view(*lead, d), post.view(*lead, hm), comb.view(*lead, hm, hm))
