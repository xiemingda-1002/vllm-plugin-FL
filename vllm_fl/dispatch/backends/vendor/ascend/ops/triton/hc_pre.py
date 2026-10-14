# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""hc_pre -- AscendC to Triton

Source: csrc/ascend/moe/hc_pre/
Interface (op_host/hc_pre_def.cpp L24-63):
    Input : x        [bs, hc_mult, d] or [b, s, hc_mult, d]   bf16
            hc_fn    [hc_mix, hc_mult*d]                      fp32
            hc_scale [3]                                      fp32
            hc_base  [hc_mix]                                 fp32
    Output: y         [bs, d]                        bf16
            post      [bs, hc_mult]                  fp32
            comb_frag [bs, hc_mult, hc_mult]         fp32
    Attr  : hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-6
    hc_mix = 2*hc_mult + hc_mult^2 (tiling.cpp L116-132 constrains hc_fn.dim1 == d*hc_mult and
    the hc_base length == hc_mix; L278 k = hcMult*d)

**Relation to hc_pre_sinkhorn / hc_pre_inv_rms** (independent implementations, not imported):
    hc_pre = hc_pre_inv_rms(x; norm_eps)          -> rsqrt[bs]
           + cube(x @ hc_fn^T)                    -> mixes[bs, hc_mix]
           + hc_pre_sinkhorn(mixes, rsqrt, hc_scale, hc_base, x; hc_eps, iters)
    Rationale: the cube computes mixes (hc_pre_cube_compute.h, k=hcMult*d); the vector side first
    computes inv_rms (hc_pre_m_k_split_core.h L366-368: Adds(normEps) -> Sqrt) and then
    multiplies rsqrt into mixes (hc_pre_base.h L428-439 ProcessPre / L519-527 ProcessPost /
    the comb segment).

Formula (op_kernel/hc_pre_base.h):
    K = hc_mult * d
    rsqrt[i]    = 1 / sqrt( sum_k x[i,k]^2 / K + norm_eps )
    mixes[i, n] = sum_k x_flat[i,k] * hc_fn[n,k]
    pre[i,j]    = sigmoid( mixes[i,j]    * rsqrt[i] * scale[0] + base[j] ) + hc_eps
    y[i,:]      = sum_j pre[i,j] * x[i,j,:] (fp32 accumulation, written back as bf16)
    post[i,j]   = 2 * sigmoid( mixes[i,hm+j] * rsqrt[i] * scale[1] + base[hm+j] )
    A[i,j,k]    = mixes[i,2hm+j*hm+k] * rsqrt[i] * scale[2] + base[2hm+j*hm+k]
    A           = softmax_k(A) + hc_eps ;  A = A / (sum_j A + hc_eps)
    iters-1 times: A = A/(sum_k A + hc_eps) ; A = A/(sum_j A + hc_eps)
    comb_frag   = A

Known Ascend backend issues (this file is designed around them):
  1. `tl.dot` crashes outright (BiShengIR/LLVM crash, both fp32 and bf16) -> use 1-D broadcast
     reductions instead.
  2. A **dynamic `while` loop combined with multi-row 2-D offsets is computed incorrectly**
     (observed: 15 of 16 rows wrong and only row 0 correct; replacing the loop with
     `tl.static_range` gave 0 errors) -> every loop in this file uses static_range, with the trip
     count computed on the host from the shape and passed in as a constexpr.
  3. 2-D masks (`rmask[:,None] & km[None,:]`) are unreliable -> host-side zero padding removes
     the masks.
  4. Loop variables are not used as pointer offsets inside a loop (the column index is passed
     through program_id(1)).

Semantic assumptions:
  * hc_mult must be a power of two (the comb matrix has to be dense, to avoid 2-D masks).
  * y is written back as bf16 (round-to-nearest); post/comb_frag stay fp32.
  * 4-D inputs are flattened over the first two dims and treated as bs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["hc_pre"]


@triton.jit
def _hc_pre_rsqrt_kernel(
    X, RSQRT,
    BS, K, KS,
    NORM_EPS,
    BLK_M: tl.constexpr,
    BLK_K: tl.constexpr,
    KCH: tl.constexpr,
):
    """inv_rms: rsqrt[i] = 1/sqrt(sum_k x^2 / K + norm_eps).

    X is zero padded (row stride KS=KCH*BLK_K).
    """
    pid = tl.program_id(0)
    rows = pid * BLK_M + tl.arange(0, BLK_M)
    ss = tl.zeros([BLK_M], dtype=tl.float32)
    for i in tl.static_range(KCH):
        kk = i * BLK_K + tl.arange(0, BLK_K)
        xv = tl.load(X + rows[:, None] * KS + kk[None, :]).to(tl.float32)
        ss += tl.sum(xv * xv, axis=1)
    tl.store(RSQRT + rows, 1.0 / tl.sqrt(ss / K + NORM_EPS), mask=rows < BS)


@triton.jit
def _hc_pre_mixes_kernel(
    X, FN, MIXES,
    BS, K, KS, HMIX,
    BLK_M: tl.constexpr,
    BLK_K: tl.constexpr,
    KCH: tl.constexpr,
    KCH_P2: tl.constexpr,
):
    """mixes[:, n] = sum_k x_flat[:, k] * hc_fn[n, k] (program_id(1) provides the column index n).

    Both X and FN use the row stride KS = KCH*BLK_K (zero padded) -- **KS must be used, not K**,
    otherwise the last block reads across rows into neighbouring data (all wrong whenever d is
    not a multiple of BLK_K).

    The unroll count must be a **power of two** (`KCH_P2 = next_pow2(KCH)`; iterations beyond
    `KCH` are skipped): the Ascend backend computes the wrong result when a static unroll count
    is not a power of two -- for the same kernel, `hm*d = 4*7168 = 28672` gives a **mixes error
    of 5.388** at `KCH=28` (which also corrupts y/post/comb directly), whereas
    `KCH in {20, 24, 32}` (d=5120/6144/8192) all give 4e-7.  This is the root cause of the
    failures at production `d=7168` while `d=4096` was fine.
    """
    n = tl.program_id(1)
    rows = tl.program_id(0) * BLK_M + tl.arange(0, BLK_M)
    accn = tl.zeros([BLK_M], dtype=tl.float32)
    for i in range(KCH_P2):
        if i < KCH:
            kk = i * BLK_K + tl.arange(0, BLK_K)
            a = tl.load(X + rows[:, None] * KS + kk[None, :]).to(tl.float32)
            bv = tl.load(FN + n * KS + kk)
            accn += tl.sum(a * bv[None, :], axis=1)
    tl.store(MIXES + rows * HMIX + n, accn, mask=rows < BS)


@triton.jit
def _hc_pre_sinkhorn_v2_kernel(
    MIXES, RSQRT, SCALE, BASE, X, Y, POST, COMB,
    BS, HMIX, HC_EPS,
    HM: tl.constexpr,
    ITERS: tl.constexpr,
    BLK_D: tl.constexpr,
    DCH: tl.constexpr,
):
    # the d dim is zero padded on the host to DCH*BLK_D -> no masks inside the kernel
    row = tl.program_id(0)
    if row >= BS:
        return
    rq = tl.load(RSQRT + row)
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    jj = tl.arange(0, HM)
    D = DCH * BLK_D

    # --- pre and y = sum_j pre[j] * x[row,j,:] ---
    for i in tl.static_range(DCH):
        dd = i * BLK_D + tl.arange(0, BLK_D)
        yacc = tl.zeros([BLK_D], dtype=tl.float32)
        for j in tl.static_range(HM):
            mj = tl.load(MIXES + row * HMIX + j)
            bj = tl.load(BASE + j)
            pj = 1.0 / (1.0 + tl.exp(0.0 - (mj * rq * s0 + bj))) + HC_EPS
            xv = tl.load(X + (row * HM + j) * D + dd).to(tl.float32)
            yacc += pj * xv
        tl.store(Y + row * D + dd, yacc.to(Y.dtype.element_ty))

    # --- post = 2*sigmoid(mix*rsqrt*scale1 + base1) ---
    b1 = tl.load(BASE + HM + jj)
    m1 = tl.load(MIXES + row * HMIX + (HM + jj))
    post = 2.0 / (1.0 + tl.exp(0.0 - (m1 * rq * s1 + b1)))
    tl.store(POST + row * HM + jj, post)

    # --- comb_frag: alternate row/column normalization after softmax (Sinkhorn) ---
    a = tl.load(MIXES + row * HMIX + (2 * HM + jj[:, None] * HM + jj[None, :]))
    b2 = tl.load(BASE + 2 * HM + jj[:, None] * HM + jj[None, :])
    a = a * rq * s2 + b2
    e = tl.exp(a - tl.max(a, axis=1)[:, None])
    A = e / tl.sum(e, axis=1)[:, None] + HC_EPS
    A = A / (tl.sum(A, axis=0)[None, :] + HC_EPS)
    for _ in tl.static_range(ITERS - 1):
        A = A / (tl.sum(A, axis=1)[:, None] + HC_EPS)
        A = A / (tl.sum(A, axis=0)[None, :] + HC_EPS)
    tl.store(COMB + row * (HM * HM) + jj[:, None] * HM + jj[None, :], A)


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    hc_sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
):
    """Equivalent to the AscendC HcPre.

    Args:
        x: [bs, hc_mult, d] bf16 (or [b, s, hc_mult, d], with the first two dims flattened)
        hc_fn: [hc_mix, hc_mult*d] fp32
        hc_scale: [3] fp32
        hc_base: [hc_mix] fp32
        hc_mult / hc_sinkhorn_iters / hc_eps / norm_eps: attributes

    Returns:
        (y [bs,d] bf16, post [bs,hc_mult] fp32, comb_frag [bs,hc_mult,hc_mult] fp32)
    """
    assert x.dim() in (3, 4), f"x must be 3-D/4-D, got {tuple(x.shape)}"
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
    hm = int(hc_mult)
    assert hm in (1, 2, 4, 8, 16), \
        f"hc_mult={hm}: only powers of two are supported (comb must be dense to avoid 2-D masks)"
    lead = tuple(int(v) for v in x.shape[:-2])
    bs = 1
    for v in lead:
        bs *= v
    d = int(x.shape[-1])
    assert int(x.shape[-2]) == hm, f"x hc dim {x.shape[-2]} != hc_mult={hm}"
    hmix = 2 * hm + hm * hm
    assert tuple(hc_fn.shape) == (hmix, hm * d), \
        f"hc_fn must be [{hmix},{hm * d}], got {tuple(hc_fn.shape)}"
    assert hc_scale.numel() == 3 and hc_base.numel() == hmix
    K = hm * d

    dev = x.device
    xc = x.contiguous().view(bs, hm, d)
    fn = hc_fn.contiguous().float()
    mixes = torch.empty((bs, hmix), dtype=torch.float32, device=dev)
    rsqrt = torch.empty((bs,), dtype=torch.float32, device=dev)
    y = torch.empty((bs, d), dtype=x.dtype, device=dev)
    post = torch.empty((bs, hm), dtype=torch.float32, device=dev)
    comb = torch.empty((bs, hm, hm), dtype=torch.float32, device=dev)

    blk_m = 16
    blk_k = 64
    # `_hc_pre_rsqrt_kernel` / `_hc_pre_mixes_kernel` unroll the K loop at compile
    # time (`tl.static_range(KCH)`), so KCH drives the build time directly.  DeepSeek-V4
    # uses hm=4, d=4096 -> K=16384, i.e. KCH=256 with BLK_K=64, which made the Ascend
    # compiler run for tens of minutes on the first call.  Grow BLK_K until the loop
    # needs at most 32 blocks; small shapes (the tested ones) keep BLK_K=64 untouched.
    while ceil_div(K, blk_k) > 32:
        blk_k *= 2
    kch = ceil_div(K, blk_k)
    # host-side zero padding of the rows and of K (removes the 2-D masks inside the kernel);
    # the trip count kch is passed in as a constexpr
    bs_pad = ceil_div(bs, blk_m) * blk_m
    k_pad = kch * blk_k
    if bs_pad != bs or k_pad != K:
        xpad = torch.zeros((bs_pad, k_pad), dtype=x.dtype, device=dev)
        xpad[:bs, :K] = xc.view(bs, K)
        fnpad = torch.zeros((hmix, k_pad), dtype=torch.float32, device=dev)
        fnpad[:, :K] = fn
    else:
        xpad, fnpad = xc.view(bs, K), fn
    _hc_pre_rsqrt_kernel[(bs_pad // blk_m,)](
        xpad, rsqrt, bs, K, k_pad, float(norm_eps),
        BLK_M=blk_m, BLK_K=blk_k, KCH=kch,
    )
    _hc_pre_mixes_kernel[(bs_pad // blk_m, hmix)](
        xpad, fnpad, mixes, bs, K, k_pad, hmix,
        BLK_M=blk_m, BLK_K=blk_k, KCH=kch, KCH_P2=next_pow2(kch),
    )

    # Ascend UB=192KB: one row per program + d blocking; the d dim is zero padded on the host to
    # DCH*BLK_D (no masks inside the kernel)
    blk_d = 128 if d > 128 else max(2, next_pow2(d))
    # Same reasoning as BLK_K above: the d loop and its inner HM loop are unrolled,
    # so cap DCH at 8 blocks (=32 unrolled iterations for hm=4).
    while ceil_div(d, blk_d) > 8:
        blk_d *= 2
    dch = ceil_div(d, blk_d)
    d_pad = dch * blk_d
    if d_pad != d:
        xpad2 = torch.zeros((bs, hm, d_pad), dtype=x.dtype, device=dev)
        xpad2[:, :, :d] = xc
        ypad = torch.empty((bs, d_pad), dtype=x.dtype, device=dev)
    else:
        xpad2, ypad = xc, y
    _hc_pre_sinkhorn_v2_kernel[(bs,)](
        mixes, rsqrt, hc_scale.contiguous().view(3), hc_base.contiguous().view(-1),
        xpad2, ypad, post, comb,
        bs, hmix, float(hc_eps),
        HM=hm,
        ITERS=int(hc_sinkhorn_iters),
        BLK_D=blk_d,
        DCH=dch,
    )
    if d_pad != d:
        y[:, :] = ypad[:, :d].reshape(bs, d)
    return (y.view(*lead, d), post.view(*lead, hm), comb.view(*lead, hm, hm))
