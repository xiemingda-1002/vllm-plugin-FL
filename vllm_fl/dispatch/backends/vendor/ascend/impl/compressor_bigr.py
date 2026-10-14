# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""`coff=1` + `cmp_ratio > 16` (long window) path of compressor -- row-wise blocked online
softmax.

================================================================================
Why this module exists
================================================================================
`compressor_tiling.cpp:958-964` explicitly allows three combinations:
    {r=4,   coff=2, D=512}
    {r=4,   coff=2, D=128}
    {r=128, coff=1, D=512}     <- this module
On the model side, `vllm_ascend/models/deepseek_v4/model.py:548` uses
`self.compress_ratio = get_dsv4_compress_ratio(config, config_layer_idx)`, and the
`compress_ratios` of the config alternate between `4 / 128` layer by layer (see
`tests/ut/test_platform.py:40  compress_ratios=[0, 0] + [4, 128] * 20 + [4]`)
=> **r=128 is a production configuration, not a "legal but unused" one**.

The coff=1 kernel of the original `compressor.py` keeps **a whole block of r rows** in
registers (two `[M, BLK_D]` fp32 accumulators, `M=16` fixed); with r=128 and D=512 that is
`128*512*4 = 256KB each`, far beyond the 192KB UB => **row-wise blocking is required**.

================================================================================
Semantics (identical to coff=1, only the implementation differs)
================================================================================
For an absolute chunk `c` (covering absolute positions `[c*r, (c+1)*r)`):

    kv_j[d]    = (x[p_j] @ wkv^T)[d]
    score_j[d] = (x[p_j] @ wgate^T)[d] + ape[j % r, d]
    o[d]       = sum_j softmax_j(score[:, d]) * kv_j[d]     # softmax column-wise over r rows
    y          = RMSNorm_D(o) * norm_weight
    out        = RoPE(y)                                    # row base = output chunk row index
    p_j = c*r + j; rows with p_j < sp are read back from state_cache; rows with p_j < 0 do
    not exist.

**online / streaming softmax** (mathematically equivalent to a one-shot softmax; only the
floating-point association order differs):

    run_max = -big ; run_sum = 0 ; run_acc = 0
    for each row-block of BM rows:
        blk_max = max(score_blk, axis=0)
        new_max = max(run_max, blk_max)
        alpha   = exp(run_max - new_max)                    # first block: run_max=-big => 0
        e       = where(good, exp(score_blk - new_max), 0)  # invalid rows are explicitly 0
        run_acc = run_acc * alpha + sum(e * kv_blk, axis=0)
        run_sum = run_sum * alpha + sum(e,          axis=0)
        run_max = new_max
    o = run_acc / run_sum

WARNING: **two easy-to-miss points** (handled explicitly here):
1. `exp(run_max - new_max)` for `alpha`: on the first block `run_max = -1e30` and `new_max` is
   finite => `exp(-1e30) = 0` (correct); if the whole block is invalid then `new_max = -1e30`
   and `alpha = exp(0) = 1`, but then `e == 0` => `run_acc/run_sum` are unchanged (correct).
2. `e` must be explicitly zeroed with `tl.where(good, ..., 0.0)` -- otherwise, for an invalid
   row where `score = -1e30` equals `new_max = -1e30`, `exp(0) = 1` is produced, which inflates
   `run_sum` without contributing to `run_acc`, and **the result becomes wrong**.

================================================================================
Backend constraints
================================================================================
* Store indices are `tl.arange` (unit stride) or "scalar + constant".
* **Do not use `tl.reshape` / `tl.join`**; RoPE reuses the already-verified per-column scalar
  expansion.
* `BLK_H` is adapted to keep UB usage in check: two `[BM, BLK_D]` fp32 accumulators plus two
  weight tiles.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["compressor_bigr"]

_MIN_SCORE = -1.0e30


@triton.jit
def _compressor_bigr_kernel(
    X, WKV, WGATE, APE, NORMW, RSIN, RCOS, STATE, BLKT, CMPKV,
    X_BASE, SP, SU, C0, NC, OB,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr, BM: tl.constexpr, HALF: tl.constexpr,
    EPS: tl.constexpr, MIN_SCORE: tl.constexpr,
    HALF_MODE: tl.constexpr, HAS_STATE: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    b = tl.program_id(0)
    ct = tl.program_id(1)

    sp = tl.load(SP + b)
    su = tl.load(SU + b)
    c0 = tl.load(C0 + b)
    nc = tl.load(NC + b)
    xbase = tl.load(X_BASE + b)
    obase = tl.load(OB + b)

    ok = ct < nc
    c = c0 + ct
    sc = ct

    d = tl.arange(0, BLK_D)
    dmask = d < D
    h = tl.arange(0, BLK_H)
    m = tl.arange(0, BM)

    run_max = tl.zeros((BLK_D,), dtype=tl.float32) + MIN_SCORE
    run_sum = tl.zeros((BLK_D,), dtype=tl.float32)
    run_acc = tl.zeros((BLK_D,), dtype=tl.float32)

    for m0 in range(0, R, BM):
        mm = m0 + m
        in_chunk = mm < R
        # WARNING: the in-window index is **reversed** (REV): the column-wise softmax and the
        #    weighted sum are insensitive to row order => the semantics are exactly equivalent,
        #    but the history (head) rows at the front of the chunk go from a tile **prefix**
        #    to a **suffix** (when a masked lane lands in the prefix this backend returns wrong
        #    values that drift with the allocation).
        #    Rows where in_chunk is false keep their original p_abs (no new addresses).
        jwin = tl.where(in_chunk, R - 1 - mm, mm)
        p_abs = c * R + jwin
        real = in_chunk & (p_abs >= sp) & (p_abs < sp + su) & ok
        head = in_chunk & (p_abs >= 0) & (p_abs < sp) & ok
        have = real | head
        tok = p_abs - sp
        xrow = xbase + tl.where(tok >= 0, tok, 0)

        acc_kv = tl.zeros((BM, BLK_D), dtype=tl.float32)
        acc_sc = tl.zeros((BM, BLK_D), dtype=tl.float32)
        if HAS_STATE:
            # WARNING: this must be written as "initialize the accumulators first, then let
            #    the dots accumulate into them": BiSheng does not support a select of the form
            #    `tl.where(mask, <vector load>, <dot result>)` (it reports "Unsupported copy
            #    from cc to cbuf"). real and head are mutually exclusive => equivalent.
            p_safe = tl.where(p_abs >= 0, p_abs, 0)
            blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_safe // BLK_SIZE))
            blk_id = tl.load(BLKT + blk_idx, mask=head, other=0)
            soff = blk_id[:, None] * STRIDE0 + (p_safe % BLK_SIZE)[:, None] * (2 * D) + d[None, :]
            sm = head[:, None] & dmask[None, :]
            acc_kv = tl.load(STATE + soff, mask=sm, other=0.0)
            acc_sc = tl.load(STATE + soff + D, mask=sm, other=0.0)

        for h0 in range(0, H, BLK_H):
            hh = h0 + h
            hmask = hh < H
            xm = real[:, None] & hmask[None, :]
            xv = tl.load(X + xrow[:, None] * H + hh[None, :], mask=xm, other=0.0)
            wm = hmask[:, None] & dmask[None, :]
            acc_kv += tl.dot(xv, tl.load(WKV + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
            acc_sc += tl.dot(xv, tl.load(WGATE + d[None, :] * H + hh[:, None], mask=wm, other=0.0))

        # after reversing the row order this must come from p_abs (the two are equivalent
        # before reversing)
        j = p_abs % R
        # equivalent to tl.where(real, acc_sc + apev, acc_sc), but avoids a select on the cube
        # result
        apev = tl.load(APE + j[:, None] * D + d[None, :],
                       mask=real[:, None] & dmask[None, :], other=0.0)
        acc_sc = acc_sc + apev

        good = have[:, None] & dmask[None, :] & ok
        s = tl.where(good, acc_sc, MIN_SCORE)
        blk_max = tl.max(s, axis=0)
        new_max = tl.maximum(run_max, blk_max)
        alpha = tl.exp(run_max - new_max)
        e = tl.where(good, tl.exp(s - new_max[None, :]), 0.0)
        run_acc = run_acc * alpha + tl.sum(e * acc_kv, axis=0)
        run_sum = run_sum * alpha + tl.sum(e, axis=0)
        run_max = new_max

    o = run_acc / tl.where(run_sum > 0.0, run_sum, 1.0)

    # --- RMSNorm over D ---
    o2 = tl.where(dmask, o, 0.0)
    ss = tl.sum(o2 * o2, axis=0)
    inv = 1.0 / tl.sqrt(ss * (1.0 / D) + EPS)
    nw = tl.load(NORMW + d, mask=dmask, other=0.0)
    y = o2 * inv * nw

    orow = obase + sc
    tl.store(CMPKV + orow * D + d, y.to(CMPKV.dtype.element_ty),
             mask=ok & dmask & (d < D - RD))

    ROFF: tl.constexpr = D - RD
    if HALF_MODE:
        hr: tl.constexpr = RD // 2
        for i in range(0, hr):
            va = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vb = tl.sum(tl.where(d == ROFF + hr + i, y, 0.0), axis=0)
            ca = tl.load(RCOS + orow * RD + i)
            sa = tl.load(RSIN + orow * RD + i)
            cb = tl.load(RCOS + orow * RD + hr + i)
            sb = tl.load(RSIN + orow * RD + hr + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ca * va - sa * vb).to(CMPKV.dtype.element_ty),
                     mask=ok)
            tl.store(CMPKV + orow * D + ROFF + hr + i,
                     (sb * va + cb * vb).to(CMPKV.dtype.element_ty), mask=ok)
    else:
        for i in range(0, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i + 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi - si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)
        for i in range(1, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i - 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi + si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)


@triton.jit
def _compressor_bigr_state_kernel(
    X, WKV, WGATE, APE, STATE, BLKT,
    X_BASE, SP, WLO, WHI,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D_ST: tl.constexpr,
    P: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)

    sp = tl.load(SP + b)
    wlo = tl.load(WLO + b)
    whi = tl.load(WHI + b)
    xbase = tl.load(X_BASE + b)

    rows = t * P + tl.arange(0, P)
    p_abs = wlo + rows
    ok = p_abs < whi
    tok = p_abs - sp

    h = tl.arange(0, BLK_H)
    p_safe = tl.where(p_abs >= 0, p_abs, 0)
    blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_safe // BLK_SIZE))
    blk_id = tl.load(BLKT + blk_idx, mask=ok, other=0)
    base = blk_id[:, None] * STRIDE0 + (p_safe % BLK_SIZE)[:, None] * (2 * D)
    j = p_abs % R

    for d0 in range(0, D, BLK_D_ST):
        d = d0 + tl.arange(0, BLK_D_ST)
        dmask = d < D
        sm = ok[:, None] & dmask[None, :]
        kv = tl.zeros((P, BLK_D_ST), dtype=tl.float32)
        scc = tl.zeros((P, BLK_D_ST), dtype=tl.float32)
        for h0 in range(0, H, BLK_H):
            hh = h0 + h
            hmask = hh < H
            xm = ok[:, None] & hmask[None, :]
            xv = tl.load(X + (xbase + tl.where(tok >= 0, tok, 0))[:, None] * H + hh[None, :],
                         mask=xm, other=0.0)
            wm = hmask[:, None] & dmask[None, :]
            kv += tl.dot(xv, tl.load(WKV + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
            scc += tl.dot(xv, tl.load(WGATE + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
        scc = scc + tl.load(APE + j[:, None] * D + d[None, :], mask=sm, other=0.0)
        tl.store(STATE + base + d[None, :], kv, mask=sm)
        tl.store(STATE + base + D + d[None, :], scc, mask=sm)


def _as_int32(t, device):
    if t is None:
        return None
    return t.to(device=device, dtype=torch.int32).contiguous()


# ===========================================================================
# Three-stage split (the structure follows the upstream vllm-ascend Triton
#          implementation `ops/triton/compressor/compressor_triton.py`:
#          `_proj_kernel` / `_pool_kernel` / `_cache_update_kernel`)
#
# Motivation: combining `tl.dot` + "load(block table) -> form addresses -> load again" + an
#       unrolled row-block loop in a single kernel makes `bishengir-compile` SIGSEGV.
#       The key of the upstream form is that **`tl.dot` appears only in `_proj_kernel`; the
#       pooling kernel is pure vector code**, where even the "inside/outside the segment"
#       choice is `tl.where(is_seg, <vector load>, <vector load>)` -- both operands are vector
#       loads, so it naturally avoids the `cc -> cbuf` defect.
# ===========================================================================
@triton.jit
def _bigr_proj_kernel(
    X, WKV, WGATE, APE, KVWS, SCWS, X_BASE, SP, SU,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr, PT: tl.constexpr,
):
    """K1: pure projection + ape. `kv = x@wkv^T`, `score = x@wgate^T + ape[p % r]` -> fp32
    workspace."""
    b = tl.program_id(0)
    t0 = tl.program_id(1) * PT
    sp = tl.load(SP + b)
    su = tl.load(SU + b)
    xbase = tl.load(X_BASE + b)

    t = t0 + tl.arange(0, PT)
    ok = t < su
    p = sp + t
    d = tl.arange(0, BLK_D)
    dmask = d < D
    h = tl.arange(0, BLK_H)

    kv = tl.zeros((PT, BLK_D), dtype=tl.float32)
    sc = tl.zeros((PT, BLK_D), dtype=tl.float32)
    for h0 in range(0, H, BLK_H):
        hh = h0 + h
        hmask = hh < H
        xm = ok[:, None] & hmask[None, :]
        xv = tl.load(X + (xbase + t)[:, None] * H + hh[None, :], mask=xm, other=0.0)
        wm = hmask[:, None] & dmask[None, :]
        kv += tl.dot(xv, tl.load(WKV + d[None, :] * H + hh[:, None], mask=wm, other=0.0))
        sc += tl.dot(xv, tl.load(WGATE + d[None, :] * H + hh[:, None], mask=wm, other=0.0))

    apev = tl.load(APE + (p % R)[:, None] * D + d[None, :],
                   mask=ok[:, None] & dmask[None, :], other=0.0)
    sc = sc + apev

    row = (xbase + t)[:, None] * D + d[None, :]
    sm = ok[:, None] & dmask[None, :]
    tl.store(KVWS + row, kv, mask=sm)
    tl.store(SCWS + row, sc, mask=sm)


@triton.jit
def _bigr_pool_kernel(
    KVWS, SCWS, STATE, BLKT, NORMW, RSIN, RCOS, CMPKV,
    X_BASE, SP, SU, C0, NC, OB,
    BLK_SIZE, STRIDE0, MAX_BLK,
    D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_D: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
    MIN_SCORE: tl.constexpr, HALF_MODE: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    """K2: pure-vector online softmax + RMSNorm + RoPE (**no `tl.dot`**)."""
    b = tl.program_id(0)
    ct = tl.program_id(1)

    sp = tl.load(SP + b)
    su = tl.load(SU + b)
    c0 = tl.load(C0 + b)
    nc = tl.load(NC + b)
    xbase = tl.load(X_BASE + b)
    obase = tl.load(OB + b)

    ok = ct < nc
    c = c0 + ct
    sc_idx = ct

    d = tl.arange(0, BLK_D)
    dmask = d < D
    m = tl.arange(0, BM)

    run_max = tl.zeros((BLK_D,), dtype=tl.float32) + MIN_SCORE
    run_sum = tl.zeros((BLK_D,), dtype=tl.float32)
    run_acc = tl.zeros((BLK_D,), dtype=tl.float32)

    su_safe = tl.maximum(su - 1, 0)
    for m0 in range(0, R, BM):
        mm = m0 + m
        in_c = mm < R
        # WARNING: REV, same as _compressor_bigr_kernel; production (has_state=True) goes
        #    through this pool kernel, so this is the rewrite that actually matters for the
        #    r=128 layers.
        jwin = tl.where(in_c, R - 1 - mm, mm)
        p = c * R + jwin
        is_seg = in_c & (p >= sp) & (p < sp + su) & ok        # in segment -> read workspace
        is_cache = in_c & (p >= 0) & (p < sp) & ok            # before segment -> read state
        have = is_seg | is_cache

        # Clamp the workspace row index back to [0, su): the backend forms the DMA address
        # before applying the mask (upstream comment: "Ascend may form a DMA address for a
        # masked lane before applying the load mask")
        seg_off = tl.minimum(tl.maximum(p - sp, 0), su_safe)
        wrow = (xbase + seg_off)[:, None] * D + d[None, :]
        wm2 = is_seg[:, None] & dmask[None, :]

        p_safe = tl.where(p >= 0, p, 0)
        blk = tl.load(BLKT + tl.where(CACHE_CYCLE, b, b * MAX_BLK + p_safe // BLK_SIZE),
                      mask=is_cache, other=0)
        soff = blk[:, None] * STRIDE0 + (p_safe % BLK_SIZE)[:, None] * (2 * D) + d[None, :]
        cm2 = is_cache[:, None] & dmask[None, :]

        # WARNING: the two masks are mutually exclusive => "adding two masked loads" is exactly
        #    equivalent to a select, but it brings the number of **live [BM, BLK_D] tiles down
        #    from 4 to 2** (192KB of UB is tight), and it also avoids the compiler-crashing
        #    "select between a vector load and a tl.dot result" pattern.
        kv = tl.load(KVWS + wrow, mask=wm2, other=0.0) \
            + tl.load(STATE + soff, mask=cm2, other=0.0)
        s = tl.load(SCWS + wrow, mask=wm2, other=0.0) \
            + tl.load(STATE + soff + D, mask=cm2, other=0.0)

        good = have[:, None] & dmask[None, :] & ok
        s = tl.where(good, s, MIN_SCORE)
        new_max = tl.maximum(run_max, tl.max(s, axis=0))
        alpha = tl.exp(run_max - new_max)
        e = tl.where(good, tl.exp(s - new_max[None, :]), 0.0)
        run_acc = run_acc * alpha + tl.sum(e * kv, axis=0)
        run_sum = run_sum * alpha + tl.sum(e, axis=0)
        run_max = new_max

    o = run_acc / tl.where(run_sum > 0.0, run_sum, 1.0)
    o2 = tl.where(dmask, o, 0.0)
    ss = tl.sum(o2 * o2, axis=0)
    inv = 1.0 / tl.sqrt(ss * (1.0 / D) + EPS)
    nw = tl.load(NORMW + d, mask=dmask, other=0.0)
    y = o2 * inv * nw

    orow = obase + sc_idx
    tl.store(CMPKV + orow * D + d, y.to(CMPKV.dtype.element_ty),
             mask=ok & dmask & (d < D - RD))

    ROFF: tl.constexpr = D - RD
    if HALF_MODE:
        hr: tl.constexpr = RD // 2
        for i in range(0, hr):
            va = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vb = tl.sum(tl.where(d == ROFF + hr + i, y, 0.0), axis=0)
            ca = tl.load(RCOS + orow * RD + i)
            sa = tl.load(RSIN + orow * RD + i)
            cb = tl.load(RCOS + orow * RD + hr + i)
            sb = tl.load(RSIN + orow * RD + hr + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ca * va - sa * vb).to(CMPKV.dtype.element_ty),
                     mask=ok)
            tl.store(CMPKV + orow * D + ROFF + hr + i,
                     (sb * va + cb * vb).to(CMPKV.dtype.element_ty), mask=ok)
    else:
        for i in range(0, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i + 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi - si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)
        for i in range(1, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i - 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i, (ci * vi + si * vp).to(CMPKV.dtype.element_ty),
                     mask=ok)


@triton.jit
def _bigr_cache_kernel(
    KVWS, SCWS, STATE, BLKT, X_BASE, SP, WLO, WHI,
    BLK_SIZE, STRIDE0, MAX_BLK,
    D: tl.constexpr, BLK_D: tl.constexpr, P: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    """K3: **pure copy** from workspace to state_cache (no dot).

    Must be launched after K2 (following the upstream "read before write" convention; K2 reads
    `p < sp` while K3 writes `p >= wlo >= sp`, so they do not actually conflict).
    """
    b = tl.program_id(0)
    t = tl.program_id(1)
    sp = tl.load(SP + b)
    wlo = tl.load(WLO + b)
    whi = tl.load(WHI + b)
    xbase = tl.load(X_BASE + b)

    p = wlo + t * P + tl.arange(0, P)
    ok = p < whi
    d = tl.arange(0, BLK_D)
    dmask = d < D

    seg_off = tl.minimum(tl.maximum(p - sp, 0), tl.maximum(whi - sp - 1, 0))
    wrow = (xbase + seg_off)[:, None] * D + d[None, :]
    sm = ok[:, None] & dmask[None, :]
    kv = tl.load(KVWS + wrow, mask=sm, other=0.0)
    scc = tl.load(SCWS + wrow, mask=sm, other=0.0)

    p_safe = tl.where(p >= 0, p, 0)
    blk = tl.load(BLKT + tl.where(CACHE_CYCLE, b, b * MAX_BLK + p_safe // BLK_SIZE),
                  mask=ok, other=0)
    soff = blk[:, None] * STRIDE0 + (p_safe % BLK_SIZE)[:, None] * (2 * D) + d[None, :]
    tl.store(STATE + soff, kv, mask=sm)
    tl.store(STATE + soff + D, scc, mask=sm)


def compressor_bigr(
    x, wkv, wgate, ape, norm_weight, rope_sin, rope_cos,
    state_cache=None, state_block_table=None, cu_seqlens=None, seqused=None,
    start_pos=None, rope_head_dim=64, cmp_ratio=128, coff=1, norm_eps=1e-6,
    rotary_mode=2, cache_mode=1, state_cache_stride_dim0=0, block_size=None,
    out_rows=None,
):
    """`coff == 1` and `cmp_ratio > 16` path of the AscendC `compressor` (online softmax)."""
    if int(coff) != 1:
        raise ValueError(f"compressor_bigr only handles coff==1, got coff={coff}")

    dev = x.device
    R = int(cmp_ratio)
    D = int(norm_weight.numel())
    RD = int(rope_head_dim)
    H = int(x.shape[-1])

    if R <= 16 or (R & (R - 1)) != 0:
        raise ValueError(f"compressor_bigr requires cmp_ratio to be a power of two > 16, got {R}")
    if wkv.shape[0] != D or wgate.shape[0] != D:
        raise ValueError(f"wkv/wgate must have coff*headDim == headDim == {D} rows, "
                         f"got {wkv.shape[0]}/{wgate.shape[0]}")
    if wkv.shape[1] != H or wgate.shape[1] != H:
        raise ValueError(f"wkv/wgate must have hidden == {H} columns")
    if tuple(ape.shape) != (R, D):
        raise ValueError(f"ape.shape={tuple(ape.shape)} must equal [cmpRatio, coff*headDim]=[{R}, {D}]")
    if state_cache is not None and state_cache.shape[-1] != 2 * D:
        raise ValueError(f"stateCache.shape[-1]={state_cache.shape[-1]} must equal 2*coff*headDim={2 * D}")
    if RD % 2 != 0:
        raise ValueError(f"rope_head_dim must be even, got {RD}")
    HALF = RD // 2

    BLK_D = next_pow2(D)
    BM = 16
    BLK_H = max(16, min(128, 8192 // BLK_D))

    is_bs = x.dim() == 3
    if is_bs:
        B, S = int(x.shape[0]), int(x.shape[1])
        Sr = int(rope_sin.shape[1])
        x_base = [b_ * S for b_ in range(B)]
        lens = [S] * B
        out_shape = (B, Sr, D)
    else:
        T = int(x.shape[0])
        if cu_seqlens is None:
            raise ValueError("the TH layout (x is [T,H]) requires cu_seqlens")
        B = int(cu_seqlens.numel()) - 1
        csl = cu_seqlens.detach().cpu().to(torch.int64).tolist()
        lens = [int(csl[b_ + 1] - csl[b_]) for b_ in range(B)]
        x_base = [int(csl[b_]) for b_ in range(B)]
        Sr = int(rope_sin.shape[0]) if out_rows is None else int(out_rows)
        out_shape = (Sr, D)

    sp_l, su_l = [], []
    for b_ in range(B):
        sp = int(start_pos[b_].item()) if start_pos is not None else 0
        su = int(seqused[b_].item()) if seqused is not None else lens[b_]
        sp_l.append(sp)
        su_l.append(su)
    c0_l = [sp_l[b_] // R for b_ in range(B)]
    nc_l = [(sp_l[b_] + su_l[b_]) // R - c0_l[b_] for b_ in range(B)]
    ob_l = []
    acc = 0
    for b_ in range(B):
        if is_bs:
            ob_l.append(b_ * Sr)
        else:
            ob_l.append(acc)
            acc += nc_l[b_]
    max_nc = max(nc_l) if nc_l else 0

    out = torch.zeros(out_shape, dtype=x.dtype, device=dev)
    x_c = x.contiguous()
    wkv_c = wkv.contiguous()
    wgate_c = wgate.contiguous()
    ape_c = ape.contiguous().to(torch.float32)
    nw_c = norm_weight.contiguous().to(torch.float32)
    rs_c = rope_sin.contiguous().to(torch.float32)
    rc_c = rope_cos.contiguous().to(torch.float32)
    if rs_c.dim() == 3:
        rs_c = rs_c.reshape(-1, RD)
        rc_c = rc_c.reshape(-1, RD)

    has_state = state_cache is not None
    if has_state:
        sc_in = state_cache.to(torch.float32)
        sc_out = sc_in.clone()
        if block_size is None:
            block_size = int(sc_in.shape[1]) if sc_in.dim() >= 2 else int(sc_in.numel() // (2 * D))
    else:
        sc_out = None
        block_size = 0
    if has_state and state_block_table is None:
        raise ValueError("state_block_table is required when state_cache is used")

    blkt = _as_int32(state_block_table, dev) if has_state else torch.zeros(1, dtype=torch.int32, device=dev)
    max_blk = int(state_block_table.shape[1]) if (state_block_table is not None
                                                  and state_block_table.dim() == 2) else 1

    dev_arrays = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731
    x_base_t = dev_arrays(x_base)
    sp_t = dev_arrays(sp_l)
    su_t = dev_arrays(su_l)
    c0_t = dev_arrays(c0_l)
    nc_t = dev_arrays(nc_l)
    ob_t = dev_arrays(ob_l)

    cache_cycle = int(cache_mode) == 2
    half_mode = int(rotary_mode) == 1

    if has_state:
        # ---- Three stages: K1 projection (with dot) -> K2 pure-vector pooling -> K3 pure
        # copy into the state cache ----
        n_tok = B * int(x.shape[1]) if is_bs else int(x.shape[0])
        PT = 16
        kvws = torch.empty((n_tok, D), dtype=torch.float32, device=dev)
        scws = torch.empty((n_tok, D), dtype=torch.float32, device=dev)
        max_su = max(su_l) if su_l else 0
        if max_su > 0:
            _bigr_proj_kernel[(B, ceil_div(max_su, PT))](
                x_c, wkv_c, wgate_c, ape_c, kvws, scws, x_base_t, sp_t, su_t,
                H=H, D=D, R=R, BLK_H=BLK_H, BLK_D=BLK_D, PT=PT)
        if max_nc > 0:
            # The pool kernel has no tl.dot, so BM is not bound by the M>=16 requirement of
            # dot; with D=512, BM=8 halves the [BM, BLK_D] tiles and avoids UB overflow
            # (measured: BM=16/BLK_D=512 needs 345KB > 192KB of UB).
            BM_POOL = 8 if BLK_D >= 512 else max(16, BM)
            _bigr_pool_kernel[(B, max_nc)](
                kvws, scws, sc_out, blkt, nw_c, rs_c, rc_c, out,
                x_base_t, sp_t, su_t, c0_t, nc_t, ob_t,
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                D=D, R=R, RD=RD, BLK_D=BLK_D, BM=BM_POOL, EPS=float(norm_eps),
                MIN_SCORE=_MIN_SCORE, HALF_MODE=bool(half_mode), CACHE_CYCLE=bool(cache_cycle))

        wlo_l, whi_l = [], []
        for b_ in range(B):
            hi = sp_l[b_] + su_l[b_]
            lo = sp_l[b_]
            if cache_cycle:
                lo = max(lo, hi - (hi % R))
            wlo_l.append(lo)
            whi_l.append(max(hi, lo))
        max_w = max((whi_l[b_] - wlo_l[b_]) for b_ in range(B)) if B else 0
        if max_w > 0:
            # The UB footprint of the [P, BLK_D] tile plus the int64 address vectors must stay
            # within 192KB. With the production shape (D=512 => BLK_D=512) the previous P=16
            # needed 1847296 bit ~ 225KB and BiShengIR reported `error: ub overflow, requires
            # 1847296 bits` for _bigr_cache_kernel, so P is shrunk here in inverse proportion
            # to BLK_D (the same approach as BM_POOL in the pool kernel above).
            P = max(8, min(32, 2048 // BLK_D))
            _bigr_cache_kernel[(B, ceil_div(max_w, P))](
                kvws, scws, sc_out, blkt,
                x_base_t, sp_t, dev_arrays(wlo_l), dev_arrays(whi_l),
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                D=D, BLK_D=BLK_D, P=P, CACHE_CYCLE=bool(cache_cycle))
    elif max_nc > 0:
        _compressor_bigr_kernel[(B, max_nc)](
            x_c, wkv_c, wgate_c, ape_c, nw_c, rs_c, rc_c,
            out, blkt, out,
            x_base_t, sp_t, su_t, c0_t, nc_t, ob_t,
            int(block_size), int(state_cache_stride_dim0), int(max_blk),
            H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, BM=BM, HALF=HALF,
            EPS=float(norm_eps), MIN_SCORE=_MIN_SCORE,
            HALF_MODE=bool(half_mode), HAS_STATE=False, CACHE_CYCLE=bool(cache_cycle),
        )

    return out, sc_out
