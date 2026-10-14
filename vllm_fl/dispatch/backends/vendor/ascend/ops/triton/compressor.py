# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""compressor -- AscendC -> Triton

Source: csrc/ascend/attention/compressor/
Interface (op_host/compressor_def.cpp L25-99):
    Input : x [B,S,H] | [T,H]           bf16/fp16
            wkv [coff*D, H]             bf16/fp16
            wgate [coff*D, H]           bf16/fp16
            state_cache                 fp32   (same address for in/out, see Output#1)
            ape [r, coff*D]             fp32
            norm_weight [D]             fp32
            rope_sin, rope_cos          fp32, [B,ceil(S/r),rD] | [min(T,T/r+B), rD]
            state_block_table (opt)     int32 [B, maxBlk] | [B]
            cu_seqlens (opt)            int32 [B+1]
            seqused (opt)               int32 [B]
            start_pos (opt)             int32 [B]
    Output: cmp_kv  [B,Sr,H] | [Sr,H]   (same dtype as x)
            state_cache                 fp32
    Attr  : rope_head_dim=64, cmp_ratio=4, coff=1, norm_eps=1e-6,
            rotary_mode=1, cache_mode=1, state_cache_stride_dim0=0

Formula (op_kernel/arch35/ compressor_block_cube.h + compressor_block_vec.h +
      vf/vf_add.h + vf/vf_softmax.h + vf/vf_mul.h + vf/vf_rms_norm.h + vf/vf_rope.h):

    Let r = cmp_ratio, D = headDim (== len(norm_weight)), H = hidden size.
    Every chunk is aligned to multiples of r by **absolute position**: chunk c covers the
    absolute positions [c*r, (c+1)*r).

    1. Linear projection (cube) -- compressor_block_cube.h ComputeMm1
           kv[p, :]    = x[p, :] @ wkv^T          # [D]
           score[p, :] = x[p, :] @ wgate^T        # [D]
    2. ape is added to score, row index = absolute position mod r -- vf_add.h AddVF /
       compressor_block_vec.h AddApeToScore (L367-399)
           score[p, :] += ape[p % r, :]
    3. state_cache read/write -- SaveState/ReadState (L831-930) + Write/ReadFromCacheState
       (L727-829): one state row holds (kv | score), 2*D fp32 in total,
       row index = absolute position mod block_size, block id comes from block_table:
           off = block_id * state_cache_stride_dim0
                 + (p % block_size) * 2*D + stateIdx*D + d      (stateIdx: kv=0, score=1)
    4. Each complete chunk applies softmax over the r positions inside the chunk, column by
       column -- compressor_block_vec.h SoftmaxDN (L933-939) + vf_softmax.h
           p[j, d] = exp(score[c*r+j, d] - max_j score) / sum_j exp(...)
    5. Weighted sum -- KvMulReduceScore (L941-947) + vf_mul.h MulReduceSumbaseVF
           o[d] = sum_j p[j, d] * kv[c*r+j, d]
    6. RMSNorm -- MultRowRmsNorm (L1300-1313) + vf_rms_norm.h RmsNormVFImpl
           y = o / sqrt(sum(o^2)/D + eps) * norm_weight
    7. RoPE (only on the last rope_head_dim dimensions) -- CalRope (L1316-1349) + vf_rope.h
       rotary_mode=1 -> HALF: let half = RD/2
           out[i]      = cos[i]   * y[i]      - sin[i]   * y[i+half]    i in [0,half)
           out[half+i] = sin[half+i] * y[i]   + cos[half+i] * y[half+i]
       rotary_mode=2 -> INTERLEAVE:
           out[2i]   = cos[2i]   * y[2i]   - sin[2i]   * y[2i+1]
           out[2i+1] = cos[2i+1] * y[2i+1] + sin[2i+1] * y[2i]
       the [0, D-RD) dimensions outside rope are passed through (HalfModeRopeVF L84-90)
    8. Output row index (compressor_tools.h Vec2SliceIterator::Reset L804-813)
           BSH layout: padScIdx = b * Sr + scIdx
           TH  layout: padScIdx = global count of processed chunks (accumulated across batches)
       the sin/cos row index equals the output row index.

Implementation notes / simplifications in this file (see the function docstring for details):
    * only the coff == 1 (COFF::DISABLE) path is implemented; coff == 2 (OVERLAP, cross-chunk
      overlap + workspace cache) is not supported and raises NotImplementedError explicitly.
    * state_cache uses out-of-place semantics (cloned internally, written, and returned); the
      original op is in-place.
    * each program handles 1 chunk, the row tile is fixed at 16 rows (r <= 16), and the extra
      rows are masked out.
    * standard triton APIs only.

rotary_mode coverage (**INTERLEAVE does not pass**)
    * `rotary_mode=1 (HALF)` -- **bit-identical** (several cases report a post-quantization error
      of 0.000e+00 against the reference).
    * `rotary_mode=2 (INTERLEAVE)` -- **does not pass and is not counted as a PASS** (the tests
      mark it `[unverified - not counted]`).
    Established facts (no need to re-investigate):
      1. **The convention is settled**: 910C runs arch32, and the authority is
         `op_kernel/arch32/rope.h` L94-114 `RotaryPosEmb<INTERLEAVE>`: `gatherOffset[k]=k^1`
         (swap adjacent columns) plus `Muls(-1)` applied to even positions only
         => `out[k] = cos[k]*x[k] + sin[k]*(k even ? -x[k+1] : +x[k-1])`,
         with **cos/sin read in natural order and a table width of col = RD** (arch35
         `vf/vf_rope.h` L96-138 has the same semantics).
      2. That convention is implemented here; replacing `tl.split` with the HALF-style
         `tl.sum(tl.where(...))` leaves the error **completely unchanged**
         (3.235e-01 -> 3.235e-01) => the cause is neither the cos/sin indexing nor tl.split.
      3. The **store of the non-rope region shares one pure 1-D path with HALF**, HALF is
         bit-identical while INTERLEAVE differs by 0.324 => **this is not the rope math but a
         backend codegen issue with "same kernel body + different constexpr"**.
      4. This version therefore switches to **kernel 1b `_compressor_interleave_kernel`: a
         standalone kernel body without the `ROTARY_INTERLEAVE` constexpr** (the HALF path in
         `_compressor_chunk_kernel` has that branch removed). If it still fails, the variant is
         reported as not passing, honestly, and no further effort is spent on it.
    * A distinguishable oracle has been added to the tests
      (`x=[1,2,3,4], cos=sin=1, RD=4 -> [-1,3,-1,7]`, passing).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2
from .compressor_interleave import _compressor_interleave_kernel

__all__ = ["compressor"]

_MIN_SCORE = -1.0e30  # no float("-inf") here (it degrades on Ascend); a large negative value instead


# ---------------------------------------------------------------------------
# kernel 1: one program computes one output chunk (softmax + weighted sum + rmsnorm + rope)
# ---------------------------------------------------------------------------
@triton.jit
def _compressor_chunk_kernel(
    X, WKV, WGATE, APE, NORMW, RSIN, RCOS, STATE, BLKT, CMPKV,
    X_BASE, SP, SU, C0, NC, OB,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr, HALF: tl.constexpr,
    M: tl.constexpr, EPS: tl.constexpr, MIN_SCORE: tl.constexpr,
    HAS_STATE: tl.constexpr, CACHE_CYCLE: tl.constexpr,
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
    sc = ct  # output chunk index inside this batch

    rows = tl.arange(0, M)
    d = tl.arange(0, BLK_D)
    dmask = d < D

    in_chunk = rows < R                       # only the first R rows belong to this chunk
    # The in-window index is **reversed** (jwin = R-1-rows): a per-column softmax and weighted sum
    # are invariant to row order => the semantics stay exactly equivalent, but the history (head)
    # rows at the start of the chunk move from the **prefix** to the **suffix** of the tile
    # (BACKEND_NOTES issue 7: masked lanes in the prefix return wrong values that drift with the
    # allocation). Lanes with in_chunk false get jwin=R so they never enter the head range.
    jwin = tl.where(in_chunk, R - 1 - rows, R)
    p_abs = c * R + jwin                      # [M] absolute positions (reversed)
    real = in_chunk & (p_abs >= sp) & ok      # real tokens inside this batch
    head = in_chunk & (p_abs < sp) & ok       # chunk head, must be read from state_cache

    tok = p_abs - sp
    xrow = xbase + tl.where(tok >= 0, tok, 0)

    h = tl.arange(0, BLK_H)
    acc_kv = tl.zeros((M, BLK_D), dtype=tl.float32)
    acc_sc = tl.zeros((M, BLK_D), dtype=tl.float32)
    for h0 in range(0, H, BLK_H):
        hh = h0 + h
        hmask = hh < H
        xm = real[:, None] & hmask[None, :]
        xv = tl.load(X + xrow[:, None] * H + hh[None, :], mask=xm, other=0.0)
        woff = d[None, :] * H + hh[:, None]          # [BLK_H, BLK_D]
        wm = hmask[:, None] & dmask[None, :]
        wkv = tl.load(WKV + woff, mask=wm, other=0.0)
        wgt = tl.load(WGATE + woff, mask=wm, other=0.0)
        acc_kv += tl.dot(xv, wkv)
        acc_sc += tl.dot(xv, wgt)

    # --- history positions at the chunk head: read back from state_cache (already post-ape,
    # --- pre-softmax) ---
    if HAS_STATE:
        blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_abs // BLK_SIZE))
        blk_id = tl.load(BLKT + blk_idx, mask=head, other=0)
        row_in = p_abs % BLK_SIZE
        soff = blk_id[:, None] * STRIDE0 + row_in[:, None] * (2 * D) + d[None, :]
        # Unmasked load: for non-head rows blk_id comes from other=0 => soff lands inside block 0
        # (verified against the addressing formula, so it stays in bounds), and whether a row is
        # used is decided entirely by the tl.where below => this sidesteps "2-D masked load +
        # indirect addressing" (issue 1.3).
        kv_h = tl.load(STATE + soff)
        sc_h = tl.load(STATE + soff + D)
        acc_kv = tl.where(head[:, None], kv_h, acc_kv)
        acc_sc = tl.where(head[:, None], sc_h, acc_sc)

    # --- ape: row index = absolute position mod r ---
    j = p_abs % R                             # after reversing the row order this must come from
                                              # p_abs (the two agree before the reversal)
    apev = tl.load(APE + j[:, None] * D + d[None, :],
                   mask=(in_chunk & ok)[:, None] & dmask[None, :], other=0.0)
    acc_sc = tl.where((real & in_chunk)[:, None], acc_sc + apev, acc_sc)

    # --- softmax over the r positions inside the chunk (column-wise) ---
    good = (in_chunk & ok)[:, None] & dmask[None, :]
    s_masked = tl.where(good, acc_sc, MIN_SCORE)
    mx = tl.max(s_masked, axis=0)                       # [BLK_D]
    ex = tl.exp(s_masked - mx[None, :])                 # masked rows give exp(-1e30)=0
    sm_ = tl.sum(ex, axis=0)                            # [BLK_D]
    pj = ex / sm_[None, :]

    o = tl.sum(pj * acc_kv, axis=0)                     # [BLK_D] fp32

    # --- RMSNorm over D ---
    o2 = tl.where(dmask, o, 0.0)
    ss = tl.sum(o2 * o2, axis=0)                        # scalar
    inv = 1.0 / tl.sqrt(ss * (1.0 / D) + EPS)
    nw = tl.load(NORMW + d, mask=dmask, other=0.0)
    y = o2 * inv * nw                                   # [BLK_D] fp32

    orow = obase + sc
    # --- dimensions outside rope are passed through ---
    tl.store(CMPKV + orow * D + d, y.to(CMPKV.dtype.element_ty),
             mask=ok & dmask & (d < D - RD))

    # --- rope part (HALF: RD splits into two segments; reshape to [BLK_D//HALF, HALF] and pick
    # --- the segments by row) ---
    # INTERLEAVE moved to the standalone kernel `_compressor_interleave_kernel` (see module docstring)
    NB: tl.constexpr = BLK_D // HALF
    Y = tl.reshape(y, (NB, HALF))
    nb = tl.arange(0, NB)
    ia: tl.constexpr = (D - RD) // HALF
    ib: tl.constexpr = ia + 1
    yA = tl.sum(tl.where(nb[:, None] == ia, Y, 0.0), axis=0)     # [HALF]
    yB = tl.sum(tl.where(nb[:, None] == ib, Y, 0.0), axis=0)     # [HALF]
    a = tl.arange(0, HALF)
    rmask = ok & (a < HALF)
    sA = tl.load(RSIN + orow * RD + a, mask=rmask, other=0.0)
    cA = tl.load(RCOS + orow * RD + a, mask=rmask, other=0.0)
    sB = tl.load(RSIN + orow * RD + HALF + a, mask=rmask, other=0.0)
    cB = tl.load(RCOS + orow * RD + HALF + a, mask=rmask, other=0.0)
    outA = cA * yA - sA * yB
    outB = sB * yA + cB * yB
    tl.store(CMPKV + orow * D + (D - RD) + a, outA.to(CMPKV.dtype.element_ty), mask=rmask)
    tl.store(CMPKV + orow * D + (D - RD) + HALF + a, outB.to(CMPKV.dtype.element_ty), mask=rmask)


# ---------------------------------------------------------------------------
# kernel 2: write the absolute positions covered by this call back to state_cache
#           (kv and the post-ape score)
# ---------------------------------------------------------------------------
@triton.jit
def _compressor_state_kernel(
    X, WKV, WGATE, APE, STATE, BLKT,
    X_BASE, SP, WLO, WHI,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr,
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

    d = tl.arange(0, BLK_D)
    dmask = d < D
    h = tl.arange(0, BLK_H)

    acc_kv = tl.zeros((P, BLK_D), dtype=tl.float32)
    acc_sc = tl.zeros((P, BLK_D), dtype=tl.float32)
    for h0 in range(0, H, BLK_H):
        hh = h0 + h
        hmask = hh < H
        xm = ok[:, None] & hmask[None, :]
        xv = tl.load(X + (xbase + tok)[:, None] * H + hh[None, :], mask=xm, other=0.0)
        woff = d[None, :] * H + hh[:, None]
        wm = hmask[:, None] & dmask[None, :]
        wkv = tl.load(WKV + woff, mask=wm, other=0.0)
        wgt = tl.load(WGATE + woff, mask=wm, other=0.0)
        acc_kv += tl.dot(xv, wkv)
        acc_sc += tl.dot(xv, wgt)

    j = p_abs % R
    apev = tl.load(APE + j[:, None] * D + d[None, :], mask=ok[:, None] & dmask[None, :], other=0.0)
    acc_sc = acc_sc + apev

    blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_abs // BLK_SIZE))
    blk_id = tl.load(BLKT + blk_idx, mask=ok, other=0)
    soff = blk_id[:, None] * STRIDE0 + (p_abs % BLK_SIZE)[:, None] * (2 * D) + d[None, :]
    sm = ok[:, None] & dmask[None, :]
    tl.store(STATE + soff, acc_kv, mask=sm)
    tl.store(STATE + soff + D, acc_sc, mask=sm)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------
def _as_int32(t, device):
    if t is None:
        return None
    return t.to(device=device, dtype=torch.int32).contiguous()


def compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    state_cache: torch.Tensor | None = None,
    state_block_table: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    seqused: torch.Tensor | None = None,
    start_pos: torch.Tensor | None = None,
    rope_head_dim: int = 64,
    cmp_ratio: int = 4,
    coff: int = 1,
    norm_eps: float = 1e-6,
    rotary_mode: int = 1,
    cache_mode: int = 1,
    state_cache_stride_dim0: int = 0,
    block_size: int | None = None,
    out_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent of the AscendC compressor (coff == 1 path).

    Args:
        x:            [B,S,H] or [T,H], bf16/fp16
        wkv, wgate:   [D,H], same dtype as x
        ape:          [r,D] fp32
        norm_weight:  [D] fp32
        rope_sin/cos: [B,ceil(S/r),RD] or [Sr_total,RD] fp32
        state_cache:  fp32; when None no state read/write happens (requires start_pos to be a
                      multiple of r)
        state_block_table: int32 [B,maxBlk] (cache_mode=1) or [B] (cache_mode=2)
        cu_seqlens:   [B+1] int32 (required for the TH layout)
        seqused:      [B] int32
        start_pos:    [B] int32
        rotary_mode:  1=HALF (default), 2=INTERLEAVE
        cache_mode:   1=CONTINUOUS, 2=CYCLE
        state_cache_stride_dim0: stride of each block inside state_cache (in elements)
        block_size:   row-count dimension of state_cache (defaults to state_cache.shape[1])
        out_rows:     number of output rows under the TH layout; defaults to rope_sin.shape[0]

    Returns:
        (cmp_kv, state_cache_out); state_cache_out is the new tensor after writing (the original
        op is in-place).
    """
    if int(coff) == 2:
        # coff==2 (COFF::OVERLAP, cross-chunk overlap window + channel halving) uses a separate module
        from .compressor_coff2 import compressor_coff2
        return compressor_coff2(
            x, wkv, wgate, ape, norm_weight, rope_sin, rope_cos,
            state_cache=state_cache, state_block_table=state_block_table,
            cu_seqlens=cu_seqlens, seqused=seqused, start_pos=start_pos,
            rope_head_dim=rope_head_dim, cmp_ratio=cmp_ratio, coff=2,
            norm_eps=norm_eps, rotary_mode=rotary_mode, cache_mode=cache_mode,
            state_cache_stride_dim0=state_cache_stride_dim0,
            block_size=block_size, out_rows=out_rows)
    if int(coff) != 1:
        raise ValueError(
            f"coff must be 1 or 2 (compressor_tiling.h:225 `const std::vector<int> COFF {{1, 2}}`), "
            f"got {coff}")

    dev = x.device
    R = int(cmp_ratio)
    D = int(norm_weight.numel())
    RD = int(rope_head_dim)
    H = int(x.shape[-1])
    if R > 16:
        # long window (production r=128 path): a full block of r rows does not fit in registers
        # => use the row-tiled online softmax implementation
        #   at r=128,D=512 the [M,BLK_D] fp32 accumulator is 128*512*4 = 256KB each > UB 192KB
        if (R & (R - 1)) != 0:
            raise ValueError(f"cmp_ratio must be a power of two, got {R}")
        from .compressor_bigr import compressor_bigr
        return compressor_bigr(
            x, wkv, wgate, ape, norm_weight, rope_sin, rope_cos,
            state_cache=state_cache, state_block_table=state_block_table,
            cu_seqlens=cu_seqlens, seqused=seqused, start_pos=start_pos,
            rope_head_dim=rope_head_dim, cmp_ratio=cmp_ratio, coff=1,
            norm_eps=norm_eps, rotary_mode=rotary_mode, cache_mode=cache_mode,
            state_cache_stride_dim0=state_cache_stride_dim0,
            block_size=block_size, out_rows=out_rows)
    if not (1 <= R <= 16) or (R & (R - 1)) != 0:
        raise ValueError(f"cmp_ratio must be a power of two and <= 16, got {R}")
    if RD % 2 != 0 or ((RD // 2) & (RD // 2 - 1)) != 0:
        raise ValueError(f"rope_head_dim/2 must be a power of two, got RD={RD}")
    HALF = RD // 2

    is_bs = x.dim() == 3
    if is_bs:
        B, S = int(x.shape[0]), int(x.shape[1])
        Sr = int(rope_sin.shape[1])
        x_base = [b * S for b in range(B)]
        lens = [S] * B
        out_shape = (B, Sr, D)
        total_rows = B * Sr
    else:
        T = int(x.shape[0])
        if cu_seqlens is None:
            raise ValueError("the TH layout (x is [T,H]) requires cu_seqlens")
        B = int(cu_seqlens.numel()) - 1
        csl = cu_seqlens.detach().cpu().to(torch.int64).tolist()
        lens = [int(csl[b + 1] - csl[b]) for b in range(B)]
        x_base = [int(csl[b]) for b in range(B)]
        Sr = int(rope_sin.shape[0])
        out_shape = (Sr, D)
        total_rows = Sr

    sp_l, su_l = [], []
    for b in range(B):
        sp = int(start_pos[b].item()) if start_pos is not None else 0
        su = int(seqused[b].item()) if seqused is not None else lens[b]
        sp_l.append(sp)
        su_l.append(su)

    c0_l = [sp_l[b] // R for b in range(B)]
    nc_l = [(sp_l[b] + su_l[b]) // R - c0_l[b] for b in range(B)]

    # output row base / rope row base
    ob_l = []
    acc = 0
    for b in range(B):
        if is_bs:
            ob_l.append(b * Sr)
        else:
            ob_l.append(acc)
            acc += nc_l[b]
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
    max_blk = int(state_block_table.shape[1]) if (state_block_table is not None and state_block_table.dim() == 2) else 1

    dev_arrays = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731
    x_base_t = dev_arrays(x_base)
    sp_t = dev_arrays(sp_l)
    su_t = dev_arrays(su_l)
    c0_t = dev_arrays(c0_l)
    nc_t = dev_arrays(nc_l)
    ob_t = dev_arrays(ob_l)

    BLK_H = 128
    BLK_D = next_pow2(D)
    M = 16

    cache_cycle = int(cache_mode) == 2
    rotary_interleave = int(rotary_mode) == 2
    if rotary_interleave and ((D - RD) % RD != 0 or BLK_D % RD != 0):
        raise NotImplementedError(
            f"compressor INTERLEAVE requires (D-RD)%%RD==0 and RD|BLK_D; D={D} RD={RD} BLK_D={BLK_D}")

    if max_nc > 0:
        grid = (B, max_nc)
        if rotary_interleave:
            # INTERLEAVE uses a **standalone kernel body** (without the ROTARY_INTERLEAVE constexpr)
            _compressor_interleave_kernel[grid](
                x_c, wkv_c, wgate_c, ape_c, nw_c, rs_c, rc_c,
                sc_out if has_state else out, blkt, out,
                x_base_t, sp_t, su_t, c0_t, nc_t, ob_t,
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D,
                M=M, EPS=float(norm_eps), MIN_SCORE=_MIN_SCORE,
                HAS_STATE=bool(has_state), CACHE_CYCLE=bool(cache_cycle),
            )
        else:
            _compressor_chunk_kernel[grid](
                x_c, wkv_c, wgate_c, ape_c, nw_c, rs_c, rc_c,
                sc_out if has_state else out, blkt, out,
                x_base_t, sp_t, su_t, c0_t, nc_t, ob_t,
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, HALF=HALF,
                M=M, EPS=float(norm_eps), MIN_SCORE=_MIN_SCORE,
                HAS_STATE=bool(has_state), CACHE_CYCLE=bool(cache_cycle),
            )

    if has_state:
        # write-back range: CONTINUOUS = [sp, sp+su); CYCLE = [max(sp, Trunc(sp+su,r)), sp+su)
        wlo_l, whi_l = [], []
        for b in range(B):
            hi = sp_l[b] + su_l[b]
            lo = sp_l[b]
            if cache_cycle:
                lo = max(lo, hi - (hi % R))
            wlo_l.append(lo)
            whi_l.append(max(hi, lo))
        max_w = max((whi_l[b] - wlo_l[b]) for b in range(B)) if B else 0
        if max_w > 0:
            P = 64
            wlo_t = dev_arrays(wlo_l)
            whi_t = dev_arrays(whi_l)
            _compressor_state_kernel[(B, ceil_div(max_w, P))](
                x_c, wkv_c, wgate_c, ape_c, sc_out, blkt,
                x_base_t, sp_t, wlo_t, whi_t,
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                H=H, D=D, R=R, BLK_H=BLK_H, BLK_D=BLK_D, P=P,
                CACHE_CYCLE=bool(cache_cycle),
            )

    return out, sc_out
