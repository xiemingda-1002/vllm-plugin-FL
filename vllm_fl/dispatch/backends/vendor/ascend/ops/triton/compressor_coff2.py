# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""The `coff=2 (COFF::OVERLAP)` path of compressor -- cross-chunk overlapping-window compression.

================================================================================
Semantics sources (three independent sources that corroborate each other)
================================================================================
Let r = cmp_ratio, D = headDim (= length of norm_weight), DC = coff*D = 2D (projection width).

1) **Upstream vLLM Triton reference (primary source: readable and semantically complete)**
   `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`
   `:156-211` `_fused_kv_compress_norm_rope_insert_sparse_attn`::

       if (position + 1) % COMPRESS_RATIO != 0: return
       start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
       tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
       pos = start + tokens
       mask_pos = pos >= 0
       head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE   # <- channel halving
       row_base = state_cache_ptr + block_numbers*stride0 + block_offsets*stride1 + head_offset
       score = load(row_base[:,None] + STATE_WIDTH + block, mask=combined_mask, other=-inf)
       score = tl.softmax(score, dim=0)
       kv    = load(row_base[:,None] + block,            mask=combined_mask, other=0.0)
       compressed_kv = tl.sum(kv * score, axis=0)
       compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO       # <- RoPE row base

   `save_partial_states.py:92-101`: a state row is `[ kv(DC) | score+ape(DC) ]`,
   `ape_row = position % compress_ratio`, `state_width = DC`.

2) **AscendC source (used to validate the boundary behavior)**
   * `op_kernel/arch35/compressor_comm.h:83-86`  `enum class COFF { DISABLE = 1, OVERLAP = 2 };`
   * `op_host/arch32/compressor_tiling.cpp:829-852` (**shape contract**)
       `wkv.shape[0] == wgate.shape[0] == coff*headDim`
       `stateCache.shape[2] == 2*coff*headDim`
       `ape.shape == [cmpRatio, coff*headDim]`
   * same file `:958-964` the only legal combinations are {r=4,coff=2,D=512} / {4,2,128} / {128,1,512}
   * same file `:179-183` with coff==2 the workspace is doubled and an extra
     `vec1TailCacheSize = cmpRatio*headDim` is allocated
   * `op_kernel/arch35/compressor_block_vec.h:906-928` (ReadState "fill left / fill right")
       "fill right" (headHolderSeqCnt>0) reads column `dStartIdx + (coff-1)*headDim` -> channels `[D,2D)`
       "fill left" reads column `dStartIdx`                                   -> channels `[0,D)`
       `isFirst = bStartPos + sIdx < cmpRatio_`
   * same file `:866-879` `DuplicateFirstBlock` -- **with no history, score is filled with
     `SOFTMAX_MIN_NUM` and kv with `FLOAT_ZERO`**
   * same file `:937` `ReduceSize = coff_ * cmpRatio_` (the softmax reduction length = 2r)
   * same file `:946` `MulReduceSumbaseVF(kv, score, dst, coff_, cmpRatio_, ...)`
   * same file `:574-582` UB layout comment for coff=2: `|---D_L---|------D_R-----|`
   * same file `:629,636` only `COMP::coff == COFF::OVERLAP` enables the cross-chunk
     `SaveToWorkSpace/LoadFromWorkSpace`

3) **Equivalent rewrite on the model side**
   (`vllm_fl/models/deepseek_v4_ascend.py:664-670 overlap_transform`)::

       # second half of the current group's channels -> the last r rows of the window
       new_tensor[:, :, ratio:]  = tensor[:, :, :, d:]
       # first half of the previous group's channels -> the first r rows of the window
       new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]

   and it is self-consistent with `vllm/models/deepseek_v4/compressor.py:141-142`
   `coff = 1 + (compress_ratio == 4)` / `sliding_window = coff*compress_ratio` (= 8 = 2r).

================================================================================
Implemented semantics (line by line against the upstream reference)
================================================================================
For **absolute chunk index c** (chunk c covers the absolute positions [c*r, (c+1)*r)), the window
of the compressed output is **previous chunk union current chunk**, i.e. 2r input positions:
`p_j = (c-1)*r + j`, `j in [0, 2r)`.
(Upstream: `position = (c+1)*r - 1` => `start = position - 2r + 1 = (c-1)*r`, `tokens = j`.)

* `j <  r` inside the window (previous chunk) takes projection channels `[0, D)` (upstream `head_offset = 0`);
* `j >= r` inside the window (current chunk) takes projection channels `[D, 2D)` (upstream `head_offset = D`).

That is::

    kv_j[d]    = (x[p_j] @ wkv^T)[half(j)*D + d]
    score_j[d] = (x[p_j] @ wgate^T)[half(j)*D + d] + ape[j % r, half(j)*D + d]
    o[d]       = sum_j softmax_j(score[:, d]) * kv_j[d]     # softmax is per column over the **2r rows**
    y          = RMSNorm_D(o) * norm_weight                 # <- unchanged
    out        = RoPE(y)                                    # <- unchanged, row base = c*r

Order: **sum over the 2r rows of kv*softmax(score) first**, then RMSNorm / RoPE.

Boundaries (`DuplicateFirstBlock` / `ReadState`):
* `p_j < 0` (the window crosses the batch start, i.e. the first r rows when `p_abs < r`) =>
  **score = -inf (`MIN_SCORE`) and kv = 0**; both are required to be equivalent to "ignore that
  row";
* `0 <= p_j < start_pos` => read back from `state_cache` (post-ape / pre-softmax), again selecting
  channels `[0,D)` or `[D,2D)` through `half(j)`.

NOTE **coff=2 and coff=1 necessarily produce different outputs for the same (r, D)** (window 2r
versus r, plus channel halving); it is not merely "the workspace got bigger". The production C4
path has `overlap = (compress_ratio == 4)`, which is always True, so coff=2 is the **only**
configuration on that path and accepting the parameter without implementing it is not an option.

================================================================================
How the backend constraints are honored
================================================================================
* The 2r-row softmax is expressed with **a single [M, BLK_D] register tensor**
  (`M = max(16, next_pow2(2r))`) and does **not** use `tl.reshape` / `tl.join`.
* Every store index is either `tl.arange` (unit stride) or "scalar + constant"; RoPE reuses the
  **per-column scalar unrolling** already validated in `compressor_interleave.py`.
* `BLK_H` adapts to `BLK_D` so that the total size of the four weight tiles stays bounded
  (UB is 192KB).
* **All program-level masking is decided by the host**: the same kernel is launched twice with a
  different FIRST constexpr -- the regular launch uses `grid=(number of plans,)` (BIDX/CTX carry
  the `(batch, chunk)` of each program) and the sequence-first-block launch uses
  `grid=(number of first blocks,)` (BIDX carries the batch index). **The kernel contains no
  runtime scalar mask for "is this program valid"**; every launched program is active. This path
  once triggered `507015 VEC instruction error: the ub address out of bounds`.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2


# -----------------------------------------------------------------------------
# Mask-free rewrite: address-legality audit
#   Iron rule: `tl.where(mask, load(p), other)` **still executes the load**, so every
#   address must be **unconditionally legal**.
#   Production shapes: D in {128, 512} (both powers of two => BLK_D == D => `dmask` is
#   **always true**); H = 4096 (BLK_H in {8,16,64}, all divisors => `hmask` is **always true**).
#   ------------------------------------------- mask -> rewrite -> bounds -> verdict
#   (1) X load (main)     xm=real&hmask -> xrow=xbase+clamp(tok,0,su-1), hh=min(hh,H-1)
#                         bounds [xbase, xbase+su-1] within [0,T)              => legal
#                         (the original clamped only the lower bound, which was a bug)
#   (2) 4x weight (main)  wm=hmask&dmask -> d=min(d,D-1), hh=min(hh,H-1)
#                         WKV + d*H+hh <= D*H-1; WKV + (D+d)*H+hh <= 2D*H-1     => legal
#   (3) BLKT (main)       head -> where(head, blk_idx, 0)
#                         true: blk_idx is in range by host construction (the earlier
#                         implementation made the same assumption); false: 0      => legal
#   (4) 2x STATE (main)   sm=head&dmask -> soff_safe=where(sm, soff, 0); soff uses _d_s
#                         false => address 0; true => blk_id*STRIDE0+row_in*4D+halfoff+d
#                         stays within the row                                     => legal
#   (5) APE (main)        real&dmask -> j=rows%R in [0,R), _d_s in [0,D-1]
#                         max = (R-1)*2D + D + (D-1) = 2RD-1 = last APE element     => legal
#   (6) NORMW (main)      dmask -> _d_s <= D-1                                      => legal
#   (7) main store (main) dmask&(d<D-RD) -> orow*D + _d_s <= orow*D+D-1             => legal
#                         false lanes **overwrite the RoPE columns**, but are then exactly
#                         overwritten by the scalar RoPE stores => measured **bit-identical**
#                         (D=512/D=128 max|diff|=0)                                   => legal
#   (8) X load (state)    hmask -> hh=min(hh,H-1); the row index xbase+xtok is guaranteed by
#                         the host to lie inside the write-back range               => legal
#   (9) 4x weight (state) same as (2)                                               => legal
#  (10) 2x APE (state)    dmask -> j is a scalar in [0,R), _d_s2 in [0,D-1] => <= 2RD-1 => legal
#  (11) 4x STATE store    dmask -> base + {0,D,2D,3D} + _d_s2 <= base+4D-1 (inside the row)
#                         false lanes rewrite the same address (harmless; and with production
#                         shapes dmask is always true => exact)                      => legal
#   ------------------------------------------- verdict: all 13 sites are legal
#   The only site that changes memory semantics is (7) (extra writes to the RoPE columns); it is
#   cancelled by the subsequent exact overwrite and was measured bit-identical.
# -----------------------------------------------------------------------------

__all__ = ["compressor_coff2"]

# Mask-free switch -- annotation only, **not wired into any kernel**:
#   the masked loads/stores in this file are already rewritten unconditionally into
#   clamp + unmasked access + tl.where.
#   Why it is not switchable: a Triton kernel cannot read the environment, so a real switch
#   would require turning NOMASK into a `tl.constexpr` parameter and changing the signatures and
#   every launch of both kernels.
_NOMASK = os.environ.get("COMPRESSOR_COFF2_NOMASK", "1") != "0"   # annotation only, unused

_MIN_SCORE = -1.0e30  # a large negative value instead of float("-inf") (which degrades on Ascend)

# Supported (cmp_ratio, coff, headDim) combinations -- compressor_tiling.cpp:958-964
_SUPPORTED = ((4, 2, 512), (4, 2, 128), (128, 1, 512))


# ---------------------------------------------------------------------------
# kernel 1: each program computes one output chunk
#           (overlapping-window softmax + weighted sum + rmsnorm + rope)
# ---------------------------------------------------------------------------
@triton.jit
def _compressor_coff2_kernel(
    X, WKV, WGATE, APE, NORMW, RSIN, RCOS, STATE, BLKT, CMPKV,
    X_BASE, SP, SU, C0, OB, BIDX, CTX,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr, HALF: tl.constexpr,
    M: tl.constexpr, EPS: tl.constexpr, MIN_SCORE: tl.constexpr,
    HALF_MODE: tl.constexpr, HAS_STATE: tl.constexpr, CACHE_CYCLE: tl.constexpr,
    FIRST: tl.constexpr, REVX: tl.constexpr, NOXMASK: tl.constexpr,
):
    # ---- program identity: **fully determined by the host**; the kernel contains no runtime
    #      scalar mask for "is this program valid" => every launched program is active (this path
    #      once triggered `507015 VEC instruction error: the ub address out of bounds`).
    # FIRST=False (regular): grid=(number of plans,), BIDX/CTX give the (batch, chunk) of each
    #   program; the served (b,ct) satisfy ct<nc[b] and exclude the "first block with c0==0".
    # FIRST=True (sequence first block, c0==0): grid=(number of first blocks,), the batch index
    #   comes from BIDX and the chunk is always the first block of that sequence.
    #   The first-block window contains no negative positions (c0==0 => p_abs=rows>=0); it uses
    #   only the r rows of the current chunk with channels [D,2D) and runs through **exactly the
    #   same code as a regular program** (same jit function, only different constexprs)
    #   => its codegen shape matches the regular program that is known to work.
    #   Why the first block is handled separately: the "2r window starting at (c-1)*r" contains a
    #   whole block of negative positions with no history when c==0, and the output of such a
    #   program was measured to drift with the allocation and contents of the output buffer.
    pid = tl.program_id(0)
    b = tl.load(BIDX + pid)

    sp = tl.load(SP + b)
    su = tl.load(SU + b)
    c0 = tl.load(C0 + b)
    xbase = tl.load(X_BASE + b)
    obase = tl.load(OB + b)

    rows = tl.arange(0, M)
    # NOTE dual mode, selected by shape (both are numerically equivalent; only "does it compile"
    # differs):
    #   * NOXMASK (BLK_D>=512, main compressor D=512): the X load drops the row mask, so the row
    #     order is untouched and UB does not grow;
    #   * REVX (BLK_D< 512, indexer compressor D=128): the in-window index is reversed and the X
    #     mask is kept as is.
    #   Both only change "where the masked lanes sit relative to the dot operand and the shape of
    #   the mask", which is numerically equivalent.
    if FIRST:
        c = c0
        sc = 0
        in_win = rows < R                       # window length = r (current chunk only => no negative positions)
        if REVX:
            jwin = tl.where(in_win, R - 1 - rows, rows)
            p_abs = c0 * R + jwin               # >= 0 (reversed order)
        else:
            p_abs = c0 * R + rows               # >= 0
        half = rows < R                         # current chunk => channels [D, 2D)
    else:
        ct = tl.load(CTX + pid)
        c = c0 + ct
        sc = ct
        in_win = rows < 2 * R                   # window length = 2r (previous chunk union current chunk)
        if REVX:
            jwin = tl.where(in_win, 2 * R - 1 - rows, rows)
            p_abs = (c - 1) * R + jwin          # reversed: history rows go from the tile prefix to the suffix
            half = rows < R                     # after reversing this is equivalent to jwin >= R (was rows >= R)
        else:
            p_abs = (c - 1) * R + rows          # j < r -> previous chunk; j >= r -> current chunk
            half = rows >= R                    # channel-half switch (upstream head_offset)

    real = in_win & (p_abs >= sp) & (p_abs < sp + su)
    head = in_win & (p_abs >= 0) & (p_abs < sp)  # history positions -> read from state_cache
    have = real | head                          # whether the row has real data; otherwise score=-inf and kv=0

    tok = p_abs - sp
    if NOXMASK:
        # Clamp the row index into [0, su) to go with the mask-free X load (Ascend emits DMA
        # addresses for masked lanes first)
        xrow = xbase + tl.minimum(tl.maximum(tok, 0), tl.maximum(su - 1, 0))
    else:
        xrow = xbase + tl.where(tok >= 0, tok, 0)

    d = tl.arange(0, BLK_D)
    dmask = d < D
    _d_s = tl.minimum(d, D - 1)   # nomask: clamp the column index (defined outside the loop, shared by both)
    h = tl.arange(0, BLK_H)
    halfoff = half[:, None] * D                 # channel-half offset (a scalar expression, not an index vector)

    acc_kv = tl.zeros((M, BLK_D), dtype=tl.float32)
    acc_sc = tl.zeros((M, BLK_D), dtype=tl.float32)
    for h0 in range(0, H, BLK_H):
        hh = h0 + h
        hmask = hh < H
        if NOXMASK:
            # The X load **drops the row mask** (only the column mask remains, explicitly
            # broadcast to [M, BLK_H]): a load carrying a leading-row mask contaminates the
            # result of tl.dot.
            xm = tl.broadcast_to(hmask[None, :], (M, BLK_H))
        else:
            xm = real[:, None] & hmask[None, :]
        # nomask rewrite: clamp the address + unmasked load + select with where (equivalent)
        _hh_s = tl.minimum(hh, H - 1)
        # Clamp **both** bounds: an unmasked load must guarantee the legality of its own
        # addresses. (Clamping only the lower bound left xrow able to run past the token count of
        # the batch on lanes where real is false, i.e. an out-of-range read.)
        # SU is available inside the kernel (`su = tl.load(SU + b)` above); the values of
        # out-of-range lanes are still discarded by where.
        _tok_s = tl.minimum(tl.maximum(tok, 0), su - 1)
        _xrow_s = xbase + _tok_s
        xv = tl.where(xm,
                      tl.load(X + _xrow_s[:, None] * H + _hh_s[None, :]),
                      0.0)
        wm = hmask[:, None] & dmask[None, :]
        # Left matrix (projection channels [0,D)) and right matrix (projection channels [D,2D));
        # wkv/wgate have shape [2D, H]
        # nomask rewrite: clamp the addresses + unmasked load + select with where (equivalent)
        wkv_a = tl.where(wm, tl.load(WKV + _d_s[None, :] * H + _hh_s[:, None]), 0.0)
        wkv_b = tl.where(wm, tl.load(WKV + (D + _d_s)[None, :] * H + _hh_s[:, None]), 0.0)
        wgt_a = tl.where(wm, tl.load(WGATE + _d_s[None, :] * H + _hh_s[:, None]), 0.0)
        wgt_b = tl.where(wm, tl.load(WGATE + (D + _d_s)[None, :] * H + _hh_s[:, None]), 0.0)
        ka = tl.dot(xv, wkv_a)
        kb = tl.dot(xv, wkv_b)
        ga = tl.dot(xv, wgt_a)
        gb = tl.dot(xv, wgt_b)
        acc_kv += tl.where(half[:, None], kb, ka)
        acc_sc += tl.where(half[:, None], gb, ga)

    # --- history rows: read back from state_cache; a state row is [ kv(2D) | score+ape(2D) ] ---
    #     channel = half*D + d (the head_offset of the upstream row_base)
    if HAS_STATE:
        p_safe = tl.where(p_abs >= 0, p_abs, 0)
        blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_safe // BLK_SIZE))
        # nomask rewrite: when the mask is false the address is clamped to 0 (value 0, later
        # discarded by where)
        blk_id = tl.load(BLKT + tl.where(head, blk_idx, 0))
        row_in = p_safe % BLK_SIZE
        # Hardening: soff uses the **clamped** _d_s (with production shapes _d_s == d, so the
        # result is bit-identical; with general shapes it prevents soff from running past the end
        # of the row when d >= D)
        soff = blk_id[:, None] * STRIDE0 + row_in[:, None] * (4 * D) + halfoff + _d_s[None, :]
        sm = head[:, None] & dmask[None, :]
        # nomask rewrite: soff is built from the clamped blk_id/d, so its own value is a legal
        # fallback when the mask is false
        _soff_ok = soff
        _soff_safe = tl.where(sm, _soff_ok, 0)
        acc_kv = tl.where(head[:, None], tl.where(sm, tl.load(STATE + _soff_safe), 0.0), acc_kv)
        acc_sc = tl.where(head[:, None],
                          tl.where(sm, tl.load(STATE + _soff_safe + 2 * D), 0.0), acc_sc)

    # --- ape: row index = p_abs % r (== rows % r), column = half*D + d; only added on rows that
    #     are actually computed ---
    if REVX:
        j = p_abs % R                             # verbatim identical to the verified rev_halfrows variant
    else:
        j = rows % R                              # verbatim identical to the verified xunmask variant
    # nomask rewrite: j is taken modulo R (in [0,R)), d is clamped, and the mask is carried by
    # where instead
    _ape_m = real[:, None] & dmask[None, :]
    apev = tl.where(_ape_m,
                    tl.load(APE + j[:, None] * (2 * D) + halfoff + _d_s[None, :]),
                    0.0)
    acc_sc = tl.where(real[:, None], acc_sc + apev, acc_sc)

    # --- per-column softmax over the 2r rows + weighted sum ---
    #     Rows without data: score -> MIN_SCORE (softmax weight 0). Their kv is already 0 (never
    #     written). Both together are equivalent to ignoring the row (matching the -inf / 0 of
    #     DuplicateFirstBlock).
    good = have[:, None] & dmask[None, :]
    s_masked = tl.where(good, acc_sc, MIN_SCORE)
    mx = tl.max(s_masked, axis=0)
    ex = tl.exp(s_masked - mx[None, :])
    den = tl.sum(ex, axis=0)
    pj = ex / tl.where(den > 0.0, den, 1.0)[None, :]
    o = tl.sum(pj * acc_kv, axis=0)

    # --- RMSNorm over D (unchanged) ---
    o2 = tl.where(dmask, o, 0.0)
    ss = tl.sum(o2 * o2, axis=0)
    inv = 1.0 / tl.sqrt(ss * (1.0 / D) + EPS)
    nw = tl.where(dmask, tl.load(NORMW + _d_s), 0.0)   # nomask rewrite
    y = o2 * inv * nw

    orow = obase + sc
    # --- the non-rope part is written straight through (unit-stride indices); there is no
    #     program-level mask (every program is active) ---
    # nomask rewrite: clamp the address + unmasked store (the extra columns are the RoPE columns,
    # which are then overwritten exactly)
    tl.store(CMPKV + orow * D + _d_s, y.to(CMPKV.dtype.element_ty))

    # --- RoPE (unchanged): scalar unrolled per column, indices = scalar + constant => absolute
    #     unit stride ---
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
            tl.store(CMPKV + orow * D + ROFF + i,
                     (ca * va - sa * vb).to(CMPKV.dtype.element_ty))
            tl.store(CMPKV + orow * D + ROFF + hr + i,
                     (sb * va + cb * vb).to(CMPKV.dtype.element_ty))
    else:
        # out[i] = cos[i]*y[i] + sin[i]*(i even ? -y[i+1] : +y[i-1])
        for i in range(0, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i + 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i,
                     (ci * vi - si * vp).to(CMPKV.dtype.element_ty))
        for i in range(1, RD, 2):
            vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
            vp = tl.sum(tl.where(d == ROFF + i - 1, y, 0.0), axis=0)
            ci = tl.load(RCOS + orow * RD + i)
            si = tl.load(RSIN + orow * RD + i)
            tl.store(CMPKV + orow * D + ROFF + i,
                     (ci * vi + si * vp).to(CMPKV.dtype.element_ty))


# ---------------------------------------------------------------------------
# kernel 2: write the absolute positions covered by this call back into state_cache
#   the state row width for coff=2 is 2*DC = 4D: [ kv(2D) | score+ape(2D) ]
#   one position per program; every write-back is "scalar address + 1-D column-masked store"
# ---------------------------------------------------------------------------
@triton.jit
def _compressor_coff2_state_kernel(
    X, WKV, WGATE, APE, STATE, BLKT,
    X_BASE, SP, WLO, RB, RQ,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D_ST: tl.constexpr, CACHE_CYCLE: tl.constexpr,
):
    # Each program handles **1 absolute position** (a scalar row index): the batch and the row
    # index come from the host-side plan table and there is no "is this program valid" mask; all
    # four write-backs are **scalar address + 1-D column-masked store** (the reliable form on this
    # backend). The previous version used a [P,BLK_D] 2-D masked store with an address that
    # depended on a blk_id loaded from memory, and was measured to write both kv halves
    # incorrectly (the score half stayed exact to 1e-8 while the wrong positions drifted from run
    # to run).
    pid = tl.program_id(0)
    b = tl.load(RB + pid)
    q = tl.load(RQ + pid)

    sp = tl.load(SP + b)
    wlo = tl.load(WLO + b)
    xbase = tl.load(X_BASE + b)

    p_abs = wlo + q                     # scalar absolute position (host guarantees it is in the write-back range)
    tok = p_abs - sp
    blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_abs // BLK_SIZE))
    blk_id = tl.load(BLKT + blk_idx)    # scalar, unmasked (no mask needed => no out-of-range address)
    base = blk_id * STRIDE0 + (p_abs % BLK_SIZE) * (4 * D)
    j = p_abs % R

    h = tl.arange(0, BLK_H)
    xtok = tl.where(tok >= 0, tok, 0)
    # Tiling along the D axis: there is no cross-column reduction, so tiling is safe and keeps UB
    # in check
    for d0 in range(0, D, BLK_D_ST):
        d = d0 + tl.arange(0, BLK_D_ST)
        dmask = d < D
        _d_s2 = tl.minimum(d, D - 1)   # nomask: clamp the column index (defined outside the h loop)
        kv_a = tl.zeros((BLK_D_ST,), dtype=tl.float32)
        kv_b = tl.zeros((BLK_D_ST,), dtype=tl.float32)
        sc_a = tl.zeros((BLK_D_ST,), dtype=tl.float32)
        sc_b = tl.zeros((BLK_D_ST,), dtype=tl.float32)
        for h0 in range(0, H, BLK_H):
            hh = h0 + h
            hmask = hh < H
            # nomask rewrite
            _hh_s2 = tl.minimum(hh, H - 1)
            xv = tl.where(hmask, tl.load(X + (xbase + xtok) * H + _hh_s2), 0.0).to(tl.float32)
            wm = hmask[:, None] & dmask[None, :]
            # The weight tiles are verbatim identical to the previous version
            # ([BLK_H, BLK_D_ST]); only the "matrix multiply" is replaced by
            # "broadcast multiply + axis=0 reduction" => the result is still a 1-D column vector
            # and all stores are 1-D.
            # nomask rewrite: clamp the addresses + where (equivalent)
            kv_a += tl.sum(tl.where(wm, tl.load(WKV + _d_s2[None, :] * H + _hh_s2[:, None]), 0.0)
                           .to(tl.float32) * xv[:, None], axis=0)
            kv_b += tl.sum(tl.where(wm, tl.load(WKV + (D + _d_s2)[None, :] * H + _hh_s2[:, None]), 0.0)
                           .to(tl.float32) * xv[:, None], axis=0)
            sc_a += tl.sum(tl.where(wm, tl.load(WGATE + _d_s2[None, :] * H + _hh_s2[:, None]), 0.0)
                           .to(tl.float32) * xv[:, None], axis=0)
            sc_b += tl.sum(tl.where(wm, tl.load(WGATE + (D + _d_s2)[None, :] * H + _hh_s2[:, None]), 0.0)
                           .to(tl.float32) * xv[:, None], axis=0)
        # ape row index = p_abs % r; each half adds its own ape columns
        sc_a = sc_a + tl.where(dmask, tl.load(APE + j * (2 * D) + _d_s2), 0.0)   # nomask
        sc_b = sc_b + tl.where(dmask, tl.load(APE + j * (2 * D) + D + _d_s2), 0.0)
        # nomask rewrite: clamp the addresses + unmasked store (the host guarantees dmask is
        # always true: D % BLK_D_ST == 0)
        tl.store(STATE + base + _d_s2, kv_a)                        # kv   [0, D)
        tl.store(STATE + base + D + _d_s2, kv_b)                    # kv   [D, 2D)
        tl.store(STATE + base + 2 * D + _d_s2, sc_a)                # score[2D, 3D)
        tl.store(STATE + base + 3 * D + _d_s2, sc_b)                # score[3D, 4D)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------
def _as_int32(t, device):
    if t is None:
        return None
    return t.to(device=device, dtype=torch.int32).contiguous()


def _check_shape_contract(x, wkv, wgate, ape, norm_weight, state_cache, coff, R, D):
    """Shape contract of compressor_tiling.cpp:829-852, checked explicitly (a violation raises)."""
    DC = coff * D
    tag = "compressor_tiling.cpp:829-852"
    if wkv.shape[0] != DC or wgate.shape[0] != DC:
        raise ValueError(
            f"[{tag}] wkv.shape[0]={wkv.shape[0]} / wgate.shape[0]={wgate.shape[0]} "
            f"must equal coff*headDim={DC} (coff={coff}, headDim={D})")
    if wkv.shape[1] != int(x.shape[-1]) or wgate.shape[1] != int(x.shape[-1]):
        raise ValueError(
            f"[{tag}] wkv.shape[1]={wkv.shape[1]} / wgate.shape[1]={wgate.shape[1]} "
            f"must equal hidden={int(x.shape[-1])}")
    if tuple(ape.shape) != (R, DC):
        raise ValueError(f"[{tag}] ape.shape={tuple(ape.shape)} must equal [cmpRatio, coff*headDim]"
                         f"=[{R}, {DC}]")
    if norm_weight.numel() != D:
        raise ValueError(f"norm_weight.numel()={norm_weight.numel()} must equal headDim={D}")
    if state_cache is not None and state_cache.shape[-1] != 2 * DC:
        raise ValueError(
            f"[{tag}] stateCache.shape[-1]={state_cache.shape[-1]} must equal 2*coff*headDim={2 * DC}")


def compressor_coff2(
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
    coff: int = 2,
    norm_eps: float = 1e-6,
    rotary_mode: int = 2,
    cache_mode: int = 1,
    state_cache_stride_dim0: int = 0,
    block_size: int | None = None,
    out_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The `coff == 2 (COFF::OVERLAP)` path of the AscendC `compressor`.

    Same signature as `compressor()`; see the module docstring for the semantics
    (overlapping window of 2r plus channel halving).
    """
    if int(coff) != 2:
        raise ValueError(f"compressor_coff2 only handles coff==2, got coff={coff}")

    dev = x.device
    R = int(cmp_ratio)
    D = int(norm_weight.numel())
    RD = int(rope_head_dim)
    H = int(x.shape[-1])

    if (R, 2, D) not in _SUPPORTED:
        raise ValueError(
            f"(cmp_ratio={R}, coff=2, headDim={D}) is not a supported combination; "
            f"the supported combinations are {_SUPPORTED} (compressor_tiling.cpp:958-964)")
    if RD % 2 != 0 or ((RD // 2) & (RD // 2 - 1)) != 0:
        raise ValueError(f"rope_head_dim/2 must be a power of two, got RD={RD}")
    HALF = RD // 2
    if D % HALF != 0:
        raise ValueError(f"headDim({D}) must be divisible by rope_head_dim/2({HALF})")

    BLK_D = next_pow2(D)
    if BLK_D > 512:
        raise NotImplementedError(
            f"the coff=2 path has BLK_D={BLK_D}, which exceeds the UB budget (production C4 uses "
            f"headDim=128); headDim>512 would need a two-pass implementation tiled along the D "
            f"axis, which is not implemented")
    # M is lowered **only for D=512** (M does not take part in the h-direction accumulation, so
    # the numerics stay bit-identical):
    #   * BLK_D>=512: M = 2R exactly (R=4 => 8) => resident UB drops from 132 to 96 KB
    #   * others (indexer D=128, ...): **keep the original structure** (no IR change)
    # NOTE do not change BLK_H: it is the step of the h loop and would change the fp32
    # accumulation order, producing a ~2.8e-3 difference
    M = next_pow2(2 * R) if BLK_D >= 512 else max(16, next_pow2(2 * R))
    # The four weight tiles (BLK_H x BLK_D, bf16/fp16) total roughly 64KB
    BLK_H = max(16, min(128, 8192 // BLK_D))

    _check_shape_contract(x, wkv, wgate, ape, norm_weight, state_cache, 2, R, D)

    is_bs = x.dim() == 3
    if is_bs:
        B, S = int(x.shape[0]), int(x.shape[1])
        Sr = int(rope_sin.shape[1])
        x_base = [b * S for b in range(B)]
        lens = [S] * B
        out_shape = (B, Sr, D)
    else:
        T = int(x.shape[0])
        if cu_seqlens is None:
            raise ValueError("the TH layout (x is [T,H]) requires cu_seqlens")
        B = int(cu_seqlens.numel()) - 1
        csl = cu_seqlens.detach().cpu().to(torch.int64).tolist()
        lens = [int(csl[b + 1] - csl[b]) for b in range(B)]
        x_base = [int(csl[b]) for b in range(B)]
        Sr = int(rope_sin.shape[0]) if out_rows is None else int(out_rows)
        out_shape = (Sr, D)

    sp_l, su_l = [], []
    for b in range(B):
        sp = int(start_pos[b].item()) if start_pos is not None else 0
        su = int(seqused[b].item()) if seqused is not None else lens[b]
        sp_l.append(sp)
        su_l.append(su)

    c0_l = [sp_l[b] // R for b in range(B)]
    nc_l = [(sp_l[b] + su_l[b]) // R - c0_l[b] for b in range(B)]

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
            block_size = int(sc_in.shape[1]) if sc_in.dim() >= 2 else int(sc_in.numel() // (4 * D))
    else:
        sc_out = None
        block_size = 0

    if has_state and state_block_table is None:
        raise ValueError("state_block_table must be provided when state_cache is used")

    blkt = _as_int32(state_block_table, dev) if has_state else torch.zeros(1, dtype=torch.int32, device=dev)
    max_blk = int(state_block_table.shape[1]) if (state_block_table is not None and state_block_table.dim() == 2) else 1

    dev_arrays = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731
    x_base_t = dev_arrays(x_base)
    sp_t = dev_arrays(sp_l)
    su_t = dev_arrays(su_l)
    c0_t = dev_arrays(c0_l)
    ob_t = dev_arrays(ob_l)
    # ---- the host decides "which (batch, chunk) pairs are actually computed" ----
    # (1) the first block of a sequence with c0==0: its window contains a prefix of negative
    #     positions without history, so it uses the dedicated FIRST form;
    # (2) every other (b, ct) with ct < nc[b].
    # Each served plan goes into the grid individually => the kernel has no runtime scalar mask
    # for "is this program valid" (the `507015 VEC instruction error: ub address out of bounds`
    # occurred precisely on a path where a whole program was masked out).
    first_b = [b for b in range(B) if c0_l[b] == 0 and nc_l[b] > 0]
    pairs = [(b, ct) for b in range(B) for ct in range(nc_l[b])
             if not (ct == 0 and c0_l[b] == 0)]
    bidx_t = dev_arrays([p[0] for p in pairs]) if pairs else dev_arrays([0])
    ctx_t = dev_arrays([p[1] for p in pairs]) if pairs else dev_arrays([0])
    first_t = dev_arrays(first_b) if first_b else dev_arrays([0])

    cache_cycle = int(cache_mode) == 2
    half_mode = int(rotary_mode) == 1
    if not half_mode and (D - RD) % RD != 0:
        raise NotImplementedError(
            f"compressor INTERLEAVE requires (D-RD) % RD == 0; D={D} RD={RD}")

    # Variant selected by shape (both are numerically equivalent; only "does it compile" differs):
    #   BLK_D>=512 (D=512 main compressor): NOXMASK (UB neutral)
    #   BLK_D< 512 (D=128 indexer)        : REVX (reversed in-window order)
    _REVX = bool(BLK_D < 512)
    _NOXMASK = bool(BLK_D >= 512)

    if pairs:
        _compressor_coff2_kernel[(len(pairs),)](
            x_c, wkv_c, wgate_c, ape_c, nw_c, rs_c, rc_c,
            sc_out if has_state else out, blkt, out,
            x_base_t, sp_t, su_t, c0_t, ob_t, bidx_t, ctx_t,
            int(block_size), int(state_cache_stride_dim0), int(max_blk),
            H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, HALF=HALF,
            M=M, EPS=float(norm_eps), MIN_SCORE=_MIN_SCORE,
            HALF_MODE=bool(half_mode), HAS_STATE=bool(has_state), CACHE_CYCLE=bool(cache_cycle),
            FIRST=False, REVX=_REVX, NOXMASK=_NOXMASK,
        )

    # Sequence first block (c0==0): the host decides the grid (= number of first-block batches)
    # and passes the batch index array, computing a clean window that contains only the r rows of
    # the current chunk and no negative positions (see the comment at the top of the kernel).
    if first_b:
        _compressor_coff2_kernel[(len(first_b),)](
            x_c, wkv_c, wgate_c, ape_c, nw_c, rs_c, rc_c,
            sc_out if has_state else out, blkt, out,
            x_base_t, sp_t, su_t, c0_t, ob_t, first_t, ctx_t,
            int(block_size), int(state_cache_stride_dim0), int(max_blk),
            H=H, D=D, R=R, RD=RD, BLK_H=BLK_H, BLK_D=BLK_D, HALF=HALF,
            M=M, EPS=float(norm_eps), MIN_SCORE=_MIN_SCORE,
            HALF_MODE=bool(half_mode), HAS_STATE=bool(has_state), CACHE_CYCLE=bool(cache_cycle),
            FIRST=True, REVX=_REVX, NOXMASK=_NOXMASK,
        )

    if has_state:
        # Write-back range: CONTINUOUS = [sp, sp+su); CYCLE = [max(sp, Trunc(sp+su,r)), sp+su)
        wlo_l, whi_l = [], []
        for b in range(B):
            hi = sp_l[b] + su_l[b]
            lo = sp_l[b]
            if cache_cycle:
                lo = max(lo, hi - (hi % R))
            wlo_l.append(lo)
            whi_l.append(max(hi, lo))
        # Plan table: one program per position to write back (batch + in-row index), so the state
        # kernel needs no "should this row be written" mask.
        rb_l, rq_l = [], []
        for b in range(B):
            for q in range(whi_l[b] - wlo_l[b]):
                rb_l.append(b)
                rq_l.append(q)
        if rb_l:
            BLK_D_ST = min(BLK_D, 128)
            wlo_t = dev_arrays(wlo_l)
            rb_t = dev_arrays(rb_l)
            rq_t = dev_arrays(rq_l)
            _compressor_coff2_state_kernel[(len(rb_l),)](
                x_c, wkv_c, wgate_c, ape_c, sc_out, blkt,
                x_base_t, sp_t, wlo_t, rb_t, rq_t,
                int(block_size), int(state_cache_stride_dim0), int(max_blk),
                H=H, D=D, R=R, BLK_H=BLK_H, BLK_D_ST=BLK_D_ST,
                CACHE_CYCLE=bool(cache_cycle),
            )

    return out, sc_out
