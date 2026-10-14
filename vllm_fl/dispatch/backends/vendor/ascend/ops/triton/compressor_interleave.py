# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""Dedicated implementation of the `rotary_mode=2 (INTERLEAVE)` path of compressor.

Split out of compressor.py to keep the main module small.

================================================================================
INTERLEAVE: investigation history and root cause
================================================================================
**Convention** (authoritative source: arch32 as run on 910C,
`op_kernel/arch32/rope.h` L94-114 `RotaryPosEmb<INTERLEAVE>`):
    gatherOffset[k] = k^1 (swap adjacent columns); Muls(-1) applies to even positions only;
    cos is read in natural order
    => out[k] = cos[k]*x[k] + sin[k]*(k even ? -x[k+1] : +x[k-1])
       cos/sin table width = col = RD (not D)
(arch35 `vf/vf_rope.h` L96-138 `InterleaveModeRopeVF` has the same semantics; it just uses the
DeInterleave/Interleave instructions.)
An identifiable oracle already passes in test_compressor.py:
x=[1,2,3,4], cos=sin=1, RD=4 -> [-1,3,-1,7].

**Dead ends explored (4 variants, bit-identical error)**:
| Variant | rope implementation | Result |
|---|---|---|
| v1 | `tl.sum(tl.where(...,axis=1))` + wrong cos index | err 1.642 |
| v2 | `tl.split` + corrected index | err 3.235e-01, non-rope 3.24e-01 |
| v3 | switched to the HALF-style `tl.sum(tl.where(...))` | **bit-identical** |
| v4 | **standalone kernel body, no constexpr** | **bit-identical** |

**Root cause (established by a discriminating experiment)**: deleting the two rope `tl.store`
lines makes the **non-rope region immediately drop from 3.24e-01 to 3.90e-03** (= 3.895e-03 for
the same configuration with HALF).
=> **the rope stores in v2/v3/v4 used the non-unit-stride index vectors `2*c2` / `2*c2+1`, which
   wrote to the wrong addresses and overwrote non-rope columns out of bounds.**
   The Ascend backend mishandles **masked-store index vectors with non-unit stride** (the same
   class of problem as the `tl.reshape` offset misalignment observed earlier: **offset/index
   vectors must have unit stride or be recomputed arithmetically**).
   This explains why "3 different rope implementations gave exactly the same wrong values" --
   **the bug was in the addresses, not in the values**.

**Fix**: first compute the rope segment as a **[RD] vector in natural order**, then store it
with a **unit-stride** `tl.arange(0, RD)`.
    Constraints: `(D-RD) % RD == 0` and `RD | BLK_D` (if violated, the main module raises
    NotImplementedError instead of silently taking a wrong branch).

**Methodology** (now standard practice): **if changing X does not change the result, X is not
the cause, but a side effect of X (such as a store address) may be.**
"""

from __future__ import annotations

import triton
import triton.language as tl

__all__ = ["_compressor_interleave_kernel"]


@triton.jit
def _compressor_interleave_kernel(
    X, WKV, WGATE, APE, NORMW, RSIN, RCOS, STATE, BLKT, CMPKV,
    X_BASE, SP, SU, C0, NC, OB,
    BLK_SIZE, STRIDE0, MAX_BLK,
    H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, RD: tl.constexpr,
    BLK_H: tl.constexpr, BLK_D: tl.constexpr,
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
    sc = ct
    rows = tl.arange(0, M)
    d = tl.arange(0, BLK_D)
    dmask = d < D
    in_chunk = rows < R                       # only the first R rows belong to this chunk
    # WARNING: the in-window index is **reversed** (jwin = R-1-rows). Column-wise softmax and
    #    the weighted sum are invariant to row order => the semantics are exactly equivalent,
    #    but this turns the history (head) rows at the front of the chunk from a tile
    #    **prefix** into a **suffix** (a known backend pitfall: a masked lane that lands in the
    #    prefix can produce wrong values that drift with the allocation).
    #    Lanes where in_chunk is false get jwin=R so that they never fall inside the head range.
    jwin = tl.where(in_chunk, R - 1 - rows, R)
    p_abs = c * R + jwin                      # [M] absolute positions (reversed)
    real = in_chunk & (p_abs >= sp) & ok
    head = in_chunk & (p_abs < sp) & ok
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
        woff = d[None, :] * H + hh[:, None]
        wm = hmask[:, None] & dmask[None, :]
        acc_kv += tl.dot(xv, tl.load(WKV + woff, mask=wm, other=0.0))
        acc_sc += tl.dot(xv, tl.load(WGATE + woff, mask=wm, other=0.0))

    if HAS_STATE:
        blk_idx = tl.where(CACHE_CYCLE, b, b * MAX_BLK + (p_abs // BLK_SIZE))
        blk_id = tl.load(BLKT + blk_idx, mask=head, other=0)
        row_in = p_abs % BLK_SIZE
        soff = blk_id[:, None] * STRIDE0 + row_in[:, None] * (2 * D) + d[None, :]
        acc_kv = tl.where(head[:, None], tl.load(STATE + soff), acc_kv)
        acc_sc = tl.where(head[:, None], tl.load(STATE + soff + D), acc_sc)

    j = p_abs % R                             # after reversing the rows this must come from
                                              # p_abs (the two are equivalent before reversing)
    apev = tl.load(APE + j[:, None] * D + d[None, :],
                   mask=(in_chunk & ok)[:, None] & dmask[None, :], other=0.0)
    acc_sc = tl.where((real & in_chunk)[:, None], acc_sc + apev, acc_sc)

    good = (in_chunk & ok)[:, None] & dmask[None, :]
    s_masked = tl.where(good, acc_sc, MIN_SCORE)
    mx = tl.max(s_masked, axis=0)
    ex = tl.exp(s_masked - mx[None, :])
    pj = ex / tl.sum(ex, axis=0)[None, :]
    o = tl.sum(pj * acc_kv, axis=0)

    o2 = tl.where(dmask, o, 0.0)
    inv = 1.0 / tl.sqrt(tl.sum(o2 * o2, axis=0) * (1.0 / D) + EPS)
    y = o2 * inv * tl.load(NORMW + d, mask=dmask, other=0.0)

    orow = obase + sc
    tl.store(CMPKV + orow * D + d, y.to(CMPKV.dtype.element_ty),
             mask=ok & dmask & (d < D - RD))

    # ---- INTERLEAVE rope: per-column scalar stores (index = scalar + constant, unit stride) ----
    ROFF: tl.constexpr = D - RD
    # No tl.reshape / tl.join / 2-D reduce -- this decouples "index correctness" from
    # "lowering correctness".
    # Columns are extracted with the 1-D `tl.sum(tl.where(d == k, y, 0.0))` (same as HALF,
    # already verified to work).
    # out[i] = cos[i]*y[i] + sin[i]*(i even ? -y[i+1] : +y[i-1]); even/odd use separate loops,
    # so there is no out-of-bounds access.
    for i in range(0, RD, 2):                       # even columns, partner = i+1 <= RD-1
        vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
        vp = tl.sum(tl.where(d == ROFF + i + 1, y, 0.0), axis=0)
        ci = tl.load(RCOS + orow * RD + i)
        si = tl.load(RSIN + orow * RD + i)
        tl.store(CMPKV + orow * D + ROFF + i, (ci * vi - si * vp).to(CMPKV.dtype.element_ty))
    for i in range(1, RD, 2):                       # odd columns, partner = i-1 >= 0
        vi = tl.sum(tl.where(d == ROFF + i, y, 0.0), axis=0)
        vp = tl.sum(tl.where(d == ROFF + i - 1, y, 0.0), axis=0)
        ci = tl.load(RCOS + orow * RD + i)
        si = tl.load(RSIN + orow * RD + i)
        tl.store(CMPKV + orow * D + ROFF + i, (ci * vi + si * vp).to(CMPKV.dtype.element_ty))
