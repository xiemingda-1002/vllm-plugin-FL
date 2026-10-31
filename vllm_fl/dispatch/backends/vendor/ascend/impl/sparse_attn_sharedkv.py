# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend Triton implementation of ``deepseek_v4_sparse_attention`` (V5.3).

One file, the way the rest of ``impl`` is laid out: the four modules that used
to form the ``sparse_attn_sharedkv`` package are merged here in dependency
order --

    device helpers                                (was ``_deps.py``)
    V5 pool / multiplicity-mask kernels           (was ``_v5.py``)
    V5.3 host wrapper + gather/pool/flash kernels (was ``kernel.py``)
    FL interface -> native ABI                     (was ``adapter.py``)

-- and the import graph is flattened: one ``from __future__``, one
``torch``/``triton`` import block, no package-relative imports, no duplicated
symbols.  No kernel math, no tuning constant and no numerical contract changed;
each section below is the original source verbatim.

The implementation is self contained: no import of the tuning workspace, no
environment switch, no host synchronisation on the hot path.

``supports()`` is the metadata-only portability probe that ``ascend.py`` uses to
route a call here or to ``sparse_attn_sharedkv_upstream`` -- the kernel this
module replaced, kept in tree as the fallback for every shape outside the
envelope.

FL ``deepseek_v4_sparse_attention`` -> V5.3 Triton kernel adapter.

The dispatch interface (``vllm_fl.dispatch.ops``) is per-token/tensor based:

    query [T, N1, D], swa_cache [P, B, D], compact_cache [Pc, Bc, D] | None,
    swa_block_table [R, Nb], compact_block_table [R, Nbc] | None,
    request_ids [T], positions [T], swa_lengths [R], compact_lengths [R] | None,
    compact_indices [T, Kc] | None, window_size, compress_ratio, scale,
    attn_sink [N1], value_dim

whereas the tuned kernel speaks the native ABI (TND q x PA_ND kv, cu_seqlens /
seqused_* / cmp_sparse_indices).  This module is the mapping and nothing else:
the numerical contract (joint SWA+compact softmax, sink only in the
denominator, invalid/-1/NaN rows masked before the dot, padding rows zero,
output in the query dtype) stays the reference's.

Mapping rules (see INTEGRATION_REPORT.md for the long form):

* ``query``            -> ``q`` [T, N1, D] (TND), untouched.
* ``swa_cache``        -> ``[P, B, 1, D]`` PA_ND view (N2 == 1).  The cache is
  addressed with **its own strides** (``page*stride(0) + row*stride(1) +
  d*stride(2)``), so it may be any view of the shared KV pool: a
  ``storage_offset`` slice, a per-head slice, a page-padded slice.  Nothing is
  copied and nothing is read back; a contiguous cache keeps the original
  address arithmetic bit for bit.
* ``compact_cache``    -> same, or ``None`` (C0 = SWA only).
* ``request_ids``      -> ``cu_seqlens_q`` [R + 1] (int32) **built on device**
  from a per-request token count (``scatter_add``) + ``cumsum``.  R is host
  known (``swa_lengths.numel()``); nothing is read back.
* ``swa_lengths``      -> ``cu_seqlens_ori_kv`` (per-request KV length).
* ``compact_lengths``  -> ``cu_seqlens_cmp_kv``.
* ``compact_indices``  -> ``cmp_sparse_indices`` [T, Kc] (int32, -1 padding).
* ``window_size``      -> ``ori_win_left = window_size - 1``, ``ori_win_right = 0``.
* ``scale``            -> ``softmax_scale`` (``None`` => ``D ** -0.5``).
* ``attn_sink``        -> ``sinks`` [N1] fp32.
* ``value_dim``        -> output slice ``[..., :value_dim]``.

Assumption (checked by the caller-side contract, documented, not enforced with
a host sync because that would break graph capture): inside one request the
query tokens occupy a contiguous run of token slots and their ``positions``
increase by one, with ``swa_lengths[r] == positions[last token of r] + 1``.
That is exactly how the scheduler lays out a prefill/decode batch
(``query_start_loc``); a batch that violates it must be routed to the
reference backend.

Contract -- ``unique_indices`` (default ``True``): a token's
``compact_indices`` row must not name the same compact slot twice.  Production
satisfies it by construction (the indexer emits a top-K selection; the C128 path
is an ``arange``).  This matters because the two sparse masks are *not*
interchangeable:

* ``"atomic"`` weights a pool slot by how many times the row selected it
  (exact for arbitrary rows, but it builds a T x R count table and costs an
  extra pass);
* ``"limit"`` ignores the row and keeps every pool slot below the causal limit,
  so it equals ``"atomic"`` only when the row happens to select **all** the
  candidates up to that limit -- true for the C128 ``arange``, and true for a
  unique top-K exactly when the row width covers the candidate count
  (``topk >= min(compact_len, kv_len / ratio)``).  That coverage test is host
  metadata, so the kernel already makes it itself: ``mask_mode=None`` picks
  ``"limit"`` when the row covers every candidate and ``"atomic"`` otherwise.

Passing ``unique_indices=True`` therefore means "uniqueness holds, so whenever
the kernel's coverage rule fires, ``"limit"`` is bit-exact"; ``False`` pins the
multiplicity mask everywhere.  The masks *do* differ on a repeated row (measured
0.65 in the output), which is why the fallback is opt-out and not inferred from
the data: it is a caller contract, never a device read.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["deepseek_v4_sparse_attention", "supports"]

# --------------------------------------------------------------------------- #
# Device-side helpers (was ``_deps.py``).
# --------------------------------------------------------------------------- #
def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())

_NEG = -1.0e30

@triton.jit
def _kv_elem_off(KV, BT, b, kvrow, n2, BLOCK_SIZE, MAX_BLOCKS, N2, D, S2,
                 PA: tl.constexpr, STRIDED: tl.constexpr = False,
                 SP: tl.constexpr = 0, SB: tl.constexpr = 0,
                 WIDE: tl.constexpr = False):
    """Linear offset of (row kvrow, head n2, element 0) inside the KV cache.

    ``kvrow`` must be a **logical row number** (under PA = block-table page * blockSize +
    in-block row) and the caller guarantees it lies in ``[0, valid rows of that batch)``. Negative
    or huge values are never accepted: on the Ascend backend both "modulo/division of negative
    values" and "masked-out addresses" go wrong (see the file header).

    ``STRIDED`` selects the general paged-cache addressing used when the cache is an
    arbitrary ``[P, B, D]`` view (a slice of the shared KV pool):

        element offset = page * SP + row_in_block * SB

    with ``SP``/``SB`` the tensor's own ``stride(0)``/``stride(1)`` (the inner-dimension
    stride is applied by the caller, which owns the ``d`` vector).  The storage offset of
    the view is already folded into the pointer Triton receives (``data_ptr()`` includes
    ``storage_offset * itemsize``), so it needs no separate term.  With ``SP = BS * D`` and
    ``SB = D`` the expression is *algebraically identical* to the contiguous one, so the
    contiguous case can use either form; ``STRIDED=False`` keeps the original code verbatim.
    ``WIDE`` widens to int64 when the view spans >= 2**31 elements (same proof as the
    paged-cache write kernel: (PAGES-1)*SP + (BLOCK-1)*SB + (DIM-1)*SD).
    """
    if PA:
        bidx = tl.minimum(tl.maximum(kvrow // BLOCK_SIZE, 0), MAX_BLOCKS - 1)
        page = tl.maximum(tl.load(BT + b * MAX_BLOCKS + bidx), 0)
        if STRIDED:
            if WIDE:
                return (page.to(tl.int64) * SP
                        + (kvrow % BLOCK_SIZE).to(tl.int64) * SB)
            return page * SP + (kvrow % BLOCK_SIZE) * SB
        row = page * BLOCK_SIZE + (kvrow % BLOCK_SIZE)
    else:
        row = b * S2 + kvrow
    return (row * N2 + n2) * D

@triton.jit
def _tok_map_bsnd(TOK_B, TOK_S1, B, S1, BLK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLK + tl.arange(0, BLK)
    m = i < B * S1
    b = i // S1
    tl.store(TOK_B + i, tl.where(m, b, 0).to(tl.int32), mask=m)
    tl.store(TOK_S1 + i, tl.where(m, i - b * S1, 0).to(tl.int32), mask=m)

@triton.jit
def _tok_map_tnd(TOK_B, TOK_S1, CU, T, B, BLK: tl.constexpr):
    """CU: cu_seqlens_q int32 [B+1]; token t lives in the first batch with
    cu[b+1] > t (batches with zero tokens are skipped naturally)."""
    pid = tl.program_id(0)
    i = pid * BLK + tl.arange(0, BLK)
    m = i < T
    b = tl.zeros_like(i)
    s1 = tl.zeros_like(i)
    found = tl.zeros_like(i) > 0
    for bb in range(0, B):
        lo = tl.load(CU + bb)
        hi = tl.load(CU + bb + 1)
        hit = (i >= lo) & (i < hi) & (~found)
        b = tl.where(hit, bb, b)
        s1 = tl.where(hit, i - lo, s1)
        found = found | hit
    tl.store(TOK_B + i, b.to(tl.int32), mask=m)
    tl.store(TOK_S1 + i, s1.to(tl.int32), mask=m)

def _tok_maps(cu_seqlens_q, T, B, S1, dev):
    tok_b = torch.empty(T, dtype=torch.int32, device=dev)
    tok_s1 = torch.empty(T, dtype=torch.int32, device=dev)
    blk = 256
    if cu_seqlens_q is not None:
        _tok_map_tnd[(ceil_div(T, blk),)](tok_b, tok_s1, cu_seqlens_q, T, B, BLK=blk)
    else:
        _tok_map_bsnd[(ceil_div(T, blk),)](tok_b, tok_s1, B, S1, BLK=blk)
    return tok_b, tok_s1

def _as_len_tensor(seq, B, name, dev):
    """[B] lengths or [B+1] cu_seqlens -> int32 length tensor **on device**."""
    if seq is None:
        return None
    s = seq.detach()
    if s.dtype != torch.int32:
        s = s.to(torch.int32)
    s = s.flatten()
    n = s.numel()
    if n == B:
        return s
    if n == B + 1:
        return (s[1:] - s[:-1]).to(torch.int32)
    raise ValueError(f"{name} length must be B={B} or B+1={B + 1}, got {n}")

def _prep_kv(kv, bt, nm, N2, D, pa):
    """(S2/block_size, max_blocks, S2_or_0, block_table int32 contiguous)."""
    if kv is None:
        return 1, 0, 0, None
    if pa:
        nb, bs, n2k, dk = kv.shape
        if bt is None:
            raise ValueError(f"layout_kv=PA_ND requires the block_table for {nm}")
        if n2k != N2 or dk != D:
            raise ValueError(f"{nm} N2/D does not match q: {tuple(kv.shape)}")
        if bt.dtype != torch.int32:
            bt = bt.to(torch.int32)
        if not bt.is_contiguous():
            bt = bt.contiguous()
        return bs, bt.shape[1], bs * bt.shape[1], bt
    _b, s2, n2k, dk = kv.shape
    if n2k != N2 or dk != D:
        raise ValueError(f"{nm} N2/D does not match q: {tuple(kv.shape)}")
    return 1, 0, s2, None


# --------------------------------------------------------------------------- #
# V5 pool / multiplicity-mask kernels (was ``_v5.py``).
# --------------------------------------------------------------------------- #
@triton.jit
def _v5_pool_kernel(
    CKV, CPOOL, KVLEN_C, KVLEN_O, QLEN, BTC,
    BLOCK_SIZE_C, MAX_BLOCKS_C, S2C, CMP_RATIO,
    RCH, POFF,
    N_ORI: tl.constexpr, PA_C: tl.constexpr, D: tl.constexpr,
    BLK_D: tl.constexpr, R: tl.constexpr, BLOCK_ROWS: tl.constexpr,
    STRIDED_C: tl.constexpr = False, SP_C: tl.constexpr = 0,
    SB_C: tl.constexpr = 0, SD_C: tl.constexpr = 1,
    WIDE_C: tl.constexpr = False,
):
    """grid = (ceil(R/RCH), B): one program fills rows [c*RCH, ..) of request b.

    Request-major so the per-request scalars (kvlen/qlen/limit/block table) are
    loaded **once per program** instead of once per row -- the row-major form
    spent a dependent scalar-load chain per row and measured 15x slower.

    ``STRIDED_C`` (default off) is the general ``[P, B, D]``-view path; with
    ``SD_C = 1`` and the contiguous page/row strides the arithmetic is the same
    as the original code.
    """
    c = tl.program_id(0)
    b = tl.program_id(1)
    d = tl.arange(0, BLK_D)
    dm = d < D
    kvlc = tl.load(KVLEN_C + b)
    qlen = tl.load(QLEN + b)
    if N_ORI > 0:
        ref_len = tl.load(KVLEN_O + b)
    else:
        ref_len = kvlc * CMP_RATIO
    limit = tl.minimum(kvlc, ref_len // CMP_RATIO)
    r0 = c * RCH
    # rows the flash can actually reach: limit rounded up to a whole KV tile
    # (beyond `limit` the pool must be zero, not garbage: 0*NaN in the PV dot)
    lim_r = tl.minimum(R, ((limit + BLOCK_ROWS - 1) // BLOCK_ROWS) * BLOCK_ROWS)
    if r0 >= lim_r:
        return
    for i in tl.range(0, RCH):
        j = r0 + i
        vld = j < limit
        off = _kv_elem_off(CKV, BTC, b, tl.where(vld, j, 0), 0,
                           BLOCK_SIZE_C, MAX_BLOCKS_C, 1, D, S2C, PA_C,
                           STRIDED_C, SP_C, SB_C, WIDE_C)
        row = tl.load(CKV + off + d * SD_C, mask=vld & dm, other=0.0)
        tl.store(CPOOL + (POFF + b * R + j) * D + d, row,
                 mask=(j < lim_r) & dm)

@triton.jit
def _v5_cmask_atomic_kernel(
    CIDX, CMASK, KVLEN_C, KVLEN_O, QLEN, TOK_B, TOK_S1,
    K_IDX, CMP_RATIO, BTC, BS_C, MAX_BLOCKS_C, NPG_C,
    N_ORI: tl.constexpr, HAS_CIDX: tl.constexpr, PA_C: tl.constexpr,
    R: tl.constexpr, BLK: tl.constexpr,
):
    """Multiplicity histogram: one program per (token, slot chunk).

    ``CMASK`` must be zeroed by the caller.  A scatter is the natural op here --
    the same compare-in-row-space form cost 30-100 ms (measured), because every
    inner step is a 2-D broadcast + last-axis reduction on the vector unit.
    """
    tc = tl.program_id(0)
    c = tl.program_id(1)
    b = tl.load(TOK_B + tc)
    s1 = tl.load(TOK_S1 + tc)
    qlen = tl.load(QLEN + b)
    kvlc = tl.load(KVLEN_C + b)
    if N_ORI > 0:
        ref_len = tl.load(KVLEN_O + b)
    else:
        ref_len = kvlc * CMP_RATIO
    s1_abs = s1 + (ref_len - qlen)
    limit = tl.maximum(tl.minimum(kvlc, (s1_abs + 1) // CMP_RATIO), 0)
    s = c * BLK + tl.arange(0, BLK)
    if HAS_CIDX:
        ii = tl.load(CIDX + tc * K_IDX + s, mask=s < K_IDX, other=-1)
        ok = (s < K_IDX) & (ii >= 0) & (ii < limit)
        if PA_C:
            ii_s = tl.where(ok, ii, 0)
            bidx = tl.minimum(ii_s // BS_C, MAX_BLOCKS_C - 1)
            pg = tl.load(BTC + b * MAX_BLOCKS_C + bidx, mask=ok, other=-1)
            ok = ok & (pg >= 0) & (pg < NPG_C)
        tl.atomic_add(CMASK + tc * R + tl.where(ok, ii, 0), 1, mask=ok)
    else:
        tl.store(CMASK + tc * R + s, (s < limit).to(tl.int32), mask=s < R)

@triton.jit
def _v5_cmask_kernel(
    CIDX, CMASK, KVLEN_C, KVLEN_O, QLEN, TOK_B, TOK_S1,
    K_IDX, CMP_RATIO,
    N_ORI: tl.constexpr, HAS_CIDX: tl.constexpr,
    R: tl.constexpr, BJ: tl.constexpr, KCH: tl.constexpr,
):
    tc = tl.program_id(0)                 # global token id (== TOK0 base is 0 here)
    b = tl.load(TOK_B + tc)
    s1 = tl.load(TOK_S1 + tc)
    qlen = tl.load(QLEN + b)
    kvlc = tl.load(KVLEN_C + b)
    if N_ORI > 0:
        ref_len = tl.load(KVLEN_O + b)
    else:
        ref_len = kvlc * CMP_RATIO
    s1_abs = s1 + (ref_len - qlen)
    limit = tl.maximum(tl.minimum(kvlc, (s1_abs + 1) // CMP_RATIO), 0)
    lane = tl.arange(0, BJ)
    # j tiled by BJ (static trip count: R is constexpr) -- a [BJ, KCH] compare tile
    # keeps the UB footprint at ~4 KB, the untiled [R, KCH] form overflowed UB.
    for jt in tl.range(0, R, BJ):
        j = jt + lane
        if HAS_CIDX:
            cnt = tl.zeros([BJ], tl.int32)
            for s0 in tl.range(0, K_IDX, KCH):
                ss = s0 + tl.arange(0, KCH)
                ii = tl.load(CIDX + tc * K_IDX + ss, mask=ss < K_IDX, other=-1)
                ok = (ii >= 0) & (ii < limit) & (ss < K_IDX)
                ii_s = tl.where(ok, ii, 0)   # safe: never an address, only a compare
                eq = (j[:, None] == ii_s[None, :]) & ok[None, :]
                cnt += tl.sum(eq.to(tl.int32), axis=1)
            val = cnt
        else:
            val = (j < limit).to(tl.int32)
        tl.store(CMASK + tc * R + j, val, mask=j < R)

# --------------------------------------------------------------------------- #
# V5.3 host wrapper + gather / pool / flash kernels (was ``kernel.py``).
# --------------------------------------------------------------------------- #
GRID_C = 96
POOL_GRID = 96
G_DEFAULT = 2
POOL_CH = 64          # pool rows per program in the page-level copy (power of 2)
RCH = 64              # gather rows per program (one BLOCK_N tile)
MASK_KCH = 16
# The multiplicity-mask kernel launches a (tokens, slot chunks) grid and
# triton-ascend rejects any launch whose total core count exceeds 65535.  A
# 20480-token prefill step with 6 chunks asked for 122880 and never started, so
# the token axis is split into bounded launches (the kernel is per token, so a
# slice is the same arithmetic).
CMASK_TOK_CHUNK = 4096
CMASK_GRID_MAX = 65535


# --------------------------------------------------------------------------- #
# P2: page-level 2-D compact-pool build (V5.3)
# --------------------------------------------------------------------------- #
@triton.jit
def _v53_pool_page_kernel(
    CKV, CPOOL, KVLEN_C, KVLEN_O, QLEN, BTC,
    MAX_BLOCKS_C, S2C, NPG, CMP_RATIO,
    POFF,
    N_ORI: tl.constexpr, PA_C: tl.constexpr, D: tl.constexpr,
    BLK_D: tl.constexpr, R: tl.constexpr, BS: tl.constexpr,
    CH: tl.constexpr, BLOCK_ROWS: tl.constexpr, CAPX: tl.constexpr,
    STRIDED_C: tl.constexpr = False, SP_C: tl.constexpr = 0,
    SB_C: tl.constexpr = 0, SD_C: tl.constexpr = 1,
    WIDE_C: tl.constexpr = False,
):
    """grid = (CAPX, B): programs stride over ``CH``-row chunks of request b.

    ``CH`` divides the cache ``BS``, so a chunk lies inside **one** page: under PA
    its source is ``page*BS + r0 .. + CH-1`` (contiguous) and its destination is
    ``b*R + j0 .. + CH-1`` (contiguous).  One block-table scalar load and one
    2-D load/store per chunk instead of one of each per row.

    The chunk count is taken **from the device** (``nch = ceil(lim_r/CH)``), not
    from the static ``R``: production block tables can be far wider than the live
    KV, so ``R`` over-estimates the work by >50x on the decode shapes and a
    grid of ``ceil(R/CH)`` programs would be 16k mostly idle blocks (measured
    1.455 ms vs 0.489 ms row-wise).  ``CAPX`` bounds the grid instead and the
    loop is empty for the programs with nothing to do.

    Bit-exactness with the row kernel: the load mask is ``j < limit`` (rows past
    the limit read as 0), the store mask is ``j < lim_r`` with the very same
    ``lim_r = min(R, ceil(limit/BLOCK_ROWS)*BLOCK_ROWS)``, and the *destination
    row is untouched* past ``lim_r`` -- identical to ``_v5_pool_kernel``.

    ``STRIDED_C`` selects the general ``[P, B, D]`` view: the source element
    offset becomes ``page * SP_C + (r0 + rr) * SB_C`` with the inner dimension
    strided by ``SD_C``, instead of the contiguous ``(page * BS + r0 + rr) * D``.
    """
    pid = tl.program_id(0)
    b = tl.program_id(1)
    d = tl.arange(0, BLK_D)
    dm = d < D
    kvlc = tl.load(KVLEN_C + b)
    if N_ORI > 0:
        ref_len = tl.load(KVLEN_O + b)
    else:
        ref_len = kvlc * CMP_RATIO
    limit = tl.minimum(kvlc, ref_len // CMP_RATIO)
    lim_r = tl.minimum(R, ((limit + BLOCK_ROWS - 1) // BLOCK_ROWS) * BLOCK_ROWS)
    nch = (lim_r + CH - 1) // CH
    cpp = BS // CH                           # chunks per page (constexpr >= 1)
    rr = tl.arange(0, CH)
    n_it = (tl.maximum(nch - pid, 0) + CAPX - 1) // CAPX
    for t in tl.range(0, n_it):
        c = pid + t * CAPX
        j0 = c * CH
        p = c // cpp                         # page holding this chunk
        r0 = (c - p * cpp) * CH              # first row inside that page
        if PA_C:
            bidx = tl.minimum(tl.maximum(p, 0), MAX_BLOCKS_C - 1)
            page = tl.minimum(tl.maximum(tl.load(BTC + b * MAX_BLOCKS_C + bidx), 0),
                              NPG - 1)
            if STRIDED_C:
                if WIDE_C:
                    eoff = (page.to(tl.int64) * SP_C
                            + (r0 + rr).to(tl.int64) * SB_C)
                else:
                    eoff = page * SP_C + (r0 + rr) * SB_C
            else:
                eoff = (page * BS + r0 + rr) * D
        else:
            # clamp per row so the tail chunk never leaves b's slab (masked anyway)
            eoff = (b * S2C + tl.minimum(j0 + rr, S2C - 1)) * D
        row = j0 + rr
        vld = row < limit
        val = tl.load(CKV + eoff[:, None] + d[None, :] * SD_C,
                      mask=vld[:, None] & dm[None, :], other=0.0)
        js = tl.minimum(row, R - 1)          # address-safe; j < lim_r <= R
        tl.store(CPOOL + (POFF + b * R + js)[:, None] * D + d[None, :],
                 val, mask=(row < lim_r)[:, None] & dm[None, :])


def _pick_g_bn(T, B):
    """G/BN from host-known ``T``/``B`` only (no device sync, graph safe).

    Tokens per request (``T/B``) bound how much a group can share: with <= 1
    token per request a group is pointless and BN must stay 64; with many tokens
    per request G=4 halves the group count, and the UB ceiling forces BN=32 at
    D=512 once BLK_M=32.
    """
    T = int(T)
    B = max(1, int(B))
    if T > 2 * B:
        return 4, 32
    if T > B:
        return 2, 64
    return 1, 64


# --------------------------------------------------------------------------- #
# G: group table + union-window gather (one program per (group, row chunk))
# --------------------------------------------------------------------------- #
@triton.jit
def _v52_gather_kernel(
    OKV, BTO, GKV, GMASK, TBL, NGT, QLEN, KVLEN_O, CUQ,
    NB, RCHV, NPG,
    G_MAX: tl.constexpr, WIN_LEFT: tl.constexpr, RW: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr, MAX_BLOCKS_O: tl.constexpr, S2O: tl.constexpr,
    PA_O: tl.constexpr, D: tl.constexpr, BLK_D: tl.constexpr,
    BN: tl.constexpr,
    STRIDED_O: tl.constexpr = False, SP_O: tl.constexpr = 0,
    SB_O: tl.constexpr = 0, SD_O: tl.constexpr = 1,
    WIDE_O: tl.constexpr = False,
):
    gid = tl.program_id(0)
    ch = tl.program_id(1)
    # ---- cumulative group count over requests -> (b, rb) for this gid
    b_out = 0
    rb = 0
    prev = 0
    for bb in tl.range(0, NB):
        q_l = tl.load(QLEN + bb)
        ngb = (q_l + G_MAX - 1) // G_MAX
        cum = prev + ngb
        hit = (gid >= prev) & (gid < cum)
        b_out = tl.where(hit, bb, b_out)
        rb = tl.where(hit, gid - prev, rb)
        prev = cum
    # program (0,0) publishes the exact group count for the flash kernel
    if (gid == 0) & (ch == 0):
        tl.store(NGT, prev)
    live_g = gid < prev
    b = b_out
    qlen = tl.load(QLEN + b)
    kvlo = tl.load(KVLEN_O + b)
    s1f = rb * G_MAX
    ntok = tl.maximum(tl.minimum(G_MAX, qlen - s1f), 1)
    # rows the flash can reach: whole tiles of the union window
    nr_need = ((WIN_LEFT + ntok + BN - 1) // BN) * BN
    r0 = ch * RCHV
    if r0 >= nr_need:
        return
    # group table (identical values from every chunk: benign duplicate stores)
    tl.store(TBL + gid * 4 + 0, tl.load(CUQ + b) + s1f)
    tl.store(TBL + gid * 4 + 1, b)
    tl.store(TBL + gid * 4 + 2, s1f)
    tl.store(TBL + gid * 4 + 3, ntok)
    d = tl.arange(0, BLK_D)
    dm = d < D
    s1_abs = s1f + (kvlo - qlen)
    base_o = s1_abs - WIN_LEFT
    for i in tl.range(0, RCHV):
        r = r0 + i
        pos = base_o + r
        vld = (pos >= 0) & (pos < kvlo) & (r < nr_need) & live_g
        if PA_O:
            # FL contract (reference _paged_read): a null or out-of-range page
            # id makes the row invalid.  Clamp the address as well so a bad
            # block-table entry can never leave the cache allocation.
            pw = tl.where((pos >= 0) & (pos < kvlo), pos, 0)
            bidx = tl.minimum(pw // BLOCK_SIZE_O, MAX_BLOCKS_O - 1)
            page = tl.load(BTO + b * MAX_BLOCKS_O + bidx, mask=live_g, other=-1)
            vld = vld & (page >= 0) & (page < NPG)
            pg = tl.minimum(tl.maximum(page, 0), NPG - 1)
            if STRIDED_O:
                # arbitrary [P, B, D] view: page/row strides come from the tensor
                if WIDE_O:
                    off = (pg.to(tl.int64) * SP_O
                           + (pw % BLOCK_SIZE_O).to(tl.int64) * SB_O)
                else:
                    off = pg * SP_O + (pw % BLOCK_SIZE_O) * SB_O
            else:
                off = (pg * BLOCK_SIZE_O + pw % BLOCK_SIZE_O) * D
        else:
            off = (b * S2O + tl.where(vld, pos, 0)) * D
        row = tl.load(OKV + off + d * SD_O, mask=vld & dm, other=0.0)
        tl.store(GKV + (gid * RW + r) * D + d, row, mask=(r < nr_need) & dm)
        tl.store(GMASK + gid * RW + r, vld.to(tl.int8), mask=r < nr_need)


# --------------------------------------------------------------------------- #
# F: multi-token flash over [shared window union] + [shared compact pool]
# --------------------------------------------------------------------------- #
@triton.jit
def _v52_flash_kernel(
    Q, GKV, GMASK, TBL, NGT, CPOOL, CMASK,
    KVLEN_C, KVLEN_O, OUT, LSE, SINK, QLEN,
    N1, SCALE, CMP_RATIO,
    N_ORI: tl.constexpr, HAS_CMP: tl.constexpr, MASK_MODE: tl.constexpr,
    HAS_SINK: tl.constexpr, RETURN_LSE: tl.constexpr,
    D: tl.constexpr, BLK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    BLK_M: tl.constexpr, G_MAX: tl.constexpr, RW: tl.constexpr,
    GRID: tl.constexpr, R: tl.constexpr, WIN_LEFT: tl.constexpr,
    NEG: tl.constexpr, NEG_INF: tl.constexpr,
):
    pid = tl.program_id(0)
    mr = tl.arange(0, BLK_M)
    sub = mr // N1                  # token slot of each M row
    hin = mr - sub * N1             # head of each M row
    d = tl.arange(0, BLK_D)
    dm = d < D
    lane = tl.arange(0, BLOCK_N)
    ng = tl.load(NGT)
    n_iter = (ng + GRID - 1) // GRID
    for it in tl.range(0, n_iter):
        gid_raw = pid + it * GRID
        live_g = gid_raw < ng
        gid = tl.where(live_g, gid_raw, 0)
        tok0 = tl.load(TBL + gid * 4 + 0, mask=live_g, other=0)
        b = tl.load(TBL + gid * 4 + 1, mask=live_g, other=0)
        s1f = tl.load(TBL + gid * 4 + 2, mask=live_g, other=0)
        ntok = tl.load(TBL + gid * 4 + 3, mask=live_g, other=1)
        qlen = tl.load(QLEN + b, mask=live_g, other=0)
        live = live_g & (mr < ntok * N1)
        q = tl.load(Q + (tok0 * N1 + mr)[:, None] * D + d[None, :],
                    mask=live[:, None] & dm[None, :], other=0.0)
        qdtype = Q.dtype.element_ty
        m = tl.full([BLK_M], NEG, tl.float32)
        l = tl.zeros([BLK_M], tl.float32)
        acc = tl.zeros([BLK_M, BLK_D], dtype=tl.float32)
        # ---------------- union window: one contiguous range per group
        if N_ORI > 0:
            n_tile_o = (WIN_LEFT + ntok + BLOCK_N - 1) // BLOCK_N
            for c in tl.range(0, n_tile_o, num_stages=1):
                r0 = c * BLOCK_N
                rr = r0 + lane
                kv = tl.load(GKV + (gid * RW + r0 + lane)[:, None] * D + d[None, :],
                             mask=dm[None, :], other=0.0)
                gv = tl.load(GMASK + gid * RW + r0 + lane).to(tl.int1)
                # token g sees rows [g, WIN_LEFT+g] of the union
                keep = ((rr[None, :] >= sub[:, None])
                        & (rr[None, :] <= (WIN_LEFT + sub)[:, None])
                        & gv[None, :] & live[:, None])
                sc = tl.dot(q, tl.trans(kv)) * SCALE
                sc = tl.where(keep, sc, NEG)
                m_new = tl.maximum(m, tl.max(sc, axis=1))
                alpha = tl.exp(m - m_new)
                p = tl.where(keep, tl.exp(sc - m_new[:, None]), 0.0)
                l = alpha * l + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(qdtype), kv, acc)
                m = m_new
        # ---------------- compact pool: shared tiles, per-token mask
        if HAS_CMP:
            kvlc = tl.load(KVLEN_C + b, mask=live_g, other=0)
            if N_ORI > 0:
                ref_len = tl.load(KVLEN_O + b, mask=live_g, other=0)
            else:
                ref_len = kvlc * CMP_RATIO
            off = ref_len - qlen
            lim = tl.maximum(tl.minimum(kvlc, (s1f + off + sub + 1) // CMP_RATIO), 0)
            lim_last = tl.maximum(tl.minimum(kvlc, (s1f + off + ntok) // CMP_RATIO), 0)
            n_tile_c = (tl.minimum(lim_last, R) + BLOCK_N - 1) // BLOCK_N
            for c in tl.range(0, n_tile_c):
                jj = c * BLOCK_N + lane
                ok = jj < R
                js = tl.where(ok, jj, 0)
                kv = tl.load(CPOOL + (b * R + js)[:, None] * D + d[None, :],
                             mask=ok[:, None] & dm[None, :] & live_g, other=0.0)
                sc = tl.dot(q, tl.trans(kv)) * SCALE
                if MASK_MODE == 1:
                    use = (jj[None, :] < lim[:, None]) & ok[None, :]
                else:
                    cnt = tl.load(CMASK + (tok0 + sub)[:, None] * R + js[None, :],
                                  mask=live[:, None] & ok[None, :], other=0)
                    use = cnt > 0
                sc = tl.where(use, sc, NEG)
                m_new = tl.maximum(m, tl.max(sc, axis=1))
                alpha = tl.exp(m - m_new)
                if MASK_MODE == 1:
                    p = tl.where(use, tl.exp(sc - m_new[:, None]), 0.0)
                else:
                    p = tl.where(use,
                                 tl.exp(sc - m_new[:, None]) * cnt.to(tl.float32),
                                 0.0)
                l = alpha * l + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(qdtype), kv, acc)
                m = m_new
        if HAS_SINK:
            sink = tl.load(SINK + hin)
            m_new = tl.maximum(m, sink)
            alpha = tl.exp(m - m_new)
            l = alpha * l + tl.exp(sink - m_new)
            acc = acc * alpha[:, None]
            m = m_new
        s1_rows = s1f + sub
        live_out = (l > 0.0) & (s1_rows < qlen) & live
        res = tl.where(live_out[:, None], acc / tl.maximum(l, 1e-30)[:, None],
                       tl.zeros([BLK_M, BLK_D], dtype=tl.float32))
        tl.store(OUT + (tok0 * N1 + mr)[:, None] * D + d[None, :],
                 res.to(qdtype),
                 mask=live[:, None] & dm[None, :])
        if RETURN_LSE:
            lse_val = tl.where(live_out, m + tl.log(tl.maximum(l, 1e-30)), NEG_INF)
            tl.store(LSE + tok0 * N1 + mr, lse_val, mask=live)


def _pool_rows(T, S2O, S2C, ratio, has_ori):
    """Static upper bound of ``max_b limit_b`` (no host sync)."""
    if S2C <= 0:
        return 1
    if not has_ori:
        return max(1, int(S2C))
    r = -(-(int(T) + int(S2O)) // max(1, int(ratio)))
    return max(1, min(int(S2C), int(r)))


def paged_layout(kv, strides, name, pa):
    """Resolve how a paged ``[P, B, N2, D]`` cache is addressed.

    Returns ``(strided, sp, sb, sd, wide)``:

    * ``strided`` -- take the general path (``page*sp + row*sb + d*sd``).  It is
      forced on whenever the view is **not** the plain contiguous layout *or* the
      view spans >= 2**31 elements (int32 address arithmetic would wrap), and it
      is off only for the small contiguous case, which therefore keeps executing
      the exact code the L1 suite validated.
    * ``wide`` -- widen the address arithmetic to int64.
    * ``sp``/``sb``/``sd`` -- element strides of the 3-D ``[P, B, D]`` view
      (``sd`` is the inner-dimension stride; ``N2 == 1`` so the head stride is
      never used).  ``strides`` may be given explicitly (the adapter does) and is
      then cross-checked against the tensor, or left ``None`` to derive them.

    The ``storage_offset`` of the view needs no term: Triton receives
    ``data_ptr()``, which already includes ``storage_offset * itemsize``.
    """
    if kv is None:
        return False, 0, 0, 1, False
    pages, block, heads, dim = (int(v) for v in kv.shape)
    if not pa:
        if strides is not None:
            raise NotImplementedError(
                f"{name}: explicit strides require layout_kv='PA_ND'"
            )
        return False, 0, 0, 1, False
    if heads != 1:
        raise NotImplementedError(f"{name}: N2==1 is required for paged addressing")
    actual = (int(kv.stride(0)), int(kv.stride(1)), int(kv.stride(3)))
    if strides is None:
        sp, sb, sd = actual
    else:
        sp, sb, sd = (int(v) for v in strides)
        if (sp, sb, sd) != actual:
            raise ValueError(
                f"{name}: strides {(sp, sb, sd)} do not match the tensor "
                f"{actual} (expected the [pages, block, dim] view's strides)"
            )
    if sp < 0 or sb < 0 or sd < 0:
        raise ValueError(f"{name}: negative strides are not supported")
    span = (pages - 1) * sp + (block - 1) * sb + (dim - 1) * sd
    wide = span >= 2**31
    contiguous = (sp, sb, sd) == (block * dim, dim, 1)
    return (not contiguous) or wide, sp, sb, sd, wide



def sparse_attn_sharedkv(
    q: torch.Tensor,
    ori_kv: torch.Tensor | None = None,
    cmp_kv: torch.Tensor | None = None,
    cmp_sparse_indices: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    softmax_scale: float = 1.0,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    cmp_ratio: int = 1,
    return_softmax_lse: bool = False,
    ori_sparse_indices: torch.Tensor | None = None,
    ori_block_table: torch.Tensor | None = None,
    cmp_block_table: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_ori_kv: torch.Tensor | None = None,
    cu_seqlens_cmp_kv: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_kv: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_kv_stride: int = 0,
    cmp_kv_stride: int = 0,
    ori_kv_strides: tuple | None = None,
    cmp_kv_strides: tuple | None = None,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    block_n: int | None = None,
    group_size: int | None = None,
    mask_mode: str | None = None,
    pool_mode: str | None = None,
):
    if layout_q not in ("BSND", "TND"):
        raise NotImplementedError(f"layout_q={layout_q!r} is not implemented")
    if layout_kv not in ("BSND", "PA_ND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"layout_kv={layout_kv!r} is not implemented")
    if ori_sparse_indices is not None:
        raise NotImplementedError("ori_sparse_indices is not implemented")
    if int(ori_kv_stride) != 0 or int(cmp_kv_stride) != 0:
        raise NotImplementedError("ori/cmp_kv_stride != 0 is not implemented")
    if int(ori_mask_mode) != 4:
        raise NotImplementedError(f"ori_mask_mode={ori_mask_mode} is not implemented")
    if int(cmp_mask_mode) != 3:
        raise NotImplementedError(f"cmp_mask_mode={cmp_mask_mode} is not implemented")
    if cmp_kv is None and ori_kv is None:
        raise ValueError("at least one of ori_kv and cmp_kv is required")
    # ``ori_win_left`` is a kernel constexpr, not a fixed constant: the union
    # window is ``[s1 - ori_win_left, s1]`` so any left width works.  A right
    # window is a different gather (rows *after* the query) and is not modelled.
    if int(ori_win_left) < 0:
        raise ValueError(f"ori_win_left must be >= 0, got {ori_win_left}")
    if int(ori_win_right) != 0:
        raise NotImplementedError(
            f"ori_win_right={ori_win_right} is not implemented (left window only)")
    pa = layout_kv in ("PA_ND", "PA_BSND", "PA_BBND")
    tnd = layout_q == "TND"
    dev = q.device
    if tnd:
        if cu_seqlens_q is None:
            raise ValueError("layout_q=TND requires cu_seqlens_q")
        T, N1, D = q.shape
        B = cu_seqlens_q.numel() - 1
        S1 = None
        qlen_b = _as_len_tensor(cu_seqlens_q, B, "cu_seqlens_q", dev)
        qbase = cu_seqlens_q.detach().flatten().to(torch.int32)
        tok_b, tok_s1 = _tok_maps(cu_seqlens_q, T, B, 0, dev)
    else:
        B, S1, N1, D = q.shape
        T = B * S1
        qlen_b = None
        qbase = torch.arange(0, (B + 1) * int(S1), int(S1), dtype=torch.int32,
                             device=dev)
        tok_b, tok_s1 = _tok_maps(None, T, B, S1, dev)
    N2 = (cmp_kv if cmp_kv is not None else ori_kv).shape[2]
    if N2 != 1:
        raise NotImplementedError(f"V5.2 requires N2==1, got {N2}")
    if N1 > 16:
        raise NotImplementedError(f"V5.2 requires N1<=16, got {N1}")
    if D > 512:
        raise NotImplementedError(f"V5.2 requires D<=512, got {D}")

    bso, mbo, S2O, BTO = _prep_kv(ori_kv, ori_block_table, "ori_block_table", N2, D, pa)
    bsc, mbc, S2C, BTC = _prep_kv(cmp_kv, cmp_block_table, "cmp_block_table", N2, D, pa)
    # ---- paged-cache addressing: the cache may be any [P, B, D] view of the
    # shared KV pool (page stride != block*dim), so never require contiguity and
    # never copy: the address becomes page*SP + row*SB + d*SD.
    str_o, sp_o, sb_o, sd_o, wide_o = paged_layout(
        ori_kv, ori_kv_strides, "ori_kv", pa)
    str_c, sp_c, sb_c, sd_c, wide_c = paged_layout(
        cmp_kv, cmp_kv_strides, "cmp_kv", pa)
    if sinks is not None and tuple(sinks.shape) != (N1,):
        raise ValueError(f"sinks must be [N1]={(N1,)}")
    if sinks is not None and sinks.dtype != torch.float32:
        sinks = sinks.float()
    kvlo_b = _as_len_tensor(cu_seqlens_ori_kv, B, "cu_seqlens_ori_kv", dev)
    if kvlo_b is None:
        kvlo_b = _as_len_tensor(seqused_kv, B, "seqused_kv", dev)
    kvlc_b = _as_len_tensor(cu_seqlens_cmp_kv, B, "cu_seqlens_cmp_kv", dev)
    sq = _as_len_tensor(seqused_q, B, "seqused_q", dev)
    if sq is not None:
        qlen_b = sq
    if qlen_b is not None and qlen_b.dtype != torch.int32:
        qlen_b = qlen_b.to(torch.int32)
    if kvlo_b is None:
        kvlo_b = torch.full((B,), S2O, dtype=torch.int32, device=dev)
    if kvlc_b is None:
        kvlc_b = torch.full((B,), S2C if cmp_kv is not None else 0,
                            dtype=torch.int32, device=dev)
    if qlen_b is None:
        qlen_b = torch.full((B,), int(S1), dtype=torch.int32, device=dev)

    n_ori = int(ori_win_left) + int(ori_win_right) + 1
    if ori_kv is None:
        n_ori = 0
    has_cmp = cmp_kv is not None
    k_idx = int(cmp_sparse_indices.shape[-1]) if cmp_sparse_indices is not None else 0
    if not has_cmp:
        k_idx = 0
    if n_ori + (1 if has_cmp else 0) == 0:
        raise ValueError("at least one branch needed")
    ratio = max(1, int(cmp_ratio))
    R = _pool_rows(T, S2O, S2C, ratio, n_ori > 0) if has_cmp else 1

    q2 = q.reshape(T, N1, D)
    if not q2.is_contiguous():
        q2 = q2.contiguous()
    # A paged cache addressed with explicit strides is never copied; only the
    # non-paged layout keeps the historical contiguous fallback.
    okv = ori_kv if (ori_kv is not None and (str_o or ori_kv.is_contiguous())) else (
        ori_kv.contiguous() if ori_kv is not None else q)
    ckv = cmp_kv if (cmp_kv is not None and (str_c or cmp_kv.is_contiguous())) else (
        cmp_kv.contiguous() if cmp_kv is not None else q)
    if k_idx > 0:
        cidx = cmp_sparse_indices.reshape(T, k_idx)
        if cidx.dtype != torch.int32:
            # A slot outside ``[0, R)`` can never be live (``limit <= R``), and a
            # sentinel above INT32_MAX -- indexers emit ``2**40`` -- would *wrap*
            # on the int32 cast (2**40 -> 0, i.e. pool slot 0) and be counted as
            # a real hit by the multiplicity mask.  Fold every dead slot to the
            # ``-1`` padding before the cast; this is exact, not a heuristic.
            dead = (cidx < 0) | (cidx >= R)
            cidx = torch.where(dead, torch.full_like(cidx, -1), cidx).to(torch.int32)
        if not cidx.is_contiguous():
            cidx = cidx.contiguous()
    else:
        cidx = torch.zeros(1, dtype=torch.int32, device=dev)

    out = torch.empty((T, N1, D), dtype=q.dtype, device=dev)
    lse = (torch.empty((T * N1,), dtype=torch.float32, device=dev)
           if return_softmax_lse else torch.empty(0, dtype=torch.float32, device=dev))
    sink_ptr = sinks if sinks is not None else q
    blk_d = next_pow2(D)
    # ---- G/BN rule (V5.3): host known T/B only, no device sync
    g_def, bn_def = _pick_g_bn(T, B)
    block_n = int(bn_def if block_n is None else block_n)
    if block_n not in (16, 32, 64, 128):
        block_n = bn_def
    grid_c = int(GRID_C)
    g_req = int(g_def if group_size is None else group_size)
    g_req = max(1, min(8, g_req))
    blk_m = max(16, next_pow2(g_req * int(N1)))
    g_max = max(1, blk_m // int(N1))          # tokens actually fusable per M tile
    g_max = min(g_max, g_req)
    if mask_mode is None:
        cand = min(int(S2C) if S2C else 0, int(S2O) // ratio) if n_ori > 0 else int(S2C)
        mask_mode = ("limit" if (k_idx == 0 or k_idx >= max(1, cand))
                     else "atomic")
    if mask_mode not in ("limit", "atomic"):
        raise ValueError(f"mask_mode must be 'limit' or 'atomic', got {mask_mode!r}")
    mcode = 1 if mask_mode == "limit" else 0
    win = int(ori_win_left)                                   # kernel constexpr
    rw = ((win + g_max + block_n - 1) // block_n) * block_n   # union window tiles
    nch = rw // block_n
    ng_max = -(-T // g_max) + B
    pool_mode = "page" if pool_mode is None else pool_mode
    # page-level chunk: power of two, divides the cache block size
    pool_ch = int(POOL_CH)
    pool_ch = max(1, min(pool_ch, int(bsc)))
    while pool_ch > 1 and (int(bsc) % pool_ch or (pool_ch & (pool_ch - 1))):
        pool_ch //= 2
    gkv = torch.empty((ng_max * rw, D), dtype=q.dtype, device=dev)
    gmask = torch.empty((ng_max * rw,), dtype=torch.int8, device=dev)
    tbl = torch.empty((ng_max * 4,), dtype=torch.int32, device=dev)
    ngt = torch.zeros((1,), dtype=torch.int32, device=dev)
    woff = 0
    if has_cmp:
        if mcode == 0:
            cmask = (torch.zeros((T * R,), dtype=torch.int32, device=dev)
                     if mask_mode == "atomic"
                     else torch.empty((T * R,), dtype=torch.int32, device=dev))
        else:
            cmask = q2
        cpool = torch.empty((B * R, D), dtype=q.dtype, device=dev)
        capx = 0
        if pool_mode == "row":
            rch = 256
            pg = max(1, -(-R // rch))
            _v5_pool_kernel[(pg, B)](
                ckv, cpool, kvlc_b, kvlo_b, qlen_b,
                BTC if BTC is not None else q2,
                bsc, mbc, S2C, ratio,
                rch, woff,
                N_ORI=n_ori, PA_C=bool(pa), D=D, BLK_D=blk_d, R=R,
                BLOCK_ROWS=block_n,
                STRIDED_C=str_c, SP_C=sp_c, SB_C=sb_c, SD_C=sd_c, WIDE_C=wide_c,
            )
        else:
            npg = int(ckv.shape[0]) if pa else 1
            # static chunk bound for the grid; the *real* count is device side
            pch = max(1, -(-R // pool_ch))
            capx = max(1, min(pch, 16, -(-256 // max(1, int(B)))))
            _v53_pool_page_kernel[(capx, B)](
                ckv, cpool, kvlc_b, kvlo_b, qlen_b,
                BTC if BTC is not None else q2,
                mbc, S2C, npg, ratio,
                woff,
                N_ORI=n_ori, PA_C=bool(pa), D=D, BLK_D=blk_d, R=R,
                BS=int(bsc), CH=pool_ch, BLOCK_ROWS=block_n, CAPX=capx,
                STRIDED_C=str_c, SP_C=sp_c, SB_C=sb_c, SD_C=sd_c, WIDE_C=wide_c,
            )
        if mcode == 0 and mask_mode == "atomic":
            blk_mk = 512
            nchk = max(1, -(-max(R, k_idx) // blk_mk))
            # A single launch would be (T, nchk) programs, i.e. 122880 for a
            # 20480-token prefill step with 6 chunks -- over the 65535 cap, and
            # the op died before running.  Walk the token axis in bounded
            # slices: program_id(0) only indexes TOK_B / TOK_S1 / CIDX / CMASK
            # rows, so slicing those tensors leaves the math untouched.
            step = max(1, min(CMASK_TOK_CHUNK, CMASK_GRID_MAX // nchk))
            for t0 in range(0, T, step):
                tn = min(step, T - t0)
                _v5_cmask_atomic_kernel[(tn, nchk)](
                    cidx[t0:t0 + tn] if k_idx > 0 else cidx,
                    cmask[t0 * R:(t0 + tn) * R],
                    kvlc_b, kvlo_b, qlen_b,
                    tok_b[t0:t0 + tn], tok_s1[t0:t0 + tn],
                    k_idx, ratio,
                    BTC if BTC is not None else cidx,
                    int(bsc), int(mbc), int(ckv.shape[0]) if pa else 1,
                    N_ORI=n_ori, HAS_CIDX=bool(k_idx > 0), PA_C=bool(pa),
                    R=R, BLK=blk_mk,
                )
        elif mcode == 0:
            _v5_cmask_kernel[(T,)](
                cidx, cmask, kvlc_b, kvlo_b, qlen_b, tok_b, tok_s1,
                k_idx, ratio,
                N_ORI=n_ori, HAS_CIDX=bool(k_idx > 0),
                R=R, BJ=(64 if R >= 64 else 16), KCH=MASK_KCH,
            )
    else:
        cpool = q2
        cmask = q2
    if n_ori > 0:
        _v52_gather_kernel[(ng_max, nch)](
            okv, BTO, gkv, gmask, tbl, ngt,
            qlen_b, kvlo_b, qbase,
            B, block_n, int(okv.shape[0]) if pa else 1,
            G_MAX=g_max, WIN_LEFT=win, RW=rw,
            BLOCK_SIZE_O=bso, MAX_BLOCKS_O=mbo, S2O=S2O,
            PA_O=bool(pa), D=D, BLK_D=blk_d, BN=block_n,
            STRIDED_O=str_o, SP_O=sp_o, SB_O=sb_o, SD_O=sd_o, WIDE_O=wide_o,
        )
    else:
        # no window rows are needed; keep the address math inside the dummy KV
        bto0 = (BTO if BTO is not None
                else torch.zeros(max(B, 1), dtype=torch.int32, device=dev))
        _v52_gather_kernel[(ng_max, 1)](
            okv if ori_kv is not None else q, bto0, gkv, gmask, tbl, ngt,
            qlen_b, kvlo_b, qbase,
            B, block_n, 1,
            G_MAX=g_max, WIN_LEFT=0, RW=block_n,
            BLOCK_SIZE_O=1 if BTO is None else bso,
            MAX_BLOCKS_O=1 if BTO is None else mbo,
            S2O=0 if BTO is None else S2O,
            PA_O=bool(pa), D=D, BLK_D=blk_d, BN=block_n,
            STRIDED_O=str_o, SP_O=sp_o, SB_O=sb_o, SD_O=sd_o, WIDE_O=wide_o,
        )
    _v52_flash_kernel[(grid_c,)](
        q2, gkv, gmask, tbl, ngt, cpool, cmask,
        kvlc_b, kvlo_b, out, lse, sink_ptr, qlen_b,
        N1, float(softmax_scale), ratio,
        N_ORI=n_ori, HAS_CMP=bool(has_cmp), MASK_MODE=mcode,
        HAS_SINK=bool(sinks is not None),
        RETURN_LSE=bool(return_softmax_lse),
        D=D, BLK_D=blk_d, BLOCK_N=block_n, BLK_M=blk_m, G_MAX=g_max, RW=rw,
        GRID=grid_c, R=R, WIN_LEFT=win,
        NEG=_NEG, NEG_INF=float("-inf"),
    )
    if tnd:
        out = out.reshape(T, N1, D)
        if return_softmax_lse:
            lse = lse.reshape(T, N1, 1)
    else:
        out = out.reshape(B, int(S1), N1, D)
        if return_softmax_lse:
            lse = lse.reshape(B, int(S1), N1, 1)
    return out, lse

# --------------------------------------------------------------------------- #
# FL ``deepseek_v4_sparse_attention`` -> native ABI (was ``adapter.py``).
# --------------------------------------------------------------------------- #


def _active_rows(request_ids: torch.Tensor,
                 positions: torch.Tensor,
                 num_requests: int) -> torch.Tensor:
    """A row is live only if it names a real request and a real position.

    ``request_ids >= num_requests`` is a padding/sentinel value too: the kernel
    clamps such an id into the last request while building ``cu_seqlens``, so
    without this bound the row would come back as a plausible-looking (or NaN)
    vector instead of the zero the contract promises.
    """
    rid = request_ids.reshape(-1)
    return (rid >= 0) & (rid < int(num_requests)) & (positions.reshape(-1) >= 0)


def _cache_strides(cache: torch.Tensor, name: str) -> tuple[int, int, int]:
    """``[P, B, D]`` paged cache -> its ``(page, block, dim)`` element strides.

    The production cache is a **view of the shared KV pool**, so the page stride
    is generally *not* ``B * D`` (the pool holds more than one KV head / more
    than one cache, and pages may be padded).  Such a view is passed to the
    kernel as-is: the PA_ND addressing becomes ``page*sP + row*sB + d*sD`` and
    nothing is copied, so there is still no host sync and no extra HBM traffic.

    ``storage_offset`` needs no extra term -- Triton's pointer is
    ``data_ptr()``, which already includes it.  That is asserted offline by
    ``tools/check_strided_cache.py`` (it reconstructs every element's absolute
    storage offset from the decomposed indices).
    """
    if cache.ndim != 3:
        raise ValueError(f"{name} must be [pages, block, dim], got {tuple(cache.shape)}")
    strides = tuple(int(s) for s in cache.stride())
    if any(s < 0 for s in strides):
        raise NotImplementedError(f"{name}: negative strides are not supported")
    if cache.shape[2] < 1:
        raise ValueError(f"{name}: empty dim")
    return strides  # type: ignore[return-value]


def _pa_view(cache: torch.Tensor, name: str) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """``[P, B, D]`` -> ``[P, B, 1, D]`` PA_ND view (zero-copy) + its strides."""
    strides = _cache_strides(cache, name)
    return cache.unsqueeze(2), strides


def _cu_seqlens(request_ids: torch.Tensor,
                positions: torch.Tensor,
                num_requests: int) -> torch.Tensor:
    """Per-request token offsets [R + 1] int32, computed on the device.

    A row is counted only if it is *live* (``0 <= request_ids < R`` and
    ``positions >= 0``), so ``cu_seqlens`` is monotonically non-decreasing and
    ends at the number of real tokens.  Counting a dead row would shift every
    later request's slot run by one -- the caller zeroes those rows' outputs, but
    their *slots* are what the group table maps tokens from.  No ``.cpu()`` /
    ``.item()`` / boolean reduction ever leaves the device, hence this is
    capture safe.
    """
    rid = request_ids.reshape(-1)
    if rid.dtype not in (torch.int32, torch.int64):
        rid = rid.to(torch.int64)
    device = rid.device
    real = _active_rows(request_ids, positions, num_requests).to(torch.int64)
    index = rid.clamp(0, max(num_requests - 1, 0))
    counts = torch.zeros(num_requests, dtype=torch.int64, device=device)
    counts = counts.scatter_add(0, index, real)
    offsets = torch.zeros(num_requests + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return offsets


def deepseek_v4_sparse_attention(
    query: torch.Tensor,
    swa_cache: torch.Tensor,
    compact_cache: torch.Tensor | None,
    swa_block_table: torch.Tensor,
    compact_block_table: torch.Tensor | None,
    request_ids: torch.Tensor,
    positions: torch.Tensor,
    swa_lengths: torch.Tensor,
    compact_lengths: torch.Tensor | None,
    compact_indices: torch.Tensor | None,
    window_size: int,
    compress_ratio: int = 4,
    scale: float | None = None,
    attn_sink: torch.Tensor | None = None,
    value_dim: int | None = None,
    unique_indices: bool = True,
) -> torch.Tensor:
    """Reference-equivalent sparse attention through the V5.3 Triton kernel."""
    if query.ndim != 3:
        raise ValueError("query must be [tokens, heads, dim]")
    tokens, num_heads, dim = query.shape
    out_dim = dim if value_dim is None else int(value_dim)
    if not 0 < out_dim <= dim:
        raise ValueError("Value width must be within the shared KV width")
    if int(window_size) < 1 or compress_ratio not in (0, 4, 128):
        raise ValueError(
            "Expected positive SWA window and compression ratio 0, 4 or 128"
        )
    if tokens == 0:
        return query.new_empty((0, num_heads, out_dim))
    if compact_cache is not None and (compress_ratio == 0 or compact_indices is None):
        raise ValueError("Compact attention requires compressed indices and a ratio")
    # ---- capability envelope: the Triton kernel's hard limits ----
    if dim > 512:
        raise NotImplementedError(f"D={dim} exceeds the Triton kernel limit (512)")
    if num_heads > 16:
        raise NotImplementedError(f"H={num_heads} exceeds the Triton kernel limit (16)")

    num_requests = int(swa_lengths.numel())
    if num_requests == 0:
        raise NotImplementedError("a batch without requests is not supported")
    if swa_block_table.ndim != 2 or swa_block_table.shape[0] < num_requests:
        raise NotImplementedError(
            "swa_block_table must provide one row per request"
        )
    if compact_cache is not None:
        if compact_block_table is None or compact_block_table.ndim != 2:
            raise NotImplementedError(
                "compact_block_table is required for compact attention"
            )
        if compact_block_table.shape[0] < num_requests:
            raise NotImplementedError(
                "compact_block_table must provide one row per request"
            )
        if compact_indices.ndim != 2:
            raise ValueError("compact_indices must be [tokens, topk]")

    ori_kv, ori_strides = _pa_view(swa_cache, "swa_cache")
    if compact_cache is None:
        ckv, cmp_strides = None, None
    else:
        ckv, cmp_strides = _pa_view(compact_cache, "compact_cache")

    softmax_scale = float(dim**-0.5) if scale is None else float(scale)
    offsets = _cu_seqlens(request_ids, positions, num_requests)
    # V5.3's G/BN rule assumes BLOCK_N=64 fits UB at the production width; at
    # D=512 the UB ceiling only allows BLOCK_N=32 (the kernel's own tuning
    # note, and measured: BLOCK_N=64 needs 2.17M of 1.57M bits).  A tiling
    # choice, not a math change.
    block_n = 32 if dim > 256 else None

    result, _ = sparse_attn_sharedkv(
        query,
        ori_kv=ori_kv,
        cmp_kv=ckv,
        cmp_sparse_indices=compact_indices,
        sinks=attn_sink,
        softmax_scale=softmax_scale,
        ori_win_left=int(window_size) - 1,
        ori_win_right=0,
        cmp_ratio=int(compress_ratio),
        return_softmax_lse=False,
        ori_block_table=swa_block_table,
        cmp_block_table=compact_block_table,
        cu_seqlens_q=offsets,
        cu_seqlens_ori_kv=swa_lengths,
        cu_seqlens_cmp_kv=compact_lengths,
        layout_q="TND",
        layout_kv="PA_ND",
        # the paged caches are arbitrary [P, B, D] views of the KV pool: address
        # them page*stride(0) + row*stride(1) + d*stride(2) instead of demanding
        # contiguity (a .contiguous() here would copy the whole pool per call).
        ori_kv_strides=ori_strides,
        cmp_kv_strides=cmp_strides,
        # explicit sparse indices (C4 topK, C128 arange).  ``None`` lets the
        # kernel apply its own host-metadata coverage rule: "limit" (dense
        # prefix, one QK/PV pass) when the row covers every candidate, else the
        # exact multiplicity mask.  A caller that cannot guarantee unique slots
        # pins "atomic" instead, because "limit" must not be used on a row with
        # repeats (see the module contract).
        mask_mode="atomic" if (compact_cache is not None and not unique_indices)
        else None,
        block_n=block_n,
    )

    result = result[..., :out_dim]
    # Padding rows (``request_ids < 0`` or past the request count, ``positions <
    # 0``) are in no request's group table, and ``out`` is ``torch.empty`` there,
    # so they are zeroed here.  In place: ``torch.where(active, result, 0)``
    # allocated and filled a second ``[T, N1, out_dim]`` buffer, and on the
    # production path every row is active, so that buffer was pure traffic
    # (0.03-0.07 ms/call in a graph).  The mask is a device tensor -- still no
    # host sync, still capturable.
    result.masked_fill_(
        ~_active_rows(request_ids, positions, num_requests)[:, None, None], 0)
    return result


# The kernel is window-agnostic (``ori_win_left`` is a constexpr), but the
# *dispatch* envelope stays on the production window: ``supports`` is a
# metadata-only probe and cannot see the one assumption the kernel really makes
# -- that a request's tokens are a contiguous slot run with consecutive
# positions (see the module docstring).  Widening the probe beyond the layout
# the caller is known to produce would route generality the kernel does not have.
_DISPATCH_WINDOW = 128


def supports(
    query: torch.Tensor,
    swa_cache: torch.Tensor,
    compact_cache: torch.Tensor | None,
    swa_block_table: torch.Tensor,
    compact_block_table: torch.Tensor | None,
    request_ids: torch.Tensor,
    positions: torch.Tensor,
    swa_lengths: torch.Tensor,
    compact_lengths: torch.Tensor | None,
    compact_indices: torch.Tensor | None,
    window_size: int,
    compress_ratio: int = 4,
    scale: float | None = None,
    attn_sink: torch.Tensor | None = None,
    value_dim: int | None = None,
) -> bool:
    """Is this call inside the V5.3 envelope?  Metadata only -- no device read.

    ``ascend.py`` calls this to choose between this package and
    ``sparse_attn_sharedkv_upstream`` at run time.  It must never sync (no
    ``.item()`` / ``.cpu()`` / device reduction) and never raise, otherwise
    dispatch would stop being graph-capturable; anything unexpected is
    reported as "not ours" and falls back to the upstream kernel, which is the
    implementation that used to serve every call.

    The envelope is deliberately the *validated* one (the shapes the L1 suite
    covers plus the production layout); everything else -- fp32, D>512, H>16,
    the >2**31-element int64 span path, strided metadata -- keeps the upstream
    path.  The window stays pinned at ``_DISPATCH_WINDOW`` for the reason given
    above the constant: the kernel handles any width, the *layout* is what the
    probe cannot check.
    """
    try:
        if not isinstance(query, torch.Tensor) or query.ndim != 3:
            return False
        if query.dtype not in (torch.float16, torch.bfloat16):
            return False
        if int(window_size) != _DISPATCH_WINDOW:
            return False
        if int(compress_ratio) not in (0, 4, 128):
            return False
        heads, dim = int(query.shape[1]), int(query.shape[2])
        if not 0 < dim <= 512 or not 0 < heads <= 16:
            return False
        out_dim = dim if value_dim is None else int(value_dim)
        if not 0 < out_dim <= dim:
            return False
        if compact_cache is not None and (
            int(compress_ratio) == 0
            or compact_indices is None
            or compact_block_table is None
            or compact_lengths is None
        ):
            return False
        if int(swa_lengths.numel()) == 0:
            return False
        # TND query, and metadata the kernel can consume without a hidden
        # ``.contiguous()`` copy (a strided block table would still be correct,
        # but that layout is not in the L1 envelope).
        if query.stride(-1) != 1:
            return False
        for tensor in (
            swa_block_table,
            compact_block_table,
            request_ids,
            positions,
            swa_lengths,
            compact_lengths,
            compact_indices,
            attn_sink,
        ):
            if tensor is not None and not tensor.is_contiguous():
                return False
        # PA_ND caches: any non-negative page/row/inner stride is addressed
        # zero-copy, but a view spanning >= 2**31 elements takes the int64
        # widening path, which is implemented yet not part of the envelope.
        for cache in (swa_cache, compact_cache):
            if cache is None:
                continue
            if cache.ndim != 3 or int(cache.shape[2]) < 1:
                return False
            strides = tuple(int(s) for s in cache.stride())
            if any(s < 0 for s in strides):
                return False
            span = (
                (int(cache.shape[0]) - 1) * strides[0]
                + (int(cache.shape[1]) - 1) * strides[1]
                + (int(cache.shape[2]) - 1) * strides[2]
            )
            if span >= 2**31:
                return False
        return True
    except Exception:  # noqa: BLE001 -- a dispatch probe must never raise
        return False

