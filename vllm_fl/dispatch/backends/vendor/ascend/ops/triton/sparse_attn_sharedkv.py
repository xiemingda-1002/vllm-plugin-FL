# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""sparse_attn_sharedkv -- AscendC -> Triton (BSND/TND query + BSND/PA_ND KV)

Source: csrc/ascend/attention/sparse_attn_sharedkv/
Interface (op_host/sparse_attn_sharedkv_def.cpp L23-111):
    Input : q          fp16/bf16  layout_q="BSND" -> [B,S1,N1,D]; "TND" -> [T,N1,D]
            ori_kv     fp16/bf16  uncompressed KV (used by the sliding window)
                       layout_kv="BSND"  -> [B,S2o,N2,D]
                       layout_kv="PA_ND" -> [numBlocks, blockSize, N2, D] + ori_block_table
            cmp_kv     fp16/bf16  compressed KV (sparse selection), same two layouts +
                       cmp_block_table
            cmp_sparse_indices int32  BSND->[B,S1,N2,K]; TND->[T,N2,K] (-1 = invalid)
            cu_seqlens_q / cu_seqlens_ori_kv / cu_seqlens_cmp_kv / seqused_q / seqused_kv
            sinks      fp32 [N1]
    Output: attn_out same as q; softmax_lse fp32
    Attr  : softmax_scale, cmp_ratio, ori_mask_mode(4), cmp_mask_mode(3),
            ori_kv_stride(0), cmp_kv_stride(0), ori_win_left(127), ori_win_right(0),
            layout_q("BSND"), layout_kv("PA_ND"), return_softmax_lse(False)

Run modes (op_kernel/sparse_attn_sharedkv_common.h L28-32):
    SWA_MODE=0 (sliding window) / SCFA_MODE=1 (sliding window + compressed sparse) / CFA_MODE=2

Formula source (op_kernel/arch32/):
  sparse_attn_sharedkv_swa_kernel.h L690-692 (same in scfa_kernel.h L716-718):
        s2Start = s1EndIdx - oriWinLeft ; s2End = s1EndIdx + oriWinRight
        => the ori KV sliding window = [s1_abs - ori_win_left, s1_abs + ori_win_right],
           s1_abs = s1 + (kvLen_ori - qLen) (aligned to the causal position)
  sparse_attn_sharedkv_swa_block_vector.h:
        L325 Muls(mmRes, mmRes, softmaxScale) -> logits = (q.kv) * softmax_scale
        L268-293 sinks are per-q-head scalars, Brcb broadcasts them to the whole row
        L295-307 WholeReduceMax folds the sink into the max
        L687  the sink's exp contributes to the softmax denominator only
  => semantic equivalence core:
        logits = [ (q.ori_kv[j]) for j in sliding window ] concat
                 [ (q.cmp_kv[t]) for t in cmp_sparse_indices ]
        logits *= softmax_scale
        m   = max( max(logits), sinks[n1] )
        p   = exp(logits - m)
        lse = sum p + exp(sinks[n1] - m)          # the sink enters the denominator, not V
        out = (sum p . v) / lse

layout normalization (the host only does layout bookkeeping; the core math is in Triton):
    TND: cu_seqlens_q yields the (batch, in-sequence position) of each token, and
         cu_seqlens_*/seqused_* yield the kv length of each batch
    PA_ND: paged addressing is done **inside the kernel** with block_table
           ori: page=ori_block_table[b, j//bs]; row=page*bs + j%bs (same for cmp)

**Simplifications / not implemented (unsupported values raise NotImplementedError explicitly)**:
    1. ori_kv_stride / cmp_kv_stride != 0 is not implemented (non-zero means a non-compact
       block layout)
    2. metadata is unused (a multi-core scheduling descriptor that carries no math)
    3. no multi-core Cube/Vector pipeline; a two-pass algorithm (compute logits into GM first,
       then normalize and weight V)
    4. the lower triangle of cmp_mask_mode=3 folds cmp_ratio **at the original sequence position**:
       n_valid_cmp = (s1_abs_orig + 1) // cmp_ratio,
       s1_abs_orig = s1 + (kvLen_ori - qLen) (reconstructed as kvLen_cmp*cmp_ratio without an ori
       branch)
       source: swa_kernel.h L405 / scfa_kernel.h L373,589,600
             thresHold = (cmpMaskRight + s1EndIdx + 1) / cmpRatio;
       tiling validates cmp_ratio in {4, 128} (sparse_attn_sharedkv_tiling.cpp L1267)
    5. the position of the ori branch is decided by the sliding window; no external
       ori_sparse_indices is accepted

## A past mistake (kept as a warning)
The cmp lower-triangle mask in this file was once changed to "do not fold cmp_ratio"
(`j <= s1 + (kvLen_cmp - qLen)`). The reason: after the cmp mask was added the BSND cases
regressed, and **the test reference at the time did not fold cmp_ratio** -- this was misjudged as
"the implementation is wrong", so **the implementation was changed to accommodate a wrong
reference**; the tests went green and the implementation was wrong.

Four independent pieces of evidence later settled that "the reference was wrong" (priority:
repository golden > kernel source > hand-written reference):
  (1) `benchmarks/prepare_indexer_indices.py` L22-27 (repository golden, not written by the author
      of this file): `visible = (positions + 1) // compress_ratio`, with -1 as the sentinel
  (2) this operator's kernel `sparse_attn_sharedkv_swa_kernel.h` L405 (same in scfa_kernel.h
      L373/589/600): `thresHold = (cmpMaskRight + s1EndIdx + 1) / cmpRatio`
  (3) physical meaning `scfa_block_vector.h` L552: cmp cache rows = `kvSeqSize / cmpRatio`
      (every cmpRatio original tokens are compressed into 1 row)
  (4) legal values `sparse_attn_sharedkv_tiling.cpp` L1267: `cmp_ratio in {4, 128}`
      (the value 2 used in the initial test is **illegal**, so even a PASS would not represent
      production)

**Conclusion / rule**: when a regression appears, first decide "is the implementation wrong or the
reference wrong", and **never change the implementation just to make the tests green**; when
writing a reference, **prefer copying the repository's golden** over writing from one's own
understanding.

## Root cause of the TND+PA numerical error (found and fixed; do not revert)
**Symptoms**: BSND 8/8 PASS, TND+PA 4/4 FAIL, fp16 err ~1.46e-01 (threshold 1e-3),
bf16 err ~3.65e-01, plus fp16 saturation values such as `out=512.0`.

**Root cause**: in block-table addressing, **negative indices were not clamped**. At the start of a
sequence the sliding window produces lanes with `j = base_o + lane < 0` (`base_o = s1_abs -
win_left` can be negative); the old `_row_of` wrote `j_safe = tl.maximum(j, 0)`, but **that clamp
did not propagate into `j_safe // BLOCK_SIZE` / `j_safe % BLOCK_SIZE`**: those lanes read not the
logical row 0 but **other rows** (sometimes fully out-of-bounds addresses). Evidence (measured on
device):

* comparing the internal score buffer against the reference logits column by column: only columns
  with `j<0` are wrong (e.g. the 4 columns `j=-59..-56`), while columns with `j>=0` are accurate to
  ~1e-7;
* with an identity block table (page numbers 0,1,2,3) the same 4 columns are still wrong => it does
  not depend on the block-table contents;
* before the fix the score |max| error grew with the module up to 1.5e+02~1.9e+03 and the output
  contained NaN/793/512;
* reverse check: every lane with `j>=0` matches the reference => the math is fine, only addressing.

**Fix (current implementation)**: keep **every index that enters modulo/division/address arithmetic
non-negative**, and express masks with non-negative quantities only:

* sliding window: `row0 = max(base_o, 0)` (address base), `shift = max(-base_o, 0)` (mask only),
  `cnt = min(N_ORI, kvlen - row0)`; valid lanes satisfy `shift <= lane < shift + cnt`, and the
  address row is `row0 + min(max(lane - shift, 0), cnt - 1)`.
  The identity `row0 + (lane-shift) + shift == base_o + lane` keeps the math unchanged.
* sparse cmp: illegal/out-of-range indices (including -1) **fall back to row 0**, only `vld`
  decides the value, and `tl.where(vld, i0, 0)` is applied before `_kv_elem_off` is entered.
* score buffer: padded columns fall back to the last valid column (same reasoning: avoid
  out-of-bounds addresses).
* `_kv_elem_off` accepts only row numbers that are **already guaranteed legal** (it still keeps an
  internal min/max backstop).

**Why "just adding a clamp" is not enough**: an in-place clamp such as `tl.maximum(j, 0)` does not
affect the subsequent `//` and `%` on the Ascend backend (measured), whereas "removing negative
values from addressing entirely" is reliable.
"""

from __future__ import annotations

import os as _os

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["sparse_attn_sharedkv"]

_NEG = -1.0e30

# Debug slot: only when SASKV_DUMP_SCORES=1 does sparse_attn_sharedkv write the intermediate score
# buffer, for numerical localization; the production path neither reads nor writes it (zero cost).
_DBG: dict = {}


@triton.jit
def _kv_elem_off(KV, BT, b, kvrow, n2, BLOCK_SIZE, MAX_BLOCKS, N2, D, S2,
                 PA: tl.constexpr):
    """Linear offset of (row kvrow, head n2, element 0) inside the KV cache.

    ``kvrow`` must be a **logical row number** (under PA = block-table page * blockSize +
    in-block row) and the caller guarantees it lies in ``[0, valid rows of that batch)``. Negative
    or huge values are never accepted: on the Ascend backend both "modulo/division of negative
    values" and "masked-out addresses" go wrong (see the file header).
    """
    if PA:
        bidx = tl.minimum(tl.maximum(kvrow // BLOCK_SIZE, 0), MAX_BLOCKS - 1)
        page = tl.maximum(tl.load(BT + b * MAX_BLOCKS + bidx), 0)
        row = page * BLOCK_SIZE + (kvrow % BLOCK_SIZE)
    else:
        row = b * S2 + kvrow
    return (row * N2 + n2) * D


@triton.jit
def _saskv_score_kernel(
    Q, OKV, CKV, CIDX, SB, TOK_B, TOK_S1, KVLEN_O, KVLEN_C, QLEN, BTO, BTC,
    N1, N2, GS,
    BLOCK_SIZE_O, MAX_BLOCKS_O, S2O, BLOCK_SIZE_C, MAX_BLOCKS_C, S2C, CMP_RATIO,
    SCALE, WIN_LEFT,
    N_ORI: tl.constexpr,
    K_IDX: tl.constexpr,
    PA_O: tl.constexpr,
    PA_C: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_K: tl.constexpr,
    ROW_MODE: tl.constexpr,
):
    row = tl.program_id(0)                 # one row = one (token, q head)
    n1 = row % N1
    t = row // N1
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS

    d = tl.arange(0, BLK_D)
    dm = d < D
    q = tl.load(Q + row * D + d, mask=dm, other=0.0).to(tl.float32)
    total = N_ORI + K_IDX

    # ---- ori KV sliding window ----
    # ROW_MODE=1 replaces the 2-D indirect-addressing [BLK_K, BLK_D] tile with **row-wise 1-D
    # loads**. Measured on device (TND+PA, B2 qlen[5,4] kvo[64,48] kvc[24,16]): for t=0 the ori
    # lanes 68..71 (j=0..3, the 4 earliest valid rows in the window) are numerically wrong while
    # lanes 72+ are exact; for t=5 lanes 83,84 are correct but 85..87 are 0/arbitrary and 88+ are
    # all correct. => the 2-D tile form drops writes / reads dirty data on the **first few valid
    # lanes of the window**, the same family as BACKEND_NOTES issue 1 (2-D tile + indirect
    # addressing). The row-wise 1-D form is the reliable pattern already validated by
    # `_kv_cache_kernel`.
    if N_ORI > 0:
        kvlo = tl.load(KVLEN_O + b)
        base_o = s1 + (kvlo - tl.load(QLEN + b)) - WIN_LEFT
        if ROW_MODE:
            for c in range(0, tl.cdiv(N_ORI, BLK_K)):
                for i in range(BLK_K):
                    li = c * BLK_K + i
                    ji = base_o + li
                    vi = (ji >= 0) & (ji < kvlo) & (li < N_ORI)
                    off_i = _kv_elem_off(OKV, BTO, b, tl.where(vi, ji, 0), n2, BLOCK_SIZE_O,
                                         MAX_BLOCKS_O, N2, D, S2O, PA_O)
                    kvi = tl.load(OKV + off_i + d, mask=vi & dm, other=0.0).to(tl.float32)
                    tl.store(SB + row * total + li,
                             tl.where(vi, tl.sum(kvi * q, axis=0) * SCALE, NEG),
                             mask=li < N_ORI)
        else:
            for c in range(0, tl.cdiv(N_ORI, BLK_K)):
                lane = c * BLK_K + tl.arange(0, BLK_K)
                window = lane < N_ORI
                # remove negative indices per lane with where: negative values never enter
                # addressing (see "root cause" in the file header)
                j = base_o + lane
                vld = (j >= 0) & (j < kvlo)
                rw = tl.where(vld, j, 0)
                off = _kv_elem_off(OKV, BTO, b, rw, n2, BLOCK_SIZE_O, MAX_BLOCKS_O,
                                   N2, D, S2O, PA_O)
                kv = tl.load(OKV + off[:, None] + d[None, :],
                             mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
                sc = tl.sum(kv * q[None, :], axis=1) * SCALE
                tl.store(SB + row * total + lane, tl.where(vld, sc, NEG), mask=window)

    # ---- sparse selection of cmp KV ----
    if K_IDX > 0:
        kvlc = tl.load(KVLEN_C + b)
        # cmp_mask_mode=3 (lower triangle): folds cmp_ratio **at the original sequence position**.
        # source: sparse_attn_sharedkv_swa_kernel.h L405 (same in scfa_kernel.h L373/589/600)
        #     thresHold = (cmpMaskRight + s1EndIdx + 1) / cmpRatio
        # s1EndIdx is the absolute position in the **original sequence** (the same variable is used
        # for the ori sliding window, see L692), so alignment uses the ori kv length; without an
        # ori branch it is reconstructed as cmp length * cmp_ratio.
        if N_ORI > 0:
            ref_len = tl.load(KVLEN_O + b)
        else:
            ref_len = kvlc * CMP_RATIO
        n_valid = (s1 + 1 + (ref_len - tl.load(QLEN + b))) // CMP_RATIO
        if ROW_MODE:
            for c in range(0, tl.cdiv(K_IDX, BLK_K)):
                for i in range(BLK_K):
                    li = c * BLK_K + i
                    ii = tl.load(CIDX + (t * N2 + n2) * K_IDX + li, mask=li < K_IDX, other=-1)
                    vi = (ii >= 0) & (ii < tl.minimum(kvlc, n_valid)) & (li < K_IDX)
                    off_i = _kv_elem_off(CKV, BTC, b, tl.where(vi, ii, 0), n2, BLOCK_SIZE_C,
                                         MAX_BLOCKS_C, N2, D, S2C, PA_C)
                    kvi = tl.load(CKV + off_i + d, mask=vi & dm, other=0.0).to(tl.float32)
                    tl.store(SB + row * total + N_ORI + li,
                             tl.where(vi, tl.sum(kvi * q, axis=0) * SCALE, NEG),
                             mask=li < K_IDX)
        else:
            for c in range(0, tl.cdiv(K_IDX, BLK_K)):
                slot = c * BLK_K + tl.arange(0, BLK_K)
                window = slot < K_IDX
                i0 = tl.load(CIDX + (t * N2 + n2) * K_IDX + slot, mask=window, other=-1)
                # illegal indices (-1 / out of range) fall back to row 0 and only the mask decides
                # the value: addresses always stay legal
                vld = (i0 >= 0) & (i0 < tl.minimum(kvlc, n_valid))
                off = _kv_elem_off(CKV, BTC, b, tl.where(vld, i0, 0), n2, BLOCK_SIZE_C,
                                   MAX_BLOCKS_C, N2, D, S2C, PA_C)
                kv = tl.load(CKV + off[:, None] + d[None, :],
                             mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
                sc = tl.sum(kv * q[None, :], axis=1) * SCALE
                tl.store(SB + row * total + N_ORI + slot, tl.where(vld, sc, NEG), mask=window)


@triton.jit
def _saskv_out_kernel(
    OKV, CKV, SB, CIDX, SINK, OUT, LSE_MAX, LSE_SUM,
    TOK_B, TOK_S1, KVLEN_O, KVLEN_C, QLEN, BTO, BTC,
    N1, N2, GS,
    BLOCK_SIZE_O, MAX_BLOCKS_O, S2O, BLOCK_SIZE_C, MAX_BLOCKS_C, S2C, CMP_RATIO,
    WIN_LEFT, LSE_B_STRIDE, LSE_N2_STRIDE,
    N_ORI: tl.constexpr,
    K_IDX: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    PA_O: tl.constexpr,
    PA_C: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_ALL: tl.constexpr,
    ROW_MODE: tl.constexpr,
):
    row = tl.program_id(0)
    pid_d = tl.program_id(1)
    n1 = row % N1
    t = row // N1
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS
    # D-direction tiling (second grid axis): keeps the UB footprint of the [BLK_K, BLK_D] KV tile
    # down. Note: an in-block loop was tried once to save programs, but the TND+PA cases produced
    # NaN (numerically wrong), so the grid-axis scheme was restored -- the grid-axis version is the
    # one validated by all unit tests passing.
    d = pid_d * BLK_D + tl.arange(0, BLK_D)
    dm = d < D

    total = N_ORI + K_IDX
    ar = tl.arange(0, BLK_ALL)
    # Address clamp: `ar` can reach BLK_ALL-1 > total, so `row*total + ar` of the last row would
    # run past the SB buffer (masks do not stop address computation, see "root cause" in the file
    # header).
    sv = tl.load(SB + row * total + tl.minimum(ar, total - 1), mask=ar < total, other=NEG)
    if HAS_SINK:
        sink = tl.load(SINK + n1)
        mx = tl.maximum(tl.max(sv, axis=0), sink)
        p = tl.where(sv > NEG, tl.exp(sv - mx), 0.0)
        lse = tl.sum(p, axis=0) + tl.exp(sink - mx)      # the sink enters the denominator only
    else:
        mx = tl.max(sv, axis=0)
        p = tl.where(sv > NEG, tl.exp(sv - mx), 0.0)
        lse = tl.sum(p, axis=0)

    # ---- read the header scalars once (independent of D) ----
    if N_ORI > 0:
        kvlo = tl.load(KVLEN_O + b)
        base_o = s1 + (kvlo - tl.load(QLEN + b)) - WIN_LEFT
    if K_IDX > 0:
        kvlc = tl.load(KVLEN_C + b)
        # same as the score kernel: the cmp lower triangle folds cmp_ratio **at the original
        # sequence position** (swa_kernel.h L405 / scfa_kernel.h L373)
        if N_ORI > 0:
            ref_len = tl.load(KVLEN_O + b)
        else:
            ref_len = kvlc * CMP_RATIO
        n_valid = (s1 + 1 + (ref_len - tl.load(QLEN + b))) // CMP_RATIO

    acc = tl.zeros([BLK_D], dtype=tl.float32)

    # Same row-wise 1-D load with ROW_MODE=1 as the score kernel (see the comment above it).
    if N_ORI > 0:
        if ROW_MODE:
            for c in range(0, tl.cdiv(N_ORI, BLK_K)):
                for i in range(BLK_K):
                    li = c * BLK_K + i
                    ji = base_o + li
                    vi = (ji >= 0) & (ji < kvlo) & (li < N_ORI)
                    off_i = _kv_elem_off(OKV, BTO, b, tl.where(vi, ji, 0), n2, BLOCK_SIZE_O,
                                         MAX_BLOCKS_O, N2, D, S2O, PA_O)
                    kvi = tl.load(OKV + off_i + d, mask=vi & dm, other=0.0).to(tl.float32)
                    pci = tl.load(SB + row * total + li, mask=li < N_ORI, other=NEG)
                    acc += tl.where(vi, tl.exp(pci - mx), 0.0) * kvi
        else:
            for c in range(0, tl.cdiv(N_ORI, BLK_K)):
                lane = c * BLK_K + tl.arange(0, BLK_K)
                window = lane < N_ORI
                j = base_o + lane
                vld = (j >= 0) & (j < kvlo)
                rw = tl.where(vld, j, 0)
                pc = tl.load(SB + row * total + lane, mask=window, other=NEG)
                pc = tl.where(vld, tl.exp(pc - mx), 0.0)
                off = _kv_elem_off(OKV, BTO, b, rw, n2, BLOCK_SIZE_O, MAX_BLOCKS_O,
                                   N2, D, S2O, PA_O)
                kv = tl.load(OKV + off[:, None] + d[None, :],
                             mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(pc[:, None] * kv, axis=0)

    if K_IDX > 0:
        if ROW_MODE:
            for c in range(0, tl.cdiv(K_IDX, BLK_K)):
                for i in range(BLK_K):
                    li = c * BLK_K + i
                    ii = tl.load(CIDX + (t * N2 + n2) * K_IDX + li, mask=li < K_IDX, other=-1)
                    vi = (ii >= 0) & (ii < tl.minimum(kvlc, n_valid)) & (li < K_IDX)
                    off_i = _kv_elem_off(CKV, BTC, b, tl.where(vi, ii, 0), n2, BLOCK_SIZE_C,
                                         MAX_BLOCKS_C, N2, D, S2C, PA_C)
                    kvi = tl.load(CKV + off_i + d, mask=vi & dm, other=0.0).to(tl.float32)
                    pci = tl.load(SB + row * total + N_ORI + li, mask=li < K_IDX, other=NEG)
                    acc += tl.where(vi, tl.exp(pci - mx), 0.0) * kvi
        else:
            for c in range(0, tl.cdiv(K_IDX, BLK_K)):
                slot = c * BLK_K + tl.arange(0, BLK_K)
                window = slot < K_IDX
                i0 = tl.load(CIDX + (t * N2 + n2) * K_IDX + slot, mask=window, other=-1)
                vld = (i0 >= 0) & (i0 < tl.minimum(kvlc, n_valid))
                pc = tl.load(SB + row * total + N_ORI + slot, mask=window, other=NEG)
                pc = tl.where(vld, tl.exp(pc - mx), 0.0)
                off = _kv_elem_off(CKV, BTC, b, tl.where(vld, i0, 0), n2, BLOCK_SIZE_C,
                                   MAX_BLOCKS_C, N2, D, S2C, PA_C)
                kv = tl.load(CKV + off[:, None] + d[None, :],
                             mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(pc[:, None] * kv, axis=0)

    res = tl.where(lse > 0.0, acc / lse, tl.zeros([BLK_D], dtype=tl.float32))
    tl.store(OUT + row * D + d, res.to(OUT.dtype.element_ty), mask=dm)

    if RETURN_LSE:
        if pid_d == 0:      # LSE is independent of D, so write it only once
            tl.store(LSE_MAX + b * LSE_B_STRIDE + n2 * LSE_N2_STRIDE + s1 * GS + (n1 % GS), mx)
            tl.store(LSE_SUM + b * LSE_B_STRIDE + n2 * LSE_N2_STRIDE + s1 * GS + (n1 % GS), lse)


def _lens_per_batch(seq, B, name) -> torch.Tensor | None:
    """[B] per-batch lengths or [B+1] cu_seqlens -> per-batch lengths (int32 CPU)."""
    if seq is None:
        return None
    s = seq.detach().to(torch.int64).cpu().flatten()
    if s.numel() == B:
        return s.to(torch.int32)
    if s.numel() == B + 1:
        return (s[1:] - s[:-1]).to(torch.int32)
    raise ValueError(f"{name} length must be B={B} or B+1={B + 1}, got {s.numel()}")


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
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
):
    """Equivalent of the AscendC SparseAttnSharedkv (BSND/TND x BSND/PA_ND).

    Args:
        q: BSND [B,S1,N1,D] / TND [T,N1,D]
        ori_kv/cmp_kv: BSND [B,S2*,N2,D] / PA_ND [numBlocks, blockSize, N2, D]
        cmp_sparse_indices: BSND [B,S1,N2,K] / TND [T,N2,K] (-1 = invalid)
        sinks: [N1] fp32
        cu_seqlens_q: required when layout_q="TND" (cu_seqlens, length B+1, last entry = T)
        cu_seqlens_ori_kv/cu_seqlens_cmp_kv: kv length per batch ([B] or [B+1] cu_seqlens)
        seqused_q/seqused_kv: used q/kv length per batch ([B])
        ori_block_table/cmp_block_table: required when layout_kv="PA_ND"
        layout_q/layout_kv: "BSND"/"TND" and "BSND"/"PA_ND" ("PA_BSND" is a synonym)
        ori_kv_stride/cmp_kv_stride: only 0 is supported

    Returns:
        (attn_out same as q, lse_max, lse_sum)
    """
    if layout_q not in ("BSND", "TND"):
        raise NotImplementedError(f"layout_q={layout_q!r} is not implemented (supported: BSND/TND)")
    if layout_kv not in ("BSND", "PA_ND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"layout_kv={layout_kv!r} is not implemented (supported: BSND/PA_ND)")
    if ori_sparse_indices is not None:
        raise NotImplementedError("ori_sparse_indices is not implemented (ori branch positions are decided by the sliding window)")
    if int(ori_kv_stride) != 0 or int(cmp_kv_stride) != 0:
        raise NotImplementedError("ori/cmp_kv_stride != 0 is not implemented (non-compact block layout)")
    if int(ori_mask_mode) != 4:
        raise NotImplementedError(f"ori_mask_mode={ori_mask_mode} is not implemented (only 4: sliding window is supported)")
    if int(cmp_mask_mode) != 3:
        raise NotImplementedError(f"cmp_mask_mode={cmp_mask_mode} is not implemented (only 3: lower triangle is supported)")
    if metadata is not None and metadata.dtype != torch.int32:
        raise ValueError(f"metadata must be an int32 descriptor tensor, got {metadata.dtype}")
    if cmp_kv is None and ori_kv is None:
        raise ValueError("at least one of ori_kv and cmp_kv is required")
    pa = layout_kv in ("PA_ND", "PA_BSND", "PA_BBND")
    tnd = layout_q == "TND"

    # ---------------- layout normalization ----------------
    if tnd:
        if cu_seqlens_q is None:
            raise ValueError("layout_q=TND requires cu_seqlens_q (length B+1, last entry = T)")
        T, N1, D = q.shape
        cu = cu_seqlens_q.detach().to(torch.int64).cpu().flatten()
        if cu.numel() < 2 or int(cu[-1]) != T:
            raise ValueError(f"cu_seqlens_q must be cu_seqlens (length B+1, last entry {T})")
        B = cu.numel() - 1
        lens = cu[1:] - cu[:-1]
        tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64) - cu[:-1].repeat_interleave(lens)).to(torch.int32)
        qlen_b = lens.to(torch.int32)
        S1 = None
    else:
        B, S1, N1, D = q.shape
        T = B * S1
        tok_b = torch.arange(B, dtype=torch.int32).repeat_interleave(S1)
        tok_s1 = torch.arange(S1, dtype=torch.int32).repeat(B)
        qlen_b = None

    N2 = (cmp_kv if cmp_kv is not None else ori_kv).shape[2]
    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} must be divisible by N2={N2}")
    gs = N1 // N2

    def _kv_geom(kv, bt, nm):
        """Return (block_size, max_blocks, S2, bt_tensor)."""
        if kv is None:
            return 1, 0, 0, q
        if pa:
            nb, bs, n2k, dk = kv.shape
            if bt is None:
                raise ValueError(f"layout_kv={layout_kv} requires the block_table for {nm}")
            if n2k != N2 or dk != D:
                raise ValueError(f"{nm} N2/D does not match q: {tuple(kv.shape)} vs N2={N2},D={D}")
            return bs, bt.shape[1], bs * bt.shape[1], bt.contiguous().to(torch.int32)
        _b, s2, n2k, dk = kv.shape
        if n2k != N2 or dk != D:
            raise ValueError(f"{nm} N2/D does not match q: {tuple(kv.shape)} vs N2={N2},D={D}")
        return 1, 0, s2, q

    bso, mbo, S2O, BTO = _kv_geom(ori_kv, ori_block_table, "ori_block_table")
    bsc, mbc, S2C, BTC = _kv_geom(cmp_kv, cmp_block_table, "cmp_block_table")

    if ori_kv is not None and S2O <= 0:
        raise ValueError("invalid kv length for ori_kv")
    if cmp_kv is not None and cmp_sparse_indices is None:
        raise ValueError("cmp_sparse_indices is required when cmp_kv is given")
    if sinks is not None and tuple(sinks.shape) != (N1,):
        raise ValueError(f"sinks must be [N1]={(N1,)}")
    if sinks is not None and sinks.dtype != torch.float32:
        sinks = sinks.float()

    kvlo_b = _lens_per_batch(cu_seqlens_ori_kv, B, "cu_seqlens_ori_kv")
    kvlc_b = _lens_per_batch(cu_seqlens_cmp_kv, B, "cu_seqlens_cmp_kv")
    if kvlo_b is None:
        kvlo_b = _lens_per_batch(seqused_kv, B, "seqused_kv")
    if kvlo_b is None:
        kvlo_b = torch.full((B,), S2O, dtype=torch.int32)
    if kvlc_b is None:
        kvlc_b = torch.full((B,), S2C if cmp_kv is not None else 0, dtype=torch.int32)
    sq = _lens_per_batch(seqused_q, B, "seqused_q")
    if sq is not None:
        qlen_b = sq
    if qlen_b is None:
        qlen_b = torch.full((B,), S1, dtype=torch.int32)

    n_ori = int(ori_win_left) + int(ori_win_right) + 1
    if ori_kv is None:
        n_ori = 0
    k_idx = cmp_sparse_indices.shape[-1] if cmp_sparse_indices is not None else 0
    if cmp_kv is None:
        k_idx = 0
    if n_ori + k_idx == 0:
        raise ValueError("at least one of ori_kv and cmp_kv must provide a valid branch")

    dev = q.device
    q2 = q.contiguous().reshape(T, N1, D)
    okv = ori_kv.contiguous() if ori_kv is not None else q
    ckv = cmp_kv.contiguous() if cmp_kv is not None else q
    if cmp_sparse_indices is not None:
        cidx = cmp_sparse_indices.contiguous().reshape(T, N2, k_idx).to(torch.int32)
    else:
        cidx = torch.zeros(1, dtype=torch.int32, device=dev)

    n_rows = T * N1
    sb = torch.empty((n_rows, n_ori + k_idx), dtype=torch.float32, device=dev)
    out = torch.empty((n_rows, D), dtype=q.dtype, device=dev)
    blk_d = next_pow2(D)
    # The UB footprint of the [BLK_K, BLK_D] KV/score tile must stay under 192KB: with D=576,
    # BLK_D=1024 and BLK_K=16 report `ub overflow, requires 8691712 bits` (5.5x); shrinking is
    # allowed through an environment variable (the default was chosen after measurement).
    blk_k = int(_os.environ.get("SASKV_BLK_K", "8"))
    # ROW_MODE=1 (default): row-wise 1-D loads, avoiding the Ascend backend issue where
    # "2-D tile + indirect addressing" drops writes on the first few valid lanes of the window
    # (see the comments inside both kernels). Setting SASKV_ROW_MODE=0 restores the 2-D tile form,
    # for A/B comparison only.
    row_mode = 0 if _os.environ.get("SASKV_ROW_MODE") == "0" else 1
    common = (tok_b.to(dev), tok_s1.to(dev), kvlo_b.to(dev), kvlc_b.to(dev), qlen_b.to(dev))
    geom = (bso, mbo, S2O, bsc, mbc, S2C, int(cmp_ratio))

    _saskv_score_kernel[(n_rows,)](
        q2, okv, ckv, cidx, sb, *common, BTO, BTC,
        N1, N2, gs, *geom,
        float(softmax_scale), int(ori_win_left),
        N_ORI=n_ori, K_IDX=k_idx,
        PA_O=bool(pa), PA_C=bool(pa), NEG=_NEG, D=D, BLK_D=blk_d, BLK_K=blk_k,
        ROW_MODE=row_mode,
    )
    if _os.environ.get("SASKV_DUMP_SCORES"):
        # debug slot (see the _DBG definition): used to compare the internal scores with the
        # reference logits
        _DBG["sb"] = sb
    if tnd:
        lse_shape = (N2, T, gs)
    else:
        lse_shape = (B, N2, S1, gs)
    if return_softmax_lse:
        lse_max = torch.empty(lse_shape, dtype=torch.float32, device=dev)
        lse_sum = torch.empty(lse_shape, dtype=torch.float32, device=dev)
    else:
        lse_max = torch.empty(0, dtype=torch.float32, device=dev)
        lse_sum = torch.empty(0, dtype=torch.float32, device=dev)
    sink_ptr = sinks if sinks is not None else q
    lse_b_stride = 0 if tnd else (N2 * S1 * gs)
    lse_n2_stride = (T * gs) if tnd else (S1 * gs)

    # D-direction tiling: the [BLK_K, BLK_D] KV tile plus the int64 offset vector of
    # `_saskv_out_kernel` need 8,691,712 bits of UB at BLK_D=next_pow2(D)=1024 (5.5x), and
    # shrinking only BLK_K to 2,886,144 bits is still 1.8x over => tile along D.
    # The tiling runs in an **in-block loop** (the grid stays n_rows): putting it on a grid axis
    # would multiply the program count by ceil(D/BLK_D) and recompute the scores in every copy,
    # which times out the whole production run (RC=124).
    blk_d_out = max(16, min(blk_d, int(_os.environ.get("SASKV_BLK_D_OUT", "128"))))
    _saskv_out_kernel[(n_rows, ceil_div(D, blk_d_out))](
        okv, ckv, sb, cidx, sink_ptr, out, lse_max, lse_sum,
        *common, BTO, BTC,
        N1, N2, gs, *geom,
        int(ori_win_left), lse_b_stride, lse_n2_stride,
        N_ORI=n_ori, K_IDX=k_idx,
        HAS_SINK=bool(sinks is not None), RETURN_LSE=bool(return_softmax_lse),
        PA_O=bool(pa), PA_C=bool(pa), NEG=_NEG, D=D, BLK_D=blk_d_out, BLK_K=blk_k,
        BLK_ALL=next_pow2(max(1, n_ori + k_idx)),
        ROW_MODE=row_mode,
    )
    if tnd:
        out = out.reshape(T, N1, D)
    else:
        out = out.reshape(B, S1, N1, D)
    return out, lse_max, lse_sum
