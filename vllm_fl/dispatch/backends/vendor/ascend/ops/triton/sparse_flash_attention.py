# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""sparse_flash_attention -- AscendC -> Triton (supports BSND / TND query + BSND / PA_BSND KV)

Source: csrc/ascend/attention/sparse_flash_attention/
Interface (op_host/sparse_flash_attention_def.cpp L23-88):
    Input : query      fp16/bf16   layout_query="BSND" -> [B,S1,N1,512]
                                   layout_query="TND"  -> [T,N1,512] (tokens of all
                                                          sequences concatenated)
            key        fp16/bf16   layout_kv="BSND"    -> [B,S2,N2,512]
                                   layout_kv="PA_BSND" -> [numBlocks, blockSize, N2, 512]
            value      same as key
            sparse_indices int32   BSND -> [B,S1,N2,K]; TND -> [T,N2,K]
            query_rope fp16/bf16   same layout as query, last dim 64
            key_rope   same as key
            block_table int32 [B, maxBlockNumPerBatch] (required for PA)
            actual_seq_lengths_query [B] or [B+1] (cumulative)
            actual_seq_lengths_kv    [B] or [B+1] (cumulative)
    Attr  : scale_value, sparse_block_size(1), sparse_mode(3, lower triangular),
            attention_mode(2), return_softmax_lse(False), pre/next_tokens(INT64_MAX)

Formula source:
  op_kernel/arch35/sparse_flash_attention_service_vector_mla.h
    L790  dSizeRope = 64                          -> KV row = [512 dense | 64 rope]
    L791  softmaxScale = scaleValue
    L470-529 online softmax; L572 LastDivNew: output = acc / sum
    L266-310 keyRopeGm[keyOffset * 64] / blkTableSrcStride:
        PA page addressing maps a logical block to a physical block through block_table
  => MLA sparse attention core:
        j is taken from sparse_indices[b, s1, n2, :] (-1 means invalid)
        kv_row(j) = BSND: b*S2 + j
                    PA_BSND: block_table[b, j//blockSize]*blockSize + j%blockSize
        qk    = ( q.kv[j] + q_rope.kv_rope[j] ) * scale_value
        lower triangular: j <= s1 + (kvLen[b] - qLen[b])
        softmax only over the selected positions; out = sum(p.v) / sum(p)

Layout normalization (the host only does **layout bookkeeping**; the core math still runs in
Triton):
    - TND: actual_seq_lengths_query gives the (batch, in-sequence position) of every token and
      the q length of every batch (equivalent to a searchsorted over cu_seqlens; vLLM likewise
      precomputes seq_idx on the host)
    - PA_BSND: page addressing is done **inside the kernel** with block_table

**Implemented layout combinations**: query in {BSND,TND} x kv in {BSND,PA_BSND}.
**Simplifications**: no multi-core Cube/Vector pipeline; a two-pass algorithm (score first into
          GM, then normalize and weight V); `sparse_block_size>1` is implemented as "each index
          expands into bs consecutive KV rows".

Backend pitfall (910_93 + BiShengIR):
  With `sparse_block_size > 1`, if the intermediate scratch `SB` of the two-pass algorithm is
  written in an **interleaved layout** `SB[row, slot, sub]` (store offset `slot*BS + sub`,
  i.e. a strided write of 1-D stride BS plus a mask), BiShengIR fails the
  `InterleaveStatusWithMaskOptimization` assertion in `InterleaveOptimization.cpp:635`
  -> **SIGABRT (RC=134, a compile-time crash)**:
      Assertion `llvm::isa<Attribute>(extractSizes.back())' failed
  The trigger condition is having **stride + mask** at the same time (removing the mask,
  switching to contiguous, or switching to a 2-D [BLK,BS] store does not trigger it; all
  measured). `SB` is internal scratch, so it is written here in a **blocked layout**
  `SB[row, sub, slot]` instead (contiguous store, no stride), and the lane -> (slot, sub)
  mapping in `_sfa_out_kernel` is changed to sub-major accordingly; sums and maxes do not
  depend on lane order, so the numerical semantics are unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["sparse_flash_attention"]

_NEG = -1.0e30
_D_ROPE = 64


@triton.jit
def _kv_row(KV, BT, b, j, n2, BLOCK_SIZE, MAX_BLOCKS, N2, D, S2,
            PA: tl.constexpr, vld):
    """Resolve the logical KV position j into the **element offset of the row start** in the
    paged/contiguous KV cache.

        BSND    : row = b*S2 + j
        PA_BSND : row = block_table[b, j//BLOCK_SIZE]*BLOCK_SIZE + (j % BLOCK_SIZE)
    Returns (row * N2 + n2) * D, i.e. the element offset of that (row, head n2) row in the
    KV cache.
    """
    # Clamp j to a legal value unconditionally.
    #   The outer code computes j = i0*BS + sub, and i0 is -1 for invalid slots, so **j can
    #   be negative**. Computing `j // BLOCK_SIZE` from an unclamped j yields a negative block
    #   number and an address before BT; on Ascend a masked load still computes addresses, so
    #   the device reports `aivec error / The DDR address of the MTE instruction is out of
    #   range` (observed with GLM-5.2-w8a8). Invalid lanes use 0 (always a legal address); the
    #   values are still masked out by vld, so the result is unaffected.
    j = tl.where(vld, j, 0)
    if PA:
        page = tl.load(BT + b * MAX_BLOCKS + j // BLOCK_SIZE, mask=vld, other=0)
        kvrow = page * BLOCK_SIZE + (j % BLOCK_SIZE)
    else:
        kvrow = b * S2 + j
    return (kvrow * N2 + n2) * D


@triton.jit
def _sfa_score_kernel(
    Q, QR, KV, KVR, IDX, SB, TOK_B, TOK_S1, KVLEN, QLEN, BT,
    N1, N2, GS, K_IDX, N_TOK,
    BLOCK_SIZE, MAX_BLOCKS, S2,
    SCALE,
    PA: tl.constexpr,
    CAUSAL: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr, DR: tl.constexpr,
    BLK_D: tl.constexpr, BLK_R: tl.constexpr,
    BLK_K: tl.constexpr, BS: tl.constexpr,
):
    row = tl.program_id(0)                 # one row = one (token, q head)
    n1 = row % N1
    t = row // N1
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS

    d = tl.arange(0, BLK_D)
    dm = d < D
    dr = tl.arange(0, BLK_R)
    drm = dr < DR

    q = tl.load(Q + row * D + d, mask=dm, other=0.0).to(tl.float32)
    qr = tl.load(QR + row * DR + dr, mask=drm, other=0.0).to(tl.float32)
    idx_base = (t * N2 + n2) * K_IDX

    kvlen = tl.load(KVLEN + b)
    if CAUSAL:
        j_hi = s1 + (kvlen - tl.load(QLEN + b))
    else:
        j_hi = S2 - 1

    for c in range(0, tl.cdiv(K_IDX, BLK_K)):
        slot = c * BLK_K + tl.arange(0, BLK_K)
        sm = slot < K_IDX
        i0 = tl.load(IDX + idx_base + slot, mask=sm, other=-1)
        for sub in tl.static_range(BS):
            j = i0 * BS + sub
            vld = sm & (i0 >= 0) & (j >= 0) & (j <= j_hi) & (j < kvlen)
            base = _kv_row(KV, BT, b, j, n2, BLOCK_SIZE, MAX_BLOCKS, N2, D, S2, PA, vld)
            k = tl.load(KV + base[:, None] + d[None, :],
                        mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
            baser = _kv_row(KVR, BT, b, j, n2, BLOCK_SIZE, MAX_BLOCKS, N2, DR, S2, PA, vld)
            kr = tl.load(KVR + baser[:, None] + dr[None, :],
                         mask=vld[:, None] & drm[None, :], other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) + tl.sum(kr * qr[None, :], axis=1)
            s = tl.where(vld, s * SCALE, NEG)
            # WARNING: the internal layout of SB is **blocked**, `SB[row, sub, slot]` (sub
            # outer, slot inner), and it must not be written in the interleaved form
            # `slot*BS + sub`: a store with "1-D stride (stride = BS) + mask" makes
            # BiShengIR fail the `InterleaveOptimization.cpp:635
            # InterleaveStatusWithMaskOptimization` assertion and **SIGABRT (RC=134, a
            # compile-time crash)**:
            #     Assertion `llvm::isa<Attribute>(extractSizes.back())' failed
            # All measured in the same kernel:
            #   stride2_masked -> RC=134 (reproduced); stride2_nomask -> OK (**removing the
            #   mask avoids it**); contig_masked -> OK (contiguous); twod -> OK (2-D [BLK,BS]).
            # SB is only internal scratch of this operator, so its layout is free; hence the
            # contiguous write here, and the lane -> (slot, sub) mapping in `_sfa_out_kernel`
            # is changed to sub-major accordingly.
            tl.store(SB + row * (K_IDX * BS) + sub * K_IDX + slot, s, mask=sm)


@triton.jit
def _sfa_out_kernel(
    KV, SB, IDX, OUT, LSE_MAX, LSE_SUM, TOK_B, TOK_S1, BT,
    N1, N2, GS, K_IDX, N_TOK,
    BLOCK_SIZE, MAX_BLOCKS, S2, LSE_B_STRIDE, LSE_N2_STRIDE,
    RETURN_LSE: tl.constexpr,
    PA: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_ALL: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0)
    n1 = row % N1
    t = row // N1
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)
    n2 = n1 // GS

    d = tl.arange(0, BLK_D)
    dm = d < D
    total = K_IDX * BS
    ar = tl.arange(0, BLK_ALL)
    sv = tl.load(SB + row * total + ar, mask=ar < total, other=NEG)
    mx = tl.max(sv, axis=0)
    p = tl.where(sv > NEG, tl.exp(sv - mx), 0.0)
    lse = tl.sum(p, axis=0)

    acc = tl.zeros([BLK_D], dtype=tl.float32)
    for c in range(0, tl.cdiv(total, BLK_K)):
        lane = c * BLK_K + tl.arange(0, BLK_K)
        sm = lane < total
        pc = tl.load(SB + row * total + lane, mask=sm, other=NEG)
        pc = tl.where(sm, tl.exp(pc - mx), 0.0)
        # SB uses the **blocked** layout SB[row, sub, slot] (consistent with the store in
        # _sfa_score_kernel): sub is the high part and slot the low part. Going back to the
        # interleaved layout triggers the FLIR compile-time crash; see the note there.
        slot = lane % K_IDX
        sub = lane // K_IDX
        i0 = tl.load(IDX + (t * N2 + n2) * K_IDX + slot, mask=sm, other=-1)
        j = i0 * BS + sub
        vld = sm & (i0 >= 0) & (j >= 0) & (j < S2)
        base = _kv_row(KV, BT, b, j, n2, BLOCK_SIZE, MAX_BLOCKS, N2, D, S2, PA, vld)
        v = tl.load(KV + base[:, None] + d[None, :],
                    mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(vld, pc, 0.0)[:, None] * v, axis=0)

    res = tl.where(lse > 0.0, acc / lse, tl.zeros([BLK_D], dtype=tl.float32))
    tl.store(OUT + row * D + d, res.to(OUT.dtype.element_ty), mask=dm)
    if RETURN_LSE:
        # Both layouts are unified as off = b*LSE_B_STRIDE + n2*LSE_N2_STRIDE + s1*GS + (n1%GS)
        # BSND: [B,N2,S1,gs] -> (N2*S1*gs, S1*gs); TND: [N2,T,gs] -> (0, T*gs)
        tl.store(LSE_MAX + b * LSE_B_STRIDE + n2 * LSE_N2_STRIDE + s1 * GS + (n1 % GS), mx)
        tl.store(LSE_SUM + b * LSE_B_STRIDE + n2 * LSE_N2_STRIDE + s1 * GS + (n1 % GS), lse)


def _as_cumulative(seq, B, name, T_expected=None):
    """Normalize actual_seq_lengths into cumulative form (= cu_seqlens, length B+1, last = T).

    Accepted conventions (the native tiling only requires `size == batch size`, see
    `op_host/sparse_flash_attention_tiling.cpp:1239-1244` "should be equal to batch size"; for
    sibling operators both B and B+1 are supported, see
    `vllm_quant_lightning_indexer.py:243`):
      * length B+1: already cu_seqlens (first entry 0);
      * length B  : **both meanings are accepted**, disambiguated from the values (when
        T_expected is given):
          - last == T  => already **cumulative** (exclusive prefix, last entry = total token
            count T);
          - sum == T   => **per-sequence lengths** (cumsum).
        For B=1 the two are equivalent (a single GLM-5.2 sequence looks like this: [T]).
    If it cannot be disambiguated (neither last == T nor sum == T), raise an error reporting
    the actual numel/last/sum against the expectation, so that the failure stays diagnosable.
    """
    if seq is None:
        return None
    s = seq.detach().to(torch.int64).cpu().flatten()
    if s.numel() == B + 1:
        cu = s
    elif s.numel() == B:
        if T_expected is None:
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), s.cumsum(0)])
        elif int(s[-1]) == int(T_expected):
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), s])
        elif int(s.sum()) == int(T_expected):
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), s.cumsum(0)])
        else:
            raise ValueError(
                f"{name}: length {s.numel()} (=B) is neither cumulative "
                f"(last={int(s[-1])} != T={int(T_expected)})"
                f" nor per-sequence lengths (sum={int(s.sum())} != T={int(T_expected)})")
    else:
        raise ValueError(
            f"{name} length must be B={B} (per-sequence lengths or cumulative) or "
            f"B+1={B + 1} (cu_seqlens), got {s.numel()}")
    if T_expected is not None and int(cu[-1]) != int(T_expected):
        raise ValueError(f"{name} last entry {int(cu[-1])} != T={int(T_expected)}")
    return cu


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    scale_value: float = 1.0,
    sparse_block_size: int = 1,
    sparse_mode: int = 3,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    attention_mode: int = 2,
    pre_tokens: int | None = None,
    next_tokens: int | None = None,
    return_softmax_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equivalent of the AscendC SparseFlashAttention.

    Call-site signature (vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py
    L682-698):
        npu_sparse_flash_attention(
            query=..., key=..., value=..., sparse_indices=..., scale_value=...,
            sparse_block_size=1, block_table=..., actual_seq_lengths_query=...,
            actual_seq_lengths_kv=..., query_rope=..., key_rope=...,
            layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2)
    That combination (TND + PA_BSND) is implemented.

    Returns:
        (attention_out, softmax_max, softmax_sum); the latter two are empty tensors when
        return_softmax_lse=False
    """
    if layout_query not in ("BSND", "TND"):
        raise NotImplementedError(f"unsupported layout_query={layout_query!r} (BSND/TND)")
    if layout_kv not in ("BSND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"unsupported layout_kv={layout_kv!r} (BSND/PA_BSND)")
    if int(attention_mode) != 2:
        raise NotImplementedError(f"unsupported attention_mode={attention_mode} "
                                  f"(2 is the op default and the only implemented mode)")
    for nm, v in (("pre_tokens", pre_tokens), ("next_tokens", next_tokens)):
        if v is not None and int(v) != (1 << 63) - 1:
            raise NotImplementedError(f"unsupported {nm}={v} (only INT64_MAX)")
    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode must be 0 or 3, got {sparse_mode}")
    pa = layout_kv in ("PA_BSND", "PA_BBND")
    tnd = layout_query == "TND"

    # ---------------- Layout normalization ----------------
    if tnd:
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND requires actual_seq_lengths_query "
                             "(length B or B+1)")
        T, N1, D = query.shape
        raw_q = actual_seq_lengths_query.detach().to(torch.int64).cpu().flatten()
        # Where B comes from: with the PA layout, dim 0 of block_table is the batch size (the
        # native tiling does the same: tiling.cpp:959 "blockTable's first dimension should be
        # equal to batch size"); otherwise fall back to the earlier convention (a cu_seqlens
        # of length B+1 => B = numel-1).
        if block_table is not None:
            B = int(block_table.shape[0])
        elif raw_q.numel() >= 2 and int(raw_q[-1]) == T:
            B = raw_q.numel() - 1
        else:
            B = raw_q.numel()
        cu_q = _as_cumulative(raw_q, B, "actual_seq_lengths_query", T_expected=T)
        lens = cu_q[1:] - cu_q[:-1]
        tok_b = torch.repeat_interleave(
            torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64) - cu_q[:-1].repeat_interleave(lens)).to(torch.int32)
        qlen_b = lens.to(torch.int32)
        S1 = None
    else:
        B, S1, N1, D = query.shape
        T = B * S1
        tok_b = torch.arange(B, dtype=torch.int32).repeat_interleave(S1)
        tok_s1 = torch.arange(S1, dtype=torch.int32).repeat(B)
        qlen_b = None

    dev = query.device
    if pa:
        if block_table is None:
            raise ValueError("layout_kv=PA_BSND requires block_table")
        num_blocks, block_size, N2, Dk = key.shape
        max_blocks = block_table.shape[1]
        S2 = block_size * max_blocks
    else:
        Bk, S2, N2, Dk = key.shape
        block_size, max_blocks = 1, 0
    if Dk != D:
        raise ValueError(f"key D {Dk} does not match query D {D}")
    if tuple(value.shape) != tuple(key.shape):
        raise ValueError("value must have the same shape as key")

    cu_kv = _as_cumulative(actual_seq_lengths_kv, B, "actual_seq_lengths_kv")
    if cu_kv is None:
        kvlen_b = torch.full((B,), S2, dtype=torch.int32)
    else:
        kvlen_b = (cu_kv[1:] - cu_kv[:-1]).to(torch.int32)
    if qlen_b is None:
        qlen_b = torch.full((B,), S1, dtype=torch.int32)

    dr = _D_ROPE
    if query_rope is None or key_rope is None:
        raise ValueError("this implementation requires query_rope / key_rope "
                         "(the 64-dim rope part of MLA)")
    if key_rope.shape[:-1] != key.shape[:-1] or key_rope.shape[-1] != dr:
        raise ValueError(f"key_rope shape must be {tuple(key.shape[:-1]) + (dr,)}")
    if query_rope.shape[:-1] != query.shape[:-1] or query_rope.shape[-1] != dr:
        raise ValueError(f"query_rope shape must be {tuple(query.shape[:-1]) + (dr,)}")
    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} must be divisible by N2={N2}")
    if sparse_indices.dtype != torch.int32:
        raise ValueError("sparse_indices must be int32")
    bs = int(sparse_block_size)
    if bs < 1:
        raise ValueError("sparse_block_size must be >= 1")

    gs = N1 // N2
    k_idx = sparse_indices.shape[-1]
    # Flatten to [n_tokens, ...]
    q2 = query.contiguous().reshape(T, N1, D)
    qr2 = query_rope.contiguous().reshape(T, N1, dr)
    idx2 = sparse_indices.contiguous().reshape(T, N2, k_idx)
    kv = key.contiguous()
    kvr = key_rope.contiguous()
    val = value.contiguous()
    bt = block_table.contiguous().to(torch.int32) if pa else key.contiguous()
    tok_b = tok_b.to(dev)
    tok_s1 = tok_s1.to(dev)
    kvlen_d = kvlen_b.to(dev)
    qlen_d = qlen_b.to(dev)

    n_rows = T * N1
    sb_buf = torch.empty((n_rows, k_idx * bs), dtype=torch.float32, device=dev)
    out = torch.empty((n_rows, D), dtype=query.dtype, device=dev)

    blk_d = next_pow2(D)
    blk_r = next_pow2(dr)
    blk_k = 16
    _sfa_score_kernel[(n_rows,)](
        q2, qr2, kv, kvr, idx2, sb_buf, tok_b, tok_s1, kvlen_d, qlen_d, bt,
        N1, N2, gs, k_idx, T,
        block_size, max_blocks, S2, float(scale_value),
        PA=bool(pa), CAUSAL=bool(sparse_mode == 3), NEG=_NEG,
        D=D, DR=dr, BLK_D=blk_d, BLK_R=blk_r, BLK_K=blk_k, BS=bs,
    )

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

    total = k_idx * bs
    lse_b_stride = 0 if tnd else (N2 * S1 * gs)
    lse_n2_stride = (T * gs) if tnd else (S1 * gs)
    _sfa_out_kernel[(n_rows,)](
        val, sb_buf, idx2, out, lse_max, lse_sum, tok_b, tok_s1, bt,
        N1, N2, gs, k_idx, T,
        block_size, max_blocks, S2, lse_b_stride, lse_n2_stride,
        RETURN_LSE=bool(return_softmax_lse), PA=bool(pa), NEG=_NEG,
        D=D, BLK_D=blk_d, BLK_K=blk_k, BLK_ALL=next_pow2(total), BS=bs,
    )
    if tnd:
        out = out.reshape(T, N1, D)
    else:
        out = out.reshape(B, S1, N1, D)
    return out, lse_max, lse_sum
