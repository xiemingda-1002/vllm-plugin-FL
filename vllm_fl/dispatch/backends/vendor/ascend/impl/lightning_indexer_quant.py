# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""lightning_indexer_quant -- AscendC -> Triton

Source: vllm-ascend/csrc/attention/lightning_indexer_quant/
Interface (op_host/lightning_indexer_quant_def.cpp L24-70):
    Input : query               int8   BSND [B,S1,N1,D] / TND [T,N1,D]
            key                 int8   BSND [B,S2,N2,D] / PA_BSND [numBlocks, blockSize, N2, D]
            weights             fp16   per q head: [B,S1,N1] / [T,N1]
            query_dequant_scale fp16   same as weights
            key_dequant_scale   fp16   BSND [B,S2] / PA_BSND [numBlocks, blockSize] (per key position)
            actual_seq_lengths_query/key int32 (optional)
            block_table         int32 [B, maxBlockNumPerBatch] (required when layout_key=PA_BSND)
    Output: sparse_indices int32  BSND [B,S1,N2,K] / TND [T,N2,K]
    Attr  : query_quant_mode(0=per-token-head), key_quant_mode(0),
            layout_query("BSND"), layout_key("PA_BSND"), sparse_count(2048), sparse_mode(3)

Formula source:
  op_kernel/lightning_indexer_quant_service_cube.h
    L542-543  params.quantPre = DEQF16 ; **params.reluPre = 1**  -> ReLU is fused into the Cube QK path
    L564-566  second stage (weighted sum), reluPre = 0
  op_kernel/lightning_indexer_quant_service_vector.h
    L287/301-302  weights and qScale are loaded at the same offset (per q head)
    L383          Mul(mm1Res, mm1Res, **kScale**)      -> finally multiplied by the key dequant scale
    L389-400      before sorting, invalid positions are filled with NEG_INF and their index with
                  -1 (INVALID_IDX); the valid range gets cuBaseS2Idx added
    L403-406      SortAll + MergeSort, multi-stage merge to select top-K
  => identical formula to `vllm_quant_lightning_indexer`:
        QK[s1,n2,g,s2]    = sum_d query[s1,n2*gs+g,d] * key[s2,n2,d]        (int8 dot product, exact in int32)
        wq[s1,g']         = weights[s1,g'] * query_dequant_scale[s1,g']     (g' = n2*gs+g)
        score[s1,n2,s2]   = key_dequant_scale[s2] * sum_g relu(QK[g,s2]) * wq[g]
        sparse_mode=3 (lower triangular): s2 <= s1 + (kvLen - qLen)
        top-K (descending, K=sparse_count); missing entries are padded with **-1** (INVALID_IDX)

**Tie-breaking**: the relative order of equal scores in AscendC's SortAll/MergeSort is
**undefined**; this implementation picks the **smaller index** (`tl.min` over the indices), and
the test data keeps a gap between rank K and rank K+1 (the test prints gapK as evidence).
**PA_BSND page addressing**: `page = block_table[b, j//blockSize] ; row = page*blockSize + (j % blockSize)`.

**Simplifications**: no multi-core Cube/Vector pipeline; two-pass kernels (score first and store
          to GM, then per-row top-K); the int8 dot product uses a broadcast fp32 multiply-add
          (exact, but the Cube int8 matmul is not used).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["lightning_indexer_quant"]

_NEG = -1.0e30


@triton.jit
def _liq_score_kernel(
    Q, K, W, QS, KS, SC, TOK_B, TOK_S1, KVLEN, QLEN, BT,
    N1, N2, GS,
    BLOCK_SIZE, MAX_BLOCKS, S2,
    PA: tl.constexpr,
    CAUSAL: tl.constexpr,
    NEG: tl.constexpr,
    HD: tl.constexpr,
    BLK_S2: tl.constexpr,
    BLK_D: tl.constexpr,
):
    row = tl.program_id(0)              # one row = one (token, n2)
    n2 = row % N2
    t = row // N2
    b = tl.load(TOK_B + t)
    s1 = tl.load(TOK_S1 + t)

    h0 = n2 * GS
    d = tl.arange(0, BLK_D)
    dm = d < HD
    q_base = (t * N1 + h0) * HD
    w_base = t * N1 + h0

    kvlen = tl.load(KVLEN + b)
    if CAUSAL:
        j_hi = s1 + (kvlen - tl.load(QLEN + b))
    else:
        j_hi = S2 - 1

    for c in range(0, tl.cdiv(S2, BLK_S2)):
        s2 = c * BLK_S2 + tl.arange(0, BLK_S2)
        s2m = s2 < S2
        vld = s2m & (s2 <= j_hi) & (s2 < kvlen)
        if PA:
            page = tl.load(BT + b * MAX_BLOCKS + s2 // BLOCK_SIZE, mask=vld, other=0)
            kvrow = page * BLOCK_SIZE + (s2 % BLOCK_SIZE)
        else:
            kvrow = b * S2 + s2
        k_off = (kvrow * N2 + n2)[:, None] * HD + d[None, :]
        kblk = tl.load(K + k_off, mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)

        acc = tl.zeros([BLK_S2], dtype=tl.float32)
        for g in range(0, GS):
            qg = tl.load(Q + q_base + g * HD + d, mask=dm, other=0.0).to(tl.float32)
            wq = tl.load(W + w_base + g).to(tl.float32) * tl.load(QS + w_base + g).to(tl.float32)
            qk = tl.sum(kblk * qg[None, :], axis=1)
            acc += tl.where(qk > 0.0, qk, 0.0) * wq          # weighted accumulation of relu(QK)
        ksc = tl.load(KS + kvrow, mask=vld, other=0.0).to(tl.float32)
        sc = tl.where(vld, acc * ksc, NEG)
        tl.store(SC + row * S2 + s2, sc, mask=s2m)


@triton.jit
def _liq_topk_kernel(
    SC, IDX, S2,
    K: tl.constexpr,
    BLK_ALL: tl.constexpr,
    BLK_K: tl.constexpr,
    NEG: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.arange(0, BLK_ALL)
    s = tl.load(SC + pid * S2 + off, mask=off < S2, other=NEG)
    ar = tl.arange(0, BLK_K)
    oi = tl.full([BLK_K], -1, tl.int32)          # INVALID_IDX = -1
    for k in tl.static_range(K):
        mx = tl.max(s, axis=0)
        lane = tl.min(tl.where(s >= mx, off, BLK_ALL))   # on ties, the smaller index wins
        oi = tl.where(ar == k, tl.where(mx > NEG, lane, -1), oi)
        s = tl.where(off == lane, NEG, s)
    tl.store(IDX + pid * K + ar, oi, mask=ar < K)


def _lens(seq, B, name):
    if seq is None:
        return None
    s = seq.detach().to(torch.int64).cpu().flatten()
    if s.numel() == B:
        return s.to(torch.int32)
    if s.numel() == B + 1:
        return (s[1:] - s[:-1]).to(torch.int32)
    raise ValueError(f"{name} length must be B={B} or B+1={B + 1}, got {s.numel()}")


def lightning_indexer_quant(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    layout_query: str = "BSND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
) -> torch.Tensor:
    """Equivalent of the AscendC LightningIndexerQuant; returns sparse_indices (int32).

    The argument names follow the call-site style used by def.cpp.
    `layout_query="TND"` with `layout_key="PA_BSND"` (the default) is implemented.
    """
    if layout_query not in ("BSND", "TND"):
        raise NotImplementedError(f"unsupported layout_query={layout_query!r} (BSND/TND)")
    if layout_key not in ("BSND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"unsupported layout_key={layout_key!r} (BSND/PA_BSND)")
    for nm, v in (("query_quant_mode", query_quant_mode), ("key_quant_mode", key_quant_mode)):
        if int(v) != 0:
            raise NotImplementedError(f"unsupported {nm}={v} (only 0: per-token-head)")
    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode must be 0 or 3, got {sparse_mode}")
    pa = layout_key in ("PA_BSND", "PA_BBND")
    tnd = layout_query == "TND"

    if tnd:
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND requires actual_seq_lengths_query (cu_seqlens)")
        T, N1, D = query.shape
        cu = actual_seq_lengths_query.detach().to(torch.int64).cpu().flatten()
        if cu.numel() < 2 or int(cu[-1]) != T:
            raise ValueError(f"actual_seq_lengths_query must be cu_seqlens (length B+1, last entry {T})")
        B = cu.numel() - 1
        lens = cu[1:] - cu[:-1]
        tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64) - cu[:-1].repeat_interleave(lens)).to(torch.int32)
        qlen_b = lens.to(torch.int32)
        S1 = None
    else:
        B, S1, N1, D = query.shape
        T = B * S1
        tok_b = torch.arange(B, dtype=torch.int32).repeat_interleave(S1)
        tok_s1 = torch.arange(S1, dtype=torch.int32).repeat(B)
        qlen_b = None

    if pa:
        if block_table is None:
            raise ValueError("layout_key=PA_BSND requires block_table")
        num_blocks, block_size, N2, Dk = key.shape
        max_blocks = block_table.shape[1]
        S2 = block_size * max_blocks
    else:
        Bk, S2, N2, Dk = key.shape
        block_size, max_blocks = 1, 0
    if Dk != D:
        raise ValueError(f"key D {Dk} does not match query D {D}")
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise ValueError("query/key must be int8")
    if query_dequant_scale.dtype != torch.float16 or key_dequant_scale.dtype != torch.float16:
        raise ValueError("query/key_dequant_scale must be fp16 (def.cpp L41/46)")
    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} must be divisible by N2={N2}")
    gs = N1 // N2
    if weights.numel() != T * N1 or query_dequant_scale.numel() != T * N1:
        raise ValueError(f"weights/query_dequant_scale must have T*N1={T * N1} elements")
    k = int(sparse_count)
    if k <= 0:
        raise ValueError("sparse_count must be > 0")

    kvlen_b = _lens(actual_seq_lengths_key, B, "actual_seq_lengths_key")
    if kvlen_b is None:
        kvlen_b = torch.full((B,), S2, dtype=torch.int32)
    if qlen_b is None:
        qlen_b = torch.full((B,), S1, dtype=torch.int32)

    dev = query.device
    q2 = query.contiguous().reshape(T, N1, D)
    w2 = weights.contiguous().reshape(T, N1)
    qs2 = query_dequant_scale.contiguous().reshape(T, N1)
    kv = key.contiguous()
    bt = block_table.contiguous().to(torch.int32) if pa else key.contiguous()
    ks = key_dequant_scale.contiguous()
    if pa:
        if ks.dim() != 2 or ks.shape[1] != block_size:
            raise ValueError(f"with PA, key_dequant_scale should be [numBlocks, blockSize]="
                             f"{(num_blocks, block_size)}, got {tuple(key_dequant_scale.shape)}")
    else:
        if ks.dim() == 1:
            ks = ks.unsqueeze(0).expand(B, -1).contiguous()
        if tuple(ks.shape) != (B, S2):
            raise ValueError(f"key_dequant_scale must be [B,S2] or [S2], got {tuple(key_dequant_scale.shape)}")

    n_rows = T * N2
    blk_d = next_pow2(D)
    blk_s2 = min(64, next_pow2(S2))          # UB is limited to 192 KB
    scores = torch.empty((n_rows, S2), dtype=torch.float32, device=dev)
    _liq_score_kernel[(n_rows,)](
        q2, kv, w2, qs2, ks, scores,
        tok_b.to(dev), tok_s1.to(dev), kvlen_b.to(dev), qlen_b.to(dev), bt,
        N1, N2, gs,
        block_size, max_blocks, S2,
        PA=bool(pa), CAUSAL=bool(sparse_mode == 3), NEG=_NEG,
        HD=D, BLK_S2=blk_s2, BLK_D=blk_d,
    )
    idx = torch.empty((n_rows, k), dtype=torch.int32, device=dev)
    _liq_topk_kernel[(n_rows,)](
        scores, idx, S2,
        K=k, BLK_ALL=next_pow2(S2), BLK_K=next_pow2(k), NEG=_NEG,
    )
    if tnd:
        return idx.view(T, N2, k)
    return idx.view(B, S1, N2, k)
