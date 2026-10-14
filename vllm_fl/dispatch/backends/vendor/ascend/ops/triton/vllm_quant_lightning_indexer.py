# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""vllm_quant_lightning_indexer -- AscendC -> Triton

Source: csrc/ascend/attention/vllm_quant_lightning_indexer/
Interface (op_host/vllm_quant_lightning_indexer_def.cpp L23-81, arch32/910B configuration):
    Input : query               int8   [B, S1, N1, 128]   (layout_query="BSND")
            key                 int8   [B, S2, N2, 128]   (layout_key="BSND")
            weights             fp16   [B, S1, N1]
            query_dequant_scale fp16   [B, S1, N1]
            key_dequant_scale   fp16   [B, S2]            (per key position)
            actual_seq_lengths_query/key, block_table, metadata  (optional)
    Output: sparse_indices int32 [B, S1, N2, sparse_count]   <- InferShape L64-75
            sparse_values  fp32  [B, S1, N2, sparse_count]   <- only when return_values=True;
                                                                otherwise an empty tensor (L76-83)
    Attr  : query/key_quant_mode=0 (per-token-head), sparse_count (default 2048),
            sparse_mode=3 (lower triangle only), pre_tokens/next_tokens=INT64_MAX, cmp_ratio=1,
            return_values=False, stride, scale_stride

Constant constraints (op_host/vllm_quant_lightning_indexer_tiling.cpp):
    - head_dim must be 128 (L549 HEAD_DIM_LIMIT)
    - gSize = n1Size_ / n2Size_ must be G_SIZE_LIMIT (L483-484)
    - the N1 dimension of weights must equal n1Size_ (L720), i.e. it is indexed by **q head**
    - sparse_mode in {0, 3} (L249), pre/next_tokens must be INT64_MAX (L252-257)

Formula source (op_kernel/arch35/vf/quant_lightning_indexer_vector1.h):
    MulWeightAndReduceSum L329-353 (fp32 version) and L206-280:
        regW    = weights * qScale                      # L243 / L331 per (s1,qhead)
        for g in 0..gSize-1:  Relu is folded into the cube path   # L341 comment
            MulAddDst(sum, qk[g], w_brc[g])             # L181-185 WeightedAccum: Relu, then MulAdd
        sum = sum * kScale                              # L352-353
    The same formula in arch32: op_kernel/arch32/quant_lightning_indexer_service_vector.h
        L265 Mul(weights, weights, qScale) ; L360 Mul(mm1Res, mm1Res, kScale)

    => core computation (**semantically equivalent version**):
        QK[s1, n2, g, s2] = sum_d query[s1, n2*gSize+g, d] * key[s2, n2, d]   (int8 dot, exact in int32)
        wq[s1, g']        = weights[s1, g'] * query_dequant_scale[s1, g']     (g' = n2*gSize+g)
        score[s1, n2, s2] = key_dequant_scale[s2] * sum_g relu(QK[g, s2]) * wq[g]
        sparse_mode=3 (lower triangle): score = -inf  when s2 > s1 + (S2 - S1)
                                  (tiling kernel L327: cuRealAcSeq = actS2SizeOrig - actS1Size
                                   + cuS1BeginIdx)
        top-k: take the sparse_count key positions with the largest score in descending order;
               pad with -1 when there are fewer
               (INVALID_IDX = -1, see arch32/quant_lightning_indexer_common.h L79)

    Note on sort precision: AscendC's topk converts scores to **bfloat16**, then to sortable keys,
    and sorts them (L273-277 FloatToSortableKey<bfloat16_t>), so ordering compares at bf16
    precision. This implementation compares in fp32; for data where "the best K scores are clearly
    separated" both agree (the tests print those gaps as evidence).

**Simplifications** (relative to the full AscendC implementation):
    1. only layout_query/layout_key = "BSND" (no TND / PA_BSND paged KV)
    2. no cmp_ratio != 1 compression and no actual_seq_lengths variable length (treated as full BSND)
    3. no multi-core Cube/Vector pipelining and no fusion; the score matrix is written to GM once
       (two-pass kernel: score -> topk)
    4. topk is implemented as repeated argmax (O(K) rounds); on ties the **smaller index** wins
       (AscendC leaves the tie order undefined)
    5. the int8 query/key dot product uses fp32 broadcast multiply-add (exact: the int8 product sum
       stays below 2^24); the Cube int8 matmul is not used
    6. sparse_values emits 0.0 in invalid slots (index -1); AscendC leaves that value undefined
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["vllm_quant_lightning_indexer"]

_NEG = -1.0e30


@triton.jit
def _qli_score_kernel(
    Q, K, W, QS, KS, SC, TOK_B, TOK_S1, KVLEN, QLEN, BT,
    N1, N2, GS,
    BLOCK_SIZE, MAX_BLOCKS, S2, CMP_RATIO,
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
        # matches AscendC: cuRealAcSeq = (count + (actS2Orig - actS1 + s1)) / cmpRatio
        j_hi = (s1 + 1 + (kvlen - tl.load(QLEN + b))) // CMP_RATIO - 1
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
            qk = tl.sum(kblk * qg[None, :], axis=1)          # int8 dot product (exact in fp32)
            acc += tl.where(qk > 0.0, qk, 0.0) * wq          # ReLU + weighted accumulation
        ksc = tl.load(KS + kvrow, mask=vld, other=0.0).to(tl.float32)
        sc = tl.where(vld, acc * ksc, NEG)
        tl.store(SC + row * S2 + s2, sc, mask=s2m)


@triton.jit
def _qli_topk_kernel(
    SC, IDX, VAL, S2,
    K: tl.constexpr,
    BLK_ALL: tl.constexpr,
    BLK_K: tl.constexpr,
    WRITE_VAL: tl.constexpr,
    NEG: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.arange(0, BLK_ALL)
    s = tl.load(SC + pid * S2 + off, mask=off < S2, other=NEG)

    ar = tl.arange(0, BLK_K)
    oi = tl.full([BLK_K], -1, tl.int32)
    ov = tl.zeros([BLK_K], dtype=tl.float32)
    for k in tl.static_range(K):
        mx = tl.max(s, axis=0)
        lane = tl.min(tl.where(s >= mx, off, BLK_ALL))    # on ties take the smaller index
        sel = tl.where(mx > NEG, lane, -1)
        oi = tl.where(ar == k, sel, oi)
        if WRITE_VAL:
            ov = tl.where(ar == k, tl.where(mx > NEG, mx, 0.0), ov)
        s = tl.where(off == lane, NEG, s)

    m = ar < K
    tl.store(IDX + pid * K + ar, oi, mask=m)
    if WRITE_VAL:
        tl.store(VAL + pid * K + ar, ov, mask=m)


def vllm_quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    return_value: bool = False,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    pre_tokens: int | None = None,
    next_tokens: int | None = None,
    cmp_ratio: int = 1,
    return_values: bool | None = None,
    out_dtype: torch.dtype = torch.int32,
):
    """Equivalent of the AscendC VllmQuantLightningIndexer (BSND, semantically equivalent core).

    Call-site signature (vllm_fl/attention/ascend/dsa_v1.py L2658-2678):
        npu_vllm_quant_lightning_indexer(
            query=..., key=..., weights=..., query_dequant_scale=..., key_dequant_scale=...,
            actual_seq_lengths_query=..., actual_seq_lengths_key=..., block_table=...,
            metadata=..., query_quant_mode=0, key_quant_mode=0,
            layout_query="TND", layout_key="PA_BSND", sparse_count=...,
            sparse_mode=3, pre_tokens=INT64_MAX, next_tokens=INT64_MAX,
            cmp_ratio=4, return_value=False)
    Note: the real call site uses **TND + PA_BSND + cmp_ratio=4**, while this core version only
    implements BSND + cmp_ratio=1; unsupported values raise NotImplementedError explicitly and are
    never silently ignored.

    Args:
        query:               int8 [B, S1, N1, 128]
        key:                 int8 [B, S2, N2, 128]
        weights:             fp16/bf16/fp32 [B, S1, N1]
        query_dequant_scale: same as weights
        key_dequant_scale:   [B, S2] or [S2]
        sparse_count:        top-k count K
        sparse_mode:         0 (all keys) or 3 (lower triangle, the AscendC default)
        return_value:        whether to return sparse_values (fp32 scores); False -> empty tensor
        layout_query/layout_key: only "BSND" is supported
        actual_seq_lengths_query/key, block_table, metadata: only None is supported
        query_quant_mode/key_quant_mode: only 0 (per-token-head) is supported
        pre_tokens/next_tokens: only None or INT64_MAX is supported
        cmp_ratio: only 1 is supported
        return_values: alias of return_value (backward compatibility)

    Returns:
        (sparse_indices [B,S1,N2,K] int32, sparse_values [B,S1,N2,K] fp32 or torch.empty(0))
    """
    if return_values is not None:
        return_value = return_values
    if layout_query not in ("BSND", "TND"):
        raise NotImplementedError(f"layout_query={layout_query!r} is not implemented (supported: BSND/TND)")
    if layout_key not in ("BSND", "PA_BSND", "PA_BBND"):
        raise NotImplementedError(f"layout_key={layout_key!r} is not implemented (supported: BSND/PA_BSND)")
    if metadata is not None and metadata.dtype != torch.int32:
        raise ValueError(f"metadata must be an int32 descriptor tensor, got {metadata.dtype}")
    for nm, v in (("query_quant_mode", query_quant_mode), ("key_quant_mode", key_quant_mode)):
        if int(v) != 0:
            raise NotImplementedError(f"{nm}={v} is not implemented (tiling only accepts 0: per-token-head)")
    for nm, v in (("pre_tokens", pre_tokens), ("next_tokens", next_tokens)):
        if v is not None and int(v) != (1 << 63) - 1:
            raise NotImplementedError(f"{nm}={v} is not implemented (tiling only accepts INT64_MAX)")
    return_values = bool(return_value)
    pa = layout_key in ("PA_BSND", "PA_BBND")
    tnd = layout_query == "TND"

    # ---------------- layout normalization (the host only does layout bookkeeping; the core
    # ---------------- math lives in Triton) ----------------
    if tnd:
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND requires actual_seq_lengths_query (cu_seqlens)")
        T, N1, D = query.shape
        cu_q = actual_seq_lengths_query.detach().to(torch.int64).cpu().flatten()
        if cu_q.numel() == 0:
            raise ValueError("actual_seq_lengths_query must not be empty for layout_query=TND")
        # Both conventions must be accepted:
        #   * AscendC / unit tests: ``[0, a1, a1+a2, ...]`` (length B+1, leading 0)
        #   * production call site: ``dsa_v1.py`` L2653/L2664 passes ``query_start_loc[1:]``,
        #                     i.e. ``[a1, a1+a2, ...]`` (length B, **no** leading 0)
        # For single-sequence decode production passes a single element (``[1]``); the previous
        # implementation raised ValueError here, the worker died inside the collective and the
        # other ranks reported 507035.
        if int(cu_q[0]) != 0:
            cu_q = torch.cat([torch.zeros(1, dtype=torch.int64), cu_q])
        if int(cu_q[-1]) != T:
            raise ValueError(
                f"with TND, actual_seq_lengths_query must be cu_seqlens (length B or B+1, last "
                f"entry {T}), got last entry {int(cu_q[-1])}"
            )
        B = cu_q.numel() - 1
        lens = cu_q[1:] - cu_q[:-1]
        tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64) - cu_q[:-1].repeat_interleave(lens)).to(torch.int32)
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
        raise ValueError(f"key D does not match query: {Dk} vs {D}")
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise ValueError("query/key must be int8")
    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} must be divisible by N2={N2}")
    gs = N1 // N2
    if sparse_mode not in (0, 3):
        raise ValueError(f"sparse_mode only supports 0 or 3, got {sparse_mode}")
    # weights/query_dequant_scale are per q head: BSND passes [B,S1,N1], TND passes [T,N1];
    # both are flattened to [T,N1]
    if weights.numel() != T * N1 or query_dequant_scale.numel() != T * N1:
        raise ValueError(
            f"weights/query_dequant_scale numel must be T*N1={T * N1} "
            f"(BSND->[B,S1,N1], TND->[T,N1]), got {tuple(weights.shape)} / "
            f"{tuple(query_dequant_scale.shape)}")
    k = int(sparse_count)
    if k <= 0:
        raise ValueError("sparse_count must be > 0")

    sq_kv = actual_seq_lengths_key
    if sq_kv is None:
        kvlen_b = torch.full((B,), S2, dtype=torch.int32)
    else:
        s = sq_kv.detach().to(torch.int64).cpu().flatten()
        if s.numel() == B:
            kvlen_b = s.to(torch.int32)
        elif s.numel() == B + 1:
            kvlen_b = (s[1:] - s[:-1]).to(torch.int32)
        else:
            raise ValueError(f"actual_seq_lengths_key length must be B={B} or B+1, got {s.numel()}")
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
        # Under PA, key_dequant_scale is laid out by (physical block, in-block offset)
        # (GetKeyScale: kScaleGm[blockId*scaleStride+off]). The production call site carries an
        # extra trailing singleton dimension (e.g. (21321, 32, 1)) -- squeeze it away; the
        # addressing below only looks at [numBlocks, blockSize], so the extra dimension has no
        # semantic effect.
        while ks.dim() > 2 and ks.shape[-1] == 1:
            ks = ks.squeeze(-1)
        if ks.dim() != 2 or ks.shape[1] != block_size:
            raise ValueError(f"under PA, key_dequant_scale should be [numBlocks, blockSize]="
                             f"{(num_blocks, block_size)} (a trailing singleton may be omitted), "
                             f"got {tuple(key_dequant_scale.shape)}")
        ks = ks.contiguous()
    else:
        if ks.dim() == 1:
            ks = ks.unsqueeze(0).expand(B, -1).contiguous()
        if tuple(ks.shape) != (B, S2):
            raise ValueError(f"key_dequant_scale must be [B,S2] or [S2], got {tuple(key_dequant_scale.shape)}")

    n_rows = T * N2
    blk_d = next_pow2(D)
    # BLK_S2 is limited by UB (192KB): the fp32 intermediate of BLK_S2*BLK_D must fit
    # (128x128 overflows UB)
    blk_s2 = min(64, next_pow2(S2))
    scores = torch.empty((n_rows, S2), dtype=torch.float32, device=dev)
    _qli_score_kernel[(n_rows,)](
        q2, kv, w2, qs2, ks, scores,
        tok_b.to(dev), tok_s1.to(dev), kvlen_b.to(dev), qlen_b.to(dev), bt,
        N1, N2, gs,
        block_size, max_blocks, S2, int(cmp_ratio),
        PA=bool(pa), CAUSAL=bool(sparse_mode == 3), NEG=_NEG,
        HD=D, BLK_S2=blk_s2, BLK_D=blk_d,
    )
    idx = torch.empty((n_rows, k), dtype=torch.int32, device=dev)
    blk_k = next_pow2(k)
    blk_all = next_pow2(S2)
    # the topk kernel writes VAL with a pid*K stride, so the buffer must be [n_rows, k]
    vals = torch.empty((n_rows, k), dtype=torch.float32, device=dev)
    _qli_topk_kernel[(n_rows,)](
        scores, idx, vals, S2,
        K=k, BLK_ALL=blk_all, BLK_K=blk_k,
        WRITE_VAL=bool(return_values), NEG=_NEG,
    )
    if return_values:
        out_v = vals.view(n_rows, k).to(torch.float32)
    else:
        out_v = torch.empty(0, dtype=torch.float32, device=dev)
    if tnd:
        return idx.view(T, N2, k), out_v.view(T, N2, k) if return_values else out_v
    return idx.view(B, S1, N2, k), (out_v.view(B, S1, N2, k) if return_values else out_v)
