# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""sparse_attention_score -- AscendC -> Triton

Source: csrc/ascend/attention/sparse_attention_score/
Interface (op_host/sparse_attention_score_def.cpp L19-72):
    Input : query [T, N, D]                 fp16/bf16/fp8_e4m3fn
            key   [numBlocks, blockSize, kvHeads, D]
            value [numBlocks, blockSize, kvHeads, D]
            selectIdx     [kvHeads, maxQSeqlen, topK]  int32
            blockTable    [B, maxBlocksPerBatch]       int32
            selectNumIdx  [kvHeads, maxQSeqlen]        int32 (optional; required for fp8)
            actualSeqLengths   [B] int32 (optional)
            actualSeqLengthsKv [B] int32 (optional)
            qDequantScale [B, numHeads, ceil(maxQSeqlen/blockSize)] fp32 (optional)
            kDequantScale [B, kvHeads, maxBlocksPerBatch]           fp32 (optional)
            vDequantScale [B, kvHeads, maxBlocksPerBatch]           fp32 (optional)
    Output: attentionOut [T, N, D] (same dtype as query; fp16 for fp8 inputs)
            softmaxLse   [T, N, 1] fp32 (optional)
    Attr  : numKeyValueHeads=1, scaleValue=0.0(-> 1/sqrt(D)), blockSize=128,
            topK=16, innerPrecise=4

Formula (Regular path: op_kernel/arch35/sparse_attention_score_kernel_arch35.h L112-289)
    for taskIdx in [0, T*kvHeads):                     # one task per (qToken, kvHead)
        qToken  = taskIdx // kvHeads ; kvHead = taskIdx % kvHeads
        groupSize = N // kvHeads ; qHeadStart = kvHead * groupSize
        G  = query[qToken, qHeadStart : qHeadStart+groupSize, :]        # [groupSize, D]
        batchIdx / qTokenInBatch come from the prefix sum of actualSeqLengths (L124-135)
        validTopK = selectNumIdx[kvHead, qToken] if present else topK
        kvSeqlen  = actualSeqLengthsKv[batchIdx] ; qSeqlen = actualSeqLengths[batchIdx]
        historyLen = kvSeqlen - qSeqlen
        lastLogicalBlockId = (historyLen + qTokenInBatch) // blockSize
        lastBlockTileSize  = (historyLen + qTokenInBatch) %  blockSize + 1     # causal
        for i in range(min(validTopK, topK)):          # L160-169
            logicalId = selectIdx[kvHead, qToken, i] ; if logicalId < 0: continue
            physId = blockTable[batchIdx, logicalId]
            tile   = blockSize if logicalId != lastLogicalBlockId else lastBlockTileSize
            K_b = key[physId][:, kvHead, :] ; S = (G @ K_b^T) * scaleValue
            S[.., j >= tile] = -inf
            per-block online softmax (EpilogueAtlasA2OnlineSoftmax; scaleValue is folded
            into S at L538-541)
            O += P @ value[physId][:, kvHead, :]
        attentionOut[qToken, qHeadStart + r, :] = O / l
        softmaxLse  [qToken, qHeadStart + r, 0] = m + log(l)

fp8 FullQuant path (op_kernel/arch35/sparse_attention_score_kernel_arch35_full_quant.h L146-344)
    Differences from the Regular path (**the code is authoritative**):
      * Dequantization is applied to the **matmul result**: deqScalar is passed to
        blockMmadQK / blockMmadPV and TileCopy scales it at PER_TENSOR granularity while
        copying L0C -> UB (`ScaleGranularity::PER_TENSOR` in
        sparse_attention_score_kernel_interface.cpp L204-207 / L221-224), that is
             S = (Q_fp8 . K_fp8^T) * (qDequantScale * kDequantScale)   # L251-263
             O = (P . V_fp8)          * vDequantScale                  # L315-326
      * **Indexing and shapes** of the three scales (L210-213 / L251-254 / L315-318):
             maxQBlocksPerBatch = ceil(maxQSeqlen / blockSize)
             logicalQBlock      = (historyLen + qTokenInBatch) // blockSize
             sq = qDequantScale[batchIdx * numHeads * maxQBlocksPerBatch
                                + qHeadIdx * maxQBlocksPerBatch + logicalQBlock]
                                             # shape [B, numHeads, maxQBlocksPerBatch]
             sk = kDequantScale[batchIdx * kvHeads * maxBlocksPerBatch
                                + kvHeadIdx * maxBlocksPerBatch + logicalId]
             sv = vDequantScale[batchIdx * kvHeads * maxBlocksPerBatch
                                + kvHeadIdx * maxBlocksPerBatch + logicalId]
                                             # shape [B, kvHeads, maxBlocksPerBatch]
        So the Q side is per (batch, qHead, qBlock) and the K/V side is per
        (batch, kvHead, logicalBlock); all of them are **per-tensor (one scalar per tile)**,
        with no per-token/per-head channel-wise scaling.
      * For fp8, `effectiveGroupSize = 1` (L126-128: a per-head dequant scale cannot pack
        several heads), so heads are handled one by one; this implementation processes a whole
        GQA group at once (mathematically equivalent, see item 5 of "Implementation notes").
      * P (the softmax probability) is written to L1 as **ElementP = InDtype = fp8_e4m3fn**
        before the PV product (kernel_interface.cpp L189/L213; epilogue
        block_epilogue_online_softmax_arch35_reg_low_prec.hpp L41/L81
        `FULL_QUANT_FP8 = IsSameType<ElementOutput, fp8_e4m3fn_t>`, converted to e4m3 with
        CAST_RINT by `ComputeExpSubSum16FP8` L483-598).
        This implementation reproduces that e4m3 quantization by default (`p_fp8=True`, using
        `tl.float8e4nv`); if the backend does not support the fp8 dtype it automatically falls
        back to fp16 and emits a warning.
      * The fp8 path **does not write softmaxLse** (the FullQuant kernel has no LSE output).

Implementation notes (what is explicitly not covered, and the trade-offs)
    1. Only the `saca_regular` and `full_quant(fp8)` paths are supported; arch22 (the
       fp16/bf16 branch on 910B/910C) is mathematically equivalent to arch35, and this
       implementation follows the arch35 description.
    2. The `innerPrecise` attribute is unused (the kernel takes the same path).
    3. LSE: the Regular kernel uses LseMode::NONE and does not write softmaxLse; this
       implementation still returns the mathematical LSE = m + log(l) (an explicitly
       documented extension); the fp8 path behaves the same way.
    4. When `numKeyValueHeads` <= 0 it is inferred from key.shape[2].
    5. For fp8 the whole GQA group (G qHeads) is processed at once instead of head-by-head;
       `S = (Q.K^T)*(sq*sk)` requires each head of the group under the same (batch, qBlock) to
       use its own sq, and this implementation takes the per-row (per-head) sq, which is
       exactly equivalent to head-by-head.
    6. Inside the kernel, fp8 inputs are widened losslessly to bf16 first (e4m3 is a subset of
       bf16, so the conversion is exact) and then fed to a dot with fp32 accumulation, which
       matches AscendC numerically ("fp8 cube multiply + fp32 accumulate, then multiply by
       deqScalar").
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["sparse_attention_score"]

_MIN_SCORE = -1.0e30  # WARNING: do not use float("-inf") (it degrades on Ascend)

# Whether fp8 probability (P) quantization is usable: set to False after the first failure
# (i.e. when the backend does not support the fp8 dtype)
_P_FP8_OK = True


@triton.jit
def _sasa_kernel(
    Q, K, V, SEL, BT, SELNUM, ACTQ, ACTKV, QSC, KSC, VSC, OUT, LSE,
    Q_BATCH, Q_INBATCH,
    KVH, MQB, MQ, N, D, BS, SCALE, MAXQBLK,
    TOP_K: tl.constexpr,
    BLK_G: tl.constexpr, BLK_D: tl.constexpr, BLK_BS: tl.constexpr,
    HAS_SELNUM: tl.constexpr, WRITE_LSE: tl.constexpr,
    IS_FP8: tl.constexpr, P_FP8: tl.constexpr,
    MIN_SCORE: tl.constexpr,
):
    task = tl.program_id(0)
    qToken = task // KVH
    h = task % KVH

    G = N // KVH
    gstart = h * G
    rg = tl.arange(0, BLK_G)
    rd = tl.arange(0, BLK_D)
    gm = rg < G
    dm = rd < D

    # Q: [G, D]
    qraw = tl.load(Q + qToken * N * D + (gstart + rg)[:, None] * D + rd[None, :],
                   mask=gm[:, None] & dm[None, :], other=0.0)
    if IS_FP8:
        q = qraw.to(tl.bfloat16)          # e4m3 -> bf16 is lossless
    else:
        q = qraw

    batchIdx = tl.load(Q_BATCH + qToken)
    qInBatch = tl.load(Q_INBATCH + qToken)
    kvSeqlen = tl.load(ACTKV + batchIdx)
    qSeqlen = tl.load(ACTQ + batchIdx)
    hist = kvSeqlen - qSeqlen
    lastLogical = (hist + qInBatch) // BS
    lastTile = (hist + qInBatch) % BS + 1

    if HAS_SELNUM:
        vtopk = tl.load(SELNUM + h * MQ + qToken)
    else:
        vtopk = TOP_K

    # fp8: the Q dequant scale is taken per (batch, qHead, qBlock) (L210-213)
    sq = tl.zeros((BLK_G,), dtype=tl.float32) + 1.0
    if IS_FP8:
        logicalQBlock = (hist + qInBatch) // BS
        qs_off = batchIdx * N * MAXQBLK + (gstart + rg) * MAXQBLK + logicalQBlock
        sq = tl.load(QSC + qs_off, mask=gm, other=1.0)

    sel_base = h * MQ * TOP_K + qToken * TOP_K
    bt_base = batchIdx * MQB

    rb = tl.arange(0, BLK_BS)
    m_i = tl.full((BLK_G,), MIN_SCORE, dtype=tl.float32)
    l_i = tl.zeros((BLK_G,), dtype=tl.float32)
    o_i = tl.zeros((BLK_G, BLK_D), dtype=tl.float32)

    for i in range(TOP_K):
        logical = tl.load(SEL + sel_base + i, mask=i < vtopk, other=-1)
        active = (i < vtopk) & (logical >= 0)
        phys = tl.load(BT + bt_base + logical, mask=active, other=0)
        tile = tl.where(logical == lastLogical, lastTile, BS)
        koff = phys * (BS * KVH * D) + h * D
        km = active & (rb[:, None] < tile) & dm[None, :]
        kraw = tl.load(K + koff + rb[:, None] * (KVH * D) + rd[None, :], mask=km, other=0.0)
        vraw = tl.load(V + koff + rb[:, None] * (KVH * D) + rd[None, :], mask=km, other=0.0)
        if IS_FP8:
            k = kraw.to(tl.bfloat16)
            vv = vraw.to(tl.bfloat16)
            # K/V dequant scales are taken per (batch, kvHead, logicalBlock)
            # (L251-254 / L315-318)
            kv_off = batchIdx * KVH * MQB + h * MQB + logical
            ksc = tl.load(KSC + kv_off, mask=active, other=1.0)
            vsc = tl.load(VSC + kv_off, mask=active, other=1.0)
        else:
            k = kraw
            vv = vraw
            ksc = 1.0
            vsc = 1.0

        if IS_FP8:
            # S = (Q.K^T) * (sq * sk) * scaleValue   (L251-263 + epilogue L538-541)
            s = tl.dot(q, tl.trans(k)) * (SCALE * ksc) * sq[:, None]
        else:
            s = tl.dot(q, tl.trans(k)) * SCALE
        s = tl.where(active & (rb[None, :] < tile) & gm[:, None], s, MIN_SCORE)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if P_FP8:
            pd = p.to(tl.float8e4nv).to(tl.bfloat16)     # ElementP = fp8_e4m3fn (CAST_RINT)
        elif IS_FP8:
            pd = p.to(tl.bfloat16)
        else:
            pd = p.to(Q.dtype.element_ty)
        o_i = o_i * alpha[:, None] + tl.dot(pd, vv) * vsc
        m_i = m_new

    has_tile = m_i > (MIN_SCORE * 0.5)
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o_i = tl.where(has_tile[:, None], o_i / l_safe[:, None], 0.0)
    olm = gm[:, None] & dm[None, :]
    tl.store(OUT + qToken * N * D + (gstart + rg)[:, None] * D + rd[None, :],
             o_i.to(OUT.dtype.element_ty), mask=olm)
    if WRITE_LSE:
        lse = tl.where(has_tile, m_i + tl.log(l_safe), 0.0)
        tl.store(LSE + qToken * N + gstart + rg, lse, mask=gm)


def _launch(query, key, value, select_idx, block_table, selnum_c, actq_c, actkv_c,
            qsc, ksc, vsc, out, lse, qb_c, qi_c, KVH, MQB, maxQ, Nh, D, BS, scale,
            maxqblk, topK, has_selnum, write_lse, is_fp8, BLK_G, BLK_D, BLK_BS, p_fp8):
    _sasa_kernel[(int(query.shape[0]) * KVH,)](
        query, key, value, select_idx, block_table, selnum_c, actq_c, actkv_c,
        qsc, ksc, vsc, out, lse, qb_c, qi_c,
        KVH, MQB, maxQ, Nh, D, BS, scale, maxqblk,
        TOP_K=topK, BLK_G=BLK_G, BLK_D=BLK_D, BLK_BS=BLK_BS,
        HAS_SELNUM=has_selnum, WRITE_LSE=write_lse,
        IS_FP8=is_fp8, P_FP8=p_fp8, MIN_SCORE=_MIN_SCORE,
    )


def sparse_attention_score(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    select_idx: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    select_num_idx: torch.Tensor | None = None,
    actual_seq_lengths: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    q_dequant_scale: torch.Tensor | None = None,
    k_dequant_scale: torch.Tensor | None = None,
    v_dequant_scale: torch.Tensor | None = None,
    num_key_value_heads: int | None = None,
    scale_value: float = 0.0,
    block_size: int = 128,
    top_k: int = 16,
    inner_precise: int = 4,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    write_lse: bool = True,
    p_fp8: bool = True,
    fp8_mode: bool | None = None,
    # ---- Aliases using the original AscendC names (real call sites use these); when passed
    # explicitly they override the snake_case parameter of the same meaning ----
    selectIdx: torch.Tensor | None = None,
    blockTable: torch.Tensor | None = None,
    selectNumIdx: torch.Tensor | None = None,
    actualSeqLengths: torch.Tensor | None = None,
    actualSeqLengthsKv: torch.Tensor | None = None,
    qDequantScale: torch.Tensor | None = None,
    kDequantScale: torch.Tensor | None = None,
    vDequantScale: torch.Tensor | None = None,
    numKeyValueHeads: int | None = None,
    scaleValue: float | None = None,
    blockSize: int | None = None,
    topK: int | None = None,
    innerPrecise: int | None = None,
    attentionOut: torch.Tensor | None = None,
    softmaxLse: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent of the AscendC sparse_attention_score (fp16/bf16 Regular + fp8 FullQuant).

    Both snake_case and the original AscendC (camelCase) parameter names are accepted:
    `selectIdx/blockTable/selectNumIdx/actualSeqLengths/actualSeqLengthsKv/
     qDequantScale/kDequantScale/vDequantScale/numKeyValueHeads/scaleValue/
     blockSize/topK/innerPrecise/attentionOut/softmaxLse`.

    With fp8_e4m3fn inputs, query/key/value are fp8 and the attentionOut output is fp16
    (op_host/sparse_attention_score_infershape.cpp L91-96); the shapes of the three dequant
    scales are described in the module docstring.

    Returns:
        (attentionOut [T, N, D], softmaxLse [T, N, 1] fp32)
    """
    # ---- Alias back-fill ----
    if selectIdx is not None:
        select_idx = selectIdx
    if blockTable is not None:
        block_table = blockTable
    if selectNumIdx is not None:
        select_num_idx = selectNumIdx
    if actualSeqLengths is not None:
        actual_seq_lengths = actualSeqLengths
    if actualSeqLengthsKv is not None:
        actual_seq_lengths_kv = actualSeqLengthsKv
    if qDequantScale is not None:
        q_dequant_scale = qDequantScale
    if kDequantScale is not None:
        k_dequant_scale = kDequantScale
    if vDequantScale is not None:
        v_dequant_scale = vDequantScale
    if numKeyValueHeads is not None:
        num_key_value_heads = numKeyValueHeads
    if scaleValue is not None:
        scale_value = scaleValue
    if blockSize is not None:
        block_size = blockSize
    if topK is not None:
        top_k = topK
    if innerPrecise is not None:
        inner_precise = innerPrecise
    if attentionOut is not None:
        out = attentionOut
    if softmaxLse is not None:
        lse = softmaxLse
    if select_idx is None or block_table is None:
        raise ValueError("sparse_attention_score requires select_idx/selectIdx and "
                         "block_table/blockTable")
    global _P_FP8_OK
    # fp8 path selection: detected automatically from the input dtype by default, and it can
    # also be forced with fp8_mode (useful when fp8 tensors cannot be materialized on the NPU:
    # the dequantization math can then be checked with "bf16 tensors already quantized to
    # e4m3", since e4m3 is a subset of bf16 and the two are numerically identical)
    is_fp8 = (query.dtype == torch.float8_e4m3fn) if fp8_mode is None else bool(fp8_mode)
    T, Nh, D = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
    BS = int(block_size)
    BSz = int(key.shape[1])
    KVH = int(num_key_value_heads) if num_key_value_heads else int(key.shape[2])
    if KVH <= 0:
        KVH = int(key.shape[2])
    Bt, MQB = int(block_table.shape[0]), int(block_table.shape[1])
    maxQ = int(select_idx.shape[1])
    topK = int(top_k)
    if int(select_idx.shape[2]) != topK:
        raise ValueError(f"topK attribute ({topK}) must equal the last dim of "
                         f"selectIdx ({int(select_idx.shape[2])})")
    # dim2 length of qDequantScale (the stride of the head dimension): the kernel reads
    #   offset = batchIdx*numHeads*maxQBlocksPerBatch + qHead*maxQBlocksPerBatch + logicalQBlock
    # (full_quant L210-213), and logicalQBlock uses the **absolute-position** block number
    # (historyLen + qTokenInBatch)//blockSize, so dim2 must cover that upper bound.
    # The actual dim2 of the given tensor is preferred; otherwise it is built from the upper
    # bound of the KV sequence length (see below, after act_kv).

    scale = float(scale_value)
    if abs(scale) < 1e-9 and D > 0:
        scale = 1.0 / math.sqrt(float(D))

    act_q = (actual_seq_lengths.detach().cpu().to(torch.int64).tolist()
             if actual_seq_lengths is not None else [T])
    act_kv = (actual_seq_lengths_kv.detach().cpu().to(torch.int64).tolist()
              if actual_seq_lengths_kv is not None else act_q)

    if q_dequant_scale is not None and q_dequant_scale.dim() == 3:
        maxqblk = int(q_dequant_scale.shape[2])
    else:
        # logicalQBlock is an **absolute-position** block number whose upper bound is set by
        # the KV sequence length (not by maxQSeqlen)
        maxqblk = max(1, max(ceil_div(int(v), BS) for v in act_kv) if act_kv else 1)

    q_batch, q_inb = [], []
    for b in range(Bt):
        for t in range(int(act_q[b]) if b < len(act_q) else 0):
            q_batch.append(b)
            q_inb.append(t)
    if len(q_batch) < T:
        q_batch += [Bt - 1] * (T - len(q_batch))
        q_inb += [0] * (T - len(q_inb))
    q_batch = q_batch[:T]
    q_inb = q_inb[:T]

    dev = query.device
    if out is None:
        out = torch.zeros((T, Nh, D), dtype=(torch.float16 if is_fp8 else query.dtype), device=dev)
    if lse is None:
        lse = torch.zeros((T, Nh, 1), dtype=torch.float32, device=dev)

    G = Nh // KVH
    BLK_G = max(16, next_pow2(G))
    BLK_D = max(16, next_pow2(D))
    BLK_BS = max(16, next_pow2(BSz))
    assert topK <= 16, "topK must be <= 16 (see MAX_TOP_K in the op_host tiling)"

    q_c = query.contiguous()
    k_c = key.contiguous()
    v_c = value.contiguous()
    sel_c = select_idx.to(torch.int32).contiguous()
    bt_c = block_table.to(torch.int32).contiguous()
    selnum_c = (select_num_idx.to(torch.int32).contiguous() if select_num_idx is not None
                else torch.zeros(1, dtype=torch.int32, device=dev))
    actq_c = torch.tensor(act_q, dtype=torch.int32, device=dev)
    actkv_c = torch.tensor(act_kv, dtype=torch.int32, device=dev)
    qb_c = torch.tensor(q_batch, dtype=torch.int32, device=dev)
    qi_c = torch.tensor(q_inb, dtype=torch.int32, device=dev)

    def _sc(t, shape):
        # When None is passed, build an all-ones scale over the real addressing range of the
        # kernel (equivalent to "no dequantization")
        if t is None:
            return torch.ones(shape, dtype=torch.float32, device=dev).reshape(-1).contiguous()
        return t.to(torch.float32).reshape(-1).contiguous()

    qsc = _sc(q_dequant_scale, (Bt, Nh, maxqblk))
    ksc = _sc(k_dequant_scale, (Bt, KVH, MQB))
    vsc = _sc(v_dequant_scale, (Bt, KVH, MQB))

    args = (q_c, k_c, v_c, sel_c, bt_c, selnum_c, actq_c, actkv_c, qsc, ksc, vsc,
            out, lse, qb_c, qi_c, KVH, MQB, maxQ, Nh, D, BS, scale, maxqblk,
            topK, bool(select_num_idx is not None), bool(write_lse), bool(is_fp8),
            BLK_G, BLK_D, BLK_BS)
    if is_fp8 and p_fp8 and _P_FP8_OK:
        try:
            _launch(*args, p_fp8=True)
            return out, lse
        except Exception as e:  # backend has no fp8 dtype: fall back to fp16 probabilities
            _P_FP8_OK = False
            import warnings
            warnings.warn(f"sparse_attention_score: fp8 probability quantization unavailable, "
                          f"falling back to fp16: {e}")
    _launch(*args, p_fp8=False)
    return out, lse
