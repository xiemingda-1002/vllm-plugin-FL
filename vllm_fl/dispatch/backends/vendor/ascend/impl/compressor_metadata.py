# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""compressor_metadata -- AscendC to Triton.

Source: csrc/ascend/attention/compressor_metadata/
Interface (op_host/compressor_metadata_def.cpp L12-61):
    Input : ropeCos      [ropeRows, ropeDim]  T(f32/f16/bf16)
            ropeSin      [ropeRows, ropeDim]  T
            cuSeqlens    [numReqs+1]          int32
            startPos     [>=actualNumReqs]    int32
            kvBlockTable [rows, stride]       int32
    Output: compressCos  [numRows, ropeDim]   T
            compressSin  [numRows, ropeDim]   T
            slotMapping  [numRows]      int32 (slotMappingFormat==1, flat)
                         [numRows, 2]   int32 (slotMappingFormat==2, block_offset)
    Attr  : kvBlockSize, slotMappingFormat(1=flat/2=block_offset), cmpRatio, actualNumReqs

Formulas (op_kernel/compressor_metadata.h):
  L145-170 BuildCompressedPrefix
      prefix[0] = 0
      seqLen_i   = cuSeqlens[i+1] - cuSeqlens[i]
      cnt_i      = ((startPos_i + seqLen_i)/cmpRatio) - (startPos_i/cmpRatio)   if startPos_i>=0 and seqLen_i>0
                 = 0                                                            otherwise
      prefix[i+1] = prefix[i] + cnt_i
  L138-140 validRows = min(prefix[actualNumReqs], numRows)
  L181-188 FindRequest: reqIdx = #{ j in [1,actualNumReqs] : prefix[j] <= row }
                         (binary search over the non-decreasing prefix)
  L204-218 per request:
      localCompressedIdx = row - prefix[reqIdx]
      compressedPos      = startPos[reqIdx]/cmpRatio + localCompressedIdx
  Note: compressedPos is the *compressed* position; RoPE reads the original group start
  position compressedPos*cmpRatio (comment at L214).
  L221-237 ProcessRequestRows: splits on kvBlockSize boundaries plus tileRows
           (a chunk never crosses a block boundary)
  L239-281 ProcessTile validity decision (whole chunk granularity):
      blockIdOffset = compressedPos / kvBlockSize
      (a) blockIdOffset >= kvBlockTableStride                          -> whole chunk invalid
      (b) blockId = kvBlockTable[reqIdx*stride + blockIdOffset] < 0    -> whole chunk invalid
      (c) if format==flat: blockId*kvBlockSize + blockOffset + rows-1 > INT32_MAX -> whole chunk invalid
      (d) (compressedPos + rows - 1)*cmpRatio >= ropeRows              -> whole chunk invalid
      valid  : L283-306 copies rope[(compressedPos+i)*cmpRatio, :] row by row, writes slot
      invalid: L335-371 cos=1.0 / sin=0.0 / slot=-1 (block_offset: slot=[-1, kvBlockSize-1])
  L373-389 output rows [validRows, numRows) are all filled as invalid chunks

Triton implementation notes / semantic assumptions:
  1. AscendC further splits the row range across AIV cores (SplitRange); the core
     boundaries truncate chunks and therefore affect the two chunk-level decisions
     (c)(d). The core split is only an artifact of the parallel partition, so this
     implementation uses "no core boundaries"; the two differ only on pathological
     data (int32 overflow / rope out of range), which the tests deliberately avoid.
  2. Integer division: for startPos>=0 the C truncating division matches floor
     division (any request that produces rows necessarily has startPos>=0).
  3. The prefix scan is done with tl.cumsum inside a single program, which requires
     actualNumReqs <= 4096.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["compressor_metadata", "SLOT_MAPPING_FLAT", "SLOT_MAPPING_BLOCK_OFFSET"]

SLOT_MAPPING_FLAT = 1
SLOT_MAPPING_BLOCK_OFFSET = 2
_MAX_PREFIX_REQS = 4096
INT32_MAX = 2147483647


@triton.jit
def _prefix_kernel(
    CU_SEQLENS, START_POS, PREFIX,
    N, CMP_RATIO,
    BLK: tl.constexpr,
):
    """prefix[0]=0, prefix[i+1]=prefix[i]+cnt_i (single program, vectorized + cumsum)."""
    j = tl.arange(0, BLK)
    m = j < N
    sp = tl.load(START_POS + j, mask=m, other=0)
    cs_lo = tl.load(CU_SEQLENS + j, mask=m, other=0)
    cs_hi = tl.load(CU_SEQLENS + j + 1, mask=m, other=0)
    seq_len = cs_hi - cs_lo
    cnt = tl.where(
        (sp >= 0) & (seq_len > 0),
        ((sp + seq_len) // CMP_RATIO) - (sp // CMP_RATIO),
        0,
    )
    incl = tl.cumsum(cnt, axis=0)
    excl = incl - cnt
    tl.store(PREFIX + j, excl, mask=m)
    tl.store(PREFIX + N, tl.sum(cnt, axis=0))


@triton.jit
def _compressor_metadata_kernel(
    ROPE_COS, ROPE_SIN, START_POS, KV_BLOCK_TABLE, PREFIX,
    C_COS, C_SIN, SLOT,
    NUM_ROWS, ACTUAL_NUM_REQS, ROPE_ROWS, ROPE_DIM,
    KV_BLOCK_SIZE, KV_BLOCK_TABLE_STRIDE, CMP_RATIO,
    SLOT_FORMAT: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    BLK_R: tl.constexpr,
    BLK_D: tl.constexpr,
    BIN_STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLK_R + tl.arange(0, BLK_R)
    rmask = rows < NUM_ROWS

    total = tl.load(PREFIX + ACTUAL_NUM_REQS)
    valid_rows = tl.minimum(total, NUM_ROWS)

    # --- binary search: reqIdx = #{ j in [1,N] : prefix[j] <= row } ---
    lo = tl.zeros([BLK_R], dtype=tl.int32)
    hi = tl.zeros([BLK_R], dtype=tl.int32) + ACTUAL_NUM_REQS
    for _ in tl.static_range(BIN_STEPS):
        active = lo < hi
        mid = (lo + hi + 1) // 2
        midc = tl.minimum(tl.maximum(mid, 1), ACTUAL_NUM_REQS)
        # Unmasked gather (index already clamped) plus tl.where: a masked gather trips the
        # FlagTree FLIR InterleaveStatusWithMaskOptimization assertion
        # (InterleaveOptimization.cpp:635).
        pv = tl.where(active & (mid >= 1), tl.load(PREFIX + midc), 0)
        go_right = active & (pv <= rows)
        go_left = active & (pv > rows)
        lo = tl.where(go_right, mid, lo)
        hi = tl.where(go_left, mid - 1, hi)

    req = tl.minimum(tl.maximum(lo, 0), ACTUAL_NUM_REQS - 1)
    req_begin = tl.load(PREFIX + req)
    req_end = tl.load(PREFIX + req + 1)
    sp = tl.load(START_POS + req)

    is_valid_row = rows < valid_rows
    comp0 = sp // CMP_RATIO                       # compressed position where the request starts
    cpos = comp0 + (rows - req_begin)            # compressed position of the current row

    # --- chunk partition (kvBlock boundary + TILE_ROWS; core boundaries ignored) ---
    blk = (cpos // KV_BLOCK_SIZE) * KV_BLOCK_SIZE
    blk0 = (comp0 // KV_BLOCK_SIZE) * KV_BLOCK_SIZE
    base = tl.where(blk == blk0, comp0, blk)
    cstart = base + ((cpos - base) // TILE_ROWS) * TILE_ROWS
    cend = tl.minimum(cstart + TILE_ROWS, blk + KV_BLOCK_SIZE)
    cend = tl.minimum(cend, comp0 + (req_end - req_begin))

    # --- validity (chunk level, see the module docstring L239-281) ---
    blk_idx = cpos // KV_BLOCK_SIZE
    ok_stride = blk_idx < KV_BLOCK_TABLE_STRIDE
    safe_blk_idx = tl.minimum(blk_idx, KV_BLOCK_TABLE_STRIDE - 1)
    block_id = tl.where(
        ok_stride,
        tl.load(KV_BLOCK_TABLE + req * KV_BLOCK_TABLE_STRIDE + safe_blk_idx),
        -1,
    )
    ok_block = block_id >= 0
    last_off = cend - 1 - blk
    max_slot = block_id.to(tl.int64) * KV_BLOCK_SIZE + last_off
    ok_slot = max_slot <= 2147483647  # MAX_INT32_VALUE (compressor_metadata.h L12)
    last_rope = (cend - 1).to(tl.int64) * CMP_RATIO
    ok_rope = last_rope < ROPE_ROWS
    valid = is_valid_row & ok_stride & ok_block & ok_slot & ok_rope

    # --- copy rope (valid rows take rope[cpos*cmpRatio]; invalid rows fall back to 1.0/0.0) ---
    rope_pos = tl.where(valid, cpos * CMP_RATIO, 0)
    d = tl.arange(0, BLK_D)
    dmask = d < ROPE_DIM
    d_safe = tl.minimum(d, ROPE_DIM - 1)
    in_offs = rope_pos[:, None] * ROPE_DIM + d_safe[None, :]
    in_mask = valid[:, None] & dmask[None, :]
    cos = tl.where(in_mask, tl.load(ROPE_COS + in_offs), 1.0)
    sin = tl.where(in_mask, tl.load(ROPE_SIN + in_offs), 0.0)

    out_offs = rows[:, None] * ROPE_DIM + d[None, :]
    out_mask = rmask[:, None] & dmask[None, :]
    tl.store(C_COS + out_offs, cos, mask=out_mask)
    tl.store(C_SIN + out_offs, sin, mask=out_mask)

    # --- slot mapping ---
    if SLOT_FORMAT == 1:  # SLOT_MAPPING_FLAT
        slot_v = block_id * KV_BLOCK_SIZE + (cpos - blk)
        tl.store(SLOT + rows, tl.where(valid, slot_v, -1), mask=rmask)
    else:
        slot_blk = tl.where(valid, block_id, -1)
        slot_off = tl.where(valid, cpos - blk, KV_BLOCK_SIZE - 1)
        # A 2-D [BLK_R,2] store instead of two "strided + masked" stores (the latter trips
        # the same FLIR assertion)
        col = tl.arange(0, 2)
        off2 = rows[:, None] * 2 + col[None, :]
        v2 = tl.where(col[None, :] == 0, slot_blk[:, None], slot_off[:, None])
        tl.store(SLOT + off2, v2, mask=rmask[:, None])


def compressor_metadata(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_pos: torch.Tensor,
    kv_block_table: torch.Tensor,
    num_rows: int,
    kv_block_size: int,
    slot_mapping_format: int = SLOT_MAPPING_FLAT,
    cmp_ratio: int = 1,
    actual_num_reqs: int | None = None,
    tile_rows: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equivalent to the AscendC CompressorMetadata operator.

    Args:
        rope_cos/rope_sin: [ropeRows, ropeDim], f32/f16/bf16 (same dtype)
        cu_seqlens: [numReqs+1] int32
        start_pos:  [>=actualNumReqs] int32
        kv_block_table: [rows, stride] int32
        num_rows: number of output rows (AscendC takes it from compressCos shape[0])
        kv_block_size / slot_mapping_format / cmp_ratio / actual_num_reqs: attributes
        tile_rows: block size determined by UB tiling (default 512, matches MAX_TILE_ROWS)

    Returns:
        (compressCos [num_rows, ropeDim], compressSin with the same shape, slotMapping)
        slotMapping: [num_rows] int32 (format=1) or [num_rows,2] int32 (format=2)
    """
    assert rope_cos.shape == rope_sin.shape, "ropeCos/ropeSin must have the same shape"
    assert rope_cos.dim() == 2, f"ropeCos must be 2D, got {tuple(rope_cos.shape)}"
    assert rope_cos.dtype == rope_sin.dtype, "ropeCos/ropeSin must have the same dtype"
    rope_rows, rope_dim = rope_cos.shape
    num_reqs = cu_seqlens.numel() - 1
    if actual_num_reqs is None:
        actual_num_reqs = min(num_reqs, start_pos.numel(), kv_block_table.shape[0])
    assert 1 <= actual_num_reqs <= _MAX_PREFIX_REQS, \
        f"actualNumReqs={actual_num_reqs} is outside the supported range"
    assert actual_num_reqs < cu_seqlens.numel()
    assert slot_mapping_format in (SLOT_MAPPING_FLAT, SLOT_MAPPING_BLOCK_OFFSET)
    assert kv_block_size >= 1 and cmp_ratio >= 1 and tile_rows >= 1

    dev = rope_cos.device
    rope_cos = rope_cos.contiguous()
    rope_sin = rope_sin.contiguous()
    cu_seqlens = cu_seqlens.contiguous().to(torch.int32)
    start_pos = start_pos.contiguous().to(torch.int32)
    kv_block_table = kv_block_table.contiguous().to(torch.int32)

    prefix = torch.empty(actual_num_reqs + 1, dtype=torch.int32, device=dev)
    _prefix_kernel[(1,)](
        cu_seqlens, start_pos, prefix,
        actual_num_reqs, cmp_ratio,
        BLK=next_pow2(actual_num_reqs),
    )

    compress_cos = torch.empty((num_rows, rope_dim), dtype=rope_cos.dtype, device=dev)
    compress_sin = torch.empty((num_rows, rope_dim), dtype=rope_cos.dtype, device=dev)
    slot_cols = 1 if slot_mapping_format == SLOT_MAPPING_FLAT else 2
    slot_shape = (num_rows,) if slot_cols == 1 else (num_rows, 2)
    slot_mapping = torch.empty(slot_shape, dtype=torch.int32, device=dev).reshape(-1)

    blk_d = next_pow2(rope_dim)
    blk_r = max(1, min(8, 4096 // blk_d))
    bin_steps = (actual_num_reqs + 1).bit_length()

    _compressor_metadata_kernel[(ceil_div(num_rows, blk_r),)](
        rope_cos, rope_sin, start_pos, kv_block_table, prefix,
        compress_cos, compress_sin, slot_mapping,
        num_rows, actual_num_reqs, rope_rows, rope_dim,
        kv_block_size, kv_block_table.shape[1], cmp_ratio,
        SLOT_FORMAT=slot_mapping_format,
        TILE_ROWS=tile_rows,
        BLK_R=blk_r,
        BLK_D=blk_d,
        BIN_STEPS=bin_steps,
    )
    return compress_cos, compress_sin, slot_mapping.reshape(slot_shape)
