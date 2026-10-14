# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""indexer_compress_epilog_v2 -- AscendC -> Triton

Source: vllm-ascend/csrc/attention/indexer_compress_epilog_v2/
Interface (op_host/indexer_compress_epilog_v2_def.cpp L22-46):
    Input : indexer_compress_cache  uint8  4D [numBlocks, cacheBs, 1, cacheD] (written in place)
            x                       fp16/bf16  [rows, d]
            slot_mapping            int32      [rows] (-1 = skip)
    Attr  : layout (default 2), block_stride (default 0)
    Output: indexer_compress_cache (in place)

Shape / geometry rules (op_host/indexer_compress_epilog_v2_tiling.cpp L94-140):
    cacheBs   = cache.shape[1]
    cache.shape[2] must be 1; cacheD = cache.shape[3]
    d         = x.shape[-1]
    scaleCol  = ceil(d / PER_BLOCK_FP16)          # one scale per 128 elements
    blockStride default (0) -> cacheBs*cacheD; when non-zero it must be >= cacheBs*cacheD
    layout only supports 2

Formula source (op_kernel/indexer_compress_epilog_v2_base.h L133-243
                VFProcessDynamicBlockQuant, and
                op_kernel/indexer_compress_epilog_v2_single_row.h L49-98 Process):
    Single row (slot != -1):
        # each 128-element sub-block computes its own scale (the fold loop merges two
        # VL_FP32=64 blocks into 128 and writes one scale)
        scale = max(|x|) * (1/448)      # INV_FP8_E4M3_MAX_VALUE; max==0 -> scale=0
        y     = x / scale               # when scale==0 the Div yields NaN -> Select keeps x (=0)
        store y as fp8_e4m3 (RoundMode::CAST_RINT, i.e. round-to-nearest-even; NO_SAT)
    Store addressing (single_row.h L83-93, units of **bytes**):
        blkNumIdx = slot // cacheBs ; blkSizeIdx = slot % cacheBs
        valueOffset = blkNumIdx*blockStride + blkSizeIdx*d
        scaleOffset = blkNumIdx*blockStride + cacheBs*d + blkSizeIdx*scaleCol*4
      => per-block layout = [cacheBs*d fp8 bytes][cacheBs*scaleCol fp32 scales]

Implementation notes:
    - cache is treated as a flat **uint8** buffer; fp8 values are written byte-wise and scales as
      fp32 little-endian 4 bytes
    - one program handles one row (1-D, so a scalar is never broadcast through [:, None] into a
      2-D base address)
    - no tl.reshape is used to build offsets (avoids the backend's "reshaped broadcast offset is
      silently misaligned" issue)

**Semantic assumption (not verified on the AscendC side)**: the default `blockStride` of
cacheBs*cacheD makes the scale region **overlap** the value region of the next block (tiling only
checks >= cacheBs*cacheD). This implementation follows the kernel formula for the byte offsets
exactly; callers that need non-overlapping value/scale regions should pass
block_stride = cacheBs*cacheD + cacheBs*scaleCol*4 explicitly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div

__all__ = ["indexer_compress_epilog_v2"]

_PER_BLOCK_FP16 = 128          # tiling: PER_BLOCK_FP16
_INV_FP8_E4M3_MAX = 1.0 / 448.0


@triton.jit
def _ice_to_e4m3_bits(y):
    """fp32 -> 8-bit e4m3fn encoding (RNE, matching the kernel's CAST_RINT).

    e4m3fn: sign1/exp4/man3, bias=7, **no inf**, E=15&m=7 is NaN, largest finite value 448
    (=0x7E). This backend has no fp8 cast ('unrecognized float type f8E4M3FN'), so the encoding
    is written by hand:
      - |y| >= 2^-6: normalize into [8,16), m = round(x*2^(3-e)) - 8, m==8 -> carry
      - |y| <  2^-6: subnormal, code = round(x / 2^-9)
      - 0 -> 0; above 448 -> saturate to 448 (scale=max/448 makes this unreachable; guard only)
    """
    b = y.to(tl.int32, bitcast=True)
    sign = (b >> 24) & 0x80
    x = (b & 0x7FFFFFFF).to(tl.float32, bitcast=True)
    x = tl.minimum(x, 448.0)
    ex = ((b >> 23) & 0xFF) - 127

    # Normal path: q = x * 2^(3-ex) in [8,16)
    sc_bits = ((3 - ex) + 127) << 23
    q = x * sc_bits.to(tl.float32, bitcast=True)
    fl = tl.floor(q)
    diff = q - fl
    odd = (fl - tl.floor(fl * 0.5) * 2.0) != 0.0     # fl is odd
    r = tl.where(diff > 0.5, fl + 1.0,
                 tl.where(diff < 0.5, fl, fl + tl.where(odd, 1.0, 0.0)))
    m = r - 8.0
    carry = m >= 8.0
    e_final = tl.where(carry, ex + 8, ex + 7)
    m_final = tl.where(carry, 0.0, m)
    code_norm = (e_final << 3) + m_final.to(tl.int32)

    # Subnormal path: qs = x / 2^-9 = x*512
    qs = x * 512.0
    fls = tl.floor(qs)
    difs = qs - fls
    odds = (fls - tl.floor(fls * 0.5) * 2.0) != 0.0
    code_sub = tl.where(difs > 0.5, fls + 1.0,
                        tl.where(difs < 0.5, fls, fls + tl.where(odds, 1.0, 0.0))).to(tl.int32)

    code = tl.where(x < 0.015625, code_sub, code_norm)      # 2^-6 boundary
    code = tl.where(x == 0.0, 0, code)
    return (sign | code) & 0xFF


@triton.jit
def _ice_kernel(
    X, SLOT, CACHE,
    ROWS, D, CACHE_BS, SCALE_COL, STRIDE,
    BLK_Q: tl.constexpr,
):
    r = tl.program_id(0)          # one program per row
    if r < ROWS:
        slot = tl.load(SLOT + r)
        if slot >= 0:
            blk_num = slot // CACHE_BS
            blk_off = slot % CACHE_BS
            v_base = blk_num * STRIDE + blk_off * D
            s_base = blk_num * STRIDE + CACHE_BS * D + blk_off * SCALE_COL * 4
            for s in range(0, SCALE_COL):
                cols = s * 128 + tl.arange(0, BLK_Q)
                m = cols < D
                v = tl.load(X + r * D + cols, mask=m, other=0.0).to(tl.float32)
                mx = tl.max(tl.where(m, tl.abs(v), 0.0), axis=0)
                scale = tl.where(mx > 0.0, mx * (1.0 / 448.0), 0.0)
                # scale==0 (all-zero sub-block) -> emit 0 (mirrors Select keeping 0 in AscendC)
                y = tl.where(scale > 0.0, v / scale, 0.0)
                tl.store(CACHE + v_base + cols, _ice_to_e4m3_bits(y).to(tl.uint8), mask=m)
                # The scale is written as one little-endian 4-byte vector; the scale of sub-block
                # s lands at s_base + s*4
                si = scale.to(tl.int32, bitcast=True)
                sh = tl.arange(0, 4) * 8
                b4 = ((si >> sh) & 0xFF).to(tl.uint8)
                tl.store(CACHE + s_base + s * 4 + tl.arange(0, 4), b4)


def indexer_compress_epilog_v2(
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    indexer_compress_cache: torch.Tensor,
    layout: int = 2,
    block_stride: int = 0,
) -> torch.Tensor:
    """Equivalent of the AscendC IndexerCompressEpilogV2 (writes the cache in place, returns it).

    Args:
        x: [rows, d] fp16/bf16 (rows = slot_mapping.numel())
        slot_mapping: [rows] int32; -1 skips
        indexer_compress_cache: uint8 4D [numBlocks, cacheBs, 1, cacheD] (may be a view of a
            larger flat buffer)
        layout: only 2 is supported
        block_stride: 0 -> cacheBs*cacheD; non-zero must be >= cacheBs*cacheD (in bytes)

    Returns:
        indexer_compress_cache (updated)
    """
    if int(layout) != 2:
        raise NotImplementedError(f"layout={layout} is not implemented (tiling only accepts 2)")
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [rows, d], got {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"x only supports fp16/bf16 (def.cpp L30), got {x.dtype}")
    if slot_mapping.dtype != torch.int32:
        raise ValueError(f"slot_mapping must be int32, got {slot_mapping.dtype}")
    if indexer_compress_cache.dtype != torch.uint8:
        raise ValueError(f"indexer_compress_cache must be uint8, got {indexer_compress_cache.dtype}")
    if indexer_compress_cache.dim() != 4:
        raise ValueError("indexer_compress_cache must be 4D [numBlocks, cacheBs, 1, cacheD]")
    if indexer_compress_cache.shape[2] != 1:
        raise ValueError("dimension 2 of indexer_compress_cache must be 1")
    rows, d = x.shape
    if slot_mapping.numel() != rows:
        raise ValueError(f"slot_mapping numel {slot_mapping.numel()} != rows={rows}")
    cache_bs = int(indexer_compress_cache.shape[1])
    cache_d = int(indexer_compress_cache.shape[3])
    if cache_d != d:
        raise ValueError(f"cacheD={cache_d} must equal the last dimension of x, d={d}")
    scale_col = ceil_div(d, _PER_BLOCK_FP16)
    # The call site (csrc/torch_binding.cpp L1902-1904) passes indexer_compress_cache.stride(0)
    # as block_stride; only the default attr value 0 falls back to cacheBs*cacheD (tiling L181-183)
    stride0 = int(indexer_compress_cache.stride(0))
    stride = int(block_stride) if int(block_stride) != 0 else max(stride0, cache_bs * cache_d)
    if stride < cache_bs * cache_d:
        raise ValueError(f"block_stride={stride} must be >= cacheBs*cacheD={cache_bs * cache_d}")

    num_blocks = int(indexer_compress_cache.shape[0])
    need = (num_blocks - 1) * stride + cache_bs * d + cache_bs * scale_col * 4
    # cache is often a view of a larger buffer (indexer_full_cache.view(torch.uint8)); size the
    # capacity from the underlying storage
    capacity = (indexer_compress_cache.untyped_storage().nbytes()
                - indexer_compress_cache.storage_offset())
    if capacity < need:
        raise ValueError(
            f"cache backing buffer is too small: need {need} bytes (blockStride={stride}, "
            f"cacheBs={cache_bs}, cacheD={cache_d}, scaleCol={scale_col}), backing buffer has "
            f"{capacity} bytes")

    blk_q = 128
    _ice_kernel[(rows,)](
        x.contiguous(), slot_mapping.contiguous(), indexer_compress_cache,
        rows, d, cache_bs, scale_col, stride,
        BLK_Q=blk_q,
    )
    return indexer_compress_cache
