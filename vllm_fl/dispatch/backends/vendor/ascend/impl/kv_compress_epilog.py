# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""kv_compress_epilog -- AscendC to Triton (quant_mode=1 GROUP_FP8 + layout=1)

Source: vllm-ascend/csrc/attention/kv_compress_epilog/
Interface (op_host/kv_compress_epilog_def.cpp L24-50):
    Input : kv_compress_cache  fp8_e5m2 / fp8_e4m3fn (written in place)
            x                  bf16  [rows, d]
            slot_mapping       int32 or int64 (the def declares 4 dtype combinations)
    Attr  : quant_group_size(128), quant_mode(1), round_scale(1), layout(1), block_stride(0)
    Output: kv_compress_cache

Geometry and formulas:
  op_kernel/kv_compress_epilog.h
    L104-124  **rows with slot_mapping==-1 are skipped and "compacted"**: valid rows are
              written tightly into xLocal (validIdx mechanism) and each is flushed with its
              own slot -- this is not "aligned by row index"
    L130-156  quantMode dispatch: 1 -> VFProcessDynamicBlockQuant; 2 -> MXFP8 variant
    L184-196  layout=1 flush: rowStride = block_stride>0 ? block_stride : kvCacheCol
                              gmOffset  = slot * rowStride
  op_kernel/kv_compress_epilog_common.h L127-266 VFProcessDynamicBlockQuant:
    # one row of x = [quantized d-64 cols | 64 rope cols passed through as-is]
    quantColNum = d - 64
    scaleColNum = ceil(quantColNum / 128)     # one fp32 scale per 128 elements
    scale = max(|x_sub|) * coeff              # coeff = (float)(1/448), a **multiply, not a divide**
    y     = x / scale  -> fp8 (CAST_RINT); max==0 -> 0
    one output row (kvCacheCol bytes, see the pointer arithmetic at L189/211/249-260):
        [0, quantColNum)                    fp8 quantized values
        [quantColNum, +128)                 last 64 cols of x (BF16 passed through = 128 bytes)
        [+128, +scaleColNum*4)              fp32 scale per sub-block (little endian)
        [concatCol, kvCacheCol)             zero padding
  op_host/kv_compress_epilog_tiling_arch35.cpp L211-263:
    concatCol = (d-64) + 128 + scaleCol*4 ; kvCacheCol = RoundUp(concatCol, 128) ; padCol = kvCacheCol-concatCol

**Not implemented (explicit NotImplementedError)**:
    - quant_mode != 1 (=2 is MXFP8, E8M0 power-of-two scale, two other implementations in
      common.h L269-724)
    - layout != 1 (the value/scale partitioning of layout=2 is described in tiling L220-248)
    - fp8_e5m2 (only e4m3fn is implemented; e5m2 needs a different encoding, coeff=1/57344)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["kv_compress_epilog"]

_ROPE = 64                      # SLICE_SIZE: number of rope columns passed through as-is
_PER_BLOCK = 128                # one scale per 128 elements
_FP8_MAX = 448.0                # FP8_E4M3FN_MAX_VALUE


@triton.jit
def _kce_to_e4m3_bits(y):
    """fp32 -> e4m3fn 8-bit encoding (RNE/round-half-to-even, matching the kernel's CAST_RINT).

    Same implementation as in indexer_compress_epilog_v2 (verified byte by byte against
    torch float8_e4m3fn).

    NOTE **ties (round-half-to-even) must round towards the even neighbour**:
    `fl = floor(q)` is an integer, so the two neighbours are necessarily one odd and one
    even, hence taking the even neighbour resolves the tie: if `fl` is **odd** round up to
    `fl + 1`; if `fl` is **even** keep `fl`. This is the IEEE-754 RNE definition and also
    the semantics of AscendC `CAST_RINT`.
    The older form `r = fl + tl.where(odd, 1.0, 0.0)` states literally the same thing, but
    was measured to differ by 1 (see the evidence below), so the explicit three-branch form
    is used to avoid depending on the backend behavior of that expression.

    Impact and evidence: the result differs by 1 only when `|x|/scale` falls exactly between
    two fp8 codes (`diff == 0.5`). The `kv_compress_epilog` case
    `d=576 group=64 rscale=0 rows=9 containing -1` happens to have 5/15360 bytes with
    `got = ref - 1`, at offsets 1690/1725/13225/13517/13531, corresponding to
    q = 11.5 / 9.5 / 10.5 / ... -- **all exact ties** (`diff - 0.5 == 0`).
    With `rscale=1` the scale is rounded to a power of two, so the quotient practically
    never lands on a tie; only rscale=0 exposes the issue.
    """
    b = y.to(tl.int32, bitcast=True)
    sign = (b >> 24) & 0x80
    x = (b & 0x7FFFFFFF).to(tl.float32, bitcast=True)
    x = tl.minimum(x, 448.0)
    ex = ((b >> 23) & 0xFF) - 127
    sc_bits = ((3 - ex) + 127) << 23
    q = x * sc_bits.to(tl.float32, bitcast=True)
    fl = tl.floor(q)
    diff = q - fl
    odd = (fl - tl.floor(fl * 0.5) * 2.0) != 0.0
    r = tl.where(diff > 0.5, fl + 1.0,
                 tl.where(diff < 0.5, fl, fl + tl.where(odd, 1.0, 0.0)))
    m = r - 8.0
    carry = m >= 8.0
    e_final = tl.where(carry, ex + 8, ex + 7)
    m_final = tl.where(carry, 0.0, m)
    code_norm = (e_final << 3) + m_final.to(tl.int32)
    qs = x * 512.0
    fls = tl.floor(qs)
    difs = qs - fls
    odds = (fls - tl.floor(fls * 0.5) * 2.0) != 0.0
    code_sub = tl.where(difs > 0.5, fls + 1.0,
                        tl.where(difs < 0.5, fls, fls + tl.where(odds, 1.0, 0.0))).to(tl.int32)
    code = tl.where(x < 0.015625, code_sub, code_norm)
    code = tl.where(x == 0.0, 0, code)
    return (sign | code) & 0xFF


@triton.jit
def _kce_kernel(
    X, SLOT, CACHE,
    ROWS, D, QUANT_COL, SCALE_COL, CONCAT_COL, KV_COL, ROW_STRIDE,
    BLK_Q: tl.constexpr,
    BLK_PAD: tl.constexpr,
):
    r = tl.program_id(0)                 # one program per row
    if r < ROWS:
        slot = tl.load(SLOT + r)
        if slot >= 0:
            base = slot * ROW_STRIDE
            # ---- quantized segment: one scale per 128 elements ----
            for s in range(0, SCALE_COL):
                cols = s * 128 + tl.arange(0, BLK_Q)
                m = cols < QUANT_COL
                v = tl.load(X + r * D + cols, mask=m, other=0.0).to(tl.float32)
                mx = tl.max(tl.where(m, tl.abs(v), 0.0), axis=0)
                scale = tl.where(mx > 0.0, mx * (1.0 / 448.0), 0.0)   # Mul, not Div
                y = tl.where(mx > 0.0, v / scale, 0.0)
                tl.store(CACHE + base + cols, _kce_to_e4m3_bits(y).to(tl.uint8), mask=m)
                si = scale.to(tl.int32, bitcast=True)
                tl.store(CACHE + base + QUANT_COL + 128 + s * 4 + tl.arange(0, 4),
                         ((si >> (tl.arange(0, 4) * 8)) & 0xFF).to(tl.uint8))
            # ---- rope pass-through segment: last 64 cols kept as BF16 (64*2=128 bytes, little endian) ----
            off = tl.arange(0, 64)
            rv = tl.load(X + r * D + QUANT_COL + off)                 # bf16
            u = rv.to(tl.uint16, bitcast=True).to(tl.int32)
            tl.store(CACHE + base + QUANT_COL + off * 2, (u & 0xFF).to(tl.uint8))
            tl.store(CACHE + base + QUANT_COL + off * 2 + 1, ((u >> 8) & 0xFF).to(tl.uint8))
            # ---- zero padding ----
            pc = tl.arange(0, BLK_PAD)
            tl.store(CACHE + base + CONCAT_COL + pc,
                     tl.zeros([BLK_PAD], dtype=tl.uint8), mask=CONCAT_COL + pc < KV_COL)


@triton.jit
def _kce_mxfp8_kernel(
    X, SLOT, CACHE,
    ROWS, D, QUANT_COL, SCALE_COL, CONCAT_COL, KV_COL, ROW_STRIDE,
    ROUND_SCALE: tl.constexpr,
    BLK_G: tl.constexpr,
    BLK_PAD: tl.constexpr,
):
    """quant_mode=2 (MXFP8): **the in-row layout is the opposite of group_fp8** -- rope comes
    first, in the leading 128 bytes.

    Source kv_compress_epilog_common.h L269-433 (VFProcessDynamicMxFP8Quant):
        L281 quantColNum = curColNum - 64 ; L282 scaleColNum = CeilDiv(quantColNum, perGroupSize)
        L338 rope is written to `ropeYLocalAddr + i*dstCol/2` (**byte offset 0**)
        L380/386 quantized values are written to `j*VL_FP32 + i*dstCol + **ropeNum(128)**`
        L391 scale is written to `quantColNum + ropeNum + scaleIdx` (**1 byte per group**)
        L361-362 sf = max(groupMax, 1e-4) * coeff
        L363-372 roundScale=True: e=(bits>>23)&0xFF; man=bits&0x7FFFFF;
                   eb = (e - 127) + (man!=0) + 127  => **scale = 2^ceil(log2(sf))** (round up)
        L376-379 y = clamp(x/scale, fp8Min, fp8Max)
        L389-391 scale byte = (scale_bits >> 23) & 0xFF (E8M0)
    """
    r = tl.program_id(0)
    if r < ROWS:
        slot = tl.load(SLOT + r)
        if slot >= 0:
            base = slot * ROW_STRIDE
            # ---- rope pass-through segment: **leading 128 bytes** (64 bf16 values, little endian) ----
            off = tl.arange(0, 64)
            rv = tl.load(X + r * D + QUANT_COL + off)
            u = rv.to(tl.uint16, bitcast=True).to(tl.int32)
            tl.store(CACHE + base + off * 2, (u & 0xFF).to(tl.uint8))
            tl.store(CACHE + base + off * 2 + 1, ((u >> 8) & 0xFF).to(tl.uint8))
            # ---- quantized segment: starts at byte 128, one shared E8M0 scale per BLK_G elements ----
            for s in range(0, SCALE_COL):
                cols = s * BLK_G + tl.arange(0, BLK_G)
                m = cols < QUANT_COL
                v = tl.load(X + r * D + cols, mask=m, other=0.0).to(tl.float32)
                mx = tl.max(tl.where(m, tl.abs(v), 0.0), axis=0)
                sf = tl.maximum(mx, 1.0e-4) * (1.0 / 448.0)      # Mul, not Div
                bits = sf.to(tl.int32, bitcast=True)
                e0 = (bits >> 23) & 0xFF
                man = bits & 0x7FFFFF
                if ROUND_SCALE:
                    eb = e0 + tl.where(man != 0, 1, 0)           # ceil(log2(sf))
                    scale = (eb << 23).to(tl.float32, bitcast=True)
                else:
                    eb = e0
                    scale = sf
                y = tl.minimum(tl.maximum(v / scale, -448.0), 448.0)
                tl.store(CACHE + base + 128 + cols, _kce_to_e4m3_bits(y).to(tl.uint8), mask=m)
                tl.store(CACHE + base + 128 + QUANT_COL + s, (eb & 0xFF).to(tl.uint8))
            # ---- zero padding ----
            pc = tl.arange(0, BLK_PAD)
            tl.store(CACHE + base + CONCAT_COL + pc,
                     tl.zeros([BLK_PAD], dtype=tl.uint8), mask=CONCAT_COL + pc < KV_COL)


def kv_compress_epilog(
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_compress_cache: torch.Tensor,
    quant_mode: int = 1,
    layout: int = 1,
    quant_group_size: int = 128,
    round_scale: int = 1,
    block_stride: int = 0,
) -> torch.Tensor:
    """Equivalent to the AscendC KvCompressEpilog (writes in place).

    **Production configuration** (`dsa_attn_kv_plan.py` L195-203 / vllm-fl
    `device_op.py` L1542-1550):
        `quant_mode=2, quant_group_size=64, round_scale_flag=True, layout=1`
    => **quant_mode=2 is the production path**; quant_mode=1 is kept for compatibility.

    Args:
        x: bf16 [rows, d]; one row = [quantized d-64 cols | 64 rope cols passed through as-is]
        slot_mapping: [rows] int32/int64; -1 means skip
        kv_compress_cache: fp8_e4m3fn (or a uint8 view) flat buffer
        quant_mode: 1 = GROUP_FP8 (fp32 scale, one per 128 elements);
                    2 = MXFP8 (**single-byte E8M0 scale**, one per `quant_group_size` elements)
        layout: 1 (production) or 2
        quant_group_size: fixed to 128 for mode 1 (hard-coded in the kernel); **mode 2 supports
                    64 / 128** (production uses 64)
        round_scale: 1 = E8M0 rounds to `2^ceil(log2(sf))`; 0 = use sf directly
                    (only meaningful for mode 2)
        block_stride: 0 -> kvCacheCol; non-zero must be >= kvCacheCol

    **Byte layout of one row (differs between the two modes)**:
        mode 1: `[fp8 quantCol][rope 128][fp32 scale x4B][pad]`
        mode 2: `[rope 128][fp8 quantCol][E8M0 scale x1B][pad]`   <- rope first
    """
    if int(quant_mode) not in (1, 2):
        raise NotImplementedError(
            f"quant_mode={quant_mode} is not implemented (only 1=GROUP_FP8 / 2=MXFP8)")
    if int(layout) not in (1, 2):
        raise NotImplementedError(
            f"layout={layout} is not implemented (only 1 or 2)")
    group = int(quant_group_size)
    if int(quant_mode) == 1:
        if group != _PER_BLOCK:
            raise NotImplementedError(
                f"quant_group_size for quant_mode=1 is fixed to 128 (hard-coded in the "
                f"kernel, L146), got {group}")
    elif group not in (64, 128):
        raise NotImplementedError(
            f"quant_group_size for quant_mode=2 only supports 64/128 (production uses 64), "
            f"got {group}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [rows, d], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bf16 (def.cpp L32), got {x.dtype}")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"slot_mapping must be int32 or int64, got {slot_mapping.dtype}")
    if kv_compress_cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise NotImplementedError(
            f"kv_compress_cache dtype={kv_compress_cache.dtype} is not implemented "
            f"(only fp8_e4m3fn is implemented; e5m2 needs a different encoding)")

    rows, d = x.shape
    if slot_mapping.numel() != rows:
        raise ValueError(f"slot_mapping numel {slot_mapping.numel()} != rows={rows}")
    if d <= _ROPE:
        raise ValueError(f"d={d} must be > 64 (one row = quantized segment d-64 + rope segment 64)")
    quant_col = d - _ROPE
    # Geometry depends on the mode (common.h L145-146 vs L281-282; scale width 4B(fp32) vs 1B(E8M0))
    if int(quant_mode) == 1:
        scale_col = ceil_div(quant_col, _PER_BLOCK)
        concat_col = quant_col + 128 + scale_col * 4
    else:
        scale_col = ceil_div(quant_col, group)
        concat_col = quant_col + 128 + scale_col
    kv_col = ceil_div(concat_col, 128) * 128
    # Flush row stride (kv_compress_epilog.h L184-196):
    #   layout==1 -> blockStride>0 ? blockStride : kvCacheCol
    #   layout!=1 (with quant_mode=1) -> kvCacheCol directly (the blockStride branch is skipped)
    if int(layout) == 1:
        row_stride = int(block_stride) if int(block_stride) != 0 else kv_col
    else:
        row_stride = kv_col
    if row_stride < kv_col:
        raise ValueError(f"block_stride={row_stride} must be >= kvCacheCol={kv_col}")

    slot = slot_mapping.contiguous()
    if slot.dtype == torch.int64:
        smax = int(slot.detach().abs().max().item()) if slot.numel() else -1
        if smax >= (1 << 31):
            raise ValueError("slot_mapping(int64) exceeds the int32 range, not implemented")
        slot = slot.to(torch.int32)

    cache_u8 = (kv_compress_cache if kv_compress_cache.dtype == torch.uint8
                else kv_compress_cache.view(torch.uint8))
    cap = cache_u8.numel()
    smax = int(slot.detach().max().item()) if slot.numel() else -1
    if smax >= 0:
        need = smax * row_stride + kv_col
        if cap < need:
            raise ValueError(f"cache too small: {need} bytes required "
                             f"(slot_max={smax}, rowStride={row_stride}), got {cap}")

    if int(quant_mode) == 1:
        _kce_kernel[(rows,)](
            x.contiguous(), slot, cache_u8,
            rows, d, quant_col, scale_col, concat_col, kv_col, row_stride,
            BLK_Q=128, BLK_PAD=128,
        )
    else:
        _kce_mxfp8_kernel[(rows,)](
            x.contiguous(), slot, cache_u8,
            rows, d, quant_col, scale_col, concat_col, kv_col, row_stride,
            ROUND_SCALE=bool(int(round_scale)),
            BLK_G=group, BLK_PAD=128,
        )
    return kv_compress_cache
