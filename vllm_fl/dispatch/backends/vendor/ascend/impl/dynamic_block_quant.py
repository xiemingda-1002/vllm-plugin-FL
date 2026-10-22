# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the CANN operator of the same name (NOT vendored in csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""dynamic_block_quant -- ``torch_npu.npu_dynamic_block_quant`` -> Triton

## Provenance: no AscendC body in the tree

    grep -rn "dynamic_block_quant|DynamicBlockQuant|BlockQuant" csrc  -> no hit
    find csrc -iname "*block_quant*"                                  -> no hit

``npu_dynamic_block_quant`` is a CANN built-in. The arithmetic is recovered from three
local sources that pin the layout from both ends:

1. The **reference implementation used by the sibling project's unit test** of the very
   same call path (``_repos/vllm-ascend/tests/ut/attention/test_sfa_nope_forward.py``
   L111-118, patched in as ``torch_npu.npu_dynamic_block_quant`` at L349):

       def _dynamic_int8_block_quant(values, *, dst_type, row_block_size, col_block_size):
           assert dst_type == torch.int8
           assert row_block_size == 1
           assert col_block_size == 128
           tiles = values.float().reshape(*values.shape[:-1], -1, col_block_size)
           scale = tiles.abs().amax(dim=-1).clamp_min(1e-12) / 127
           quantized = (tiles / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
           return quantized.reshape(values.shape), scale

   (``.round()`` is round-half-to-even, i.e. AscendC ``CAST_RINT``.)

2. The **producer** in this tree: ``vllm_fl/attention/ascend/sfa_v1.py`` L1992-2028
   (``custom_kv_rmsnorm_rope``) calls it and immediately consumes the pair:
       L2009-2014  npu_dynamic_block_quant(k_nope.view(-1, 1, kv_lora_rank),
                                            dst_type=dst_type, row_block_size=1,
                                            col_block_size=tile_size)
       L2015-2021  int8 branch: knope_scale.to(torch.float32) and k_nope are returned as
                   bytes; the caller (L1179-1185) passes dst_type = 1 on non-A5 and
                   torch.float8_e4m3fn on A5.
3. The **consumer**: ``impl/kv_quant_sparse_flash_attention.py`` L42-63 documents the
   packed cache layout and the dequantization rule this scale feeds:
       k_dense[d] = q[d] * scale[d // col_block_size]        (no zero point)
       scale_metadata_bytes = (kv_lora_rank // tile_size) * 4   (sfa_v1 utils L82-93)
   i.e. **4 bytes per block scale in both branches** -> the returned scale is fp32 even on
   the fp8 path (the A5 branch only re-views the scale tensor as fp8 bytes without a cast,
   ``sfa_v1.py`` L2027, which is consistent with 4-byte scales).

## Semantics

    nblk = cdiv(D, col_block_size)
    for each (row r, block b):
        amax  = max_{d in block b} |x[r, d]|
        scale = max(amax, 1e-12) / qmax
        q     = clamp(rint(x[r, d] / scale), -127, 127)      # dst_type int8, qmax = 127
        q     = fp8_encode(x[r, d] / scale)                  # dst_type fp8_e4m3fn, qmax = 448
    y     shape = x.shape
    scale shape = x.shape[:-1] + (nblk,)   fp32

## Call sites (one executable call, in vllm_fl)

    vllm_fl/attention/ascend/sfa_v1.py:2009-2014   (inside custom_kv_rmsnorm_rope,
        called from L1179-1185 with dst_type = 1 on A2/A3 and torch.float8_e4m3fn on A5;
        x is 3-D [tokens, 1, kv_lora_rank], col_block_size = tile_size = 128)

    Note: ``impl/kv_quant_sparse_flash_attention.py`` L42 quotes this same call inside its
    module docstring (the docstring ends at L137 and the module never imports the op), so
    L42 is documentation, not a second call site; that file belongs to another task and was
    only read here.

## Implemented vs refused

Implemented: ``row_block_size == 1``; any power-of-two ``col_block_size <= 4096``;
``dst_type`` in {``torch.int8``, ``torch.qint8``, the ScalarType-style int ``1``,
``torch.float8_e4m3fn``}; x of any rank >= 1 in fp16 / bf16 / fp32 (leading dimensions are
flattened, the scale keeps them).

Refused (``NotImplementedError`` naming the value): ``row_block_size != 1`` (neither call
site uses it and the scale layout of a multi-row block is not pinned by any local source);
non-power-of-two ``col_block_size``; every other destination dtype (fp8_e5m2, int4/uint8 --
the int4 packing and the e5m2 scale convention are not documented in this tree).

## Uncertainties (also listed in RESULT.md)

* The int8 saturating range is ``[-127, 127]`` as in the reference above; a range of
  ``[-128, 127]`` would differ only for a value that lands exactly on -128 (impossible with
  ``|x / scale| <= 127`` unless rounding pushes it, which the clamp then handles).
* The fp8 branch's ``qmax = 448`` is the e4m3 format maximum (the value
  ``impl/dynamic_mx_quant.py`` and ``impl/dynamic_quant.py`` use for the same format). The
  exact qmax the CANN operator picks for an fp8 destination is **not confirmed**; because
  the pair (q, scale) stays self-consistent, a different qmax changes the scale byte and
  the rounding granularity but not the dequantized value.
* The fp8 encoder below is copied verbatim from ``impl/dynamic_quant.py`` L134-176 where it
  is documented as byte-identical to torch's own fp32 -> fp8 cast on every finite e4m3 value
  and 200000 random values. It is copied (not imported) so this module stays self-contained.

Backend notes: the tile is a pure 1-D ``[BLOCK_C]`` vector per block, every load and store
is masked, and the grid is capped at 32768 programmes with a row stride loop inside the
kernel (TRITON_OPERATOR_PITFALLS.md C1: 65535 is the measured ceiling).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b



__all__ = [
    "dynamic_block_quant",
    "npu_dynamic_block_quant",
    "INT8_BLOCK_QMAX",
    "FP8_E4M3_MAX",
    "BLOCK_QUANT_MIN_AMAX",
    "MAX_GRID",
]

#: int8 dynamic-quant dividend (csrc .../rms_norm_dynamic_quant_helper.h L28-29).
INT8_BLOCK_QMAX = 127.0
#: Largest finite e4m3 magnitude (same value the two sibling quant ports use).
FP8_E4M3_MAX = 448.0
#: The reference clamps the block amax to this before dividing (avoids a 0/0 scale).
BLOCK_QUANT_MIN_AMAX = 1e-12
#: Measured AI-core grid ceiling is 65535 (TRITON_OPERATOR_PITFALLS.md C1); stay below it.
MAX_GRID = 32768

_MAX_COL_BLOCK = 4096


@triton.jit
def _round_half_to_even(v):
    """AscendC ``RoundMode::CAST_RINT`` (round to nearest, ties to even)."""
    f = tl.floor(v)
    fr = v - f
    odd = (f * 0.5) != tl.floor(f * 0.5)
    return tl.where((fr > 0.5) | ((fr == 0.5) & odd), f + 1.0, f)


@triton.jit
def _fp8_rne_codes(v, IS_E4M3: tl.constexpr):
    """Saturating float32 -> fp8 code byte, round half to even (CAST_RINT + SAT).

    Verbatim from ``impl/dynamic_quant.py`` L134-176 (validated there against torch's own
    fp32 -> fp8 cast on every finite magnitude of both formats plus 200000 random values).
    """
    MANT: tl.constexpr = 3 if IS_E4M3 else 2
    BIAS8: tl.constexpr = 7 if IS_E4M3 else 15
    VMAX: tl.constexpr = 448.0 if IS_E4M3 else 57344.0
    CMAX: tl.constexpr = 0x7E if IS_E4M3 else 0x7B
    SH: tl.constexpr = 20 if IS_E4M3 else 21
    REBIAS: tl.constexpr = (127 - BIAS8) << MANT
    SUBN: tl.constexpr = (1 - BIAS8 + 127) << 23
    SUBSCALE: tl.constexpr = float(1 << (MANT + BIAS8 - 1))

    b = v.to(tl.int32, bitcast=True)
    sign = (b >> 24) & 0x80
    ab = b & 2147483647
    x = tl.minimum(ab.to(tl.float32, bitcast=True), VMAX)
    ab = x.to(tl.int32, bitcast=True)
    r = ab + ((1 << (SH - 1)) - 1) + ((ab >> SH) & 1)
    code_n = (r >> SH) - REBIAS
    xs = x * SUBSCALE
    fl = tl.floor(xs)
    fr = xs - fl
    odd = (fl - tl.floor(fl * 0.5) * 2.0) != 0.0
    code_s = tl.where(
        fr > 0.5, fl + 1.0,
        tl.where(fr < 0.5, fl, fl + tl.where(odd, 1.0, 0.0))).to(tl.int32)
    code = tl.where(ab < SUBN, code_s, code_n)
    code = tl.minimum(tl.maximum(code, 0), CMAX)
    code = tl.where(v != v, 0x7F, code)
    return (sign | code) & 0xFF


@triton.jit
def _dynamic_block_quant_kernel(
    X, Y, S,
    ROWS, D, NBLK, NPROG,
    QMAX,
    BLOCK_C: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """One row per iteration, ``NBLK`` column blocks of ``BLOCK_C`` elements inside it.

    X   [ROWS, D] fp16 / bf16 / fp32
    Y   [ROWS, D] int8, or the uint8 view of the fp8 destination
    S   [ROWS, NBLK] fp32 block scale, S = max(block amax, 1e-12) / QMAX
    """
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    for r in range(pid, ROWS, NPROG):
        for b in range(0, NBLK):
            d = b * BLOCK_C + c
            mask = d < D
            v = tl.load(X + r * D + d, mask=mask, other=0.0).to(tl.float32)
            amax = tl.max(tl.where(mask, tl.abs(v), 0.0))
            scale = tl.maximum(amax, 1e-12) / QMAX
            q = v / scale
            if IS_FP8:
                code = _fp8_rne_codes(q, IS_E4M3=True)
                tl.store(Y + r * D + d, code.to(tl.uint8), mask=mask)
            else:
                q = tl.minimum(tl.maximum(_round_half_to_even(q), -127.0), 127.0)
                tl.store(Y + r * D + d, q.to(tl.int8), mask=mask)
            tl.store(S + r * NBLK + b, scale)


def _resolve_dst_type(dst_type):
    """Map ``dst_type`` (torch dtype or ScalarType-style int) to int8 / fp8_e4m3fn."""
    if dst_type is None:
        return torch.int8
    if isinstance(dst_type, bool):
        raise NotImplementedError(f"dst_type={dst_type!r} is not a dtype")
    if isinstance(dst_type, int):
        if dst_type == 1:
            # sfa_v1.py L1184 passes the ScalarType-style int 1 for the int8 destination.
            return torch.int8
        raise NotImplementedError(
            f"dst_type={dst_type} (ScalarType-style int) is not implemented; only 1 (int8) "
            "is used by any call site"
        )
    if dst_type in (torch.int8, torch.qint8):
        return torch.int8
    if dst_type is torch.float8_e4m3fn:
        return torch.float8_e4m3fn
    raise NotImplementedError(
        f"dst_type={dst_type} is not implemented: only int8 and float8_e4m3fn are used by "
        "the call sites, and neither the int4 packing nor the e5m2 convention is documented "
        "in this tree"
    )


def dynamic_block_quant(
    x: torch.Tensor,
    *,
    dst_type=None,
    row_block_size: int = 1,
    col_block_size: int = 128,
):
    """Drop-in equivalent of ``torch_npu.npu_dynamic_block_quant`` for row-block size 1.

    Args:
        x: fp16 / bf16 / fp32, at least 1-D; the last dimension is quantized in blocks of
            ``col_block_size``, the leading dimensions are only flattened.
        dst_type: ``torch.int8`` / ``torch.qint8`` / the int ``1`` (int8, qmax 127) or
            ``torch.float8_e4m3fn`` (qmax 448). None means int8.
        row_block_size: must be 1 (both call sites).
        col_block_size: block width along the last dimension; a power of two, 128 in
            production.

    Returns:
        ``(y, scale)``: ``y`` has x's shape and the destination dtype; ``scale`` is fp32
        with shape ``x.shape[:-1] + (cdiv(D, col_block_size),)`` and holds
        ``max(block amax, 1e-12) / qmax``, i.e. the dequantization scale
        (``x ~= q * scale``, no zero point).
    """
    if row_block_size != 1:
        raise NotImplementedError(
            f"row_block_size={row_block_size} is not implemented: both call sites use 1, "
            "and the scale layout of a multi-row block is not pinned by any local source"
        )
    if not isinstance(col_block_size, int) or isinstance(col_block_size, bool):
        raise ValueError(f"col_block_size must be an int, got {col_block_size!r}")
    if col_block_size <= 0 or (col_block_size & (col_block_size - 1)) != 0:
        raise NotImplementedError(
            f"col_block_size={col_block_size} is not implemented: it must be a power of two "
            "(the kernel uses it as a compile-time tile width)"
        )
    if col_block_size > _MAX_COL_BLOCK:
        raise NotImplementedError(
            f"col_block_size={col_block_size} exceeds the {_MAX_COL_BLOCK} cap that keeps "
            "one block inside the 192 KiB UB"
        )
    if x.dim() < 1:
        raise ValueError("dynamic_block_quant expects at least 1-D input")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"input dtype {x.dtype} is not implemented (fp16/bf16/fp32)")

    out_dtype = _resolve_dst_type(dst_type)
    is_fp8 = out_dtype is torch.float8_e4m3fn
    qmax = FP8_E4M3_MAX if is_fp8 else INT8_BLOCK_QMAX

    shape = tuple(int(s) for s in x.shape)
    d = shape[-1]
    rows = 1
    for s in shape[:-1]:
        rows *= s
    nblk = _ceil_div(d, col_block_size)

    x2 = x.contiguous().reshape(rows, d)
    y = torch.empty(shape, dtype=out_dtype, device=x.device)
    scale = torch.empty((*shape[:-1], nblk), dtype=torch.float32, device=x.device)
    if rows == 0 or d == 0 or nblk == 0:
        return y, scale

    y2 = y if not is_fp8 else y.view(torch.uint8)
    y2 = y2.reshape(rows, d)
    s2 = scale.reshape(rows, nblk)
    nprog = min(rows, MAX_GRID)
    _dynamic_block_quant_kernel[(nprog,)](
        x2, y2, s2,
        rows, d, nblk, nprog,
        qmax,
        BLOCK_C=col_block_size,
        IS_FP8=is_fp8,
    )
    return y, scale


#: ``torch_npu`` spelling of the same operator.
npu_dynamic_block_quant = dynamic_block_quant
