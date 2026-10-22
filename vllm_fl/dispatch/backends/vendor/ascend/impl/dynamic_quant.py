# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""dynamic_quant -- AscendC -> Triton (per-row / per-token symmetric dynamic quant)

Operator: ``torch_npu.npu_dynamic_quant`` (CANN ``aclnnDynamicQuant``). The aclnn op itself
is **not** vendored in this repo, so the formula below is taken from the three AscendC
operators in ``csrc/`` that implement the same per-row dynamic quant:

    csrc/ascend/moe/moe_init_routing_custom/op_kernel/moe_custom_full_load_dynamic_quant.h
        L137-141  Abs(tempLocal, inLocal); ReduceMax(dynamicQuantLocal, ...)   -> amax
        L143      float maxValue = dynamicQuantLocal.GetValue(0) / MAX_INT8;   -> dequant scale
        L150      Div(tempLocal, inLocal, tempLocal)                           -> q = x / scale
        L154      Cast(intLocal, tempLocal, RoundMode::CAST_RINT)              -> round half to even
        L157-159  int32 -> half (CAST_ROUND) -> int8 (CAST_TRUNC)
    csrc/ascend/moe/moe_init_routing_custom/op_kernel/moe_custom_common.h
        L36       constexpr float MAX_INT8 = 127.0f;
    csrc/ascend/moe/moe_init_routing_custom/op_kernel/moe_custom_gather_dynamic_quant.h
        L134/L382/L430  scale = reduceMax / MAX_INT8   (same rule, gather/scatter variants)
    csrc/ascend/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_normal_kernel.h
        L301-308  Abs + ReduceMaxInplace                                        -> maxTemp
        L309      scaleTemp = this->quantMaxVal / maxTemp;
        L310      scaleTensor.SetValue(rid, 1 / scaleTemp);   -> the *stored* value is amax/qmax
        L315      Muls(srcSlice, srcSlice, scaleTemp)         -> q = x * (qmax / amax)
        L218/L226 CAST_RINT then CAST_TRUNC
    csrc/ascend/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_helper.h
        L28-29    DYNAMIC_QUANT_DIVIDEND = 127.0 (int8), DYNAMIC_QUANT_DIVIDEND_INT4 = 7.0
    csrc/ascend/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h
        L33       DYNAMIC_QUANT_FACTOR = 1.0 / 127.0
        L706-712  WholeReduceMax then Muls(scaleOut, reduceMax, DYNAMIC_QUANT_FACTOR) -> amax/127
    csrc/ascend/attention/common/op_kernel/arch35/vf/vf_basic_block_utils.h
        L28-29    fp8e4m3MaxValue = 448.0f, int8MaxValue = 127.0f

Formula (per row, i.e. per token; the reduction is over the **last** dimension):
    amax  = max_d |x[d]|                                     (fp32)
    scale = amax / qmax                                      (the value that is returned)
    y     = round_half_to_even(x / scale)                     (AscendC CAST_RINT)
    qmax  = 127 (int8), 448 (float8_e4m3fn), 57344 (float8_e5m2)
    the quantized value is exact (|x / scale| <= qmax), so no saturation is involved and
    int8 can never produce -128 -- the sign range is [-127, 127].

Call contract -- the 10 ``torch_npu.npu_dynamic_quant`` sites in
``vendor/ascend/device/device_op.py`` (all arguments are positional or ``dst_type=``):

    L215  torch_npu.npu_dynamic_quant(hidden_states, dst_type=act_quant_type)
    L328  torch_npu.npu_dynamic_quant(hidden_states)
    L480  torch_npu.npu_dynamic_quant(hidden_states.contiguous())
    L825  torch_npu.npu_dynamic_quant(q, dst_type=torch.int8)
    L833  torch_npu.npu_dynamic_quant(q, dst_type=torch.int8)
    L839  torch_npu.npu_dynamic_quant(kv, dst_type=torch.int8)
    L855  torch_npu.npu_dynamic_quant(kv, dst_type=torch.int8)
    L873  torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)
    L1577 torch_npu.npu_dynamic_quant(q, dst_type=torch.float8_e4m3fn)
    L1586 torch_npu.npu_dynamic_quant(q, dst_type=torch.float8_e4m3fn)

so the entry point accepts the 1-argument form, the ``dst_type=`` form (the dtype may be
int8, float8_e4m3fn or float8_e5m2) and the ``act_quant_type=`` spelling that
``BaseDeviceAdaptor.npu_dynamic_quant`` (device_op L204-217) uses for the same parameter.

Semantics / deviations (all deliberate):
  1. Rows are the product of the leading dimensions and the reduction runs over the last
     one, so a [batch, heads, head_dim] query yields a [batch, heads] scale -- which is what
     the call sites consume (``q_scale.to(torch.float16)``, ``dynamic_scale.view(-1, 1)``).
  2. ``amax == 0`` (all-zero row): AscendC divides by zero (scale = 0, y = NaN). Here the
     quantized value is mapped to 0 and the scale to 0.0, i.e. a safe-guarded row -- the
     same choice as the sibling ``rms_norm_dynamic_quant`` port.
  3. ``dst_type=torch.int8`` is the default because 8 of the 10 sites quantize to int8.
     int4 / uint8 are **not** implemented: no call site uses them, and neither the rounding
     of the uint8 range nor the int4 packing layout could be confirmed from a local
     AscendC source. Passing them raises ``NotImplementedError`` with the value.
  4. ``quant_mode``: all ten sites are plain per-token symmetric quant. ``None`` and 0 are
     accepted; any other value raises -- the per-block/per-tensor variants of the aclnn op
     are not vendored and their exact scale rule could not be read from source.
  5. The fp8 branch (2 of 10 sites, the A5 indexer path) writes the destination bytes with
     the hand-rolled saturating round-half-to-even encoder ``_fp8_rne_codes`` instead of
     relying on the backend's own fp32 -> fp8 cast; that encoder is shared with
     ``dynamic_mx_quant`` and is byte-identical to the one validated in
     ``impl/swiglu_group_quant.py``. ``qmax = 448 / 57344`` for the fp8 branch is the
     format maximum (vf_basic_block_utils.h L28); the aclnn op's own choice of qmax for an
     fp8 destination is **not** verified against hardware here.
  6. ``dynamic_scale`` is accepted as an optional **pre-allocated output buffer** (the
     device_op wrapper at L214-217 returns early when it is not None, so it never reaches
     the operator). It is filled with the computed scale and returned.

Backend notes: every load and every store carries a mask; the tile shape is
[BLK_R, BLK_D] with BLK_R == 1, so the 2-D masked form is the same one measured reliable
for ``rms_norm_dynamic_quant`` on 910_93 + BiShengIR/FlagTree. The grid is capped at 65535
and a stride loop inside the kernel covers the remaining row blocks.

Validated locally on CPU (no NPU, no Triton -- see `_agent_logs/T-209_quant_ops/`):
  * ``_round_half_to_even`` is the same expression the ``rms_norm_dynamic_quant`` port uses
    and reproduces ``torch.round`` (which is round-half-to-even) exactly, including the
    -63.5 -> -64 tie of the hand-worked example;
  * ``_fp8_rne_codes`` reproduces torch's own float32 -> float8_e4m3fn cast byte for byte
    on every finite e4m3 value, both signs, every midpoint (exact tie) and those midpoints
    shifted by -2..+2 fp32 ulp, 200000 random values over 2**-20..2**20, the subnormal range
    and 0 / -0 / inf / NaN.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["dynamic_quant", "npu_dynamic_quant", "INT8_QUANT_MAX", "FP8_E4M3_MAX", "FP8_E5M2_MAX"]

#: MAX_INT8 (csrc .../moe_custom_common.h L36) == DYNAMIC_QUANT_DIVIDEND (helper.h L28).
INT8_QUANT_MAX = 127.0
#: fp8e4m3MaxValue (csrc .../vf/vf_basic_block_utils.h L28).
FP8_E4M3_MAX = 448.0
#: Largest finite e5m2 magnitude.
FP8_E5M2_MAX = 57344.0

#: 1-D grid limit of the backend; the kernel strides over the rest.
_MAX_GRID = 65535


@triton.jit
def _round_half_to_even(v):
    """AscendC ``RoundMode::CAST_RINT``: round to nearest, ties to even.

    The backend's own float -> integer cast rounds toward zero, so the rounding is spelled
    out: floor, look at the fraction, and send an exact .5 to the even neighbour.
    """
    f = tl.floor(v)
    fr = v - f
    odd = (f * 0.5) != tl.floor(f * 0.5)
    return tl.where((fr > 0.5) | ((fr == 0.5) & odd), f + 1.0, f)


@triton.jit
def _fp8_rne_codes(v, IS_E4M3: tl.constexpr):
    """Saturating float32 -> fp8 code byte, round half to even (CAST_RINT + SAT).

    Integer arithmetic on the |v| bit pattern, so the result does not depend on the
    backend's fp8 cast. Identical to ``_sgq_fp8_code`` in ``impl/swiglu_group_quant.py``
    (which was validated there against torch's own fp32 -> fp8 cast on every finite
    magnitude of both formats plus 200000 random values, 0 differing bytes):

      * normal path: round the fp32 mantissa to MANT bits with the standard
        "add half-1, then add the kept LSB" RNE form (a carry propagates into the
        exponent), then rebase the exponent bias 127 -> BIAS8;
      * subnormal path: scale by 2 ** (MANT + BIAS8 - 1), which maps the fp8 subnormal grid
        onto the integers, and round that with the explicit floor / half / odd form;
      * NaN returns the canonical fp8 NaN code; +/-inf saturates through the VMAX clamp.
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
def _dynamic_quant_kernel(
    X, Y, S,
    ROWS, D,
    NPROG,
    QMAX,
    BLK_R: tl.constexpr,
    BLK_D: tl.constexpr,
    OUT_INT8: tl.constexpr,
    IS_E4M3: tl.constexpr,
):
    """One row block per inner iteration: pass 1 takes the row amax, pass 2 quantizes.

    X        [ROWS, D] float16 / bfloat16 / float32
    Y        [ROWS, D] int8, or the uint8 view of the fp8 destination
    S        [ROWS]    float32 dequant scale, S = amax / QMAX
    NPROG    the grid size, used as the row-block stride
    """
    pid = tl.program_id(0)
    dr = tl.arange(0, BLK_D)
    nloop = tl.cdiv(D, BLK_D)

    # Pass 1: per-row maximum of |x|.
    for rb in range(pid, tl.cdiv(ROWS, BLK_R), NPROG):
        rows = rb * BLK_R + tl.arange(0, BLK_R)
        rmask = rows < ROWS
        amax = tl.zeros([BLK_R], dtype=tl.float32)
        for i in range(0, nloop):
            d = i * BLK_D + dr
            m2 = rmask[:, None] & (d < D)[None, :]
            v = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.where(m2, tl.abs(v), 0.0), axis=1))

        # amax == 0 (all-zero row) is guarded with 1.0 so the reciprocal stays finite; the
        # quantized value of such a row is 0 anyway and the stored scale is 0.0.
        nz = amax > 0.0
        inv = QMAX / tl.where(nz, amax, 1.0)
        tl.store(S + rows, tl.where(nz, amax / QMAX, 0.0), mask=rmask)

        # Pass 2: q = round_half_to_even(x * qmax / amax), written as int8 or fp8 bytes.
        for i in range(0, nloop):
            d = i * BLK_D + dr
            m2 = rmask[:, None] & (d < D)[None, :]
            v = tl.load(X + rows[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
            q = _round_half_to_even(v * inv[:, None])
            if OUT_INT8:
                # |q| <= 127 by construction; the clamp only guards fp32 edge effects and
                # keeps the value inside the range the AscendC op can emit ([-127, 127]).
                q = tl.minimum(tl.maximum(q, -QMAX), QMAX)
                tl.store(Y + rows[:, None] * D + d[None, :], q.to(tl.int8), mask=m2)
            else:
                code = _fp8_rne_codes(q, IS_E4M3=IS_E4M3)
                tl.store(Y + rows[:, None] * D + d[None, :], code.to(tl.uint8), mask=m2)


def _resolve_dst_type(dst_type, act_quant_type):
    """Pick the destination dtype from ``dst_type`` / the ``act_quant_type`` alias."""
    if act_quant_type is not None:
        if dst_type is not None and dst_type != act_quant_type:
            raise ValueError(
                f"conflicting destination dtypes: dst_type={dst_type}, act_quant_type={act_quant_type}"
            )
        dst_type = act_quant_type
    if dst_type is None:
        return torch.int8
    if not isinstance(dst_type, torch.dtype):
        raise NotImplementedError(
            f"dst_type={dst_type!r} (a ScalarType code) is not supported; pass a torch.dtype "
            "(torch.int8 / torch.float8_e4m3fn / torch.float8_e5m2)"
        )
    if dst_type in (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2):
        return dst_type
    raise NotImplementedError(
        f"dst_type={dst_type} is not implemented for dynamic_quant: no call site in "
        "vendor/ascend/device/device_op.py uses it, and neither the uint8 range nor the int4 "
        "packing/rounding could be confirmed from a local AscendC source"
    )


def dynamic_quant(
    x: torch.Tensor,
    dynamic_scale: torch.Tensor | None = None,
    *,
    dst_type: torch.dtype | None = None,
    act_quant_type: torch.dtype | None = None,
    quant_mode: int | None = 0,
    round_mode: str = "rint",
    scale_dtype: torch.dtype = torch.float32,
):
    """Drop-in equivalent of ``torch_npu.npu_dynamic_quant`` (per-token symmetric quant).

    Args:
        x: [..., D] float16 / bfloat16 / float32. Every row (the last dimension) gets its own
            scale; the leading dimensions are only flattened away.
        dynamic_scale: optional pre-allocated [..., 1] / [rows] float32 output buffer for the
            scale. The device_op wrapper returns early when this is not None, so it is never
            forwarded to the operator; here it is treated as an output buffer, filled with
            the computed scale and returned.
        dst_type: None (resolves to torch.int8, the dtype of 8 of the 10 sites) /
            torch.int8 / torch.float8_e4m3fn / torch.float8_e5m2.
        act_quant_type: alias for ``dst_type`` (the spelling used by
            ``BaseDeviceAdaptor.npu_dynamic_quant``, device_op L204-217). Giving both is only
            allowed when they are equal.
        quant_mode: None / 0 (per-token symmetric). Anything else raises.
        round_mode: only "rint" (round half to even, AscendC CAST_RINT) is implemented.
        scale_dtype: dtype of the returned scale when ``dynamic_scale`` is None (default
            float32, which is what all ten call sites consume).

    Returns:
        ``(y, scale)`` -- ``y`` has x's shape and the destination dtype, ``scale`` has x's
        shape without the last dimension (float32 by default) and holds ``amax / qmax``,
        i.e. the dequantization scale, exactly like the AscendC operator.
    """
    if x.dim() < 2:
        raise ValueError(f"dynamic_quant expects at least 2 dimensions [..., D], got {tuple(x.shape)}")
    if quant_mode not in (None, 0):
        raise NotImplementedError(
            f"quant_mode={quant_mode!r} is not implemented; all ten npu_dynamic_quant call "
            "sites are per-token symmetric quant (quant_mode 0/None)"
        )
    if round_mode != "rint":
        raise NotImplementedError(
            f"round_mode={round_mode!r} is not implemented; the AscendC dynamic quant "
            "operators use RoundMode::CAST_RINT (round half to even)"
        )
    out_dtype = _resolve_dst_type(dst_type, act_quant_type)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"dynamic_quant input dtype {x.dtype} is not supported")

    shape = tuple(x.shape)
    rows = 1
    for s in shape[:-1]:
        rows *= s
    d = shape[-1]
    dev = x.device
    x = x.contiguous()

    is_int8 = out_dtype is torch.int8
    qmax = INT8_QUANT_MAX if is_int8 else (FP8_E4M3_MAX if out_dtype is torch.float8_e4m3fn else FP8_E5M2_MAX)
    is_e4m3 = out_dtype is not torch.float8_e5m2

    y = torch.empty(shape, dtype=out_dtype, device=dev)
    scale_buf = torch.empty(shape[:-1], dtype=torch.float32, device=dev)
    if rows == 0 or d == 0:
        # Nothing to reduce (an empty row reduces to amax == 0), so the scale is 0.0 and the
        # quantized output is left as allocated, exactly like the empty case of the formula.
        empty = torch.zeros(shape[:-1], dtype=torch.float32, device=dev)
        if dynamic_scale is not None:
            dynamic_scale.copy_(empty.to(dynamic_scale.dtype))
            empty = dynamic_scale
        return y, empty

    blk_r = 1
    blk_d = _next_pow2(min(d, 2048)) if d > 1 else 1
    nblocks = _ceil_div(rows, blk_r)
    grid = (min(nblocks, _MAX_GRID),)

    # The fp8 bytes are written through a uint8 view; int8 goes through Y directly.
    y_arg = y if is_int8 else y.view(torch.uint8)
    _dynamic_quant_kernel[grid](
        x, y_arg, scale_buf,
        rows, d,
        grid[0],
        qmax,
        BLK_R=blk_r, BLK_D=blk_d,
        OUT_INT8=is_int8, IS_E4M3=is_e4m3,
    )

    if dynamic_scale is not None:
        dynamic_scale.copy_(scale_buf.to(dynamic_scale.dtype))
        return y, dynamic_scale
    if scale_dtype != torch.float32:
        return y, scale_buf.to(scale_dtype)
    return y, scale_buf


#: ``torch_npu`` spelling of the same operator.
npu_dynamic_quant = dynamic_quant
