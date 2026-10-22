# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the CANN operator of the same name (NOT vendored in csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""quantize -- ``torch_npu.npu_quantize`` (aclnnAscendQuantV3) -> Triton

## Provenance: no AscendC body in the tree

    grep -rn "format_cast|npu_quantize|aclnnAscendQuant" csrc   -> no hit
    find csrc -iname "*.h" | grep -i ascendquant               -> no hit

``npu_quantize`` is a CANN built-in. The semantics below come from the single call site in
this tree plus the sibling checkout's comment that documents ``div_mode``:

    vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/linear_utils.py
        L36-48   def _quantize_impl(in_tensor, input_scale, input_scale_reciprocal,
                                    input_offset):
                     return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal,
                                                   input_offset, torch.qint8, -1, False)
        L51-58   _quantize_impl_fake -- the fake/registration twin, same call
        L61-77   register_quantize() -> torch.ops.vllm.quantize (PrivateUse1)
    callers of torch.ops.vllm.quantize:
        vllm_fl/.../impl/quantization/linear.py:144-149 (W8A8 static, non-int8 x)
        vllm_fl/compilation/passes/norm_quant_fusion_pass.py L67/L129/L191/L255 (the
        compile-time norm+quant fusion emits the same wrapper with the same
        (scale, scale_reciprocal, offset) argument order)
        (_repos/vllm-ascend/vllm_ascend/ops/register_custom_ops.py L178-187 has the same
        wrapper plus the reason: "aclnnAscendQuantV3(npu_quantize) use div_mode=False,
        while aclnnAddRmsNormQuantV2(npu_add_rms_norm_quant) use div_moe=True. We have to
        pass input_scale and input_scale_reciprocal at the same time ...")

=> ``div_mode=False`` means ``scales`` is a **multiplier** (the reciprocal of the model's
input scale), not a divisor; ``axis=-1`` means the scale/offset are per-input-channel along
the innermost dimension. ``linear.py`` L165-173 builds those tensors:
``input_scale.repeat(expanding_factor)`` with ``expanding_factor = weight.shape[1] = K``,
so the scale and the offset both have length K.

## Semantics

    div_mode=False:  y = clamp(rint(x * scales + zero_points), -128, 127) -> int8
    div_mode=True :  y = clamp(rint(x / scales + zero_points), -128, 127) -> int8

``zero_points`` is an int8 zero point; ``linear.py`` L171-173 stores it cast to the scale
dtype (fp16/bf16), where small integers are exact, so "add before rounding" and "add after
rounding" are the same number. The saturation range is the full int8 range.

Rounding: AscendC's quant path uses ``RoundMode::CAST_RINT`` (round half to even) --
``impl/rms_norm_dynamic_quant.py`` L32/L166 and ``impl/dynamic_quant.py`` L14 record the
same convention for the two sibling operators. For *this* aclnn op there is no local source
pin, so the rounding mode is a documented assumption; it only matters on exact ties.

## Implemented vs refused

Implemented: input fp16 / bf16 / fp32; ``scales`` ``[K]`` or ``[1]`` (per-tensor); optional
``zero_points`` of the same two shapes; ``dtype`` ``torch.qint8`` / ``torch.int8`` (the only
value any call site passes); any ``axis`` (non-last axes are moved to the last dimension on
the host and moved back, the kernel itself always works on the innermost dimension).

Refused: ``torch.quint4x2`` / ``torch.quint8`` / fp8 destinations -- no call site in
``vllm_fl`` uses them and the int4 packing layout is not documented in this tree.

Deviation worth knowing: a real ``qint8`` torch tensor needs a quantization scheme and
cannot be allocated with ``torch.empty``; the CANN op returns a plain byte tensor that the
next operator (``npu_quant_matmul``) consumes as int8, so this port returns ``torch.int8``.
The consumer checks ``x.dtype != torch.int8`` (``linear.py`` L134), which the int8 result
satisfies.

Kernel note: one program handles one row of ``BLOCK`` columns (``BLOCK`` = next power of two
of the channel count, capped at 4096 so a row always fits the 192 KiB UB); every load and
store is masked, and rows beyond the 32768-programme cap are covered by a stride loop.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["quantize", "npu_quantize", "INT8_QMIN", "INT8_QMAX", "MAX_GRID"]

INT8_QMIN = -128.0
INT8_QMAX = 127.0
#: Measured AI-core grid ceiling is 65535 (TRITON_OPERATOR_PITFALLS.md C1); stay below it.
MAX_GRID = 32768

_ROW_BLOCK_CAP = 4096


@triton.jit
def _round_half_to_even(v):
    """AscendC ``RoundMode::CAST_RINT``: round to nearest, ties to even.

    The backend's float -> integer cast truncates, so the rounding is spelled out; this is
    the same expression as ``impl/dynamic_quant.py`` L121-131.
    """
    f = tl.floor(v)
    fr = v - f
    odd = (f * 0.5) != tl.floor(f * 0.5)
    return tl.where((fr > 0.5) | ((fr == 0.5) & odd), f + 1.0, f)


@triton.jit
def _quantize_kernel(
    X, S, Z, Y,
    OUTER, COLS, NPROG,
    S_STRIDE: tl.constexpr,
    Z_STRIDE: tl.constexpr,
    HAS_ZP: tl.constexpr,
    DIV_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """y[r, c] = clamp(rint(x[r, c] * s[c] + z[c])) for one row per iteration.

    X / Y are [OUTER, COLS] contiguous; S is [COLS] (S_STRIDE=1) or [1] (S_STRIDE=0);
    Z is the same shape as S when HAS_ZP, otherwise the unused dummy pointer.
    """
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    mask = c < COLS
    for r in range(pid, OUTER, NPROG):
        v = tl.load(X + r * COLS + c, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(S + c * S_STRIDE, mask=mask, other=1.0).to(tl.float32)
        if DIV_MODE:
            t = v / s
        else:
            t = v * s
        if HAS_ZP:
            z = tl.load(Z + c * Z_STRIDE, mask=mask, other=0).to(tl.float32)
            t = t + z
        q = _round_half_to_even(t)
        # Full int8 saturation range (INT8_QMIN / INT8_QMAX are the host-side twins).
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(Y + r * COLS + c, q.to(tl.int8), mask=mask)


def _channel_tensor(t: torch.Tensor | None, cols: int, name: str) -> tuple[torch.Tensor | None, int]:
    """Flatten a per-channel / per-tensor parameter and return it with its element stride."""
    if t is None:
        return None, 0
    t = t.contiguous().reshape(-1)
    if t.numel() == 1:
        return t, 0
    if t.numel() != cols:
        raise ValueError(f"{name} has {t.numel()} elements, expected 1 or {cols} (the axis length)")
    return t, 1


def quantize(
    x: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    axis: int = -1,
    div_mode: bool = False,
):
    """Drop-in equivalent of ``torch_npu.npu_quantize`` for the per-channel int8 form.

    Args:
        x: input activations, fp16 / bf16 / fp32, at least 1-D.
        scales: per-channel scale along ``axis``, ``[K]``, or a one-element per-tensor scale.
            With ``div_mode=False`` (the only mode any call site uses) this is a
            multiplier, i.e. the reciprocal of the model's input scale.
        zero_points: optional int8 zero point, ``[K]`` or ``[1]``; may be None.
        dtype: destination dtype; ``torch.qint8`` / ``torch.int8`` (None means int8, which
            is the only destination any call site asks for).
        axis: the channel axis (``-1`` at the only call site).
        div_mode: ``False`` -> multiply by ``scales``; ``True`` -> divide.

    Returns:
        int8 tensor with ``x``'s shape.
    """
    if dtype is None:
        dtype = torch.int8
    if dtype not in (torch.qint8, torch.int8):
        raise NotImplementedError(
            f"dtype={dtype} is not implemented: the only npu_quantize call site "
            "(linear_utils.py L46-48) quantizes to torch.qint8/int8, and the packing layout "
            "of the int4 (quint4x2) destination is not documented in this tree"
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"input dtype {x.dtype} is not implemented (fp16/bf16/fp32)")
    if x.dim() < 1:
        raise ValueError("quantize expects at least 1-D input")

    ndim = x.dim()
    ax = axis + ndim if axis < 0 else axis
    if ax < 0 or ax >= ndim:
        raise IndexError(f"axis={axis} is out of range for a {ndim}-D input")

    xt = x.movedim(ax, -1).contiguous() if ax != ndim - 1 else x.contiguous()
    cols = int(xt.shape[-1])
    outer = 1
    for s in xt.shape[:-1]:
        outer *= int(s)

    s, s_stride = _channel_tensor(scales, cols, "scales")
    if s is not None and s.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"scales dtype {s.dtype} is not implemented")
    z, z_stride = _channel_tensor(zero_points, cols, "zero_points")
    if z is not None and z.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.int8):
        raise NotImplementedError(f"zero_points dtype {z.dtype} is not implemented")

    y = torch.empty(xt.shape, dtype=torch.int8, device=x.device)
    if outer == 0 or cols == 0:
        out = y
    else:
        x2 = xt.reshape(outer, cols)
        y2 = y.reshape(outer, cols)
        blk = _next_pow2(min(cols, _ROW_BLOCK_CAP))
        nprog = min(outer, MAX_GRID)
        _quantize_kernel[(nprog,)](
            x2, s, z if z is not None else x2, y2,
            outer, cols, nprog,
            S_STRIDE=s_stride,
            Z_STRIDE=z_stride,
            HAS_ZP=z is not None,
            DIV_MODE=bool(div_mode),
            BLOCK=blk,
        )
        out = y
    if ax != ndim - 1:
        out = out.movedim(-1, ax).contiguous()
    return out


#: ``torch_npu`` spelling of the same operator.
npu_quantize = quantize
