# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the CANN operator of the same name (NOT vendored in csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""quant_matmul -- ``torch_npu.npu_quant_matmul`` (aclnnQuantMatmul) -> Triton

## Provenance: there is no AscendC kernel for this operator in the tree

    grep -rn "QuantMatmul|quant_matmul" csrc   -> only csrc/ascend/cmake/opbuild.cmake
                                                  L194/195/209/210, four *different* fused
                                                  ops (weight_quant_matmul_all_reduce*,
                                                  inplace_quant_matmul_all_reduce*)
    find csrc -iname "*quant_matmul*"          -> no hit
    find csrc -iname "*format_cast*"           -> no hit (see format_cast.py)

``npu_quant_matmul`` is a CANN built-in exposed through torch_npu, so the arithmetic is
recovered from local sources instead of a kernel body:

1. The 22 call sites in ``vllm_fl`` (table below) pin the argument names, the shapes and
   the dtypes. Every one of them routes int8 activations / int8 weights into it.
2. The producer of ``pertoken_scale`` in this same tree is ``rms_norm_dynamic_quant``
   (``impl/rms_norm_dynamic_quant.py`` L31: ``outScale = maxAbs / quantMaxVal``) or
   ``npu_dynamic_quant`` -- i.e. the per-token value is a **dequantization** scale
   (``x ~= q * scale``), not a multiplier.
3. The consumer of the ``output_dtype=torch.int32`` form is ``npu_dequant_swiglu_quant``,
   whose AscendC body applies both scales *after* the int32 input:
   ``csrc/ascend/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h``
       L367-368  int32 input + non-empty activationScale -> activationScale is read
       L404-407  xF32 = weightScale[n] * xF32
       L409-412  xF32 = activationScale[m] * xF32
   => the int32 result of the matmul is the **raw integer accumulator**: no scale and no
   bias dequantization may be applied on the matmul side for that dtype.
4. The int64 ``deq_scale`` variant (fp16 ModelSlim static) is decoded in the sibling
   checkout ``_repos/vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_static.py``
   L144-146: "FP16 quant_matmul scales encode FP32 bits in the low 32 bits" ->
   ``deq_scale.to(torch.int32).view(torch.float32)``. The same conversion is used here.
5. ``_repos/vllm-ascend/.../w8a8/w8a8_static.py`` L105-116 and ``linear.py`` L151-161 of
   this tree show the two bias flavours: an int32 ``quant_bias`` (ModelSlim) and a
   bf16/fp16 ``bias`` (compressed-tensors, where ``deq_scale = input_scale * weight_scale``
   already folds the **activation** scale in, so the bias cannot be pre-dequant there).

## Semantics (what this implementation computes)

    acc[m, n] = sum_k x[m, k] * w[k, n]                            # int32 accumulation
    out[m, n] = (acc[m, n] + bias_i32[n]) * scale[n] * pertoken_scale[m]
    out[m, n] = acc[m, n] * scale[n] * pertoken_scale[m] + bias_float[n]
    out[m, n] = acc[m, n] + bias_i32[n]                            # output_dtype=int32

``bias`` is applied **before** the dequantization when it is an integer tensor and
**after** it when it is a float tensor; ``output_dtype=torch.int32`` is the raw integer
accumulator, where ``scale`` and ``pertoken_scale`` are deliberately ignored (see
evidence 3 above).

## Call sites (22, all in vllm_fl)

    vllm_fl/attention/ascend/dsa_v1.py:1986, 2055, 2071, 2094, 2369, 2382, 2392, 2418,
        2661, 2989
        (x int8 [T, K] / 3-D, weight int8 [K, N], weight_scale [N] float, pertoken_scale
        [T] fp32, bias bf16, output_dtype = the activation dtype)
    vllm_fl/attention/ascend/sfa_v1.py:1407
        (same, plus - only when ``q_c_tensor`` is fp8 - ``scale_dtype`` /
        ``pertoken_scale_dtype`` = e8m0 and ``group_sizes=[1, 1, group_size]``)
    vllm_fl/attention/ascend/context_parallel/dsa_cp.py:1341, 1349, 1361, 1628   (same)
    vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/linear.py:155
        (x int8, weight int8 [K, N], ``deq_scale`` float32 **or int64**, bias int32 or
        bf16, output_dtype = params dtype)
    .../impl/quantization/linear.py:226, 234, 246
        (W8A8_DYNAMIC: pertoken_scale [T] fp32, weight_scale [N] bf16/fp16)
    .../impl/moe/fused_moe.py:713
        (pertoken_scale=None, bias=None, output_dtype=torch.int32)
    .../impl/moe/fused_moe.py:742
        (pertoken_scale=swiglu_out_scale, bias=None, output_dtype=bf16)
    .../ops/cv_linear.py:115
        (pertoken_scale may be [T, 1]; output_dtype = weight_scale.dtype)
    (``.../device/mxfp_compat.py`` L68/80/86 only *name* the op in feature strings.)

## Implemented vs refused

Implemented: x int8; weight int8 **[K, N] ND**; ``scale`` float (per-channel [N] or
per-tensor [1..1]) or int64 (bit-encoded, see evidence 4); ``pertoken_scale``
None / [1] / [M] / [M, 1]; ``bias`` None / float (fp16/bf16/fp32, [N] or [1]) / int32;
``output_dtype`` in {fp16, bf16, fp32, int32}; x of any rank >= 2 (leading dims are
flattened, the result is reshaped back).

Refused (``NotImplementedError`` naming the value, never silently ignored):
    * ``scale_dtype`` / ``pertoken_scale_dtype`` / ``group_sizes`` -- the MXFP8 / W4A4
      block-scaling form (sfa_v1.py:1401-1404). The e8m0 block layout is not implemented.
    * x/weight in fp8 / fp4, quantized weight (``[N, K]`` weight, NZ weight, 3-D weight).
    * integer ``bias`` that is not int32, float ``bias`` with ``output_dtype=torch.int32``.
    * ``output_dtype`` None (all 22 sites pass it, so no default is invented).

## Implementation notes

Tiling: BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, K looped at runtime. Live tiles per step:
x 32x128 fp16 = 8 KiB, w 128x64 fp16 = 16 KiB, acc 32x64 fp32 = 8 KiB (32 KiB total,
56 KiB on the exact int32 path with its four accumulators) -- far below the 192 KiB UB.
grid = (cdiv(N, BLOCK_N), m_prog); m_prog is folded down when the product would exceed
32768 programmes (the measured 65535 ceiling, see TRITON_OPERATOR_PITFALLS.md C1), and the
kernel strides over the remaining row blocks.

Two numerically different paths, chosen by ``output_dtype``:

* **float output**: one fp16 ``tl.dot`` with an fp32 accumulator (the pattern already
  deployed in ``grouped_matmul_swiglu_quant_v2.py`` L169-176). int8 -> fp16 is exact;
  the residual error is the fp32 accumulation (relative ~sqrt(K) * 2**-24). If the
  backend were to round each product to fp16 before accumulating, the relative error
  would instead be ~2**-11 (~4.9e-4) -- NOT measured here (no NPU in this session).
* **int32 output**: the exact integer split ``a = 16*aq + ar`` (``aq = a >> 4`` in
  [-8, 7], ``ar = a & 15`` in [0, 15]) gives four dots whose products are all <= 225,
  hence exactly representable in fp16 *and* whose fp32 partial sums stay below 2**24
  (<= 225 * K, i.e. 1.6e6 at K=7168), so every partial sum is an exact integer. The four
  accumulators are converted to int32 and recombined as
  ``256*qq + 16*(qr + rq) + rr``; |result| <= 127*127*K = 1.16e8 < 2**31. The int32 path
  is therefore **exact**, independent of the backend's product rounding, at 4x the dot
  cost (only the single fused_moe.py:713 site uses it).

Returns a single tensor (all 22 sites read the return value directly; none indexes [0]).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b



__all__ = ["quant_matmul", "npu_quant_matmul"]

#: Measured AI-core grid ceiling is 65535 (TRITON_OPERATOR_PITFALLS.md C1); stay below it.
_MAX_GRID = 32768

#: Tile sizes. BLOCK_N is deliberately small: the UB live set must stay far below 192 KiB
#: (see the module docstring).
_BLOCK_M = 32
_BLOCK_N = 64
_BLOCK_K = 128

_FLOAT_OUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _quant_matmul_kernel(
    X, W, SCALE, PSCALE, BIAS, OUT,
    M, N, K,
    M_PROG,
    SCALE_STRIDE: tl.constexpr,
    PSCALE_STRIDE: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    HAS_PSCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_INT: tl.constexpr,
    INT_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out[m, n] = dequant(sum_k x[m, k] * w[k, n]) for one [BLOCK_M, BLOCK_N] tile.

    X        [M, K] int8, row stride K
    W        [K, N] int8, row stride N (ND layout)
    SCALE    [N] (SCALE_STRIDE=1) or [1] (SCALE_STRIDE=0) per-output-channel dequant scale
    PSCALE   [M] (stride 1) or [1] (stride 0) per-token dequant scale
    BIAS     [N] / [1]; int32 -> added pre-dequant, float -> added post-dequant
    OUT      [M, N] in the requested output dtype
    M_PROG   the number of row programmes, used as the row-block stride
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    if not INT_OUT:
        scale = tl.load(SCALE + offs_n * SCALE_STRIDE, mask=mask_n, other=1.0).to(tl.float32)
        if HAS_BIAS:
            if BIAS_INT:
                bias_i = tl.load(BIAS + offs_n * BIAS_STRIDE, mask=mask_n, other=0).to(tl.int32)
            else:
                bias_f = tl.load(BIAS + offs_n * BIAS_STRIDE, mask=mask_n, other=0.0).to(tl.float32)
    else:
        if HAS_BIAS:
            bias_i = tl.load(BIAS + offs_n * BIAS_STRIDE, mask=mask_n, other=0).to(tl.int32)

    n_mblocks = tl.cdiv(M, BLOCK_M)
    for mb in range(pid_m, n_mblocks, M_PROG):
        offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        if INT_OUT:
            # Exact integer split a = 16 * aq + ar, see the module docstring.
            acc_qq = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_qr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_rq = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_rr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            mask_k = kk < K
            x = tl.load(
                X + offs_m[:, None] * K + kk[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0,
            )
            w = tl.load(
                W + kk[:, None] * N + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            if INT_OUT:
                xi = x.to(tl.int32)
                wi = w.to(tl.int32)
                xq = (xi >> 4).to(tl.float16)
                xr = (xi & 15).to(tl.float16)
                wq = (wi >> 4).to(tl.float16)
                wr = (wi & 15).to(tl.float16)
                acc_qq += tl.dot(xq, wq, out_dtype=tl.float32)
                acc_qr += tl.dot(xq, wr, out_dtype=tl.float32)
                acc_rq += tl.dot(xr, wq, out_dtype=tl.float32)
                acc_rr += tl.dot(xr, wr, out_dtype=tl.float32)
            else:
                acc += tl.dot(x.to(tl.float16), w.to(tl.float16), out_dtype=tl.float32)

        if INT_OUT:
            res_i = (
                acc_qq.to(tl.int32) * 256
                + (acc_qr.to(tl.int32) + acc_rq.to(tl.int32)) * 16
                + acc_rr.to(tl.int32)
            )
            if HAS_BIAS:
                res_i = res_i + bias_i[None, :]
            tl.store(
                OUT + offs_m[:, None] * N + offs_n[None, :],
                res_i,
                mask=mask_m[:, None] & mask_n[None, :],
            )
        else:
            if HAS_BIAS:
                if BIAS_INT:
                    # int32 bias is a quantized bias: it is added before the dequantization.
                    acc = acc + bias_i[None, :].to(tl.float32)
            res = acc * scale[None, :]
            if HAS_PSCALE:
                # PSCALE_STRIDE == 0 selects the per-tensor form: every in-range row reads
                # the same address (the out-of-range rows are masked in the store anyway).
                ps = tl.load(
                    PSCALE + offs_m * PSCALE_STRIDE, mask=mask_m, other=0.0
                ).to(tl.float32)
                res = res * ps[:, None]
            if HAS_BIAS:
                if not BIAS_INT:
                    res = res + bias_f[None, :]
            tl.store(
                OUT + offs_m[:, None] * N + offs_n[None, :],
                res.to(OUT.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )


def _as_dequant_scale(scale: torch.Tensor, n: int, name: str) -> tuple[torch.Tensor, int]:
    """Flatten a [N] / [1] / scalar scale and return it with its element stride.

    int64 scales are the fp16 ModelSlim encoding (fp32 bits in the low 32 bits), decoded
    exactly like ``_repos/vllm-ascend/.../w8a8/w8a8_static.py`` L144-146.
    """
    s = scale.contiguous()
    if s.dtype is torch.int64:
        s = s.to(torch.int32).contiguous().view(torch.float32)
    elif s.dtype not in _FLOAT_OUT_DTYPES:
        raise NotImplementedError(
            f"{name} dtype {s.dtype} is not supported: expected a float tensor "
            "(fp16/bf16/fp32) or the int64 bit-encoded form"
        )
    s = s.reshape(-1)
    if s.numel() == 1:
        return s, 0
    if s.numel() != n:
        raise ValueError(f"{name} has {s.numel()} elements, expected 1 or N={n}")
    return s, 1


def quant_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    pertoken_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    scale_dtype=None,
    pertoken_scale_dtype=None,
    group_sizes=None,
):
    """Drop-in equivalent of ``torch_npu.npu_quant_matmul`` for the A8W8 (int8 x int8) form.

    Args:
        x: int8 ``[..., K]``; every leading dimension is flattened to a row.
        weight: int8 ``[K, N]`` in **ND** layout (the layout every call site produces by
            transposing the loaded ``[N, K]`` weight; see ``impl/quantization/linear.py``
            L257). A `[N, K]` or NZ weight is refused rather than mis-read.
        scale: per-output-channel dequantization scale, ``[N]`` or ``[1]``; float or the
            int64 bit-encoded form.
        pertoken_scale: optional per-token dequantization scale ``[M]`` / ``[M, 1]`` /
            ``[1]``, fp32 in every call site.
        bias: optional ``[N]`` / ``[1]``. int32 -> quantized bias, added to the integer
            accumulator before the dequantization; float -> added after it.
        output_dtype: required; fp16 / bf16 / fp32 / int32. ``int32`` returns the raw
            integer accumulator (``scale`` / ``pertoken_scale`` are not applied).
        scale_dtype, pertoken_scale_dtype, group_sizes: MXFP8 / W4A4 block-scaling
            arguments. Any non-None value raises NotImplementedError.

    Returns:
        ``[..., N]`` in ``output_dtype``.
    """
    if scale_dtype is not None or pertoken_scale_dtype is not None or group_sizes is not None:
        raise NotImplementedError(
            "the MXFP8/W4A4 block-scaling form is not implemented: scale_dtype="
            f"{scale_dtype!r}, pertoken_scale_dtype={pertoken_scale_dtype!r}, "
            f"group_sizes={group_sizes!r} (used only by sfa_v1.py:1401-1404, for an fp8 "
            "activation plus an e8m0 block-scale weight; neither the e8m0 scale layout nor "
            "the [1,1,32] group semantics are documented in this tree)"
        )
    if output_dtype is None:
        raise ValueError(
            "output_dtype is required: all 22 npu_quant_matmul call sites pass it, so no "
            "default is assumed here"
        )
    if output_dtype not in _FLOAT_OUT_DTYPES and output_dtype is not torch.int32:
        raise NotImplementedError(
            f"output_dtype={output_dtype} is not implemented (supported: fp16, bf16, fp32, "
            "int32)"
        )
    if x.dim() < 2:
        raise ValueError(f"x must be at least 2-D [..., K], got {tuple(x.shape)}")
    if x.dtype is not torch.int8:
        raise NotImplementedError(
            f"x dtype {x.dtype} is not implemented: only the int8 activation form is "
            "covered (the fp8 form of sfa_v1.py:1397-1404 is refused above)"
        )
    if weight.dim() != 2:
        raise NotImplementedError(
            f"weight must be 2-D [K, N] in ND layout, got {tuple(weight.shape)}; the 3-D "
            "per-expert and the NZ (fractal) layouts are not implemented"
        )
    if weight.dtype is not torch.int8:
        raise NotImplementedError(f"weight dtype {weight.dtype} is not implemented (int8 only)")

    k = int(x.shape[-1])
    n = int(weight.shape[1])
    if int(weight.shape[0]) != k:
        raise ValueError(
            f"weight.shape[0]={int(weight.shape[0])} must equal K={k}: pass the [K, N] "
            "weight (transpose the loaded [N, K] one); a transposed or NZ weight is refused "
            "instead of being silently mis-read"
        )
    if x.device != weight.device:
        raise ValueError(f"x is on {x.device} but weight is on {weight.device}")

    x2 = x.contiguous().reshape(-1, k)
    m = int(x2.shape[0])
    out_shape = (*x.shape[:-1], n)
    if m == 0 or n == 0:
        return torch.empty(out_shape, dtype=output_dtype, device=x.device)

    w = weight.contiguous()
    s, s_stride = _as_dequant_scale(scale, n, "scale")

    ps = None
    ps_stride = 0
    if pertoken_scale is not None:
        ps = pertoken_scale.contiguous().reshape(-1)
        if ps.dtype not in _FLOAT_OUT_DTYPES:
            raise NotImplementedError(
                f"pertoken_scale dtype {ps.dtype} is not implemented (fp16/bf16/fp32 only)"
            )
        if ps.numel() == 1:
            ps_stride = 0
        elif ps.numel() == m:
            ps_stride = 1
        else:
            raise ValueError(f"pertoken_scale has {ps.numel()} elements, expected 1 or M={m}")

    b = None
    b_stride = 0
    b_int = False
    if bias is not None:
        b = bias.contiguous().reshape(-1)
        if b.numel() == 1:
            b_stride = 0
        elif b.numel() == n:
            b_stride = 1
        else:
            raise ValueError(f"bias has {b.numel()} elements, expected 1 or N={n}")
        if b.dtype in (torch.int32, torch.int64):
            if b.dtype is not torch.int32:
                raise NotImplementedError(
                    f"bias dtype {b.dtype} is not implemented: an integer bias must be int32 "
                    "(the quantized-bias convention is defined for int32)"
                )
            b_int = True
        elif b.dtype in _FLOAT_OUT_DTYPES:
            if output_dtype is torch.int32:
                raise ValueError(
                    "output_dtype=torch.int32 returns the raw integer accumulator, so a "
                    f"{b.dtype} bias cannot be applied; pass an int32 bias or None"
                )
        else:
            raise NotImplementedError(f"bias dtype {b.dtype} is not implemented")

    out = torch.empty((m, n), dtype=output_dtype, device=x.device)

    n_blocks = _ceil_div(n, _BLOCK_N)
    m_blocks = _ceil_div(m, _BLOCK_M)
    if n_blocks > _MAX_GRID:
        raise NotImplementedError(
            f"N={n} needs {n_blocks} column programmes > {_MAX_GRID}; split the output "
            "dimension at the call site (the same workaround the DSA-CP path uses)"
        )
    m_prog = min(m_blocks, max(1, _MAX_GRID // n_blocks))

    _quant_matmul_kernel[(n_blocks, m_prog)](
        x2, w, s,
        ps if ps is not None else x2,
        b if b is not None else s,
        out,
        m, n, k,
        m_prog,
        SCALE_STRIDE=s_stride,
        PSCALE_STRIDE=ps_stride,
        BIAS_STRIDE=b_stride,
        HAS_PSCALE=ps is not None,
        HAS_BIAS=b is not None,
        BIAS_INT=b_int,
        INT_OUT=output_dtype is torch.int32,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
    )
    return out.reshape(out_shape)


#: ``torch_npu`` spelling of the same operator.
npu_quant_matmul = quant_matmul
