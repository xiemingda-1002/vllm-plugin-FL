# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""transpose_quant_batchmatmul -- AscendC to Triton.

Source: this operator is the CANN / torch_npu built-in
``aclnnTransposeQuantBatchMatMul`` behind
``torch_npu.npu_transpose_quant_batchmatmul``. **There is no AscendC kernel body for it
in this tree**:

    grep -rn "transpose_quant_batchmatmul\\|TransposeQuantBatchMatmul" csrc -> no match

The semantics below are pinned from the call sites plus the two producers of the scale
tensors (``npu_dynamic_mx_quant`` and the A5 MXFP8 weight loader), and the scale layout
was measured locally by replaying that loader -- see "Scale layout" below. Every open
point is listed under "Unconfirmed" at the end.

------------------------------------------------------------------------------
Call sites -- 2 occurrences of the name in the vllm_fl tree
------------------------------------------------------------------------------
    grep -rn "npu_transpose_quant_batchmatmul" vllm_fl --include=*.py

  [1] attention/ascend/dsa_v1.py L1741                ``_forward_o_proj``, A5 branch
  [2] attention/ascend/context_parallel/dsa_cp.py L1249  PCP copy of [1]

  Both are textually identical:

      o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
      o = torch_npu.npu_transpose_quant_batchmatmul(
          o,                                  # x1, fp8_e4m3fn, [T, G, D]
          self.wo_a.weight,                   # x2, fp8_e4m3fn, [G, D, R]
          dtype=torch.bfloat16,
          bias=None,
          group_sizes=(0, 0, 32),
          x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu),
          x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
          perm_x1=(1, 0, 2),
          perm_x2=(0, 1, 2),
          perm_y=(1, 0, 2),
      )
      o = o.reshape(num_tokens, -1)           # [T, G*R]

  => the entry signature below has to accept exactly those keywords, and the two
     leading positional parameters (x1, x2).

------------------------------------------------------------------------------
Semantics
------------------------------------------------------------------------------
    a[b, m, k] = e4m3(x1)[b, m, k] * 2 ** (s1[b, m, k // 32] - 127)     # dequantized x1
    c[b, k, n] = e4m3(x2)[b, k, n] * 2 ** (s2[b, k // 32, n] - 127)     # dequantized x2
    y = bmm(a.permute(perm_x1), c.permute(perm_x2)).to(dtype).permute(perm_y)

with all blocking along the contraction axis K. The permutations follow
``npu_transpose_batchmatmul`` exactly (``y = bmm(x1.permute(perm_x1),
x2.permute(perm_x2)).permute(perm_y)``); the only difference is the operand dtype and
the per-32-block E8M0 dequantization. Evidence for the permutation part: the upstream
vllm-ascend unit tests substitute the *same* torch reference for this operator as for the
unquantized one (_repos/vllm-ascend/tests/ut/attention/test_dsa_v1.py L2725-2728, used
for the ``npu_transpose_quant_batchmatmul`` mock at L2740-2742):

    def matmul(x, weight, **kwargs):
        return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)   # [T,G,D]x[G,D,R]

------------------------------------------------------------------------------
Scale layout (derived from the producers, then measured locally)
------------------------------------------------------------------------------
Both scales arrive as a raw ``.view(torch.float8_e8m0fnu)`` of what the producer built,
so their shape is the producer's layout:

  x1_scale -- produced by ``npu_dynamic_mx_quant`` on x1 = [T, G, D]. That operator's
    port (``impl/dynamic_mx_quant.py`` L56-68, L316-321) documents and implements the
    CANN "pairs" layout: one E8M0 byte per 32-element block along D, packed two per
    entry, i.e. shape [T, G, ceil(ceil(D/32)/2), 2].

  x2_scale -- produced by the A5 MXFP8 quant method's
    ``process_weights_after_loading`` for ``wo_a``
    (_repos/vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py L611-635):
    the checkpoint scale [out, in/128] is expanded to [out, in/32], packed two per entry
    to [out, in/64, 2], transposed, and then reshaped for wo_a, ending at

        weight_scale [G, in/64, R, 2]      (in = D = hidden_size, R = o_lora_rank)
        weight       [G, D, R]             (wo_a layout, ops/linear.py L474-476)

    The axis meaning of that tensor is not written down anywhere, so it was **measured**
    by replaying exactly those ops on a position-labelled tensor with torch 2.13.0+cpu
    (no NPU needed; the command and its output are in RESULT.md): every element of the
    trailing pair belongs to the same output column n, and the pair walks the two
    adjacent 32-element blocks of one 64-element group:

        x2_scale[g, kb, r, q]  ==  scale of x2[g, 64*kb + 32*q : 64*kb + 32*q + 32, r]

    (all 30 720 entries of a 8 x 20 x 192 x 2 probe satisfy it; the two competing
    readings -- pair = adjacent N columns, or pair spans two halves of K -- are excluded
    by the same probe.)

  => both operands use one convention, which is why a single ``group_sizes=(0, 0, 32)``
     covers them: the third entry is the 32-element quantization group along K, and the
     two leading zeros carry no information for this MX path.

  This file accepts the same numbers in either of two forms:

    * "pairs" -- [..., ceil(K/64), 2], the layout both call sites pass;
    * "flat"  -- [..., ceil(K/32)], the pair axis merged (requires an even block count,
                 i.e. K % 64 == 0).

  Both are permuted with the operand's own permutation plus one trailing axis
  (``perm_x1 + (3,)`` / ``perm_x2 + (3,)``), which is exact by construction: scale axis i
  belongs to operand axis i.

------------------------------------------------------------------------------
Production shapes
------------------------------------------------------------------------------
  B = n_local_groups (``o_groups``), M = num_tokens (OTP branch:
  ``oproj_tp_size * exchange_num_tokens``), K = group_hidden_dim =
  n_local_heads * head_dim / n_local_groups, N = o_lora_rank. K must be a multiple of
  64 for the pairs layout to cover whole groups; nothing in the operator requires it,
  and a tail block is masked (only its first 32-block exists, which the pair addressing
  reads naturally).

------------------------------------------------------------------------------
Parameter coverage
------------------------------------------------------------------------------
  dtype        : torch.bfloat16 (both call sites) / torch.float16 / torch.float32.
  bias         : only ``None`` (both call sites); a non-None bias raises, its layout for
                 a transposed batch matmul is not pinned by any source in this tree.
  group_sizes  : ``None`` or ``(0, 0, 32)``. The third entry must equal the MX block
                 size 32 (see "Scale layout"); anything else raises instead of being
                 reinterpreted, because no source here defines the other slots.
  x1 / x2      : torch.float8_e4m3fn. e5m2 raises (no call site; its scale exponent
                 offset L differs, see dynamic_mx_quant.py L43-47).
  x1_scale /
  x2_scale     : 1-byte dtypes (torch.uint8, torch.float8_e8m0fnu) -- the bytes are what
                 the kernel reads; float32 scales raise.
  perm_x1 /
  perm_x2      : permutation of (0, 1, 2); ``None`` = identity.
  perm_y       : permutation of (0, 1, 2) whose last entry is 2.

------------------------------------------------------------------------------
Implementation notes
------------------------------------------------------------------------------
* The fp8 payload is read as **uint8** and decoded with integer bit arithmetic
  (``_e4m3_to_f32``): e4m3 has 4 significant bits, bias 7, subnormals m * 2**-9, and
  0x7F / 0xFF are the only NaN codes. This keeps the kernel independent of whether this
  backend can load / cast an fp8 dtype (support for that is uneven in this tree --
  ``impl/dynamic_mx_quant.py`` L404-410 documents an fp8 allocation quirk), and the
  decode rule was checked against torch's own fp8 -> fp32 cast for all 256 byte values
  on CPU (RESULT.md).
* The E8M0 byte is decoded as 2 ** (b - 127) through its exact fp32 bit pattern
  ``(b << 23)`` -- exact for b in [1, 254], which is the reachable range (the MX
  encoder cannot emit more than 246 for a finite fp16/bf16 input,
  ``impl/dynamic_mx_quant.py`` L91-95). Byte 0 occurs only for an all-zero block, where
  0.0 and 2**-127 give the same contribution; byte 255 (the inf/NaN marker) decodes to
  +inf rather than NaN, which makes the row non-finite either way.
* The scales are folded into the operands before the dot. That is exact, not an
  approximation: within one 32-element block both scales are constant, so
  ``sum_k (x1*s1)*(x2*s2) == s1*s2*sum_k x1*x2``, and ``e4m3 * 2**n`` keeps at most 4
  significant bits, i.e. it is exactly representable in fp16. fp16 operands follow the
  verified precedent of ``impl/grouped_matmul_swiglu_quant_v2.py`` (int8 -> fp16 dot,
  fp32 accumulator). Consequence to know: a *dequantized* value with |v| > 65504 would
  overflow the fp16 operand; the dequantized values are the same numbers the bf16 model
  already holds, so this is a range limit of the same kind that file accepts.
* ``grid`` layout, row-tile folding under the 65535-program cap, masked loads/stores,
  no materialised permutation (only strides are passed) and fp32 accumulation with a
  single cast at the store: same design as ``impl/transpose_batchmatmul.py``.

------------------------------------------------------------------------------
Unconfirmed (no CANN header, no NPU, no triton in this container)
------------------------------------------------------------------------------
* The scale layout above is derived from the two producers and measured locally; it was
  not read from a CANN header and not compared against the native operator.
* ``group_sizes``' first two slots -- unused by both call sites, so only the third is
  validated.
* NaN payloads: 0x7F / 0xFF decode to NaN through a ``float("nan")`` constant; whether
  the backend accepts that constant is unverified (no triton here).
* Whether the native rounds the accumulator to ``dtype`` before or after ``perm_y`` --
  irrelevant for the result (a permutation of a rounded tensor).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b



__all__ = ["transpose_quant_batchmatmul", "npu_transpose_quant_batchmatmul"]

#: MX block size: one E8M0 scale per 32 elements along K (PER_MX_FP16, see
#: impl/dynamic_mx_quant.py L126-127).
_MX_BLOCK = 32
#: Two 32-element blocks share one "pairs" entry of the native scale layout.
_MX_PAIR = 2 * _MX_BLOCK

_BLOCK_M = 64
_BLOCK_K = 64
_BLOCK_N_WIDE = 128
_BLOCK_N = 64

#: AI Core block_dim limit of this backend (see impl/grouped_matmul.py L132-139).
_MAX_PROGRAMS = 65535

_IDENTITY = (0, 1, 2)

_FP8_E4M3 = torch.float8_e4m3fn


def _scale_dtypes() -> tuple:
    """1-byte dtypes accepted for an E8M0 scale (the kernel only reads the bytes)."""
    out = [torch.uint8]
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        out.append(e8m0)
    return tuple(out)


@triton.jit
def _e4m3_to_f32(bits):
    """Decode one float8_e4m3fn code (as an int32) to float32.

    exp field 0 -> subnormal ``m * 2**-9``; exp field e >= 1 ->
    ``(8 + m) * 2**(e - 10)`` (bias 7, 3 mantissa bits); the sign is applied by
    negation, which is exact including -0.0; ``0x7F`` / ``0xFF`` are the NaN codes.
    """
    e = (bits >> 3) & 0xF
    m = bits & 0x7
    man = tl.where(e == 0, m, 8 + m)
    exp = tl.where(e == 0, -9, e - 10)
    v = man.to(tl.float32) * ((exp + 127) << 23).to(tl.float32, bitcast=True)
    v = tl.where(((bits >> 7) & 1) != 0, -v, v)
    return tl.where((bits & 0x7F) == 0x7F, float("nan"), v)


@triton.jit
def _e8m0_to_f32(bits):
    """Decode one E8M0 scale byte (as an int32) to its exact power of two.

    ``2 ** (b - 127)`` is built as the fp32 bit pattern ``b << 23``: bit-exact for
    b in [1, 254]; b == 0 gives 0.0 (only reachable for an all-zero block) and b == 255
    gives +inf (the inf/NaN marker of the MX encoder).
    """
    return (bits << 23).to(tl.float32, bitcast=True)


@triton.jit
def _tqmm_kernel(
    A, B, S1, S2, C,
    M, N, K,
    num_mt,
    stride_ab, stride_am,
    stride_bb, stride_bk,
    stride_s1b, stride_s1m, stride_s1k,
    stride_s2b, stride_s2k, stride_s2n,
    stride_cb, stride_cm,
    grid_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[b, m, n] = sum_k dq(A)[b, m, k] * dq(B)[b, k, n].

    ``A``  [B, M, K] uint8 view of the fp8 operand, strides (stride_ab, stride_am, 1).
    ``B``  [B, K, N] uint8 view of the fp8 operand, strides (stride_bb, stride_bk, 1).
    ``S1`` [B, M, ceil(K/64), 2] E8M0 bytes, strides (stride_s1b, stride_s1m,
           stride_s1k, 1).
    ``S2`` [B, ceil(K/64), N, 2] E8M0 bytes, strides (stride_s2b, stride_s2k,
           stride_s2n, 1).
    ``C``  [B, M, N] in ``dtype``, strides (stride_cb, stride_cm, 1).

    The scale of the element at contraction index k is ``S*[..., k // 64, (k % 64) //
    32]``, i.e. the pairs layout documented in the module docstring.
    """
    batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    mt = pid_m
    while mt < num_mt:
        offs_m = mt * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            mask_k = kk < K
            kb = kk // _MX_PAIR
            half = (kk % _MX_PAIR) // _MX_BLOCK
            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_k[:, None] & mask_n[None, :]
            a_bits = tl.load(
                A + batch * stride_ab + offs_m[:, None] * stride_am + kk[None, :],
                mask=mask_a, other=0,
            )
            b_bits = tl.load(
                B + batch * stride_bb + kk[:, None] * stride_bk + offs_n[None, :],
                mask=mask_b, other=0,
            )
            s1_bits = tl.load(
                S1 + batch * stride_s1b + offs_m[:, None] * stride_s1m
                + kb[None, :] * stride_s1k + half[None, :],
                mask=mask_a, other=0,
            )
            s2_bits = tl.load(
                S2 + batch * stride_s2b + kb[:, None] * stride_s2k
                + offs_n[None, :] * stride_s2n + half[:, None],
                mask=mask_b, other=0,
            )
            a = (_e4m3_to_f32(a_bits.to(tl.int32)) * _e8m0_to_f32(s1_bits.to(tl.int32)))
            b = (_e4m3_to_f32(b_bits.to(tl.int32)) * _e8m0_to_f32(s2_bits.to(tl.int32)))
            acc += tl.dot(a.to(tl.float16), b.to(tl.float16), out_dtype=tl.float32)
        tl.store(
            C + batch * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :],
            acc.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )
        mt += grid_m


def _norm_perm(perm, name: str) -> tuple[int, int, int]:
    """A 3-element permutation of (0, 1, 2); ``None`` means the identity."""
    if perm is None:
        return _IDENTITY
    try:
        p = tuple(int(v) for v in perm)
    except (TypeError, ValueError):
        raise ValueError(f"{name}={perm!r} is not a permutation of (0, 1, 2)") from None
    if len(p) != 3 or sorted(p) != [0, 1, 2]:
        raise ValueError(f"{name}={p} is not a permutation of (0, 1, 2)")
    return p


def _inverse_perm(p: tuple[int, int, int]) -> tuple[int, int, int]:
    q = [0, 0, 0]
    for j, v in enumerate(p):
        q[v] = j
    return (q[0], q[1], q[2])


def _scale_pairs(scale: torch.Tensor, x_shape, block_axis: int, name: str) -> torch.Tensor:
    """Return the scale as ``[..., ceil(K/64), 2]`` E8M0 bytes (the "pairs" layout).

    Accepts the native pairs form or the same numbers with the pair axis merged
    ("flat", shape ``[..., ceil(K/32)]``).
    """
    if scale.dim() == len(x_shape) + 1:
        expect = list(x_shape)
        expect[block_axis] = _ceil_div(int(x_shape[block_axis]), _MX_PAIR)
        expect.append(2)
        if tuple(scale.shape) != tuple(expect):
            raise ValueError(
                f"{name} must have the pairs layout {tuple(expect)} "
                f"(operand shape {tuple(x_shape)}, K axis {block_axis}), "
                f"got {tuple(scale.shape)}"
            )
        return scale
    if scale.dim() == len(x_shape):
        blocks = _ceil_div(int(x_shape[block_axis]), _MX_BLOCK)
        expect = list(x_shape)
        expect[block_axis] = blocks
        if tuple(scale.shape) != tuple(expect):
            raise ValueError(
                f"{name} must have the flat layout {tuple(expect)} "
                f"(operand shape {tuple(x_shape)}, K axis {block_axis}), "
                f"got {tuple(scale.shape)}"
            )
        if blocks % 2 != 0:
            raise ValueError(
                f"{name}: the flat layout needs an even number of 32-element blocks "
                f"(K % 64 == 0), got {blocks} block(s); pass the pairs layout instead"
            )
        # split the block axis into (pair-group, pair) and move the pair axis to the end,
        # which is where the native pairs layout keeps it
        split = scale.reshape(
            *scale.shape[:block_axis],
            blocks // 2,
            2,
            *scale.shape[block_axis + 1:],
        )
        order = list(range(split.dim()))
        order.append(order.pop(block_axis + 1))
        return split.permute(order).contiguous()
    raise ValueError(
        f"{name} must have rank {len(x_shape)} (flat) or {len(x_shape) + 1} (pairs), "
        f"got {scale.dim()} for operand shape {tuple(x_shape)}"
    )


def transpose_quant_batchmatmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    dtype: torch.dtype | None = None,
    bias: torch.Tensor | None = None,
    group_sizes=(0, 0, _MX_BLOCK),
    x1_scale: torch.Tensor | None = None,
    x2_scale: torch.Tensor | None = None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
) -> torch.Tensor:
    """Equivalent of ``torch_npu.npu_transpose_quant_batchmatmul`` (MXFP8/A8W8).

    ``y = bmm(dq(x1).permute(perm_x1), dq(x2).permute(perm_x2)).to(dtype).permute(perm_y)``
    with one E8M0 scale per 32-element block along K -- see the module docstring for the
    derivation and the exactness argument.

    Args:
        x1: [B, M, K] ``torch.float8_e4m3fn``, up to the permutations of ``perm_x1``.
        x2: [B, K, N] ``torch.float8_e4m3fn``, up to the permutations of ``perm_x2``.
        dtype: output dtype (``torch.bfloat16`` at both call sites); ``None`` = bf16.
        bias: must be ``None``.
        group_sizes: ``(0, 0, 32)`` (the MX block size in the K slot) or ``None``.
        x1_scale: E8M0 bytes for ``x1``, axes (x1 axes..., K blocks, pair).
        x2_scale: E8M0 bytes for ``x2``, axes (x2 axes..., K blocks, pair).
        perm_x1: permutation of (0, 1, 2) for ``x1``; ``None`` = identity.
        perm_x2: permutation of (0, 1, 2) for ``x2``; ``None`` = identity.
        perm_y: permutation of (0, 1, 2) for the result; its last entry must be 2.

    Returns:
        A **new contiguous** tensor of shape
        ``tuple((B, M, N)[i] for i in perm_y)`` and dtype ``dtype``.
    """
    if bias is not None:
        raise NotImplementedError(
            "bias is not None: both call sites pass bias=None, and the bias layout for a "
            "transposed batch matmul ([N] vs [B, M, N]) is not pinned by any source in "
            "this tree, so it is not implemented rather than silently dropped."
        )
    if group_sizes is not None:
        gs = tuple(int(v) for v in group_sizes)
        if gs != (0, 0, _MX_BLOCK):
            raise NotImplementedError(
                f"group_sizes={gs}: both call sites pass (0, 0, 32), where 32 is the MX "
                "block size along K (see the scale layout in the module docstring); the "
                "meaning of the other slots is not defined by any source in this tree."
            )
    if dtype is None:
        out_dtype = torch.bfloat16
    elif isinstance(dtype, torch.dtype):
        out_dtype = dtype
    else:
        raise NotImplementedError(
            f"dtype={dtype!r} (a ScalarType code) is not supported; pass a torch.dtype"
        )
    if not out_dtype.is_floating_point:
        raise NotImplementedError(f"dtype={out_dtype} is not a floating point dtype")

    if x1.dim() != 3:
        raise ValueError(f"x1 must be 3-D [B, M, K], got {tuple(x1.shape)}")
    if x2.dim() != 3:
        raise ValueError(f"x2 must be 3-D [B, K, N], got {tuple(x2.shape)}")
    if x1.dtype != _FP8_E4M3 or x2.dtype != _FP8_E4M3:
        raise NotImplementedError(
            f"x1/x2 must be {_FP8_E4M3}, got {x1.dtype} / {x2.dtype}: no call site uses "
            "another quantized dtype, and the e5m2 scale offset differs "
            "(impl/dynamic_mx_quant.py L43-47)."
        )
    if x1.device != x2.device:
        raise ValueError("x1 and x2 must live on the same device")
    if x1_scale is None or x2_scale is None:
        raise ValueError(
            "x1_scale and x2_scale are required: the operands are fp8 and carry no scale "
            "of their own; both call sites pass them."
        )

    px1 = _norm_perm(perm_x1, "perm_x1")
    px2 = _norm_perm(perm_x2, "perm_x2")
    py = _norm_perm(perm_y, "perm_y")
    if py[2] != 2:
        raise NotImplementedError(
            f"perm_y={py}: this implementation keeps the N axis innermost (it writes the "
            "result through the inverse permutation's strides); the call sites only use "
            "(1, 0, 2)."
        )

    a = x1.permute(px1)
    b = x2.permute(px2)
    batch, m, k = (int(s) for s in a.shape)
    batch_b, k_b, n = (int(s) for s in b.shape)
    if batch != batch_b:
        raise ValueError(
            f"the batch axis disagrees after perm_x1/perm_x2: {batch} vs {batch_b}"
        )
    if k != k_b:
        raise ValueError(
            f"the contraction axis disagrees after perm_x1/perm_x2: {k} vs {k_b}"
        )

    # Scale axes follow the operand axes (scale axis i <-> operand axis i), plus the
    # trailing pair axis, so the scale is permuted by the operand's permutation + (3,).
    scale_dtypes = _scale_dtypes()
    for name, sc in (("x1_scale", x1_scale), ("x2_scale", x2_scale)):
        if sc.dtype not in scale_dtypes:
            raise NotImplementedError(
                f"{name} has dtype {sc.dtype}; only 1-byte E8M0 containers are supported "
                f"({', '.join(str(d) for d in scale_dtypes)}). A float32 scale would have "
                "to be converted, and no call site passes one."
            )
        if sc.device != x1.device:
            raise ValueError(f"{name} must live on the same device as the operands")

    s1 = _scale_pairs(x1_scale.view(torch.uint8), tuple(x1.shape), px1[2], "x1_scale")
    s2 = _scale_pairs(x2_scale.view(torch.uint8), tuple(x2.shape), px2[1], "x2_scale")
    s1 = s1.permute(px1 + (3,))
    s2 = s2.permute(px2 + (3,))
    expect_kb = _ceil_div(k, _MX_PAIR)
    if tuple(s1.shape) != (batch, m, expect_kb, 2):
        raise ValueError(
            f"x1_scale must be [B, M, ceil(K/64), 2] = {(batch, m, expect_kb, 2)} after "
            f"perm_x1, got {tuple(s1.shape)}"
        )
    if tuple(s2.shape) != (batch, expect_kb, n, 2):
        raise ValueError(
            f"x2_scale must be [B, ceil(K/64), N, 2] = {(batch, expect_kb, n, 2)} after "
            f"perm_x2, got {tuple(s2.shape)}"
        )
    if s1.stride(3) != 1:
        s1 = s1.contiguous()
    if s2.stride(3) != 1:
        s2 = s2.contiguous()

    # Permutations are views; only an exotic (non-unit innermost) layout forces a copy.
    if a.stride(2) != 1:
        a = a.contiguous()
    if b.stride(2) != 1:
        b = b.contiguous()

    out_shape = (batch, m, n)
    y = torch.empty(tuple(out_shape[i] for i in py), dtype=out_dtype, device=x1.device)
    # y == out.permute(py), so out == y.permute(inverse(py)): writing through those
    # strides fills y contiguously instead of filling out and permuting afterwards.
    c = y.permute(_inverse_perm(py))
    if c.stride(2) != 1:
        raise NotImplementedError(
            f"perm_y={py}: the result's N axis is not its innermost axis; a strided "
            "store is not implemented."
        )
    if batch == 0 or m == 0 or n == 0:
        return y
    if k == 0:
        y.zero_()
        return y

    block_n = _BLOCK_N_WIDE if n >= _BLOCK_N_WIDE else _BLOCK_N
    num_mt = _ceil_div(m, _BLOCK_M)
    num_nt = _ceil_div(n, block_n)
    if batch * num_nt > _MAX_PROGRAMS:
        raise NotImplementedError(
            f"batch({batch}) * column tiles({num_nt}) = {batch * num_nt} exceeds the "
            f"backend's {_MAX_PROGRAMS}-program cap; folding the row axis cannot help."
        )
    grid_m = max(1, min(num_mt, _MAX_PROGRAMS // (batch * num_nt)))
    _tqmm_kernel[(batch, grid_m, num_nt)](
        a.view(torch.uint8), b.view(torch.uint8), s1, s2, y,
        m, n, k,
        num_mt,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        s1.stride(0), s1.stride(1), s1.stride(2),
        s2.stride(0), s2.stride(1), s2.stride(2),
        c.stride(0), c.stride(1),
        grid_m,
        BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_K=_BLOCK_K,
    )
    return y


#: ``torch_npu`` spelling of the same operator.
npu_transpose_quant_batchmatmul = transpose_quant_batchmatmul
