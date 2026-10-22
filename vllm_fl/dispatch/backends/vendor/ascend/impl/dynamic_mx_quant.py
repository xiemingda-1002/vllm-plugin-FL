# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""dynamic_mx_quant -- OCP MX (microscaling) quantisation, AscendC -> Triton

Operator: ``torch_npu.npu_dynamic_mx_quant`` (CANN ``DynamicMxQuant``). **There is no
AscendC source for this operator in csrc/** -- it ships inside CANN, so the formula below is
assembled from the sources that *are* in this tree plus the Python layer that calls it, and
every open point is listed under "Unconfirmed" at the end of this docstring.

Sources read for the formula:
  1. The sibling port ``impl/swiglu_group_quant.py`` documents the MX part of
     ``csrc/moe/swiglu_group_quant/op_kernel/swiglu_group_quant_base.h`` (that operator's C++
     is not vendored either, so the line numbers are the ones its port cites):
        base.h L30   PER_MX_FP16 = 32            -> one shared exponent per 32 elements
        base.h L359-422 VFComputeMaxExp          -> e = max bf16 exponent field of |s|
        base.h L425-490 VFComputeScale           -> byte = (e - LOWER) >> 7,
                                                    LOWER = L << 7, L = 8 (e4m3) / 15 (e5m2),
                                                    byte = 0xFF for an inf/NaN block
        base.h L493-590 VFComputeData            -> q = s * 2 ** (L - exponent), then
                                                    SatMode::SAT + RoundMode::CAST_RINT
     i.e.  scale = 2 ** (floor(log2(amax)) - L),  byte = 127 + exponent,  y = rint(s / scale)
  2. ``csrc/ascend/attention/common/op_kernel/arch35/vf/vf_basic_block_utils.h`` L28-29:
     ``fp8e4m3MaxValue = 448.0f``, ``int8MaxValue = 127.0f`` (the L above is 448's exponent,
     i.e. L = emax_elem of the destination format: 8 for e4m3, 15 for e5m2).
  3. ``_repos/vllm-ascend/vllm_ascend/quantization/utils.py`` L47-54 documents the
     ``scale_alg`` attribute of DynamicMxQuantV3:
        scale_alg=0  "derives a shared exponent directly from the block's maximum absolute
                      value"                                         -> the OCP MX rule above
        scale_alg=1  "first maps that maximum to the destination FP8 range, then rounds the
                      E8M0 scale exponent up to avoid quantization overflow"
                      -> scale = 2 ** ceil(log2(amax / 448))
  4. ``_repos/vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py`` L103-107 and
     L117-119 (``scale_dtype=torch_npu.float8_e8m0fnu``) plus
     ``.../w8a8/fp8_block.py`` L107-120, L231-235:
        "CANN 9.1 npu_dynamic_mx_quant packs two E8M0 scales per entry on a 2D weight, so the
         native result is [out, in // 64, 2]"
     -> the scale tensor has one trailing pair axis: [..., ceil(ceil(D/32) / 2), 2] bytes.

Formula (per 32-element block b along the last dimension; BLOCK = 32):
    amax        = max_{i<32} |x[b*32 + i]|
    e           = floor(log2(amax))                       (0 for amax == 0)
    byte        = clamp(e - L + 127, 0, 254)              L = 8 (e4m3) / 15 (e5m2)
    byte       += 1 when scale_alg == 1 and mantissa(amax) > 1.75
    scale       = 2 ** (byte - 127)
    y[b*32 + i] = sat_rint_fp8(x[b*32 + i] / scale)        round half to even, saturate
    an inf/NaN block gives byte = 0xFF and 32 fp8 NaN codes (base.h L474-475, L40)

The exponent is extracted with integer bit arithmetic on the fp32 bit pattern
(``(bits >> 23) & 0xFF``), not with ``log2``, because ``log2`` is not exact at powers of two
and a block maximum sitting exactly on a power of two would move the whole block's scale by
a factor of two. For scale_alg=1 the "+1" rule is exact rather than a second rounding:
``ceil(log2(amax / 448)) = floor(log2(amax)) - 8 + [mantissa(amax) > 1.75]``, so the
division by 448 never happens and cannot be perturbed by fp32 rounding.

Call contract -- the 3 ``torch_npu.npu_dynamic_mx_quant`` sites in
``vendor/ascend/device/device_op.py``:

    L1249  torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=act_quant_type)
    L1491  torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
    L1773  torch_npu.npu_dynamic_mx_quant(hidden_states_temp, dst_type=torch.float8_e4m3fn)

and, for signature coverage beyond those three, the same call in
``impl/moe/prepare_finalize.py`` L360-369 (``dst_type=`` plus ``round_mode="round"`` for the
fp4 destination) and in vllm-ascend's w8a8/fp8_block schemes (``dst_type=`` plus
``scale_alg=``). All three device_op sites immediately flatten the scale with
``dynamic_scale.reshape(tokens, -1)`` and then ``.view(torch.float8_e8m0fnu)``, so only the
byte values matter to them, not the trailing pair shape.

Semantics / deviations (all deliberate):
  1. ``dst_type`` float8_e4m3fn (default) and float8_e5m2 are implemented; float4_e2m1fn_x2
     and hifloat8 raise ``NotImplementedError`` (the fp4 packing, its round_mode and its
     scale_alg rules could not be read from any source in this tree, and no device_op call
     site uses them).
  2. The fp8 bytes are produced by the hand-rolled saturating round-half-to-even encoder
     ``_fp8_rne_codes`` (shared with ``dynamic_quant``, byte-identical to the encoder
     validated in ``impl/swiglu_group_quant.py``) rather than the backend's fp32 -> fp8 cast.
  3. One padding E8M0 byte per row is added when ceil(D/32) is odd, so the scale row is
     ``2 * ceil(ceil(D/32) / 2)`` bytes. The padding is written as 0 here; the native
     allocates with at::empty and leaves it undefined (same decision as the
     ``swiglu_group_quant`` port).
  4. The scale is allocated as uint8 and returned as ``torch.float8_e8m0fnu`` when that dtype
     exists (the bytes are identical either way, and all three call sites re-``view`` it).
  5. Blocks are quantised in a single pass: unlike per-token dynamic quant, each MX block is
     self-contained, so no second read of the input is needed.
  6. The grid is one program per row with a stride loop inside the kernel, so it never
     exceeds the backend's grid limit.

Unconfirmed (no AscendC or hardware evidence in this container):
  * the exact rounding of the fp8 element (CAST_RINT is taken from the sibling MX kernel);
    a round-half-away-from-zero variant would differ only on exact ties;
  * the treatment of a block whose amax is a **subnormal fp16/bf16** input value (the fp32
    exponent is used here, which is the mathematically correct floor(log2(amax)));
  * whether the native clamps ``byte`` at 254 the way this port does (a finite fp16/bf16
    input can only reach ``byte = 246``, so the clamp is unreachable in practice);
  * the native's scale_alg=1 exact rule -- it is derived analytically from the description in
    vllm-ascend's utils.py, not from code.

Validated locally on CPU (no NPU, no Triton -- see `_agent_logs/T-209_quant_ops/`):
  * ``_fp8_rne_codes`` reproduces torch's own float32 -> float8_e4m3fn cast byte for byte on
    every finite e4m3 value, on both signs, on every midpoint between adjacent values (the
    exact ties) and on those midpoints shifted by -2..+2 fp32 ulp, on 200000 random values
    spanning 2**-20..2**20, on the e4m3 subnormal range and on 0 / -0 / inf / NaN: 0 of
    231 000+ differing bytes.
  * for e5m2 the only divergence from torch's cast is the overflow case: torch maps a value
    above 57344 to inf, while this encoder saturates to 57344 (AscendC SatMode::SAT). An MX
    quantized e5m2 value cannot reach that range (|q| <= 2**15 = 32768 with L = 15), and an
    inf/NaN input takes the explicit 0xFF / 0x7F branch instead.
  * the scale_alg 1 rule was checked against hand-worked examples: amax = 1.9 gives
    byte 119 under alg 0 (q = 486.4, i.e. saturated) and byte 120 under alg 1 (q = 243.2),
    which is exactly the "avoid quantization overflow" behaviour the algorithm is meant to
    provide; amax = 3.0, 0.5, 448.0 and 1000.0 behave the same way.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())


from .dynamic_quant import _fp8_rne_codes

__all__ = ["dynamic_mx_quant", "npu_dynamic_mx_quant", "MX_BLOCK_SIZE"]

#: PER_MX_FP16 (swiglu_group_quant_base.h L30): one shared E8M0 exponent per 32 elements.
MX_BLOCK_SIZE = 32
#: emax_elem of the destination format, i.e. the L of ``scale = 2 ** (e - L)``:
#: 8 for e4m3 (largest magnitude 448), 15 for e5m2 (57344).
_ELEM_EMAX = {torch.float8_e4m3fn: 8, torch.float8_e5m2: 15}
#: 1-D grid limit of the backend; the kernel strides over the remaining rows.
_MAX_GRID = 65535
#: Elements per program tile: a whole number of 32-element blocks.
_BLK_BLOCKS = 32


@triton.jit
def _dynamic_mx_quant_kernel(
    X, Y, S,
    ROWS, D, G, SPAD,
    NPROG,
    SCALE_ALG: tl.constexpr,
    IS_E4M3: tl.constexpr,
    BLK: tl.constexpr,
    BLK_B: tl.constexpr,
):
    """One row per inner iteration, BLK_B blocks of BLK == 32 elements per tile.

    X     [ROWS, D] float16 / bfloat16 / float32
    Y     [ROWS, D] uint8 view of the fp8 destination
    S     [ROWS, SPAD] uint8 E8M0 bytes, SPAD = 2 * ceil(G / 2), G = ceil(D / 32)
    """
    L: tl.constexpr = 8 if IS_E4M3 else 15
    pid = tl.program_id(0)
    off = tl.arange(0, BLK)
    blk = tl.arange(0, BLK_B)
    nloop = tl.cdiv(G, BLK_B)
    for row in range(pid, ROWS, NPROG):
        base = row * D
        for s in range(0, nloop):
            b = s * BLK_B + blk
            d = b[:, None] * BLK + off[None, :]
            m2 = (b < G)[:, None] & (d < D)
            v = tl.load(X + base + d, mask=m2, other=0.0).to(tl.float32)
            amax = tl.max(tl.where(m2, tl.abs(v), 0.0), axis=1)

            ab = amax.to(tl.int32, bitcast=True)
            e = (ab >> 23) & 0xFF                                  # biased exponent of amax
            byte = e - L
            if SCALE_ALG == 1:
                # ceil(log2(amax / 448)) == floor(log2(amax)) - 8 + [mantissa(amax) > 1.75];
                # the comparison is strict, so an amax of exactly 1.75 * 2**n keeps the
                # floor exponent (amax / scale is then exactly 448, which is representable).
                byte = byte + tl.where((e > 0) & ((ab & 0x7FFFFF) > 0x600000), 1, 0)
            byte = tl.maximum(byte, 0)
            byte = tl.minimum(byte, 254)
            special = e == 0xFF                                    # inf / NaN block
            byte = tl.where(special, 0xFF, byte)                   # FP8_E8M0_NAN_VAL

            # 2 ** (127 - byte) == 1 / scale, as a power-of-two fp32 bit pattern
            inv = ((254 - byte) << 23).to(tl.float32, bitcast=True)
            code = _fp8_rne_codes(v * inv[:, None], IS_E4M3=IS_E4M3)
            code = tl.where(special[:, None], 0x7F, code)          # canonical fp8 NaN

            tl.store(Y + base + d, code.to(tl.uint8), mask=m2)
            tl.store(S + row * SPAD + b, byte.to(tl.uint8), mask=b < G)


def _e8m0_dtype():
    """``float8_e8m0fnu`` when this torch exposes it, else None (the bytes are the same)."""
    return getattr(torch, "float8_e8m0fnu", None)


def _resolve_dst_type(dst_type, act_quant_type):
    if act_quant_type is not None:
        if dst_type is not None and dst_type != act_quant_type:
            raise ValueError(
                f"conflicting destination dtypes: dst_type={dst_type}, act_quant_type={act_quant_type}"
            )
        dst_type = act_quant_type
    if dst_type is None:
        return torch.float8_e4m3fn
    if not isinstance(dst_type, torch.dtype):
        raise NotImplementedError(
            f"dst_type={dst_type!r} (a ScalarType code) is not supported; pass a torch.dtype "
            "(torch.float8_e4m3fn / torch.float8_e5m2)"
        )
    if dst_type in _ELEM_EMAX:
        return dst_type
    raise NotImplementedError(
        f"dst_type={dst_type} is not implemented for dynamic_mx_quant: no call site in "
        "vendor/ascend/device/device_op.py uses it, and the fp4/hifloat8 packing, round_mode "
        "and scale rule could not be confirmed from any source in this tree"
    )


def _resolve_scale_alg(scale_alg):
    """0 = OCP floor rule, 1 = ceil after mapping amax into the fp8 range."""
    if isinstance(scale_alg, str):
        # vllm-ascend passes the string form in some schemes (kimi_kda.py L342 "floor").
        aliases = {"floor": 0, "ceil": 1}
        if scale_alg not in aliases:
            raise NotImplementedError(f"scale_alg={scale_alg!r} is not implemented")
        return aliases[scale_alg]
    if scale_alg in (0, 1):
        return int(scale_alg)
    raise NotImplementedError(f"scale_alg={scale_alg!r} is not implemented (supported: 0, 1)")


def dynamic_mx_quant(
    x: torch.Tensor,
    dst_type: torch.dtype | None = None,
    *,
    act_quant_type: torch.dtype | None = None,
    scale_alg: int | str = 0,
    round_mode: str = "rint",
    block_size: int = MX_BLOCK_SIZE,
    scale_layout: str = "pairs",
    e8m0: bool = True,
):
    """Drop-in equivalent of ``torch_npu.npu_dynamic_mx_quant`` (OCP MXFP8).

    Args:
        x: [..., D] float16 / bfloat16 / float32. The last dimension is cut into blocks of
            ``block_size`` (32); every block gets one E8M0 shared exponent.
        dst_type: None (resolves to torch.float8_e4m3fn) / torch.float8_e4m3fn /
            torch.float8_e5m2.
        act_quant_type: alias for ``dst_type`` (the spelling used by
            ``A5DeviceAdaptor.npu_dynamic_quant``, device_op L1249). Giving both is only
            allowed when they are equal.
        scale_alg: 0 (default OCP rule) or 1 (ceil rule). "floor" / "ceil" are accepted as
            aliases of 0 / 1.
        round_mode: only "rint" (round half to even) is implemented; the fp4-only
            ``round_mode="round"`` raises.
        block_size: must be 32 (MX block size); any other value raises.
        scale_layout: "pairs" (default) returns the native layout
            [..., ceil(ceil(D/32)/2), 2]; "flat" returns [..., 2*ceil(ceil(D/32)/2)]. All
            three device_op call sites flatten the scale, so both are equivalent for them.
        e8m0: return the scale as ``torch.float8_e8m0fnu`` (default) when that dtype exists,
            otherwise as raw uint8 bytes.

    Returns:
        ``(y, scale)`` -- ``y`` has x's shape and the destination dtype, ``scale`` holds one
        E8M0 byte per 32-element block (see ``scale_layout``), i.e. the shared exponent
        ``scale = 2 ** (byte - 127)`` to divide the quantized values by.
    """
    if x.dim() < 2:
        raise ValueError(f"dynamic_mx_quant expects at least 2 dimensions [..., D], got {tuple(x.shape)}")
    if block_size != MX_BLOCK_SIZE:
        raise NotImplementedError(
            f"block_size={block_size} is not supported: the MX block size is 32 "
            "(PER_MX_FP16, swiglu_group_quant_base.h L30; group_size 32 in w8a8_mxfp8.py L70)"
        )
    if round_mode != "rint":
        raise NotImplementedError(
            f"round_mode={round_mode!r} is not implemented for the fp8 destinations; "
            "the MX kernels in this tree use RoundMode::CAST_RINT (round half to even)"
        )
    if scale_layout not in ("pairs", "flat"):
        raise ValueError(f"scale_layout={scale_layout!r} is not supported ('pairs' or 'flat')")
    out_dtype = _resolve_dst_type(dst_type, act_quant_type)
    alg = _resolve_scale_alg(scale_alg)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"dynamic_mx_quant input dtype {x.dtype} is not supported")

    shape = tuple(x.shape)
    rows = 1
    for s in shape[:-1]:
        rows *= s
    d = shape[-1]
    groups = _ceil_div(d, MX_BLOCK_SIZE)
    # The native declares two E8M0 bytes per entry, i.e. an even number of groups per row.
    spad = 2 * _ceil_div(groups, 2) if scale_layout == "pairs" else groups

    dev = x.device
    x = x.contiguous()
    y = torch.empty(shape, dtype=out_dtype, device=dev)
    scale_u8 = torch.zeros(shape[:-1] + (spad,), dtype=torch.uint8, device=dev)

    if rows != 0 and d != 0 and groups != 0:
        # One program per row; the kernel strides if there are more rows than grid slots.
        # The tile covers a power-of-two number of blocks, so a narrow row does not pay for a
        # full 32-block tile while a wide one still keeps the loop count low.
        blk_b = max(1, min(_BLK_BLOCKS, _next_pow2(groups)))
        grid = (min(rows, _MAX_GRID),)
        _dynamic_mx_quant_kernel[grid](
            x, y.view(torch.uint8), scale_u8,
            rows, d, groups, spad,
            grid[0],
            SCALE_ALG=alg,
            IS_E4M3=out_dtype is torch.float8_e4m3fn,
            BLK=MX_BLOCK_SIZE,
            BLK_B=blk_b,
        )

    scale = scale_u8
    if scale_layout == "pairs":
        scale = scale.view(shape[:-1] + (spad // 2, 2))
    e8m0_dtype = _e8m0_dtype() if e8m0 else None
    if e8m0_dtype is not None:
        scale = scale.view(e8m0_dtype)
    return y, scale


#: ``torch_npu`` spelling of the same operator.
npu_dynamic_mx_quant = dynamic_mx_quant
