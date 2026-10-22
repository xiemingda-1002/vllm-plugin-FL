# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""grouped_matmul -- AscendC -> Triton

Source: this operator is the CANN / torch_npu built-in `npu_grouped_matmul`
(aclnnGroupedMatmul). **There is no AscendC kernel body for it in this tree**:
`grep -rn "grouped_matmul" csrc/ascend --include=*.cpp` returns no match (the only
mention in a header is a stale comment, csrc/ascend/attention/recurrent_gated_delta_rule/
op_kernel/recurrent_gated_delta_rule.h L11). The semantics below are therefore pinned
from (a) the call sites, (b) the torch reference implementation shipped with the tree,
and (c) the parameter set the call sites actually pass -- not from a kernel body.

--------------------------------------------------------------------------
Call sites (the five `torch_npu.npu_grouped_matmul` sites in
dispatch/backends/vendor/ascend/device/device_op.py)
--------------------------------------------------------------------------
  [1] L291-302   BaseDeviceAdaptor.npu_grouped_matmul_gmm2        split_item=2
                 x=[h], weight=weight, scale=weight_scale, bias=bias,
                 per_token_scale=[per_token_scale], group_list_type=..., group_type=0,
                 group_list=group_list, output_dtype=fallback_output_dtype
                 -> A8W8 dynamic quantization (int8 x, int8 w, per-channel + per-token)
  [2] L1294-1308 A5DeviceAdaptor.npu_grouped_matmul_swiglu_quant   split_item=2
                 x=[x], weight=[w], scale=None, antiquant_scale=[weight_scale],
                 scale_dtype=None, per_token_scale=[x_scale],
                 per_token_scale_dtype=torch.float8_e8m0fnu, group_type=0,
                 group_list=group_list, x_dtype=torch.float8_e4m3fn,
                 weight_dtype=torch_npu.float4_e2m1fn_x2, output_dtype=torch.bfloat16
                 -> W4A8 MXFP (fp8 activation, fp4 packed weight, e8m0 scales)
  [3] L1319-1327 A5DeviceAdaptor.npu_grouped_matmul_swiglu_quant   split_item=3
                 x=[x], weight=[w], antiquant_scale=[weight_scale], group_list=...,
                 group_type=0, output_dtype=x.dtype
                 -> W4A16 MXFP (fp4 packed weight)
  [4] L1445-1455 A5DeviceAdaptor.npu_grouped_matmul_gmm2           split_item=3
                 x=[h], weight=w, antiquant_scale=scale, bias=bias, group_type=0,
                 group_list_type=..., group_list=..., output_dtype=...
                 -> W4A16 MXFP (fp4 packed weight)
  [5] L1461-1473 A5DeviceAdaptor.npu_grouped_matmul_gmm2           split_item=2
                 x=[h], weight=w, scale=scale, bias=bias, per_token_scale=[pts],
                 group_list_type=..., group_type=0, group_list=..., output_dtype=...,
                 **gmm2_kwargs (antiquant_scale, scale_dtype, per_token_scale_dtype,
                 x_dtype, weight_dtype)
                 -> W4A8 MXFP (fp4 packed weight, e8m0 scales)

Other in-tree users of the *same* op (same parameter names, so the signature here has
to cover them too): impl/fused_moe.py L124-131 / L145-152, impl/fused_moe_layer.py
L157-164 / L170-177, impl/moe/moe_mlp.py L223 / L232-240 / L287-297 / L318-328 / L410 /
L502-510 / L551-559, and the vllm-ascend reference helper `apply_mlp` in
tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_moe.py L81-101.

--------------------------------------------------------------------------
Semantics (pure grouped matmul)
--------------------------------------------------------------------------
    out[m, n] = ( sum_k x[m, k] * w_deq[g(m), k, n] ) * scale[g(m), n] * pscale[m]
                + bias[g(m), n]

g(m) is the group owning row m, i.e. rows [start_g, end_g) derived from `group_list`
(`group_list_type` 0 = inclusive cumsum / end offsets, 1 = per-group counts), and

    w_deq = w                                          (unquantized path)
    w_deq = (w + antiquant_offset) * antiquant_scale    (dequant path)

The row-range part is pinned by the reference implementation above:
`out[mask] == a[mask] @ w1[i].transpose(0, 1)` for `mask = topk_ids == i`, with the rows
already sorted by expert and `group_list = torch.bincount(...).cumsum(0)`.

The dequant sign convention (`(w + off) * s` vs `w * s + off`) is **not pinned by any
source in this tree**: every production caller passes `antiquant_offset` as an all-zero
tensor (vllm-ascend vllm_ascend/quantization/methods/wna16/w4a16.py L171/L174
"``w13_weight_offset``: ``torch.bfloat16``, all zeros"), where both conventions
coincide. This implementation uses `(w + off) * s`; a non-zero offset would distinguish
them, so the NPU equivalence test compares against the vendor op rather than assuming.

--------------------------------------------------------------------------
Scope (per-call-site coverage)
--------------------------------------------------------------------------
  supported : [1] A8W8 dynamic (int8 x / int8 w / scale / per_token_scale / bias);
              the unquantized float path (fused_moe.py, fused_moe_layer.py,
              moe_mlp.py L502/L551, test_fused_moe.py -- all split_item=2);
              a dequant path for non-packed integer weights.
  raises    : [2] [3] [4] [5] -- all four A5 MXFP sites. They need fp4 packed weights
              (`weight_dtype=torch_npu.float4_e2m1fn_x2`), fp8 activations
              (`x_dtype=torch.float8_e4m3fn`) and e8m0 block scales
              (`per_token_scale_dtype=torch.float8_e8m0fnu`). None of that is
              implemented here, and this file raises NotImplementedError naming the
              offending argument instead of guessing a packing/layout.
              Also raised: split_item not in {2, 3}, group_type != 0, act_type != 0,
              integer `output_dtype`, fp32 activations, `offset` (as opposed to
              `antiquant_offset`), and packed/fp8 dtypes.

Why the four MXFP sites are out of scope: they are A5-only (`device_op.py` L1292
"W4A8 mxfp", `QuantType.W4A8MXFP`), they carry a packed 4-bit weight container plus
e8m0 block-scale layouts that are documented nowhere in this tree, and two of them are
already served by the *fusion* operator in grouped_matmul_swiglu_quant_v2.py.

--------------------------------------------------------------------------
Precision (read this before comparing against the vendor op)
--------------------------------------------------------------------------
* `tl.dot` on this backend takes fp16/bf16 operands with an fp32 accumulator -- the
  pattern grouped_matmul_swiglu_quant_v2.py uses and which is verified PASS there. The
  int8 A8W8 path therefore feeds **fp16** operands: an int8 product can be as large as
  127*127 = 16129, which needs 14 mantissa bits and is rounded to fp16 (relative error
  <= 2**-11 ~ 4.9e-4 per product) before the fp32 accumulate over K.
  Consequence: an A8W8 result differs from the vendor's exact int32 accumulation by
  <= ~5e-4 relative -- below bf16 resolution (2**-8 ~ 3.9e-3) but **not** bit-exact.
  This is also why an integer `output_dtype` is rejected: impl/moe/moe_mlp.py L239 uses
  `output_dtype=torch.int32` to hand a raw int32 sum to npu_dequant_swiglu_quant, and
  this path cannot reproduce that exactly.
* bf16 activations keep bf16 dot operands (no downcast); fp16 and int8 use fp16.
* The masked 2-D store is the access pattern the verified
  grouped_matmul_swiglu_quant_v2.py uses; that module documents a backend hazard (a 2-D
  masked store can drop lanes for non-power-of-two widths on this backend). Here the
  store mask is `rm & nm` with a power-of-two BLK_N tile, the same shape that was
  measured clean there. test_numeric.py double-fills the output with a sentinel and
  counts dropped lanes, so the hazard is measured, not assumed away.

--------------------------------------------------------------------------
UB budget (borrowed from the measured formula in grouped_matmul_swiglu_quant_v2.py)
--------------------------------------------------------------------------
This kernel does strictly less per-tile elementwise work than that one (no swiglu, no
row max, no requantization), so that file's fitted requirement is used as a
conservative upper bound:

    ub_bits ~= BLK_N * (3243 + 86*BLK_M + 24*max(BLK_K-128, 0)),  limit 1,572,864

(Ascend 910_93 + BiShengIR, measured). The tiles here are BLK_M=16, BLK_N<=256,
BLK_K<=128, which the formula puts at 256*(3243+1376) = 1,182,464 bits < limit; the
host raises rather than launching if a shape would need a tile above the limit. This is
a **borrowed** bound, not a measurement of this kernel -- see RESULT.md.

--------------------------------------------------------------------------
Grid cap (AI Core block_dim)
--------------------------------------------------------------------------
The program count is capped at 65,535; a larger grid was measured to crash with `aivec`
on this stack (_agent_logs/TRITON_OPERATOR_PITFALLS.md C1 "grid has a limit of 65,535
(block_dim)"). The natural grid for this kernel, (E, ceil(M/BLK_M)), reaches
256 * 512 = 131,072 for a production MoE layer, so the host folds
ceil(n_row_tiles / (MAX_PROGRAMS // E)) row tiles into each program (`MT_PER_PROG`) and
the total program count stays <= 32,768.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["grouped_matmul"]

# UB budget, see the module docstring (measured limit on 910_93 + BiShengIR).
UB_LIMIT_BITS = 1_572_864
BLK_M_CAP = 16
BLK_K_CAP = 128
BLK_N_CAP = 256
# The AI Core block_dim (program count) is capped at 65,535 and a larger grid was
# measured to crash with `aivec` (documented in _agent_logs/TRITON_OPERATOR_PITFALLS.md
# C1). The grid here is (E, n_row_blocks): with E = 256 experts and M = 8192 that is
# 256 * 512 = 131,072 programs, i.e. over the cap, so each program takes several row
# tiles serially until the total fits under half the hard limit.
MAX_PROGRAMS = 32_768

# split_item values used by the call sites: 2 at device_op.py L291/L1294/L1461,
# 3 at L1319/L1445. Both mean "one 2-D result; x/weight are 1-element lists".
_SPLIT_ITEMS = (2, 3)


def ub_budget_bits(blk_m: int, blk_k: int, blk_n: int) -> int:
    """Borrowed UB estimate in bits (see the module docstring)."""
    return blk_n * (3243 + 86 * blk_m + 24 * max(blk_k - 128, 0))


@triton.jit
def _gmm_kernel(
    X, W, GSTART, GEND, Y,
    WSCALE, PSCALE, ABSCALE, ABOFF, BIAS,
    K, N, KN,
    BLK_M: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_N: tl.constexpr,
    MT_PER_PROG: tl.constexpr,  # row tiles handled serially per program (grid cap)
    DOT_BF16: tl.constexpr,     # dot operands are bf16 (else fp16)
    HAS_WSCALE: tl.constexpr,   # per-expert per-channel scale, applied after the matmul
    HAS_PSCALE: tl.constexpr,   # per-token scale, applied after the matmul
    ANTIQ: tl.constexpr,        # dequantize the weight tile with (w + off) * scale
    HAS_BIAS: tl.constexpr,
):
    """One program handles (group g, MT_PER_PROG row tiles) and loops over the N axis and
    the K axis at runtime, so a block never holds more than BLK_N x BLK_K of W and
    BLK_M x BLK_K of x (see the UB note in the module docstring).

    Row indices are absolute rows of the concatenated `x` (`rows = start + ...`), so a
    group whose row count is not a multiple of BLK_M is handled by the `rm` mask, and a
    group with zero rows writes nothing (`mt * BLK_M < end - start` is false).

    MT_PER_PROG > 1 exists only because the AI Core `block_dim` is capped at 65,535
    programs (a measured `aivec` crash above it, see the module docstring): the host
    raises BLK_M-sized row tiles per program until `E * n_row_blocks` fits.
    """
    g = tl.program_id(0)
    start = tl.load(GSTART + g)
    end = tl.load(GEND + g)
    pid_m = tl.program_id(1)

    for i in range(MT_PER_PROG):
        mt = pid_m * MT_PER_PROG + i
        # Empty / exhausted row tile: every store below would be masked out anyway.
        if mt * BLK_M < end - start:
            rows = start + mt * BLK_M + tl.arange(0, BLK_M)
            rm = rows < end
            k_off = tl.arange(0, BLK_K)
            nchunk = tl.cdiv(N, BLK_N)
            # `g * KN` must not overflow int32: KN = K * N is passed as its own argument so
            # that this multiplication is promoted to int64 for large E*K*N tensors.
            wbase = g * KN

            for nc in range(0, nchunk):
                cn = nc * BLK_N + tl.arange(0, BLK_N)
                nm = cn < N
                acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
                for k0 in range(0, K, BLK_K):
                    kk = k0 + k_off
                    km = kk < K
                    xv = tl.load(X + rows[:, None] * K + kk[None, :],
                                 mask=rm[:, None] & km[None, :], other=0)
                    wv = tl.load(W + wbase + kk[:, None] * N + cn[None, :],
                                 mask=km[:, None] & nm[None, :], other=0)
                    if ANTIQ:
                        asv = tl.load(ABSCALE + g * N + cn, mask=nm,
                                      other=0.0).to(tl.float32)
                        aov = tl.load(ABOFF + g * N + cn, mask=nm,
                                      other=0.0).to(tl.float32)
                        wv = (wv.to(tl.float32) + aov[None, :]) * asv[None, :]
                    if DOT_BF16:
                        acc += tl.dot(xv.to(tl.bfloat16), wv.to(tl.bfloat16),
                                      out_dtype=tl.float32)
                    else:
                        acc += tl.dot(xv.to(tl.float16), wv.to(tl.float16),
                                      out_dtype=tl.float32)
                if HAS_WSCALE:
                    ws = tl.load(WSCALE + g * N + cn, mask=nm, other=1.0).to(tl.float32)
                    acc = acc * ws[None, :]
                if HAS_PSCALE:
                    ps = tl.load(PSCALE + rows, mask=rm, other=1.0).to(tl.float32)
                    acc = acc * ps[:, None]
                if HAS_BIAS:
                    bv = tl.load(BIAS + g * N + cn, mask=nm, other=0.0).to(tl.float32)
                    acc = acc + bv[None, :]
                acc = tl.where(nm[None, :], acc, 0.0)
                tl.store(Y + rows[:, None] * N + cn[None, :],
                         acc.to(Y.dtype.element_ty), mask=rm[:, None] & nm[None, :])


def _ends_from_group_list(group_list, e, group_list_type):
    """group_list_type 0 => inclusive cumsum / end offsets (the default);
    1 => per-group counts (the prefix sum is computed here).

    Same convention as the helper of the same name in
    grouped_matmul_swiglu_quant_v2.py: what the kernel needs is, per group, the
    [start, end) row range. Everything stays on the device (no host sync).
    """
    gl = group_list.contiguous().to(torch.int32)
    assert gl.numel() == e, f"group_list length {gl.numel()} != number of groups {e}"
    if int(group_list_type) == 1:
        ends = torch.cumsum(gl.to(torch.int64), dim=0).to(torch.int32)
    else:
        ends = gl
    starts = torch.zeros_like(ends)
    starts[1:] = ends[:-1]
    return starts, ends


def _as_list(t, name):
    """`x` / `weight` / `bias` / `scale` / ... are TensorLists in the operator
    signature; every call site in this tree passes a 1-element list (device_op.py) or a
    bare tensor (fused_moe.py). Returns a list, or None."""
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        assert len(t) >= 1, f"{name} is an empty list"
        return list(t)
    return [t]


def _weight_shape(wl):
    """(E, K, N) implied by the weight container, without materializing it."""
    w0 = wl[0]
    if len(wl) == 1 and w0.dim() == 3:
        return int(w0.shape[0]), int(w0.shape[1]), int(w0.shape[2])
    if len(wl) == 1 and w0.dim() == 2:
        return 1, int(w0.shape[0]), int(w0.shape[1])
    assert all(t.dim() == 2 for t in wl), \
        "a per-expert weight list must hold [K, N] tensors"
    k, n = int(wl[0].shape[0]), int(wl[0].shape[1])
    for t in wl:
        assert tuple(t.shape) == (k, n), f"per-expert weights differ in shape: {tuple(t.shape)}"
    return len(wl), k, n


def _stack_weight(wl, e, k, n):
    """Return a contiguous [E, K, N] tensor.

    Two container forms occur in this tree:
      * one stacked tensor (or a 1-element list holding it), shape [E, K, N]
        -- device_op.py L1295 `weight=[weight]`, vllm-ascend w8a8_dynamic.py L497;
      * one [K, N] tensor per expert -- vllm-ascend w8a8_dynamic.py L491-493 ("w1 may be
        a per-expert weight list ... aclnn grouped_matmul requires len(scale) ==
        len(weight)") and test_fused_moe.py `weight=[w.transpose(0, 1) for w in weights]`.
    The per-expert form is stacked here, which transiently allocates the whole weight a
    second time (E*K*N*dtype bytes); the vendor op reads the list in place. Stated as a
    limitation in RESULT.md.
    """
    if len(wl) == 1 and wl[0].dim() == 3:
        w = wl[0].contiguous()
    elif len(wl) == 1 and wl[0].dim() == 2:
        w = wl[0].contiguous().unsqueeze(0)
    else:
        w = torch.stack([t.contiguous() for t in wl], dim=0)
    assert tuple(w.shape) == (e, k, n), \
        f"weight must be [{e},{k},{n}], got {tuple(w.shape)}"
    return w


def _per_channel(t, e, n, name):
    """Normalize a per-expert per-channel tensor ([E, N], [N], or a list of them) to a
    contiguous [E, N] float32 tensor."""
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        assert len(t) >= 1, f"{name} is an empty list"
        if len(t) == 1 and t[0].dim() == 2:
            v = t[0].contiguous()
        else:
            v = torch.stack([x.contiguous() for x in t], dim=0)
    else:
        v = t.contiguous()
    if v.dim() == 3 and v.shape[1] == 1:      # [E, 1, N]
        v = v.squeeze(1)
    if v.dim() == 1:
        assert v.numel() == n, f"{name} has {v.numel()} entries but N={n}"
        v = v.unsqueeze(0).expand(e, n)
    assert v.dim() == 2 and tuple(v.shape) == (e, n), \
        f"{name} must be [{e},{n}] or [{n}], got {tuple(v.shape)}"
    return v.to(torch.float32).contiguous()


def _per_token(t, m):
    """Normalize per_token_scale ([M], [M, 1] or a 1-element list of either) to [M]."""
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        assert len(t) == 1, f"per_token_scale must hold one tensor, got {len(t)}"
        v = t[0]
    else:
        v = t
    v = v.reshape(-1)
    assert v.numel() == m, f"per_token_scale has {v.numel()} entries for M={m} rows"
    return v.to(torch.float32).contiguous()


def grouped_matmul(
    x,
    weight,
    bias=None,
    scale=None,
    offset=None,
    antiquant_scale=None,
    antiquant_offset=None,
    per_token_scale=None,
    group_list: torch.Tensor | None = None,
    split_item: int = 0,
    group_type: int = 0,
    group_list_type: int = 0,
    act_type: int = 0,
    output_dtype: torch.dtype | None = None,
    x_dtype: torch.dtype | None = None,
    weight_dtype: torch.dtype | None = None,
    scale_dtype: torch.dtype | None = None,
    per_token_scale_dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    """Equivalent of `torch_npu.npu_grouped_matmul` -- pure (unfused) grouped matmul.

    Produces
        out[m, n] = ( sum_k x[m, k] * w_deq[g(m), k, n] ) * scale[g(m), n] * pscale[m]
                    + bias[g(m), n]
    where g(m) comes from `group_list` (see `group_list_type`), and
        w_deq = w                                        (unquantized)
        w_deq = (w + antiquant_offset) * antiquant_scale  (dequant)

    Returns:
        `[out]` with out of shape [M, N] -- a **one-element list**, because the operator
        returns a TensorList and all five device_op.py sites index `[0]` (L302, L1308,
        L1327, L1455, L1473). Evidence for that container: tests/unit_tests/dispatch/
        test_ascend_device_operator_quant.py L74-76 mocks the vendor op returning
        `[expected]` and L104 asserts the wrapper hands back that very element.

    Parameter names, order and defaults mirror the call sites. All five sites pass
    keyword arguments only, so the positional order is a best-effort reconstruction of
    the aclnn/torch_npu signature; torch_npu is not installed in this workspace, so its
    exact positional order is unverified.

    Unsupported arguments raise NotImplementedError naming the argument and its value;
    nothing is silently ignored.
    """
    # ---- mode / dtype validation ---------------------------------------------------
    if int(split_item) not in _SPLIT_ITEMS:
        raise NotImplementedError(
            f"split_item={split_item}: only {_SPLIT_ITEMS} occur in this tree (2 at "
            "device_op.py L291/L1294/L1461, 3 at L1319/L1445). split_item 0/1 return one "
            "output per group, which no call site uses.")
    if int(group_type) != 0:
        raise NotImplementedError(
            f"group_type={group_type}: only group_type=0 (group_list splits the token "
            "dimension) is used; group_type=1 (K split) is not covered.")
    if int(act_type) != 0:
        raise NotImplementedError(
            f"act_type={act_type}: must be 0 (no fused activation). No call site passes "
            "act_type; the in-tree fusion is a separate operator "
            "(grouped_matmul_swiglu_quant_v2.py).")
    if offset is not None:
        raise NotImplementedError(
            "offset is not None: no call site in this tree passes `offset` (the dequant "
            "argument they use is `antiquant_offset`), so its semantics are unconfirmed.")
    if x_dtype is not None or weight_dtype is not None:
        raise NotImplementedError(
            f"x_dtype={x_dtype} / weight_dtype={weight_dtype}: these declare packed or "
            "8-bit-block dtypes (device_op.py L1305-1306 passes "
            "x_dtype=torch.float8_e4m3fn, weight_dtype=torch_npu.float4_e2m1fn_x2). "
            "Packed fp4 weights and fp8 activations are outside this implementation.")
    if scale_dtype is not None or per_token_scale_dtype is not None:
        raise NotImplementedError(
            f"scale_dtype={scale_dtype} / per_token_scale_dtype={per_token_scale_dtype}: "
            "these are MXFP block-scale dtypes (device_op.py L1301 passes "
            "per_token_scale_dtype=torch.float8_e8m0fnu), which are not implemented.")

    xl = _as_list(x, "x")
    if len(xl) != 1:
        raise NotImplementedError(
            f"x holds {len(xl)} tensors: one activation per group is the split_item=0/1 "
            "form, which no call site in this tree uses (all five pass `x=[x]`).")
    xs = xl[0]
    assert xs.dim() == 2, f"x must be 2-D [M, K], got {tuple(xs.shape)}"
    if xs.dtype == torch.float32:
        raise NotImplementedError(
            "x.dtype=torch.float32: tl.dot on this backend takes fp16/bf16 operands, and "
            "a downcast would silently drop fp32 mantissa bits. No call site passes fp32.")
    if xs.dtype not in (torch.int8, torch.bfloat16, torch.float16):
        raise NotImplementedError(
            f"x.dtype={xs.dtype} is not covered (int8 / bfloat16 / float16)")
    m, k = int(xs.shape[0]), int(xs.shape[1])
    assert k > 0, "x must have K > 0"

    wl = _as_list(weight, "weight")
    e, wk, n = _weight_shape(wl)
    assert wk == k, f"weight K={wk} != x K={k}"
    assert n > 0, "weight must have N > 0"

    # ---- output dtype ---------------------------------------------------------------
    if output_dtype is None:
        if xs.dtype == torch.int8:
            raise NotImplementedError(
                "output_dtype=None with int8 x: the result dtype cannot be inferred from "
                "an int8 activation. Every int8 call site passes output_dtype "
                "(device_op.py L1307/L1326/L1454/L1471).")
        out_dtype = xs.dtype
    else:
        out_dtype = output_dtype
    if out_dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise NotImplementedError(
            f"output_dtype={out_dtype}: an exact integer accumulation is not reproducible "
            "through the fp16 MMA this backend offers (see the precision note in the "
            "module docstring); impl/moe/moe_mlp.py L239 relies on "
            "output_dtype=torch.int32.")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError(f"output_dtype={out_dtype} is not covered")

    # ---- optional tensors ------------------------------------------------------------
    ws = _per_channel(scale, e, n, "scale")
    ps = _per_token(per_token_scale, m)
    abs_ = _per_channel(antiquant_scale, e, n, "antiquant_scale")
    abo = _per_channel(antiquant_offset, e, n, "antiquant_offset")
    bi = _per_channel(bias, e, n, "bias")
    antiq = abs_ is not None or abo is not None

    w = _stack_weight(wl, e, k, n)
    if w.dtype not in (torch.int8, torch.bfloat16, torch.float16):
        raise NotImplementedError(
            f"weight.dtype={w.dtype}: only int8 (A8W8 / dequant) and bf16/fp16 are "
            "covered. Packed 4-bit containers (torch_npu.float4_e2m1fn_x2) and int32 "
            "int4 packing are not implemented.")
    if antiq:
        if w.dtype != torch.int8:
            raise NotImplementedError(
                f"antiquant_scale with weight.dtype={w.dtype}: the dequant path expects an "
                "integer weight; a float weight means the argument is being used as "
                "something else.")
        if xs.dtype == torch.int8:
            raise NotImplementedError(
                "antiquant_scale together with int8 x: two independent dequantizations at "
                "once -- no call site does this, so the order of the two scales would be a "
                "guess.")
        if abs_ is None:
            abs_ = torch.ones((e, n), dtype=torch.float32, device=xs.device)
        if abo is None:
            abo = torch.zeros((e, n), dtype=torch.float32, device=xs.device)
    if xs.dtype == torch.int8 and not antiq and ws is None:
        raise NotImplementedError(
            "int8 x without `scale`: the A8W8 path needs the per-channel dequant scale; "
            "no call site passes int8 activations without it.")

    # ---- int32 element-offset safety --------------------------------------------------
    if m * k >= 2 ** 31 or m * n >= 2 ** 31:
        raise NotImplementedError(
            f"M*K={m * k} / M*N={m * n} exceed int32; the kernel uses int32 row indexing "
            "(as grouped_matmul_swiglu_quant_v2.py does).")

    # ---- group ranges ------------------------------------------------------------------
    xs = xs.contiguous()
    if group_list is None:
        if e != 1:
            raise NotImplementedError(
                f"group_list=None with {e} experts: without group_list the whole token "
                "range is one group, so only a single expert is meaningful.")
        starts = torch.zeros(1, dtype=torch.int32, device=xs.device)
        ends = torch.full((1,), m, dtype=torch.int32, device=xs.device)
    else:
        starts, ends = _ends_from_group_list(group_list, e, group_list_type)

    y = torch.empty((m, n), dtype=out_dtype, device=xs.device)
    if m == 0:
        return [y]

    # ---- launch -------------------------------------------------------------------------
    blk_m = BLK_M_CAP
    blk_k = min(max(_next_pow2(k), 16), BLK_K_CAP)
    blk_n = min(max(_next_pow2(n), 16), BLK_N_CAP)
    need = ub_budget_bits(blk_m, blk_k, blk_n)
    if need > UB_LIMIT_BITS:
        raise NotImplementedError(
            f"grouped_matmul: tile (BLK_M={blk_m}, BLK_K={blk_k}, BLK_N={blk_n}) needs "
            f"~{need} bits of UB > {UB_LIMIT_BITS} (K={k}, N={n}).")

    w = w.contiguous()
    dummy = torch.zeros(1, dtype=torch.float32, device=xs.device)
    # bf16 activations keep bf16 dot operands; fp16 and int8 use fp16 (best for the
    # 14-bit int8 products, and the combination measured to work in
    # grouped_matmul_swiglu_quant_v2.py).
    dot_bf16 = xs.dtype == torch.bfloat16
    # Grid cap: E * n_row_blocks must stay under MAX_PROGRAMS (AI Core block_dim), so a
    # program walks several BLK_M row tiles serially when the tile count is too large.
    n_row_tiles = _ceil_div(m, blk_m)
    per_group_cap = max(1, MAX_PROGRAMS // max(e, 1))
    mt_per_prog = max(1, _ceil_div(n_row_tiles, per_group_cap))
    grid = (e, _ceil_div(n_row_tiles, mt_per_prog))
    _gmm_kernel[grid](
        xs, w, starts, ends, y,
        ws if ws is not None else dummy,
        ps if ps is not None else dummy,
        abs_ if antiq else dummy,
        abo if antiq else dummy,
        bi if bi is not None else dummy,
        k, n, k * n,
        BLK_M=blk_m, BLK_K=blk_k, BLK_N=blk_n, MT_PER_PROG=mt_per_prog,
        DOT_BF16=dot_bf16,
        HAS_WSCALE=ws is not None,
        HAS_PSCALE=ps is not None,
        ANTIQ=antiq,
        HAS_BIAS=bi is not None,
    )
    return [y]
