# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""grouped_matmul_swiglu_quant_v2 -- AscendC to Triton

Source: vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant_v2/

--------------------------------------------------------------------------
Call-site signature (verbatim from grouped_matmul_swiglu_quant_v2_torch_adpt.h L20-36)
--------------------------------------------------------------------------
    std::tuple<at::Tensor,at::Tensor> grouped_matmul_swiglu_quant_v2(
        const at::Tensor& x,                                   // [M, K] int8
        const at::TensorList& weight,                          // per expert [K, N] int8 (FRACTAL_NZ)
        const at::TensorList& weight_scale,                    // per expert [N]
        const at::Tensor& x_scale,                             // [M] fp32  per-token
        const at::Tensor& group_list,                          // [G] int64
        const c10::optional<at::Tensor>& smooth_scale,          // optional, (E,N/2) or (E,)
        const c10::optional<at::TensorList>& weight_assist_matrix,
        const c10::optional<at::Tensor>& bias,
        c10::optional<int64_t> dequant_mode, dequant_dtype, quant_mode, quant_dtype,
        bool transpose_weight, int64_t group_list_type,
        at::IntArrayRef tuning_config, double swiglu_limit)

Shape derivation (adapter L39-45): `n = weight_scale[0].sizes().back()`, `m = x.sizes()[0]`,
`k = x.sizes()[1]`; `output = [m, n/2] int8`, `output_scale = [m] fp32`.
(Note that v2 takes n from **weight_scale** while v1 takes it from weight -- one of the
interface differences between the two.)

--------------------------------------------------------------------------
Differences from v1 (`grouped_matmul_swiglu_quant_weight_nz_tensor_list`)
--------------------------------------------------------------------------
1. **`smooth_scale` (new in v2, optional)**: multiplied onto the SwiGLU result, **after
   SwiGLU and before the per-token quantization**. Source:
   op_kernel/grouped_matmul_swiglu_quant_v2_a4w4_post.h
     L137-139 `if (smoothScaleDimNum != 0) ApplySmoothScale(...)` (skipped when empty)
     L214-217 `smoothScaleDimNum == 2` => shape (E, N/2), use the row of the current group
     L221-223 `smoothScaleDimNum == 1` => shape (E,), one scalar broadcast over N/2
     L350 comment: "four vector steps (per-token dequant, Swiglu, SmoothScale, Quant)"
2. **`bias` (declared in the v2 signature but unused by the kernel)**:
   op_kernel/grouped_matmul_swiglu_quant_v2_utils.h L84 states explicitly
   `// bias is unused but the high-level template parameters require it`.
   => As in v1, **the original AscendC implementation adds no bias**; this implementation
   raises NotImplementedError for a non-empty bias (it is not silently ignored).
3. `weight_assist_matrix`: the auxiliary quantization matrix (int4 packed) of the W4A8/A4W4
   paths. This implementation targets **int8 weights (A8W8)** and raises NotImplementedError
   for a non-empty assist matrix.
4. `group_list_type`: 0 => `group_list` holds **cumulative end offsets** (cumsum, the default);
   1 => it holds **per-group counts** (the prefix sum is computed internally). Documented here
   explicitly as a semantic assumption.
5. Everything else is mathematically identical to v1 (same-family SwiGLU plus per-token
   dynamic quantization):
     C1 = (x*x_scale) @ (W_g*weight_scale_g)      # [rows, N]
     gate = C1[:, :N/2] ; up = C1[:, N/2:]         # <- up must be shifted by N/2 to align
     sw = silu(gate) * up                           # clamp(gate) when swiglu_limit>0
     sw = sw * smooth_scale_g                       # new in v2
     y = rint(sw*127/max|sw|) -> int8 ; y_scale = max|sw|/127

--------------------------------------------------------------------------
Implementation notes (following the style already verified to PASS)
--------------------------------------------------------------------------
* gate/up use the **same output column index** to read the two halves of W (two tl.dot
  calls), so they are aligned by construction -- never write `up = where(n>=I, c, 0)`
  (that supports disjoint column ranges and the product is identically 0).
* Verifiable segment by segment: this kernel only uses load -> tl.dot -> elementwise ->
  store, with no runtime loop carrying tensors.
* No "scalar mask & vector mask" broadcast; the two shapes of `smooth_scale` are handled
  separately.

Backend behavior (910_93 + BiShengIR/FlagTree):
  L129 is a "[BLK_M, BLK_N] 2-D masked store with a long value chain (dot->silu->row max->
  divide->fp16 round trip->rint)"; under a sentinel double-fill it showed **no** silent
  dropped writes: for I in {17,24,33,48,100,127} (including smooth col/row, swiglu_limit,
  K=128/256) every run had dropped=0, wrong>1LSB=0 and scale_err <= 2e-7.
  This file is therefore **not** rewritten; note however that I >= 257 (BLK_N >= 512) fails to
  compile on this backend's UB (ub overflow), so supporting it would need a smaller BLK_N tile
  -- unrelated to the above, it is a UB capacity issue.
  Note: do not `tl.reshape` 2-D->1-D before storing: the same backend drops lanes on a large
  scale.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["grouped_matmul_swiglu_quant_v2"]


# ---------------------------------------------------------------------------
# UB budget (910_93 + BiShengIR, **measured**)
# ---------------------------------------------------------------------------
# Limit 1,572,864 bits (the raw error is "ub overflow, requires N bits while 1572864 bits available").
# Measured (M=16/64, BLK_M=16, BLK_K=128):
#     BLK_N=256 -> OK ; 512 -> 2,359,808 ; 1024 -> 4,719,104 ; 2048 -> 9,437,696 ; 4096 -> 18,874,880
# Fitted formula (checks: BLK_M=16/BLK_K=512/BLK_N=256 -> 3,541,760 vs measured 3,539,456;
#                     BLK_M=16/BLK_K=1024/BLK_N=256 -> 6,685,184 vs measured 6,685,184):
#     ub ~= BLK_N x (3243 + 86 x BLK_M + 24 x max(BLK_K-128, 0))
# **Shrinking BLK_M alone does not help**: with BLK_N=512, BLK_M=1 still requires 1,704,192 bits
# (> limit)
# => production widths (I in [512,2304], K in {2048,7168}) require **runtime tiling along both
#    the N axis and the K axis**.
UB_LIMIT_BITS = 1_572_864
BLK_N_CAP = 256
BLK_K_CAP = 128


def ub_budget_bits(blk_m: int, blk_k: int, blk_n: int) -> int:
    """Measured, fitted UB requirement (bits), used for an explicit host-side rejection."""
    return blk_n * (3243 + 86 * blk_m + 24 * max(blk_k - 128, 0))


@triton.jit
def _gmm_v2_kernel(
    X, W, WS, XS, SMOOTH, GSTART, GEND, Y, YS,
    K, N, HALF_N,
    BLK_M: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_N: tl.constexpr,
    HAS_SMOOTH_COL: tl.constexpr,     # smooth_scale shape (E, N/2)
    HAS_SMOOTH_ROW: tl.constexpr,     # smooth_scale shape (E,)
    LIMIT_ON: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    """One program handles (group g, BLK_M rows) and **tiles along the N axis and the K axis at
    runtime inside the program**.

    Why tiling is needed (measured on 910_93 + BiShengIR):
      production `I = moe_intermediate_size in [512, 2304]` and `K = hidden in {2048, 7168}`.
      A single `[BLK_M, BLK_N]` block that loads the whole K at once necessarily hits
      `ub overflow` (limit 1,572,864 bits) when `BLK_N >= 512` or `BLK_K >= 512`, and
      **shrinking BLK_M does not help** (BLK_M=1, BLK_N=512 still requires 1,704,192 bits).
      N and K are therefore both turned into runtime loops, so that one block only occupies
      BLK_N <= 256 / BLK_K <= 128.

    NOTE the cost (which must be known): the per-token quantization needs max|sw| over the
    **whole row**, which one block cannot see once the row is tiled, so this is
    "**pass 1 computes the row max / pass 2 recomputes and quantizes**", doubling the tl.dot
    work. sw is not staged in scratch because that would need M x I x 4 bytes (production M can
    reach tens of thousands of tokens -> hundreds of MB).

    NOTE alignment (same as the previous version): gate takes columns [0, HALF_N) of W and up
    takes columns [HALF_N, 2*HALF_N), both expressed through the **same output column index
    cn**, so they align naturally.
    """
    g = tl.program_id(0)
    start = tl.load(GSTART + g)
    end = tl.load(GEND + g)
    rows = start + tl.program_id(1) * BLK_M + tl.arange(0, BLK_M)
    rm = rows < end
    k_off = tl.arange(0, BLK_K)
    n_off = tl.arange(0, BLK_N)
    xs = tl.load(XS + rows, mask=rm, other=1.0).to(tl.float32)
    nchunk = tl.cdiv(HALF_N, BLK_N)

    # ---- pass 1: compute sw block by block over N, accumulate the row max of |sw| (the K axis
    #      is tiled and accumulated the same way) ----
    m1 = tl.zeros([BLK_M], dtype=tl.float32)
    for nc in range(0, nchunk):
        cn = nc * BLK_N + n_off
        nm = cn < HALF_N
        cg = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
        cu = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
        for k0 in range(0, K, BLK_K):
            kk = k0 + k_off
            km = kk < K
            xi = tl.load(X + rows[:, None] * K + kk[None, :],
                         mask=rm[:, None] & km[None, :], other=0).to(tl.float16)
            wg = tl.load(W + g * K * N + kk[:, None] * N + cn[None, :],
                         mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
            wu = tl.load(W + g * K * N + kk[:, None] * N + (cn[None, :] + HALF_N),
                         mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
            cg += tl.dot(xi, wg, out_dtype=tl.float32)
            cu += tl.dot(xi, wu, out_dtype=tl.float32)
        sg = tl.load(WS + g * N + cn, mask=nm, other=0.0).to(tl.float32)
        su = tl.load(WS + g * N + cn + HALF_N, mask=nm, other=0.0).to(tl.float32)
        gate = cg * xs[:, None] * sg[None, :]
        up = cu * xs[:, None] * su[None, :]
        if LIMIT_ON:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
        sw = (gate / (1.0 + tl.exp(-gate))) * up
        if HAS_SMOOTH_COL:
            sm = tl.load(SMOOTH + g * HALF_N + cn, mask=nm, other=1.0).to(tl.float32)
            sw = sw * sm[None, :]
        if HAS_SMOOTH_ROW:
            smr = tl.load(SMOOTH + g).to(tl.float32)
            sw = sw * smr
        sw = tl.where(nm[None, :], sw, 0.0)
        m1 = tl.maximum(m1, tl.max(tl.abs(sw), axis=1))

    mxs = tl.where(m1 > 0.0, m1, 1.0)
    s = mxs / 127.0

    # ---- pass 2: recompute sw (identical to pass 1) and quantize ----
    for nc in range(0, nchunk):
        cn = nc * BLK_N + n_off
        nm = cn < HALF_N
        cg = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
        cu = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
        for k0 in range(0, K, BLK_K):
            kk = k0 + k_off
            km = kk < K
            xi = tl.load(X + rows[:, None] * K + kk[None, :],
                         mask=rm[:, None] & km[None, :], other=0).to(tl.float16)
            wg = tl.load(W + g * K * N + kk[:, None] * N + cn[None, :],
                         mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
            wu = tl.load(W + g * K * N + kk[:, None] * N + (cn[None, :] + HALF_N),
                         mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
            cg += tl.dot(xi, wg, out_dtype=tl.float32)
            cu += tl.dot(xi, wu, out_dtype=tl.float32)
        sg = tl.load(WS + g * N + cn, mask=nm, other=0.0).to(tl.float32)
        su = tl.load(WS + g * N + cn + HALF_N, mask=nm, other=0.0).to(tl.float32)
        gate = cg * xs[:, None] * sg[None, :]
        up = cu * xs[:, None] * su[None, :]
        if LIMIT_ON:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
        sw = (gate / (1.0 + tl.exp(-gate))) * up
        if HAS_SMOOTH_COL:
            sm = tl.load(SMOOTH + g * HALF_N + cn, mask=nm, other=1.0).to(tl.float32)
            sw = sw * sm[None, :]
        if HAS_SMOOTH_ROW:
            smr = tl.load(SMOOTH + g).to(tl.float32)
            sw = sw * smr
        sw = tl.where(nm[None, :], sw, 0.0)

        q = sw * (127.0 / mxs)[:, None]
        q16 = q.to(tl.float16).to(tl.float32)
        qr = tl.minimum(tl.maximum(
            tl.where(q16 >= 0, tl.floor(q16 + 0.5), tl.ceil(q16 - 0.5)), -128.0), 127.0)
        tl.store(Y + rows[:, None] * HALF_N + cn[None, :], qr.to(tl.int8),
                 mask=rm[:, None] & nm[None, :])

    tl.store(YS + rows, s, mask=rm)


def _ends_from_group_list(group_list, e, group_list_type):
    """group_list_type 0 => cumsum (cumulative end offsets); 1 => per-group counts (the prefix
    sum is computed internally)."""
    gl = group_list.contiguous().to(torch.int32)
    assert gl.numel() == e, f"group_list length {gl.numel()} != number of groups {e}"
    if int(group_list_type) == 1:
        ends = torch.cumsum(gl.to(torch.int64), dim=0).to(torch.int32)
    else:
        ends = gl
    starts = torch.zeros_like(ends)
    starts[1:] = ends[:-1]
    return starts, ends


def grouped_matmul_swiglu_quant_v2(
    x: torch.Tensor,
    weight,
    weight_scale,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    smooth_scale: torch.Tensor | None = None,
    weight_assist_matrix=None,
    bias: torch.Tensor | None = None,
    dequant_mode=None,
    dequant_dtype=None,
    quant_mode=None,
    quant_dtype=None,
    transpose_weight: bool = False,
    group_list_type: int = 0,
    tuning_config=(0,),
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent to the AscendC GroupedMatmulSwigluQuantV2 (int8/A8W8 path).

    Returns:
        (y [M, N//2] int8, y_scale [M] fp32)
    """
    if bias is not None:
        raise NotImplementedError(
            "bias is not None: op_kernel/grouped_matmul_swiglu_quant_v2_utils.h L84 states "
            "explicitly that bias is unused and only required by the high-level template "
            "parameters, so the original AscendC implementation does not use bias. This "
            "implementation keeps the same behavior and does not support passing bias.")
    if weight_assist_matrix is not None and len(weight_assist_matrix) > 0:
        raise NotImplementedError(
            "weight_assist_matrix is not None: it is the auxiliary quantization matrix of the "
            "W4A8/A4W4 paths, while this implementation only covers the int8 (A8W8) weight path.")
    if transpose_weight:
        raise NotImplementedError("transpose_weight=True is outside the scope of this implementation.")

    assert x.dim() == 2 and x.dtype == torch.int8, "x must be [M, K] int8"
    M, K = x.shape
    E = len(weight) if isinstance(weight, (list, tuple)) else weight.shape[0]
    w = (torch.stack([t.contiguous() for t in weight], dim=0)
         if isinstance(weight, (list, tuple)) else weight.contiguous())
    assert w.dtype == torch.int8 and w.shape[1] == K, f"weight must be int8 [E,{K},N]"
    ws = (torch.stack([t.contiguous() for t in weight_scale], dim=0).to(torch.float32)
          if isinstance(weight_scale, (list, tuple)) else weight_scale.contiguous().to(torch.float32))
    N = ws.shape[-1]                       # <- matches the adapter: n is taken from weight_scale
    assert N % 2 == 0, f"N={N} must be even (gate|up)"
    I = N // 2
    assert tuple(ws.shape) == (E, N), f"weight_scale must be [{E},{N}], got {tuple(ws.shape)}"
    assert tuple(w.shape) == (E, K, N), f"weight must be [{E},{K},{N}], got {tuple(w.shape)}"

    dev = x.device
    x = x.contiguous()
    xs = x_scale.contiguous().to(torch.float32)
    assert xs.numel() == M
    starts, ends = _ends_from_group_list(group_list, E, group_list_type)

    has_col = has_row = False
    sm = x
    if smooth_scale is not None:
        smt = smooth_scale.contiguous().to(torch.float32)
        if smt.dim() == 2:
            assert tuple(smt.shape) == (E, I), f"smooth_scale must be [{E},{I}], got {tuple(smt.shape)}"
            has_col = True
        elif smt.dim() == 1:
            assert smt.numel() == E, f"smooth_scale must be [{E}], got {tuple(smt.shape)}"
            has_row = True
        else:
            raise NotImplementedError(
                f"smooth_scale with {smt.dim()} dims is not supported (only (E,N/2) or (E,))")
        sm = smt

    y = torch.empty((M, I), dtype=torch.int8, device=dev)
    y_scale = torch.empty(M, dtype=torch.float32, device=dev)

    # ---- tile selection: bounded by the UB limit (see the fitted formula at the top of the file) ----
    # For small shapes blk_k / blk_n cover the whole K / I => a single block, numerically
    # identical to the untiled version; only large shapes are actually tiled (BLK_N <= 256,
    # BLK_K <= 128).
    if K <= 0 or I <= 0:
        raise NotImplementedError(f"K={K} / I={I} is invalid (must be positive)")
    blk_m = 16
    blk_k = min(max(next_pow2(K), 16), BLK_K_CAP)
    blk_n = min(max(next_pow2(I), 16), BLK_N_CAP)
    need = ub_budget_bits(blk_m, blk_k, blk_n)
    if need > UB_LIMIT_BITS:
        # Reject explicitly instead of letting BiShengIR raise a ub overflow at compile time
        # (and never silently produce wrong values)
        raise NotImplementedError(
            f"grouped_matmul_swiglu_quant_v2: tile (BLK_M={blk_m}, BLK_K={blk_k}, BLK_N={blk_n}) "
            f"estimates a UB requirement of {need} bits > limit {UB_LIMIT_BITS} bits "
            f"(K={K}, I={I}). Please lower BLK_K_CAP/BLK_N_CAP.")
    _gmm_v2_kernel[(E, ceil_div(M, blk_m))](
        x, w, ws, xs, sm, starts, ends, y, y_scale,
        K, N, I,
        BLK_M=blk_m, BLK_K=blk_k, BLK_N=blk_n,
        HAS_SMOOTH_COL=has_col, HAS_SMOOTH_ROW=has_row,
        LIMIT_ON=bool(swiglu_limit > 0.0), SWIGLU_LIMIT=float(swiglu_limit),
    )
    return y, y_scale
