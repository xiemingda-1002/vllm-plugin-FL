# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""grouped_matmul_swiglu_quant_weight_nz_tensor_list -- AscendC to Triton

Source: `csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/`

--------------------------------------------------------------------------
Call sites and alias relationships (verified against the source)
--------------------------------------------------------------------------
`grouped_matmul_swiglu_quant_torch_adpt.h`
  L20-51  `grouped_matmul_swiglu_quant_weight_nz(x, weight, weight_scale, x_scale,
                                                 group_list, bias, offset, swiglu_limit)`
          -> `aclnnGroupedMatmulSwigluQuantWeightNZ` (**single tensor** version: weight is an
             at::Tensor)
  L53-60  `grouped_matmul_swiglu_quant(...)` **returns the function above directly** => a **pure
          alias**
  L62-99  `grouped_matmul_swiglu_quant_weight_nz_tensor_list(x, weight[], weight_scale[],
                                                             x_scale, group_list, bias, offset,
                                                             swiglu_limit)`
          -> `aclnnGroupedMatmulSwigluQuantWeightNzTensorList` (**TensorList** version)

In addition, `csrc/gmm/` contains **only 4 operator directories** (`gmm_dequant_situ_quant`,
`grouped_matmul_swiglu_quant`, `grouped_matmul_swiglu_quant_v2`,
`grouped_matmul_swiglu_quant_weight_nz_tensor_list`) and **no `*_weight_nz` directory**; the only
OpDef is `class GroupedMatmulSwigluQuantWeightNzTensorList`
(`op_host/..._def.cpp` L19 / `OP_ADD` L65).

=> **Conclusion**: `grouped_matmul_swiglu_quant` and `grouped_matmul_swiglu_quant_weight_nz`
have **no standalone AscendC kernel source in this repository** (they go through the aclnn
provided by CANN/OPP); only the `_tensor_list` variant has an AscendC implementation here.  This
file therefore implements the math of `_tensor_list` and exports `grouped_matmul_swiglu_quant` /
`..._weight_nz` as **aliases of the same math** (thin wrappers around `..._tensor_list` that turn
a single weight tensor into a list internally).

--------------------------------------------------------------------------
Interface (op_host/grouped_matmul_swiglu_quant_weight_nz_tensor_list_def.cpp L23-51)
--------------------------------------------------------------------------
    Input : x            [M, K]            int8
            weight       DYNAMIC TensorList[G], FORMAT_FRACTAL_NZ, int8
            weight_scale DYNAMIC TensorList[G], fp32 / bf16 / fp16
            x_scale      [M]               fp32
            group_list   [G]               int64
    Output: y        [M, N/2]  int8
            y_scale  [M]       fp32
            (the adapter also returns output_offset: a scalar fp32, L34/L79)
    Attr  : swiglu_limit (REQUIRED, float, default 0.0)

Shape inference in the adapter (L72-79): `n = weight[0].sizes()[1]`, `m = x.sizes()[0]`,
`k = x.sizes()[1]`; `output = [m, n/2] int8`, `output_scale = [m] fp32`.

--------------------------------------------------------------------------
Math (SwiGLU + per-token dynamic quantization; same lineage as the mc2 epilogue in this repo)
--------------------------------------------------------------------------
For the row range [start_g, end_g) covered by group g:
    C   = (x[start:end] . x_scale[:, None]) @ (W_g . weight_scale_g[None, :])   # [rows, N]
        (x/W are int8 with fp32 accumulation; x_scale is per-token and weight_scale is per-N
         channel)
    gate = C[:, :N/2]; up = C[:, N/2:]
    sw  = (gate * sigmoid(gate)) * up          # clamp(gate, max=limit) first when swiglu_limit>0
    mx  = max|sw|  (per row)
    y   = rint(sw * 127 / mx)  -> int8          # per-token dynamic quantization
    y_scale = mx / 127
Formula source: `csrc/ascend/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/
block_epilogue_pertoken_swiglu.hpp` L211-265 (same SwiGLU + per-token quantization epilogue
family).

`group_list` semantics: this implementation interprets it as **cumulative end offsets**
(`group_list[g]` = the row following the last row of group g, i.e. the cumsum form), consistent
with the MoE `expert_tokens_count_or_cumsum` (CUMSUM) convention used in this repository; the
start of a group is derived from the end of the previous group (group 0 starts at 0).  This is
explicitly marked as a **semantic assumption** in this file and in the tests.

--------------------------------------------------------------------------
Parameter coverage (nothing is silently ignored)
--------------------------------------------------------------------------
x / weight / weight_scale / x_scale / group_list -- used
swiglu_limit -- used (clamps gate when > 0)
bias / offset    -- adapter L41-42/L86-87 passes them to aclnn, but **def.cpp has no bias/offset
                    parameter**, so this implementation does not support them: a non-empty
                    argument raises NotImplementedError (it is not silently ignored).
--------------------------------------------------------------------------
Implementation notes (deliberately avoiding known silent Ascend backend errors)
--------------------------------------------------------------------------
* **No runtime range loops, and nothing more complex than 2-D reduces**: grid =
  (G, row_blocks, col_blocks), where each program only does load -> tl.dot -> elementwise ->
  store.
* Known pitfalls avoided: no "scalar mask & vector mask" broadcast; no literals such as `1e-30`
  written into the kernel (`EPS` is a runtime scalar parameter); no reliance on loop-carried
  tensors.
* The logical shape of `weight` is treated as [K, N] (the AscendC side works on the FRACTAL_NZ
  fractal format; the Triton implementation uses the logically equivalent [K, N] row-major
  layout, as documented in the tests).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = [
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list",
    "grouped_matmul_swiglu_quant_weight_nz",
    "grouped_matmul_swiglu_quant",
]


@triton.jit
def _gmm_swiglu_quant_kernel(
    X, W, WS, XS, GSTART, GEND, Y, YS, YOFF,
    K, N, HALF_N, EPS,
    BLK_M: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    LIMIT_ON: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    """One program handles (group g, row block, output column block 0..HALF_N).

    SwiGLU alignment requirement: gate takes columns [0, HALF_N) of W and up takes columns
    [HALF_N, 2*HALF_N), both expressed through the **same output column index n_off**, so they
    are aligned by construction.
    The incorrect form `gate = where(n<I, c, 0)` + `up = where(n>=I, c, 0)` acts on **disjoint**
    column ranges, making the product identically 0 -- this was the root cause of the earlier
    three operators whose out was exactly 0 (it is not a tl.dot issue).
    """
    g = tl.program_id(0)
    start = tl.load(GSTART + g)
    end = tl.load(GEND + g)
    rows = start + tl.program_id(1) * BLK_M + tl.arange(0, BLK_M)
    k_off = tl.arange(0, BLK_K)
    n_off = tl.arange(0, BLK_N)          # output column index
    rm = rows < end
    km = k_off < K
    nm = n_off < HALF_N

    xi = tl.load(X + rows[:, None] * K + k_off[None, :],
                 mask=rm[:, None] & km[None, :], other=0).to(tl.float16)
    wg = tl.load(W + g * K * N + k_off[:, None] * N + n_off[None, :],
                 mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
    wu = tl.load(W + g * K * N + k_off[:, None] * N + (n_off[None, :] + HALF_N),
                 mask=km[:, None] & nm[None, :], other=0).to(tl.float16)
    cg = tl.dot(xi, wg, out_dtype=tl.float32)
    cu = tl.dot(xi, wu, out_dtype=tl.float32)

    xs = tl.load(XS + rows, mask=rm, other=1.0).to(tl.float32)
    sg = tl.load(WS + g * N + n_off, mask=nm, other=0.0).to(tl.float32)
    su = tl.load(WS + g * N + n_off + HALF_N, mask=nm, other=0.0).to(tl.float32)
    gate = cg * xs[:, None] * sg[None, :]
    up = cu * xs[:, None] * su[None, :]

    if LIMIT_ON:
        gate = tl.minimum(gate, SWIGLU_LIMIT)
    sw = (gate / (1.0 + tl.exp(-gate))) * up
    sw = tl.where(nm[None, :], sw, 0.0)

    mx = tl.max(tl.abs(sw), axis=1)
    mxs = tl.maximum(mx, EPS)
    s = mxs / 127.0
    q = sw * (127.0 / mxs)[:, None]
    q16 = q.to(tl.float16).to(tl.float32)
    qr = tl.minimum(tl.maximum(
        tl.where(q16 >= 0, tl.floor(q16 + 0.5), tl.ceil(q16 - 0.5)), -128.0), 127.0)

    tl.store(Y + rows[:, None] * HALF_N + n_off[None, :], qr.to(tl.int8),
             mask=rm[:, None] & nm[None, :])
    tl.store(YS + rows, s, mask=rm)
    if HAS_BIAS:
        tl.store(YOFF + tl.arange(0, 1), tl.sum(tl.where(rm, s, 0.0), axis=0))


def _split_groups(group_list: torch.Tensor, n_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert group_list into (start, end).  group_list is assumed to hold **cumulative end
    offsets** (cumsum)."""
    ends = group_list.contiguous().to(torch.int32)
    assert ends.numel() == n_groups, f"group_list length {ends.numel()} != groups {n_groups}"
    starts = torch.zeros_like(ends)
    starts[1:] = ends[:-1]
    return starts, ends


def _impl(
    x, weight_list, weight_scale_list, x_scale, group_list, bias, offset, swiglu_limit,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.dtype == torch.int8, "x must be [M, K] int8"
    if bias is not None:
        raise NotImplementedError(
            "non-empty bias: def.cpp of grouped_matmul_swiglu_quant_weight_nz_tensor_list "
            "(L23-51) has no bias parameter and the original AscendC implementation does not "
            "use bias; this implementation keeps that behaviour and does not accept bias.")
    if offset is not None:
        raise NotImplementedError(
            "non-empty offset: def.cpp has no offset parameter and the original AscendC "
            "implementation does not use it; this implementation does not support it.")

    M, K = x.shape
    if isinstance(weight_list, (list, tuple)):
        G = len(weight_list)
        w = torch.stack([t.contiguous() for t in weight_list], dim=0)
    else:
        assert weight_list.dim() == 3, "a single-tensor weight must be [G, K, N]"
        G = weight_list.shape[0]
        w = weight_list.contiguous()
    assert w.dtype == torch.int8 and w.shape[1] == K, f"weight must be int8 [G,{K},N]"
    N = w.shape[2]
    assert N % 2 == 0, f"N={N} must be even (gate||up)"
    I = N // 2
    if isinstance(weight_scale_list, (list, tuple)):
        assert len(weight_scale_list) == G
        ws = torch.stack([t.contiguous() for t in weight_scale_list], dim=0).to(torch.float32)
    else:
        ws = weight_scale_list.contiguous().to(torch.float32)
    assert tuple(ws.shape) == (G, N), f"weight_scale must be [{G},{N}], got {tuple(ws.shape)}"

    dev = x.device
    x = x.contiguous()
    xs = x_scale.contiguous().to(torch.float32)
    assert xs.numel() == M, f"x_scale length {xs.numel()} != M={M}"
    starts, ends = _split_groups(group_list, G)
    assert int(ends[-1].item()) <= M, "the last entry of group_list must not exceed M"

    y = torch.empty((M, I), dtype=torch.int8, device=dev)
    y_scale = torch.empty(M, dtype=torch.float32, device=dev)
    y_off = torch.zeros(1, dtype=torch.float32, device=dev)

    blk_m = 16
    blk_k = max(next_pow2(K), 16)
    blk_n = max(next_pow2(I), 16)      # the output columns only need to cover HALF_N = I
    grid = (G, ceil_div(M, blk_m))
    _gmm_swiglu_quant_kernel[grid](
        x, w, ws, xs, starts, ends, y, y_scale, y_off,
        K, N, I, 1e-30,
        BLK_M=blk_m, BLK_K=blk_k, BLK_N=blk_n,
        HAS_BIAS=True,
        LIMIT_ON=bool(swiglu_limit > 0.0), SWIGLU_LIMIT=float(swiglu_limit),
    )
    return y, y_scale, y_off[0]


def grouped_matmul_swiglu_quant_weight_nz_tensor_list(
    x: torch.Tensor,
    weight,
    weight_scale,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    bias=None,
    offset=None,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TensorList version: one int8 weight tensor per expert.  Returns (y, y_scale, y_offset)."""
    return _impl(x, weight, weight_scale, x_scale, group_list, bias, offset, swiglu_limit)


def grouped_matmul_swiglu_quant_weight_nz(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    bias=None,
    offset=None,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-tensor version (adapter L20-51).  **This repository has no standalone AscendC
    kernel for it** (see the module docstring); the math matches the TensorList version, so the
    same implementation is reused.  weight must be [G, K, N] (an unstacked 2-D tensor is treated
    as [1, K, N])."""
    if weight.dim() == 2:
        weight = weight.unsqueeze(0)
        weight_scale = weight_scale.reshape(1, -1)
    return _impl(x, weight, weight_scale, x_scale, group_list, bias, offset, swiglu_limit)


def grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    bias=None,
    offset=None,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """**Pure alias**: adapter L53-60 directly does
    `return grouped_matmul_swiglu_quant_weight_nz(...)`."""
    return grouped_matmul_swiglu_quant_weight_nz(
        x, weight, weight_scale, x_scale, group_list, bias, offset, swiglu_limit)
