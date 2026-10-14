# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""moe_gating_top_k_hash -- AscendC to Triton

Source: csrc/ascend/moe/moe_gating_top_k_hash/

Interface (op_host/moe_gating_top_k_hash_def.cpp L22-131):
    Input : x        [N, E]     fp32 / fp16 / bf16       (REQUIRED)
            bias     [E]        same dtype as x          (OPTIONAL)
            input_ids[N]        int32 / int64            (OPTIONAL, used by the hash path)
            tid2eid  [V, >=k]   int32 / int64            (OPTIONAL, used by the hash path)
    Output: y        [N, k]     same dtype as x
            expert_idx [N, k]   int32
            out      [N, E]     fp32                      (written only when out_flag=true)
    Attr  : k(int), k_group(int=1), group_count(int=1), group_select_mode(int=0),
            renorm(int=0), norm_type(int=0), out_flag(bool=false),
            routed_scaling_factor(float=1.0), eps(float=1e-20)

norm_type values (op_kernel/common.h L41-43):
    0 = SOFTMAX, 1 = SIGMOID, 2 = SOFTPLUS

--------------------------------------------------------------------------
Formula sources
--------------------------------------------------------------------------
[1] Ungrouped main body: op_kernel/moe_gating_top_k_hash_without_group.h
    L126-196  ComputeX     -- normalization (+bias)
    L212-238  SelectTopKExpertIdx   -- non-hash: take the top-k expert ids by descending
                                       (xNorm+bias)
    L294-313  SelectExpertIdxByHash -- hash: expert ids = tid2eid[input_ids[row], :k]
    L241-278  SelectTopKExpertScore -- y = xNorm[expert_idx]; sigmoid/softplus then renormalize;
                                       *rsf
    L199-209  CopuOutXNorm -- out = xNorm (**bias not included**)
    L366-383  Process      -- per row: normalize -> (hash ? table lookup : topk) -> gather the
                              score -> write back

[2] Grouped main body (group_count>1): op_kernel/moe_gating_top_k_hash_generalized.h
    L152-224  ComputeX     -- as above, grouped by perGroupExpertCount + -inf padding at the tail
                              of each group
    L242-260  SortInGroup  -- descending sort inside each group
    L263-367  SelectTopKGroupIndex -- group score = sum of the top2 inside the group (mode=1) or
                                      top1 (mode=0); take the first k_group groups by descending
                                      group score
    L370-514  SelectTopKExpertIdx  -- merge the sorted tables of the selected groups and take the
                                      global top-k
    L517-554  SelectTopKExpertScore-- same as [1]
    L557-574  CumputeActualTopKExpertId -- restore the global expert ids

[3] High-performance specialization (tiling_key=0): op_kernel/moe_gating_top_k_hash_e_k_fullload.h
    L125-144 / L147-157 / L160-232 / L235-273 / L276-317
    -- enabled for E=256,G=8,k_group=4,k<=32,with bias,group_select_mode=SUM,renorm=0,
       norm_type=SIGMOID,out_flag=false (op_host/..._tiling.cpp L576-580).
       Mathematically equivalent to [2] (group score = sum of the top2 in the group; the Sort32
       width 32 equals P), except that it assumes E%32==0 and P==32 and needs no -inf padding.

tiling constraints (op_host/moe_gating_top_k_hash_tiling.cpp L206-271):
    E <= 2048; E % group_count == 0; softplus only supports group_count==1;
    renorm only supports 0; group_select_mode in {0:max, 1:sum}; k <= k_group*perGroupCount.

--------------------------------------------------------------------------
Precision notes (important)
--------------------------------------------------------------------------
* Normalization is done in **fp32** (the AscendC kernel casts T to float first), and compared
  against an fp64 reference.
* y is taken from **xNorm (bias not added)**; bias only takes part in the top-k selection
  ([1] L180 vs L250).
* sigmoid / softplus are followed by one **division by (sum(y)+eps)** renormalization; softmax is
  not ([1] L253-268, [2] L529-544).
* The result is finally multiplied by routed_scaling_factor and written out after CAST_RINT back
  to T.
* Expert order for ties (equal scores): AscendC uses the hardware Sort32, which leaves it
  undefined; this implementation prefers **the smaller expert id on equal scores**
  (deterministic), and the tests cover it with distinguishable data.
* AscendC fills padding with -inf through `FLOAT32_NEG_INF`; the manual explicitly forbids the
  `float("-inf")` literal on Ascend, so this implementation uses a finite large negative `_NEG`
  as the sentinel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import next_pow2

__all__ = ["moe_gating_top_k_hash"]

NORM_SOFTMAX = 0
NORM_SIGMOID = 1
NORM_SOFTPLUS = 2

GROUP_SELECT_MODE_MAX = 0
GROUP_SELECT_MODE_SUM = 1

# Sentinel: no -inf literal (it degrades on Ascend); use a finite large negative value
_NEG = -1.0e30
# int32 index sentinel
_IDX_BIG = 1 << 30


@triton.jit
def _normalize_flat(XV, EMASK, NORM_TYPE: tl.constexpr, NEG: tl.constexpr):
    """Normalization of one [BLK] row (fp32), corresponding to ComputeX."""
    if NORM_TYPE == 0:  # softmax
        mx = tl.max(tl.where(EMASK, XV, NEG), axis=0)
        ev = tl.where(EMASK, tl.exp(XV - mx), 0.0)
        sm = tl.sum(ev, axis=0)
        return ev * (1.0 / sm)
    elif NORM_TYPE == 1:  # sigmoid
        return 1.0 / (1.0 + tl.exp(-XV))
    else:  # softplus: sqrt(ln(1+exp(x)))
        return tl.sqrt(tl.log(1.0 + tl.exp(XV)))


@triton.jit
def _apply_norm_to_selected(XSEL, MX, SM, NORM_TYPE: tl.constexpr):
    """Apply the same normalization to the k raw values that were gathered.

    For softmax the whole-row max/sum is reused (MX/SM are only valid for softmax).
    """
    if NORM_TYPE == 0:
        return tl.exp(XSEL - MX) * (1.0 / SM)
    elif NORM_TYPE == 1:
        return 1.0 / (1.0 + tl.exp(-XSEL))
    else:
        return tl.sqrt(tl.log(1.0 + tl.exp(XSEL)))


@triton.jit
def _gating_flat_kernel(
    X, BIAS, INPUT_IDS, TID2EID, Y, EXPERT_IDX, OUT,
    E, K, RSF, EPS,
    NORM_TYPE: tl.constexpr,
    OUT_FLAG: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HASH: tl.constexpr,
    K_C: tl.constexpr,
    BLK_E: tl.constexpr,
    BLK_K: tl.constexpr,
    NEG: tl.constexpr,
    E_BIG: tl.constexpr,
):
    """Ungrouped path (group_count==1): corresponds to the per-row loop in without_group.h
    L366-383."""
    row = tl.program_id(0)
    e_off = tl.arange(0, BLK_E)
    k_off = tl.arange(0, BLK_K)
    emask = e_off < E
    kmask = k_off < K

    xv = tl.load(X + row * E + e_off, mask=emask, other=0.0).to(tl.float32)
    xn = _normalize_flat(xv, emask, NORM_TYPE, NEG)

    mx = 0.0
    sm = 1.0
    if NORM_TYPE == 0:
        mx = tl.max(tl.where(emask, xv, NEG), axis=0)
        ev = tl.where(emask, tl.exp(xv - mx), 0.0)
        sm = tl.sum(ev, axis=0)

    if OUT_FLAG:
        tl.store(OUT + row * E + e_off, xn, mask=emask)

    if HASH:
        # without_group.h L294-313: expertIdx = tid2eid[inputIds[row], :k]
        key = tl.load(INPUT_IDS + row).to(tl.int64)
        sel = tl.load(TID2EID + key * K + k_off.to(tl.int64), mask=kmask, other=0).to(tl.int32)
    else:
        # without_group.h L212-238 + L179-183: take the top-k by descending (xNorm + bias)
        scores = xn
        if HAS_BIAS:
            scores = scores + tl.load(BIAS + e_off, mask=emask, other=0.0).to(tl.float32)
        scores = tl.where(emask, scores, NEG)
        # repeated argmax (smallest expert id on ties)
        sel = tl.zeros([BLK_K], dtype=tl.int32)
        for i in tl.static_range(K_C):
            mxi = tl.max(scores, axis=0)
            idx = tl.min(tl.where(scores == mxi, e_off, E_BIG), axis=0)
            sel = tl.where(k_off == i, idx, sel)
            scores = tl.where(e_off == idx, NEG, scores)

    # y = xNorm[expertIdx] (note that the xNorm without bias is used)
    xsel = tl.load(X + row * E + sel, mask=kmask, other=0.0).to(tl.float32)
    yv = _apply_norm_to_selected(xsel, mx, sm, NORM_TYPE)

    # without_group.h L253-268: sigmoid / softplus additionally apply one sum renormalization
    if NORM_TYPE == 1 or NORM_TYPE == 2:
        s = tl.sum(tl.where(kmask, yv, 0.0), axis=0) + EPS
        yv = yv / s
    yv = yv * RSF

    tl.store(Y + row * K + k_off, yv.to(Y.dtype.element_ty), mask=kmask)
    tl.store(EXPERT_IDX + row * K + k_off, sel, mask=kmask)


@triton.jit
def _gating_group_kernel(
    X, BIAS, Y, EXPERT_IDX, OUT,
    E, K, RSF, EPS, P,
    NORM_TYPE: tl.constexpr,
    OUT_FLAG: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GROUP_SELECT_MODE: tl.constexpr,
    K_GROUP: tl.constexpr,
    K_C: tl.constexpr,
    G_C: tl.constexpr,
    BLK_E: tl.constexpr,
    BLK_G: tl.constexpr,
    BLK_K: tl.constexpr,
    NEG: tl.constexpr,
    E_BIG: tl.constexpr,
):
    """Grouped path: corresponds to generalized.h L645-662 (EKFullload has the same math).

    Implementation note: everything is deliberately expressed with **1-D tensors plus a
    compile-time group loop**.  The Ascend Triton backend hits the LLVM assertion
    `PointerUnion::get: Invalid accessor called` (a crash) on 2-D reductions such as
    `tl.max(x, axis=1)`, so every 2-D reduce is avoided here.
    The price is O(G*E) for the group scores; G is passed as a constexpr, and G <= 128 is
    recommended.
    """
    row = tl.program_id(0)
    e_off = tl.arange(0, BLK_E)
    g_off = tl.arange(0, BLK_G)
    k_off = tl.arange(0, BLK_K)
    emask = e_off < E

    xv = tl.load(X + row * E + e_off, mask=emask, other=0.0).to(tl.float32)
    mx = 0.0
    sm = 1.0
    if NORM_TYPE == 0:  # softmax (normalizes over all experts of the row)
        mx = tl.max(tl.where(emask, xv, NEG), axis=0)
        ev = tl.where(emask, tl.exp(xv - mx), 0.0)
        sm = tl.sum(ev, axis=0)
        xn = ev * (1.0 / sm)
    elif NORM_TYPE == 1:
        xn = 1.0 / (1.0 + tl.exp(-xv))
    else:
        xn = tl.sqrt(tl.log(1.0 + tl.exp(xv)))

    if OUT_FLAG:
        tl.store(OUT + row * E + e_off, xn, mask=emask)

    scores = xn
    if HAS_BIAS:
        scores = scores + tl.load(BIAS + e_off, mask=emask, other=0.0).to(tl.float32)
    scores = tl.where(emask, scores, NEG)

    # --- group score: sum of the top2 in the group (mode=1) or top1 (mode=0)
    #     [generalized.h L284-321]
    gscore = tl.full([BLK_G], NEG, dtype=tl.float32)
    # This must be a **runtime** loop: `tl.static_range(G_C)` fully unrolls the group loop
    # (G can reach 128 in production) and every unrolled body needs its own UB buffer, which
    # overflows with `ub overflow, requires 7013888 bits` (4.5x the 192KB available).
    # With `range` the loop becomes an scf.for and UB is accounted per single iteration.
    for g in range(0, G_C):
        gm = (e_off >= g * P) & (e_off < (g + 1) * P)
        sv = tl.where(gm, scores, NEG)
        v1 = tl.max(sv, axis=0)
        i1 = tl.min(tl.where(sv == v1, e_off, E_BIG), axis=0)
        v2 = tl.max(tl.where(e_off == i1, NEG, sv), axis=0)
        if GROUP_SELECT_MODE == 1:
            gsv = v1 + v2
        else:
            gsv = v1
        gscore = tl.where(g_off == g, gsv, gscore)

    # --- take the first K_GROUP groups by descending group score and merge every expert of the
    #     selected groups into the candidate set
    #     [generalized.h L343-366 group selection + L370-514 merging of the selected tables]
    #     Note: merging the sorted tables of k_group groups and taking the top-k is equivalent to
    #     taking the global top-k over the experts covered by those groups.  Group selection and
    #     candidate collection are fused into one loop here, to avoid extracting a scalar from an
    #     int32 vector with tl.sum (that form is miscompiled into the last element).
    cand = tl.full([BLK_E], NEG, dtype=tl.float32)
    gs = gscore
    # Also a runtime loop: unrolling with `tl.static_range` makes every unrolled body take its
    # own UB (`ub overflow, requires 2819584 bits` is still reported when K_GROUP/K_C unroll).
    for i in range(0, K_GROUP):
        gmx = tl.max(gs, axis=0)
        gi = tl.min(tl.where(gs == gmx, g_off, BLK_G), axis=0)
        cand = tl.where((e_off >= gi * P) & (e_off < (gi + 1) * P), scores, cand)
        gs = tl.where(g_off == gi, NEG, gs)

    # --- merge and take the global top-k  [generalized.h L370-514]
    sel = tl.zeros([BLK_K], dtype=tl.int32)
    for i in range(0, K_C):
        mxi = tl.max(cand, axis=0)
        idx = tl.min(tl.where(cand == mxi, e_off, E_BIG), axis=0)
        sel = tl.where(k_off == i, idx, sel)
        cand = tl.where(e_off == idx, NEG, cand)

    # --- y = xNorm[sel], renormalized once more for sigmoid  [generalized.h L517-554]
    xsel = tl.load(X + row * E + sel, mask=k_off < K, other=0.0).to(tl.float32)
    yv = _apply_norm_to_selected(xsel, mx, sm, NORM_TYPE)
    if NORM_TYPE == 1:
        s = tl.sum(tl.where(k_off < K, yv, 0.0), axis=0) + EPS
        yv = yv / s
    yv = yv * RSF

    tl.store(Y + row * K + k_off, yv.to(Y.dtype.element_ty), mask=k_off < K)
    tl.store(EXPERT_IDX + row * K + k_off, sel, mask=k_off < K)


def moe_gating_top_k_hash(
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
    k: int = 1,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = GROUP_SELECT_MODE_MAX,
    renorm: int = 0,
    norm_type: int = NORM_SOFTMAX,
    out_flag: bool = False,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equivalent to the AscendC MoeGatingTopKHash.

    Args:
        x:        [N, E]  fp32/fp16/bf16
        bias:     [E] or None, same dtype as x
        input_ids:[N] int32/int64, required in hash mode
        tid2eid:  [V, >=k] int32/int64, required in hash mode
        the remaining arguments are operator attributes.

    Returns:
        (y [N,k] with the dtype of x, expert_idx [N,k] int32, out [N,E] fp32)
        With out_flag=False, out is all zeros (the AscendC kernel does not write that output).
    """
    assert x.dim() == 2, f"x must be [N,E], got {tuple(x.shape)}"
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), x.dtype
    assert renorm == 0, "AscendC tiling only supports renorm=0"
    N, E = x.shape
    assert 0 < k <= E, f"k={k} must satisfy 0<k<=E={E}"
    assert group_count >= 1 and E % group_count == 0, "E must be divisible by group_count"
    assert 1 <= k_group <= group_count, f"k_group={k_group} must be in [1, {group_count}]"
    if norm_type == NORM_SOFTPLUS:
        assert group_count == 1, "softplus only supports group_count==1"
    assert norm_type in (NORM_SOFTMAX, NORM_SIGMOID, NORM_SOFTPLUS), norm_type
    assert group_select_mode in (GROUP_SELECT_MODE_MAX, GROUP_SELECT_MODE_SUM), group_select_mode

    P = E // group_count
    HASH = input_ids is not None
    assert HASH == (tid2eid is not None), "input_ids / tid2eid must both be given or both be None"

    device = x.device
    x = x.contiguous()
    y = torch.empty((N, k), dtype=x.dtype, device=device)
    expert_idx = torch.empty((N, k), dtype=torch.int32, device=device)
    # With out_flag=False the AscendC kernel does not write this output; zero it for determinism
    out = torch.zeros((N, E), dtype=torch.float32, device=device)

    has_bias = bias is not None
    if has_bias:
        assert bias.numel() == E, f"bias length {bias.numel()} != E={E}"
        bias_t = bias.contiguous()
    else:
        bias_t = x  # placeholder pointer; not loaded when HAS_BIAS=False

    if HASH:
        input_ids_t = input_ids.contiguous()
        tid2eid_t = tid2eid.contiguous()
        assert tid2eid_t.shape[-1] >= k, f"tid2eid last dim {tid2eid_t.shape[-1]} < k={k}"
    else:
        input_ids_t = tid2eid_t = x

    if group_count == 1:
        _gating_flat_kernel[(N,)](
            x, bias_t, input_ids_t, tid2eid_t, y, expert_idx, out,
            E, k, float(routed_scaling_factor), float(eps),
            NORM_TYPE=int(norm_type),
            OUT_FLAG=bool(out_flag),
            HAS_BIAS=bool(has_bias),
            HASH=bool(HASH),
            K_C=int(k),
            BLK_E=max(next_pow2(E), 2),
            BLK_K=max(next_pow2(k), 2),
            NEG=_NEG,
            E_BIG=_IDX_BIG,
        )
    else:
        assert group_count <= 128, "the grouped path needs group_count as a constexpr, so <= 128"
        _gating_group_kernel[(N,)](
            x, bias_t, y, expert_idx, out,
            E, k, float(routed_scaling_factor), float(eps), P,
            NORM_TYPE=int(norm_type),
            OUT_FLAG=bool(out_flag),
            HAS_BIAS=bool(has_bias),
            GROUP_SELECT_MODE=int(group_select_mode),
            K_GROUP=int(k_group),
            K_C=int(k),
            G_C=int(group_count),
            BLK_E=max(next_pow2(E), 2),
            BLK_G=max(next_pow2(group_count), 2),
            BLK_K=max(next_pow2(k), 2),
            NEG=_NEG,
            E_BIG=_IDX_BIG,
        )
    return y, expert_idx, out
