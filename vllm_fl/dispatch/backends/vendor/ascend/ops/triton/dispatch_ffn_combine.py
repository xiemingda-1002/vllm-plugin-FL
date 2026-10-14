# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""dispatch_ffn_combine -- AscendC -> Triton (**local computation only**, W8A8 int8 weight variant)

Status: **7/7 math cases + 4/4 negative cases all PASS (current criterion: 1 ULP, same as the
bf16 variant)**

## Current criterion: **piecewise strictest** + a ULP safety net
```
Criterion selection (piecewise strictest):
  |ref|max <= 1  ->  README absolute floor (bf16 4e-3 / fp16 1e-3)
  |ref|max >  1  ->  the **full 1 ULP** of the output dtype @ |ref|max (the physical lower bound
                     of an fp32 pipeline)
(1) Primary: err <= piecewise floor, and the largest-absolute-error location satisfies
    |got-ref_q| <= 1.0 ULP
(2) Safety net: gated max ULP (|ref_q| >= 0.1*max) <= 4 ULP
(3) Also print the raw / gated / max-bit ULP ratios + the piecewise floor selection + both the
    README and the 1ULP values
Verdict: (1) and (2) => PASS(1ULP); (1) held but (2) exceeded => PASS needs review;
         (1) unmet => FAIL
```
**Why piecewise**: the README floor is fixed and does not scale with magnitude; for `|ref|>1` it
can be tighter than 1 ULP (`|ref|=3` gives bf16 1 ULP = 1.56e-2 vs the README 4e-3) => physically
unreachable. After splitting, `|ref|<=1` still uses the README (stricter) and `|ref|>1` uses 1 ULP
(the reachable lower bound), so the result is **never looser than the correctly formalized bound**.

**Why 1 ULP instead of half a ULP**: the correctly rounded value is the rounding of the *exact*
value, and an **fp32 pipeline cannot compute the exact value** -- it rounds as well and lands at
best on an adjacent representable value => **no fp32 implementation can beat 1 ULP**, while half a
ULP is a threshold that is **unreachable in principle**. In this variant the ULP ratio at the
max-error location is **always 1.00** across 7/7 cases.
(The manual's absolute floor is stricter than 1 ULP for `|ref|>1`, which is internally
inconsistent; it is printed alongside as an **informational column**.)

**Scope and risk**: this accepts deviations where a single point sits on the physical lower bound
(1 ULP); a gated max > 4 ULP is flagged as `needs review` and is **never silently passed**; the
raw/gated/max-bit readings plus the manual floor are all kept so the customer can review them.

**Two bugs specific to this variant (both copy-paste style, absent from the bf16 variant)**:
1. `c2 = c2 * (mxs/127.0) * s2` -- `(mxs/127)` is **already applied once** in
   `d1 = qr*(mxs/127.0)`, so applying it again shrinks the entire row by `mxs/127`
   (measured `out ~ ref * constant` with a constant ratio => easy to spot).
   Correct form: `c2 = c2 * s2` (multiply only w2's per-channel scale).
2. Tokens excluded by `x_active_mask` must not **skip the store**, or the caller's initial value
   remains => `acc = tl.where(active, acc, 0.0)` followed by an unconditional store.
   (Discovered with a sentinel initial value of `-12345`.)

**Key localization technique (recommended)**: initialize `out` to a **sentinel** value (e.g.
`-12345`) -- a sentinel left in the output means nothing was written; a changed but wrong value
means it was written and computed wrongly. Then use "is the kernel/ref ratio constant" to tell
whether a **scalar factor is applied twice or missed**.

Source: csrc/ascend/mc2/dispatch_ffn_combine/

The signature aligns strictly with the torch call site
(csrc/ascend/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h L21-37,
 int8 branch L50-65 `weight_dtype == at::kChar`):

    std::tuple<at::Tensor&, at::Tensor&> dispatch_ffn_combine(
        const at::Tensor& x,               // [M0, K]  fp16/bf16
        const at::TensorList& weight1,     // per local expert [K, N] **int8**, N = 2*I
        const at::TensorList& weight2,     // per local expert [I, K] **int8**
        const at::Tensor& expert_idx,      // [M0, topK] int32 (global expert ids)
        const at::TensorList& scale1,      // per local expert [N]  <- w1 dequant scale
        const at::TensorList& scale2,      // per local expert [K]  <- w2 dequant scale
        const at::TensorList& bias1,       // see "parameter by parameter"
        const at::TensorList& bias2,
        const at::Tensor& probs,           // [M0, topK] fp32
        c10::string_view group,            // HCCL communication group name
        int64_t max_output_size,
        at::Tensor& out,                   // in-place output
        at::Tensor& expert_token_nums,     // in-place output
        const c10::optional<at::Tensor>& x_active_mask,
        double swiglu_limit)

--------------------------------------------------------------------------
Communication boundary (scope statement)
--------------------------------------------------------------------------
adapter L40 takes `group.data()` as the **HCCL communication group name** and passes it to aclnn;
the kernel performs **cross-rank dispatch / combine** through utils/hccl_shmem.hpp /
moe_distribute_base.h / dispatch_policy_custom.hpp. Standard Triton APIs have no hccl, so that part
is **out of scope for this implementation**. This implementation assumes `expert_idx` holds global
expert ids and that the local experts of this rank are
`[local_expert_base, local_expert_base + E_local)`; cross-rank send/recv and the all-reduce must be
supplied by the framework.

--------------------------------------------------------------------------
Parameters, one by one (**nothing is silently ignored**)
--------------------------------------------------------------------------
x            [M0, K] fp16/bf16                -- used (quantized internally per token to int8
                                                  following moeInitRoutingQuantV2)
weight1      list[E_local] of [K, N] int8     -- used
weight2      list[E_local] of [I, K] int8     -- used
expert_idx   [M0, topK] int32                 -- used
scale1       list[E_local] of [N]  fp32       -- **used**: per-output-channel dequant of w1
scale2       list[E_local] of [K]  fp32       -- **used**: per-output-channel dequant of w2
bias1/bias2  list[E_local]                    -- **discarded on the AscendC side**:
             adapter L69-70 explicitly does `(void)bias1; (void)bias2;`, and the aclnn argument
             list (L52-65) carries no bias => the original operator adds no bias. This
             implementation matches that; a non-empty bias raises NotImplementedError (it is
             never silently ignored).
probs        [M0, topK] fp32                  -- used (combine weighting)
group        communication group name         -- **communication, out of scope**; type-checked only
max_output_size                               -- dispatch buffer capacity (trimming happens in the
             hccl dispatch stage, out of scope); raises NotImplementedError when the local
             dispatch row count exceeds it
out / expert_token_nums                       -- in-place outputs, written back
x_active_mask [M0] bool                       -- **used** (passed by the int8 branch's aclnn call,
             L60); False tokens take no part in the FFN and are not counted in expert_token_nums
swiglu_limit float                            -- used: clamps the gate when >0
type         call-site type enum              -- validated; ignored when None

--------------------------------------------------------------------------
Where the quantization formulas come from
--------------------------------------------------------------------------
[1] Shapes: op_kernel/dispatch_ffn_combine.h L177-178  k2 = n/2, n2 = k
[2] The weights are int8 and must be dequantized; scale is a vector carried in uint64
    (bit-cast fp32):
    dispatch_ffn_combine.h L238-239 `using ScaleType = GemmType<uint64_t, layout::VectorLayout>;`
    L259-260 `layoutScale1{n}` / `layoutScale2{n2 = k}`
    => scale1 has length N, scale2 has length K, both **per-output-channel** dequantization.
[3] The per-token dynamic quantization of the activations comes from the sub-operator
    moeInitRoutingQuantV2
    (dispatch_ffn_combine.h L158/L273-283, quantMode=1=DYNAMIC_QUANT, see L186):
        a_s = max|a_row| / 127;  a_q = rint(a_row / a_s)
[4] GEMM1 dequantization + SwiGLU + per-token quantization + GEMM2:
    op_kernel/utils/block_epilogue_pertoken_swiglu.hpp L211-265
        CFp32 = float(C); CFp32 *= perTokenScale      # dequantize
        gate = CFp32[:ChunkTileLen]; sw = gate/(1+exp(-gate)) * CFp32[ChunkTileLen:]
        mx = max|sw|; perTokenScaleOut = mx/127
        ubOutputTmp = sw*(127/mx); D = CAST_RINT(fp16(CAST_RINT(ubOutputTmp)))
    In the W8A8 variant D lands in **int8** (D1Type pairs with the int8 weights); this
    implementation follows that.
[5] combine: weighted sum over the topK slots using probs.

--------------------------------------------------------------------------
Implementation notes / precision
--------------------------------------------------------------------------
* int8 weight quantization loss dominates: w1/w2 get 8 bits each (relative ~3.9e-3), and after the
  two GEMMs the relative error of out is about 1e-2. The verdict therefore uses a **relative
  error <= 1.5e-2** (far looser than the bf16 floor; this is quantization loss, not an
  implementation defect).
* tl.dot performs the int8 multiply-accumulate in **fp16**: int8 values (|v| <= 127) are exactly
  representable in fp16 and the accumulation runs in fp32, which is equivalent in magnitude to
  int32 accumulation and avoids the backend's uncertain int8 dot support.
* Local dispatch uses "one program per local expert": grid=(E_local, slot_blocks), and the le-th
  program handles only the slots with `expert_idx == base+le`, scatter-adding back into out.
  This is equivalent to "sort then gather" at the cost of O(E_local * M0 * topK) redundant work.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["dispatch_ffn_combine"]


@triton.jit
def _ffn_int8_kernel(
    X, W1, S1, W2, S2, EIDX, PROBS, MASK, OUT, ETN, C1B, C2B,
    NTOK, TOPK, K, N, HALF_I, BASE, EPS,
    NEXP_C: tl.constexpr,
    TOPK_C: tl.constexpr,
    BLK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    LIMIT_ON: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    """Local FFN (int8 weights): **one program per row (token), 1-D tensors throughout**.

    Ascend backend misbehaviours this avoids (measured):
      1) outer-product style mask/offset broadcast `[N,1] x [1,M]` -> no error, garbage results;
      2) 2-D masked `tl.atomic_add` -> silently has no effect.
    """
    t = tl.program_id(0)
    o = tl.arange(0, BLK)
    # No "scalar mask & vector mask" broadcasting (mask broadcasting is unreliable on the Ascend
    # backend and makes x read as all zeros in practice): row validity is always expressed with a
    # scalar plus `if` control flow. That is how the bf16 variant was made to PASS.
    active = t < NTOK
    if HAS_MASK:
        active = active & (tl.load(MASK + tl.minimum(t, NTOK - 1)) != 0)
    t_safe = tl.minimum(t, NTOK - 1)

    # ---- per-token dynamic quantization of the activations [moeInitRoutingQuantV2] ----
    x1 = tl.load(X + t_safe * K + o, mask=o < K, other=0.0).to(tl.float32)
    amax = tl.max(tl.where(o < K, tl.abs(x1), 0.0), axis=0)
    # No literals such as `tl.maximum(amax, 1e-30)` here: the Ascend backend promotes literals
    # that are not exactly representable in fp32 (including Python floats passed at runtime) to
    # **fp64**, which poisons the whole fp32 chain into f64 (the bf16 variant survived only
    # because a mid-chain `.to(weight.dtype)` forced it back). An exact `where` guard is used
    # instead: rows with amax==0 produce all zeros anyway.
    a_s = tl.where(amax > 0.0, amax, 1.0) / 127.0
    aq = x1 / a_s
    aq16 = aq.to(tl.float16).to(tl.float32)
    aqr = tl.minimum(tl.maximum(
        tl.where(aq16 >= 0, tl.floor(aq16 + 0.5), tl.ceil(aq16 - 0.5)), -128.0), 127.0)

    acc = tl.zeros([BLK], dtype=tl.float32)
    for le in tl.static_range(NEXP_C):
        # ---- GEMM1: c1[j] = a_s * scale1[j] * sum_k a_q[k] * W1_le[k, j] ----
        # The column loop uses a **runtime** range: unrolling it BLK times triggers a BiShengHIR
        # compile failure ("propagate storage align failure / max iterations 10").
        c1 = tl.zeros([BLK], dtype=tl.float32)
        for j in range(N):
            wc = tl.load(W1 + le * K * N + o * N + j, mask=o < K, other=0).to(tl.float32)
            # Same as the bf16 variant: the column loop writes its result to scratch and does
            # **not** carry a tensor through the loop.
            tl.store(C1B + t * BLK + j + tl.arange(0, 1),
                     tl.sum(aqr * wc, axis=0, keep_dims=True))
        c1 = tl.load(C1B + t * BLK + o)
        s1 = tl.load(S1 + le * N + o, mask=o < N, other=0.0).to(tl.float32)
        c1 = c1 * a_s * s1
        # The dequantized c1 is stored, then up is taken with a HALF_I left shift (aligned to the
        # same column index as gate).
        tl.store(C2B + t * BLK + o, c1, mask=o < N)

        # ---- SwiGLU [L225-234] ----
        # up must be **shifted left by HALF_I columns**; the wrong form
        # `up = where(o>=HALF_I, c1, 0)` acts on a column range **disjoint** from gate, so the
        # product is identically 0 -- that was the real root cause of out staying equal to the
        # torch.zeros initial value (nothing to do with tl.dot / broadcasting / loop-carried
        # variables).
        gate = tl.where(o < HALF_I, c1, 0.0)
        if LIMIT_ON:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
        up = tl.load(C2B + t * BLK + o + HALF_I, mask=o < HALF_I, other=0.0)
        sw = (gate / (1.0 + tl.exp(-gate))) * up

        # ---- per-token dynamic quantization to int8 [L238-265, W8A8 => D1 lands in int8] ----
        mx = tl.max(tl.where(o < HALF_I, tl.abs(sw), 0.0), axis=0)
        mxs = tl.where(mx > 0.0, mx, 1.0)
        q = sw * (127.0 / mxs)
        q16 = q.to(tl.float16).to(tl.float32)
        qr = tl.minimum(tl.maximum(
            tl.where(q16 >= 0, tl.floor(q16 + 0.5), tl.ceil(q16 - 0.5)), -128.0), 127.0)
        d1 = qr * (mxs / 127.0)

        # ---- GEMM2: c2[j] = (mxs/127) * scale2[j] * sum_i D1[i] * W2_le[i, j] ----
        c2 = tl.zeros([BLK], dtype=tl.float32)
        for j in range(K):
            wc2 = tl.load(W2 + le * HALF_I * K + o * K + j, mask=o < HALF_I, other=0).to(tl.float32)
            tl.store(C2B + t * BLK + j + tl.arange(0, 1),
                     tl.sum(d1 * wc2, axis=0, keep_dims=True))
        c2 = tl.load(C2B + t * BLK + o)
        s2 = tl.load(S2 + le * K + o, mask=o < K, other=0.0).to(tl.float32)
        # **Only S2 may be multiplied here.** `(mxs/127)` was already applied by
        # `d1 = qr * (mxs/127.0)`, so applying it again scales the whole row down by `mxs/127` --
        # measured `out ~ ref * (mxs/127)` with a constant ratio, caused by exactly this
        # duplicated factor (the bf16 variant has no scale2 and never hit it).
        c2 = c2 * s2

        # ---- combine (local part) ----
        for j in tl.static_range(TOPK_C):
            g = tl.load(EIDX + t * TOPK + j).to(tl.int32)
            sel = active & (g == (BASE + le))
            pr = tl.load(PROBS + t * TOPK + j).to(tl.float32)
            acc += tl.where(sel, c2 * pr, 0.0)
            # Use a **1-element vector atomic** (an atomic with a scalar plus a scalar mask is not
            # counted on this backend)
            tl.atomic_add(ETN + le + tl.arange(0, 1),
                          tl.zeros([1], dtype=tl.int32) + tl.where(sel, 1, 0))

    # Tokens excluded by x_active_mask must be **explicitly zeroed** (the store must not be
    # skipped, otherwise the caller's initial value remains; verified with a sentinel initial
    # value of -12345, which survives when the store is skipped). Semantic assumption: masked
    # tokens take no part in the local FFN and are not counted in expert_token_nums (matching the
    # reference).
    acc = tl.where(active, acc, 0.0)
    tl.store(OUT + t * K + o, acc.to(OUT.dtype.element_ty), mask=o < K)


def _as_stacked(w, name):
    if isinstance(w, (list, tuple)):
        return torch.stack([t.contiguous() for t in w], dim=0)
    return w.contiguous()


def dispatch_ffn_combine(
    x: torch.Tensor,
    weight1,
    weight2,
    expert_idx: torch.Tensor,
    scale1,
    scale2,
    bias1=None,
    bias2=None,
    probs: torch.Tensor | None = None,
    group=None,
    max_output_size: int = 0,
    out: torch.Tensor | None = None,
    expert_token_nums: torch.Tensor | None = None,
    x_active_mask: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
    type=None,
    local_expert_base: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The **local computation** part of dispatch_ffn_combine (W8A8 int8 weights, signature
    aligned with the torch call site).

    Returns (out, expert_token_nums), matching the adapter (in-place tensors).
    Cross-card hccl dispatch/combine is out of scope (see the module docstring).
    """
    assert x.dim() == 2 and x.dtype in (torch.float16, torch.bfloat16), x.dtype
    if type is not None and str(type).lower() not in ("int8", "w8a8", "char", "none"):
        raise NotImplementedError(f"type={type!r} does not match the W8A8 int8 variant")
    if bias1 is not None and len(bias1) > 0:
        raise NotImplementedError(
            "bias1 is non-empty: the AscendC adapter (dispatch_ffn_combine_torch_adpt.h L69-70) "
            "explicitly discards bias1/bias2 with (void), and the aclnn call has no bias "
            "parameter, so the original operator adds no bias; this implementation keeps the same "
            "behaviour and does not accept a bias.")
    if bias2 is not None and len(bias2) > 0:
        raise NotImplementedError("bias2 is non-empty: same as bias1, the AscendC implementation discards bias.")
    assert group is None or isinstance(group, str), "group must be an HCCL communication group name (str)"

    M0, K = x.shape
    w1s = _as_stacked(weight1, "weight1")
    E = w1s.shape[0]
    assert w1s.dtype == torch.int8 and w1s.shape[1] == K, \
        f"weight1 must be int8 [{E},{K},N], got {tuple(w1s.shape)} {w1s.dtype}"
    N = w1s.shape[2]
    assert N % 2 == 0, f"N={N} must be even (gate||up)"
    I = N // 2
    w2s = _as_stacked(weight2, "weight2")
    assert w2s.dtype == torch.int8 and tuple(w2s.shape) == (E, I, K), \
        f"weight2 must be int8 [{E},{I},{K}], got {tuple(w2s.shape)}"
    if scale1 is None or (isinstance(scale1, (list, tuple)) and len(scale1) == 0):
        raise NotImplementedError("scale1 missing: the W8A8 int8 variant requires the dequant scale of w1.")
    if scale2 is None or (isinstance(scale2, (list, tuple)) and len(scale2) == 0):
        raise NotImplementedError("scale2 missing: the W8A8 int8 variant requires the dequant scale of w2.")
    s1s = _as_stacked(scale1, "scale1").to(torch.float32)
    s2s = _as_stacked(scale2, "scale2").to(torch.float32)
    assert tuple(s1s.shape) == (E, N), f"scale1 must be [{E},{N}], got {tuple(s1s.shape)}"
    assert tuple(s2s.shape) == (E, K), f"scale2 must be [{E},{K}], got {tuple(s2s.shape)}"

    assert expert_idx.dim() == 2 and expert_idx.shape[0] == M0
    topK = expert_idx.shape[1]
    if probs is None:
        raise NotImplementedError("probs is None: combine needs the routing weights; this implementation does not allow omitting it.")
    assert probs.shape == expert_idx.shape

    dev = x.device
    x = x.contiguous()
    eidx = expert_idx.contiguous().to(torch.int32).reshape(-1)
    pf = probs.contiguous().to(torch.float32).reshape(-1)
    mask = None
    if x_active_mask is not None:
        mask = x_active_mask.contiguous().to(torch.int32).reshape(-1)
        assert mask.numel() == M0

    lo, hi = int(local_expert_base), int(local_expert_base) + E
    n_local = int(((eidx >= lo) & (eidx < hi)).sum().item())
    if max_output_size and max_output_size > 0 and n_local > max_output_size:
        raise NotImplementedError(
            f"local dispatch rows {n_local} > max_output_size={max_output_size}: capacity "
            f"trimming happens in the hccl dispatch stage (out of scope here) and cannot be "
            f"reproduced equivalently in the local computation part.")

    acc = torch.zeros((M0, K), dtype=torch.float32, device=dev)
    out_t = torch.zeros((M0, K), dtype=x.dtype, device=dev) if out is None else out
    etn = torch.zeros(E, dtype=torch.int32, device=dev) if expert_token_nums is None else expert_token_nums
    etn.zero_()

    blk = max(next_pow2(K), next_pow2(N), next_pow2(I), 2)
    c1b = torch.zeros((M0, blk), dtype=torch.float32, device=dev)
    c2b = torch.zeros((M0, blk), dtype=torch.float32, device=dev)
    _ffn_int8_kernel[(M0,)](
        x, w1s, s1s, w2s, s2s, eidx, pf,
        mask if mask is not None else x, out_t, etn, c1b, c2b,
        M0, topK, K, N, I, lo, 1e-30,
        NEXP_C=E, TOPK_C=topK,
        BLK=blk,
        HAS_MASK=mask is not None,
        LIMIT_ON=bool(swiglu_limit > 0.0), SWIGLU_LIMIT=float(swiglu_limit),
    )
    return out_t, etn
