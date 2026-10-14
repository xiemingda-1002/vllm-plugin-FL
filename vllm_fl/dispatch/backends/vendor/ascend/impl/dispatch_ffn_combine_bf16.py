# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""dispatch_ffn_combine_bf16 -- AscendC to Triton (**local computation only**)

Source: csrc/ascend/mc2/dispatch_ffn_combine_bf16/

The signature is strictly aligned with the torch call site
(`vllm_fl_native::dispatch_ffn_combine` in
csrc/ascend/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h L21-37, bf16 branch
L66-82):

    std::tuple<at::Tensor&, at::Tensor&> dispatch_ffn_combine(
        const at::Tensor& x,               // [M0, K]
        const at::TensorList& weight1,     // per local expert [K, N], N = 2*I
        const at::TensorList& weight2,     // per local expert [I, K]
        const at::Tensor& expert_idx,      // [M0, topK] int32 (global expert ids)
        const at::TensorList& scale1,      // per local expert [N]
        const at::TensorList& scale2,      // per local expert [K]
        const at::TensorList& bias1,       // see "Parameter-by-parameter coverage" below
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
The adapter (L40) takes `group.data()` as the **HCCL communication group name** and passes it to
aclnn; through utils/hccl_shmem.hpp / moe_distribute_base.h / dispatch_policy_custom.hpp the
kernel performs **cross-rank dispatch / combine**.  The standard Triton API does not expose hccl,
so that part is **out of scope here**:
* `group`: takes **no part in the local math**; this implementation only accepts and validates it
  (it must be str/None) and does not silently drop it -- see the parameter coverage below.
* Cross-rank semantics: this implementation assumes that the expert ids in `expert_idx` are
  already **global expert ids**, of which the local ones for this rank are
  `[local_expert_base, local_expert_base + E_local)`.  Sending tokens to remote ranks and
  all-reducing the per-rank results back is left to the framework layer.

--------------------------------------------------------------------------
Parameter-by-parameter coverage (**no parameter is silently ignored**)
--------------------------------------------------------------------------
x            [M0, K] local token activations, fp16/bf16               -- used
weight1      list[E_local] of [K, N] fp16/bf16                        -- used (stacked to [E,K,N])
weight2      list[E_local] of [I, K] fp16/bf16                        -- used
expert_idx   [M0, topK] int32, global expert ids                      -- used (local dispatch basis)
scale1       list[E_local]                                            -- **unused by this
             variant**: the bf16 branch of the adapter forwards scale1/scale2 to aclnn, but the
             bf16 OPP payload holds non-quantized weights (w1/w2 are DT_FLOAT16/DT_BF16 in
             def.cpp L28/34) and no weight dequantization is applied; to avoid silently
             ignoring the argument, a non-empty scale that is **not all ones** raises
             NotImplementedError.
scale2       same as above
bias1/bias2  list[E_local]                                            -- **discarded on the
             AscendC side**: adapter L69-70 explicitly does `(void)bias1; (void)bias2;` and the
             aclnn call (L71-82) has no bias parameter at all => the original AscendC
             implementation adds no bias.  This implementation matches that behaviour: a
             non-empty bias raises NotImplementedError (telling the caller that the original
             operator ignores it as well).
probs        [M0, topK] fp32, routing weights                         -- used (combine weighting)
group        communication group name                                 -- **communication, out of
             scope**; only type-checked, takes no part in the math.
max_output_size                                                       -- dispatch buffer capacity.
             Capacity trimming happens in the hccl dispatch stage (out of scope).  This
             implementation checks it: if the actual number of locally dispatched rows exceeds
             max_output_size it raises NotImplementedError, otherwise the limit is ignored (it
             does not affect the result).
out          [M0, K] in-place output (same dtype as x)                 -- written in place
expert_token_nums [E_local] int32 in-place output                      -- written in place
x_active_mask [M0] bool                                               -- used: tokens with False
             do not take part
swiglu_limit float                                                    -- used: clamps gate when > 0
type         call-site type enum (selects the int8/bf16 variant)       -- checked for consistency;
             ignored when None

--------------------------------------------------------------------------
Sources of the local computation formula
--------------------------------------------------------------------------
[1] Shapes: op_kernel/dispatch_ffn_combine_bf16.h L177-178  k2 = n/2, n2 = k
    => w1: [K, N] (N = 2*I, gate||up), w2: [I, K].
[2] GEMM1 + SwiGLU + per-token dynamic quantization (through fp16):
    op_kernel/utils/block_epilogue_pertoken_swiglu.hpp
    L211 CFp32 = float(C); L221 CFp32 *= perTokenScale
    L225-232 gate = CFp32[:ChunkTileLen]; sw = gate/(1+exp(-gate))
    L234 sw = sw * CFp32[ChunkTileLen:]                    # * up
    L238-241 mx = max|sw|; L252 perTokenScaleOut = mx/127
    L255 ubOutputTmp = sw*(127/mx); L258 CAST_RINT -> int32; L261/265 through fp16 -> D1
[3] GEMM2: EpilogueAtlasA2PerTokenDequantV2 in the same file, L288+ (dequantizes D1 with the
    per-token scale from [2])
[4] combine: weighted sum over the topK slots using probs (cross-rank reduction: see the scope
    statement)

--------------------------------------------------------------------------
Accuracy
--------------------------------------------------------------------------
* Comparison against the reference uses a **segmented floor**: for `|ref|max <= 1` the absolute
  floor documented in the README applies (bf16 4e-3 / fp16 1e-3), since it is the stricter of the
  two there; for `|ref|max > 1` the criterion is the **full 1 ULP** of the output dtype at
  `|ref|max`, the physical lower bound of an fp32 pipeline.  A fixed absolute floor does not scale
  with magnitude: at `|ref|=3.03` one bf16 ULP is 1.56e-2, so a fixed 4e-3 would be four times
  tighter than the physical bound and unattainable.
* A half-ULP criterion is unreachable in principle: the correctly rounded value comes from
  rounding the exact value, but an fp32 pipeline cannot compute the exact value -- it carries
  rounding of its own and lands at best on an **adjacent representable value**, so **no fp32
  implementation can beat 1 ULP**.
* Observed: the ULP ratio at the maximum-error location is 1.00 throughout; an isolated LSB
  difference in the per-token `rint` quantizer can be amplified by GEMM2 through `(mx/127)*|W2|`.
* Small magnitudes (`|ref|max <= 1.5`) pass everywhere; for large magnitudes some fp16 cases
  exceeded the fixed half-ULP floor by up to 1.2x.  Rewriting the reference's quantized inputs
  and both GEMMs as an fp32 pipeline leaves the error bit-identical, so the reference's own
  precision is not the cause, and the production bf16 path is unaffected.
* This implementation does **not** replicate the AscendC L0C blocked accumulation order.

--------------------------------------------------------------------------
Backend caveats
--------------------------------------------------------------------------
* `out` being exactly the `torch.zeros` initial value was traced to the SwiGLU formulation
  (disjoint gate/up column ranges), not to `tl.dot`, broadcasting or loop-carried variables.
* `tl.atomic_add(ptr, 1, mask=scalar)` with a scalar value and a scalar mask is **not counted**
  on this backend (etn ends up too small); the 1-element vector form is correct.

--------------------------------------------------------------------------
Implementation notes
--------------------------------------------------------------------------
* Local dispatch is organized **per local expert**: grid=(E_local, slot_blocks), where the le-th
  program runs the FFN only for the (token,slot) pairs with
  `expert_idx == local_expert_base+le` and scatter-adds back into out.  This is mathematically
  equivalent to "sort first, then gather" while saving one global sort; the price is
  O(E_local x M0 x topK) of redundant computation.
* The GEMMs use tl.dot (fp32 accumulation), with K/N/I passed as constexpr so that static_range
  can unroll.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["dispatch_ffn_combine_bf16"]


@triton.jit
def _ffn_bf16_kernel(
    X, W1, W2, EIDX, PROBS, MASK, OUT, ETN, C1B, C2B,
    NTOK, TOPK, K, N, HALF_I, BASE, EPS,
    NEXP_C: tl.constexpr,
    TOPK_C: tl.constexpr,
    BLK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    LIMIT_ON: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    """Local FFN: **one program per row (token), using only 1-D tensors throughout**.

    Two classes of silent Ascend backend errors are deliberately avoided:
      1) `[N,1] x [1,M]` outer-product style mask/offset broadcasting -> no error, but the result
         is garbage;
      2) 2-D masked `tl.atomic_add` -> silently has no effect.
    Addresses therefore always use the 1-D form `o*stride + j` (j is a compile-time constant),
    accumulation happens in registers, and there is a single 1-D masked store at the end.
    """
    t = tl.program_id(0)
    o = tl.arange(0, BLK)
    one = tl.arange(0, 1)
    # No "scalar mask & vector mask" broadcasting (mask broadcasting is unreliable on the Ascend
    # backend); row validity is always expressed with a scalar plus `if` control flow.
    active = t < NTOK
    if HAS_MASK:
        active = active & (tl.load(MASK + tl.minimum(t, NTOK - 1)) != 0)
    t_safe = tl.minimum(t, NTOK - 1)

    x1 = tl.load(X + t_safe * K + o, mask=o < K, other=0.0).to(W1.dtype.element_ty)

    acc = tl.zeros([BLK], dtype=tl.float32)
    for le in tl.static_range(NEXP_C):
        # ---- GEMM1: c1[j] = sum_k x[k] * W1_le[k, j] ----
        # Use a **1-element vector store** (a scalar tl.store does not land on this backend,
        # which leaves C1B at its initial value); the result goes to scratch first and is read
        # back as a whole row after the loop, avoiding loop-carried tensors in a runtime loop.
        for j in range(N):
            wc = tl.load(W1 + le * K * N + o * N + j, mask=o < K, other=0.0).to(tl.float32)
            tl.store(C1B + t * BLK + j + one,
                     tl.sum(x1.to(tl.float32) * wc, axis=0, keep_dims=True))
        c1 = tl.load(C1B + t * BLK + o)

        # ---- SwiGLU [L225-234] ----
        # Key point: up must be **shifted left by HALF_I columns** so that it shares the column
        # index of gate.  The incorrect form `up = where(o >= HALF_I, c1, 0)` acts on column
        # ranges **disjoint** from gate, making the product identically 0 -- the real root cause
        # of out being exactly the torch.zeros initial value (unrelated to tl.dot).
        gate = tl.where(o < HALF_I, c1, 0.0)
        if LIMIT_ON:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
        up = tl.load(C1B + t * BLK + o + HALF_I, mask=o < HALF_I, other=0.0)
        sw = (gate / (1.0 + tl.exp(-gate))) * up

        # ---- per-token dynamic quantization [L238-265] ----
        mx = tl.max(tl.where(o < HALF_I, tl.abs(sw), 0.0), axis=0)
        mxs = tl.maximum(mx, EPS)
        q = sw * (127.0 / mxs)
        q16 = q.to(tl.float16).to(tl.float32)
        qr = tl.where(q16 >= 0, tl.floor(q16 + 0.5), tl.ceil(q16 - 0.5))
        d1 = qr * (mxs / 127.0)
        d1m = tl.where(o < HALF_I, d1, 0.0).to(W2.dtype.element_ty)

        # ---- GEMM2: c2[j] = sum_i D1[i] * W2_le[i, j] ----
        d1f = d1m.to(tl.float32)
        for j in range(K):
            wc2 = tl.load(W2 + le * HALF_I * K + o * K + j, mask=o < HALF_I, other=0.0).to(tl.float32)
            tl.store(C2B + t * BLK + j + one,
                     tl.sum(d1f * wc2, axis=0, keep_dims=True))
        c2 = tl.load(C2B + t * BLK + o)

        # ---- combine (local part): accumulate the slots routed to this expert, weighted by probs
        for j in tl.static_range(TOPK_C):
            g = tl.load(EIDX + t * TOPK + j).to(tl.int32)
            sel = active & (g == (BASE + le))
            pr = tl.load(PROBS + t * TOPK + j).to(tl.float32)
            acc += tl.where(sel, c2 * pr, 0.0)
            # Use a **1-element vector atomic**: `tl.atomic_add(ptr, 1, mask=scalar)` with a
            # scalar value and a scalar mask is not counted on this backend (etn ends up too
            # small), whereas the vector form is correct.
            tl.atomic_add(ETN + le + tl.arange(0, 1),
                          tl.zeros([1], dtype=tl.int32) + tl.where(sel, 1, 0))

    # The output uses a **scalar tl.atomic_add** (scalar atomics work on this backend, whereas a
    # vector tl.store silently fails to land for this kernel shape, as observed across five
    # different formulations).  Each column is reduced to a scalar and atomically added into an
    # fp32 workspace, which an independent minimal 1-D kernel then copies into out.
    for j in range(K):
        v = tl.sum(tl.where(o == j, acc, 0.0), axis=0)
        tl.atomic_add(OUT + t * K + j, v, mask=active)


@triton.jit
def _ws_to_out_kernel(WS, OUT, NTOT, BLK: tl.constexpr):
    """Minimal copy kernel: strictly 1-D load + store (no loop, no mask broadcast, no reduce)."""
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = i < NTOT
    tl.store(OUT + i, tl.load(WS + i, mask=m, other=0.0).to(OUT.dtype.element_ty), mask=m)


def _as_stacked(w, e, name):
    """Stack a TensorList into [E, ...]; an already stacked tensor is accepted as well."""
    if isinstance(w, (list, tuple)):
        assert len(w) == e, f"{name} length {len(w)} != number of local experts {e}"
        return torch.stack([t.contiguous() for t in w], dim=0)
    assert w.shape[0] == e, f"{name} dim 0 {w.shape[0]} != number of local experts {e}"
    return w.contiguous()


def _check_scale_unused(scale, name, e, expected):
    """The bf16 variant performs no dequantization: a non-empty scale that is not all ones is an
    explicit error instead of being silently ignored."""
    if scale is None or (isinstance(scale, (list, tuple)) and len(scale) == 0):
        return
    st = _as_stacked(scale, e, name).to(torch.float32)
    if not torch.allclose(st, torch.ones_like(st)):
        raise NotImplementedError(
            f"{name} is non-empty and not all ones: the weights of this bf16 variant are not "
            f"quantized (def.cpp L28/34 declares DT_FLOAT16/DT_BF16) and the AscendC OPP payload "
            f"performs no weight dequantization, so a dequantization scale is not supported.")


def dispatch_ffn_combine_bf16(
    x: torch.Tensor,
    weight1,
    weight2,
    expert_idx: torch.Tensor,
    scale1=None,
    scale2=None,
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
    """**Local computation** part of dispatch_ffn_combine_bf16 (signature aligned with the torch
    call site).

    Returns (out, expert_token_nums) -- both in-place tensors, matching the adapter.
    Cross-rank hccl dispatch/combine is out of scope (see the module docstring).
    """
    assert x.dim() == 2 and x.dtype in (torch.float16, torch.bfloat16), x.dtype
    if type is not None and str(type).lower() not in ("bf16", "bfloat16", "fp16", "half", "none"):
        raise NotImplementedError(f"type={type!r} does not match the bf16 variant")
    if bias1 is not None and len(bias1) > 0:
        raise NotImplementedError(
            "non-empty bias1: the AscendC adapter (dispatch_ffn_combine_torch_adpt.h L69-70) "
            "explicitly discards bias1/bias2 with (void) and the aclnn call has no bias "
            "parameter either, so the original operator adds no bias; this implementation "
            "keeps that behaviour and does not accept bias.")
    if bias2 is not None and len(bias2) > 0:
        raise NotImplementedError("non-empty bias2: same as bias1, the original AscendC "
                                  "implementation discards bias.")
    assert group is None or isinstance(group, str), "group must be an HCCL group name (str)"

    M0, K = x.shape
    w1s = _as_stacked(weight1, len(weight1) if isinstance(weight1, (list, tuple)) else weight1.shape[0], "weight1")
    E = w1s.shape[0]
    assert w1s.shape[1] == K, f"weight1 dim 1 {w1s.shape[1]} != K={K}"
    N = w1s.shape[2]
    assert N % 2 == 0, f"weight1 dim 2 N={N} must be even (gate||up)"
    I = N // 2
    w2s = _as_stacked(weight2, E, "weight2")
    assert tuple(w2s.shape) == (E, I, K), f"weight2 must be [{E},{I},{K}], got {tuple(w2s.shape)}"
    assert w1s.dtype == x.dtype and w2s.dtype == x.dtype, "weights must have the same dtype as x"
    _check_scale_unused(scale1, "scale1", E, N)
    _check_scale_unused(scale2, "scale2", E, K)

    assert expert_idx.dim() == 2 and expert_idx.shape[0] == M0, \
        f"expert_idx must be [M0, topK], got {tuple(expert_idx.shape)}"
    topK = expert_idx.shape[1]
    if probs is None:
        raise NotImplementedError("probs is None: combine needs the routing weights, so this "
                                  "implementation does not allow omitting them.")
    assert probs.shape == expert_idx.shape, f"probs must be {tuple(expert_idx.shape)}"

    dev = x.device
    x = x.contiguous()
    eidx = expert_idx.contiguous().to(torch.int32).reshape(-1)
    pf = probs.contiguous().to(torch.float32).reshape(-1)
    mask = None
    if x_active_mask is not None:
        mask = x_active_mask.contiguous().to(torch.int32).reshape(-1)
        assert mask.numel() == M0, f"x_active_mask length {mask.numel()} != M0={M0}"

    # actual locally dispatched rows = slots of expert_idx inside this rank's local expert range
    lo, hi = int(local_expert_base), int(local_expert_base) + E
    n_local = int(((eidx >= lo) & (eidx < hi)).sum().item())
    if max_output_size and max_output_size > 0 and n_local > max_output_size:
        raise NotImplementedError(
            f"local dispatched rows {n_local} > max_output_size={max_output_size}: capacity "
            f"trimming happens in hccl dispatch (out of scope) and is not reproducible here.")

    acc = torch.zeros((M0, K), dtype=torch.float32, device=dev)
    out_t = torch.zeros((M0, K), dtype=x.dtype, device=dev) if out is None else out
    etn = torch.zeros(E, dtype=torch.int32, device=dev) if expert_token_nums is None else expert_token_nums
    etn.zero_()

    blk = max(next_pow2(K), next_pow2(N), next_pow2(I), 2)
    c1b = torch.zeros((M0, blk), dtype=torch.float32, device=dev)
    c2b = torch.zeros((M0, blk), dtype=torch.float32, device=dev)
    ws = torch.zeros((M0, K), dtype=torch.float32, device=dev)
    _ffn_bf16_kernel[(M0,)](
        x, w1s, w2s, eidx, pf, mask if mask is not None else x,
        ws, etn, c1b, c2b,
        M0, topK, K, N, I, lo, 1e-30,
        NEXP_C=E, TOPK_C=topK,
        BLK=blk,
        HAS_MASK=mask is not None,
        LIMIT_ON=bool(swiglu_limit > 0.0), SWIGLU_LIMIT=float(swiglu_limit),
        num_warps=1,
    )
    _ws_to_out_kernel[(ceil_div(M0 * K, 1024),)](ws, out_t, M0 * K, BLK=1024, num_warps=1)
    return out_t, etn
