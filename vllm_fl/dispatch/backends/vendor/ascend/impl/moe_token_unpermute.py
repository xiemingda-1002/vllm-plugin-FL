# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""moe_token_unpermute -- AscendC -> Triton

Source: csrc/ascend/mc2/dispatch_ffn_combine/op_kernel/unpermute/moe_token_unpermute.h
(the same kernel body is duplicated under dispatch_ffn_combine_bf16/op_kernel/
unpermute/moe_token_unpermute.h; the two files are identical apart from their
include guard).

--------------------------------------------------------------------------
Call sites (the two `torch_npu.npu_moe_token_unpermute` sites in
dispatch/backends/vendor/ascend/device/device_op.py)
--------------------------------------------------------------------------
  [1] L1074-1077  BaseDeviceAdaptor.npu_moe_token_unpermute(permuted_tokens,
                      sorted_indices, probs)
                  -> torch_npu.npu_moe_token_unpermute(
                         permuted_tokens=permuted_tokens,
                         sorted_indices=torch.abs(sorted_indices), probs=probs)
  [2] L2010-2013  A5DeviceAdaptor.npu_moe_token_unpermute(permuted_tokens,
                      sorted_indices, probs)
                  -> torch_npu.npu_moe_token_unpermute(
                         permuted_tokens=permuted_tokens,
                         sorted_indices=sorted_indices, probs=probs)
(identical 3-argument form also at impl/device_operator.py L215-226, which additionally
wraps sorted_indices in torch.abs.)

Note the difference between the two: [1] passes `torch.abs(sorted_indices)`, [2] passes
it through. That is the callers' business, not this kernel's -- see the index-domain
note below.

--------------------------------------------------------------------------
Semantics (from the AscendC kernel body)
--------------------------------------------------------------------------
Tiling (op_kernel/unpermute/moe_token_unpermute_tiling.h L20-35):
    MoeTokenUnpermuteTiling(m, n, topK, ...):
        hidden_size    = n
        top_k          = topK
        num_out_tokens = m                 # m = number of rows of permuted_tokens
        outTokens      = m / topK          # number of *output* tokens
=> permuted_tokens is [m, hidden_size], sorted_indices / probs hold m = outTokens*topK
   entries, and the result is [m / topK, hidden_size].

Per output token (moe_token_unpermute.h):
    L216-218  for each out_token_idx: CalSingleOutToken(out_token_idx * top_k, ...)
    L248      cal_token_idx = indicesLocal.GetValue(start_token)
    L251      if (cal_token_idx < num_out_tokens)  ... else Duplicate(acc, 0)   [L262]
    L290      input offset = cal_token_idx * hidden_size + h_index * chunk
    L266-278  for the remaining top_k - 1 slots: gather + accumulate
    L317-320  if PROBS: Muls(token_tensor0, token_tensor0, prob_value)   [first slot]
    L332-343  if PROBS: Muls(token_tensor1, token_tensor1, prob_value) then
              Add(token_tensor0, token_tensor0, token_tensor1)
    L354-357  Cast(temp_out_tensors, token_tensor0, RoundMode::CAST_RINT)
    L365-371  store at out_token_index * hidden_size

so, written as a formula:

    out[t, :] = sum_{j < top_k} permuted_tokens[idx[t, j], :] * probs[t, j]
    out[t, :] = 0                                              if top_k == 0
    terms with idx[t, j] >= P (= rows of permuted_tokens) are skipped   (L251)

Accumulation is in **fp32** even for a bf16/fp16 token tensor (L309/L330 cast to float),
and the store rounds with CAST_RINT (L357) -- i.e. round-to-nearest, which is what
`.to(tl.bfloat16)` / `.to(tl.float16)` do in Triton. With `probs=None` (`PROBS=false`)
the slots are *summed*, not averaged.

Index domain / the `torch.abs` at the call sites: the AscendC body only tests
`cal_token_idx < num_out_tokens` (L251). It has no lower bound, so a negative index is
out-of-bounds there (which is presumably why call site [1] feeds it `torch.abs(...)`).
This implementation skips an index unless `0 <= idx < P`; for non-negative input that is
identical to L251, and for a negative index it is a deterministic, memory-safe choice
instead of the undefined read the AscendC body would perform. **Divergence noted, not
hidden.**

Cross-check of the index convention: moe_init_routing_custom.py L33-35 records that with
`row_idx_type=0` the routing op stores `expanded_row_idx[src] = dst`, i.e. at flat source
position src = t*top_k + j it holds the *row of the permuted matrix* for that (token, slot)
pair. That is exactly the index `sorted_indices[t*top_k + j]` consumed here, and
impl/fused_moe_layer.py L178-184 and impl/moe/token_dispatcher.py L428-433 feed
`expanded_row_idx` / `reversed_*_permutation_mapping` straight in.

--------------------------------------------------------------------------
Scope of this implementation
--------------------------------------------------------------------------
  supported : permuted_tokens 2-D [P, H] in bf16/fp16/fp32; sorted_indices 2-D
              [T, top_k] in int32/int64; probs None or holding T*top_k entries;
              restore_shape (a `.view`, used by impl/moe/token_dispatcher.py L702-707).
  raises    : padded_mode=True (the padded layout is a different output shape and is not
              used by the two call sites -- no source in this tree pins it);
              a 1-D sorted_indices (top_k cannot be inferred from it: the AscendC tiling
              takes topK as an explicit argument, tiling.h L20);
              a probs tensor whose element count is not T*top_k (L253/L271 read probs at
              the same flat offset as sorted_indices);
              integer / bool token dtypes.

--------------------------------------------------------------------------
Block layout (why 1-D tiles)
--------------------------------------------------------------------------
moe_init_routing_custom.py documents a measured backend hazard: a 2-D masked store can
silently drop lanes on BiShengIR/FlagTree for non-power-of-two widths (see the
"Backend pitfall" paragraph of its module docstring, which also names the FlagTree
`BlockPtrAnalysis.cpp:309 BlockData::divBlock` assertion for degenerate [1, N] blocks).
The gather here is a pure index permutation, so it is written the way that was measured
clean: one output row at a time (scalar row index), one 1-D column tile per step, and a
runtime loop over the top_k slots. No 2-D block, no 2-D masked store.

--------------------------------------------------------------------------
Grid cap (AI Core block_dim)
--------------------------------------------------------------------------
The program count is capped at 65,535, and a larger grid was measured to crash with
`aivec` on this stack (_agent_logs/TRITON_OPERATOR_PITFALLS.md C1). The grid here is
ceil(T / BLK_T); the host doubles BLK_T while that exceeds MAX_PROGRAMS, so a large
token count trades parallelism for staying inside the cap instead of crashing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["moe_token_unpermute"]

# Rows handled serially per program, and the column tile of the 1-D block. Both are
# small: the kernel is gather-bound, and a 1-D tile of BLK_C floats is all the UB a
# program needs.
BLK_T = 4
BLK_C_CAP = 1024
BLK_C_MIN = 16
# The AI Core block_dim (program count) is capped at 65,535 and a larger grid was
# measured to crash with `aivec` (_agent_logs/TRITON_OPERATOR_PITFALLS.md C1), so the
# host doubles BLK_T until ceil(T / BLK_T) fits.
MAX_PROGRAMS = 32_768


@triton.jit
def _unpermute_kernel(
    P, IDX, PRB, OUT,
    H, K, T, PROWS,
    BLK_T: tl.constexpr,
    BLK_C: tl.constexpr,
    HAS_PROBS: tl.constexpr,
):
    """out[t, :] = sum_j P[idx[t, j], :] * prb[t, j], skipping idx outside [0, PROWS).

    One program owns BLK_T consecutive output rows; inside, one row at a time, one
    1-D column tile at a time, and a runtime loop over the top_k slots (see the block
    layout note in the module docstring).

    int64 casts: `idx * H` and `t * H` reach P*H elements (tens of billions for the
    production shapes), which does not fit int32, so the row offsets are computed in
    int64 while the column offsets stay int32.
    """
    r0 = tl.program_id(0) * BLK_T
    ncol = tl.cdiv(H, BLK_C)
    for i in range(BLK_T):
        t = r0 + i
        if t < T:
            t64 = t.to(tl.int64)
            for c0 in range(0, ncol):
                c = c0 * BLK_C + tl.arange(0, BLK_C)
                mc = c < H
                acc = tl.zeros([BLK_C], dtype=tl.float32)
                for j in range(0, K):
                    idx = tl.load(IDX + t * K + j).to(tl.int32)
                    # `ok` mirrors AscendC L251 (idx < PROWS) plus a lower guard that the
                    # AscendC body does not have; see the index-domain note above.
                    ok = (idx >= 0) & (idx < PROWS)
                    if ok:
                        v = tl.load(P + idx.to(tl.int64) * H + c, mask=mc,
                                    other=0.0).to(tl.float32)
                        if HAS_PROBS:
                            pr = tl.load(PRB + t * K + j).to(tl.float32)
                            v = v * pr
                        acc = acc + v
                tl.store(OUT + t64 * H + c, acc.to(OUT.dtype.element_ty), mask=mc)


def moe_token_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor | None = None,
    *,
    padded_mode: bool = False,
    restore_shape: tuple | None = None,
) -> torch.Tensor:
    """Equivalent of `torch_npu.npu_moe_token_unpermute` (pure token unpermutation).

        out[t, :] = sum_j permuted_tokens[sorted_indices[t, j], :] * probs[t, j]

    i.e. tokens that were sorted by expert are put back in their original order, with
    the optional router weights applied.

    Args:
        permuted_tokens: [P, H] expert-sorted tokens (the rows of `permuted_tokens`,
            concatenated group by group).
        sorted_indices: [T, top_k] int32/int64. `sorted_indices[t, j]` is the row of
            `permuted_tokens` that belongs to (output token t, slot j). Entries outside
            [0, P) are skipped. This is the mapping the AscendC kernel indexes
            (moe_token_unpermute.h L248/267) and the one routing produces with
            `row_idx_type=0` (moe_init_routing_custom.py L33-35).
        probs: None (slots are summed) or [T, top_k] (slots are weighted and summed).
            torch_npu calls the weighted variant the "PROBS" kernel of
            moe_token_unpermute.h L317-343.
        padded_mode: keyword-only, default False. Only the unpadded layout exists in the
            source this operator was derived from, so True raises.
        restore_shape: keyword-only, default None. Optional output shape; the result is
            `.view(restore_shape)`-ed, as impl/moe/token_dispatcher.py L702-708 does with
            `restore_shape=combine_metadata.hidden_shape_before_permute`.

    Returns:
        [T, H] (or `restore_shape` when given), in `permuted_tokens.dtype`. Accumulation
        is fp32 and the final cast rounds to nearest, matching the CAST_RINT store of
        moe_token_unpermute.h L354-357.
    """
    if padded_mode:
        raise NotImplementedError(
            "padded_mode=True: the padded layout implies a different output shape and is "
            "not used by either call site (device_op.py L1075-1077 / L2011-2013 pass only "
            "permuted_tokens, sorted_indices and probs); no source in this tree pins it.")
    assert permuted_tokens.dim() == 2, \
        f"permuted_tokens must be 2-D [P, H], got {tuple(permuted_tokens.shape)}"
    if permuted_tokens.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError(
            f"permuted_tokens.dtype={permuted_tokens.dtype}: only bf16/fp16/fp32 are "
            "covered (kernels are fp16/bf16 on this backend).")
    rows, h = int(permuted_tokens.shape[0]), int(permuted_tokens.shape[1])

    # `torch_npu.npu_moe_init_routing_v2` hands its row indices back flat, so the
    # production MoE path (impl/fused_moe.py L57 -> L89-93) reaches this function with a
    # 1-D `sorted_indices` of length T*top_k. The last dimension is therefore not always
    # available; fall back to `probs`, which the same call site passes as [T, top_k]
    # (moe_token_unpermute.h L253/L271 read both at the same flat offset).
    if sorted_indices.dim() == 2:
        n_out, top_k = int(sorted_indices.shape[0]), int(sorted_indices.shape[1])
    elif sorted_indices.dim() == 1:
        total = int(sorted_indices.shape[0])
        if probs is not None and probs.dim() == 2:
            n_out, top_k = int(probs.shape[0]), int(probs.shape[1])
        else:
            n_out, top_k = total, 1
        if n_out * top_k != total:
            raise ValueError(
                f"sorted_indices has {total} flat entries but probs implies "
                f"{n_out}*{top_k} = {n_out * top_k}; cannot infer top_k")
    else:
        raise NotImplementedError(
            f"sorted_indices has {sorted_indices.dim()} dims: only 1-D [T*top_k] and "
            "2-D [T, top_k] are covered.")
    assert sorted_indices.dtype in (torch.int32, torch.int64), \
        f"sorted_indices must be int32/int64, got {sorted_indices.dtype}"

    out_dtype = permuted_tokens.dtype
    dev = permuted_tokens.device
    if n_out == 0 or top_k == 0 or h == 0:
        out = torch.zeros((n_out, h), dtype=out_dtype, device=dev)
        return out.view(restore_shape) if restore_shape is not None else out

    if probs is not None:
        prb = probs.reshape(-1).to(torch.float32).contiguous()
        assert prb.numel() == n_out * top_k, (
            f"probs has {prb.numel()} entries but sorted_indices implies "
            f"{n_out}*{top_k} = {n_out * top_k} (moe_token_unpermute.h L253/L271 read "
            "probs at the same flat offset as sorted_indices)")
    else:
        prb = torch.zeros(1, dtype=torch.float32, device=dev)

    idx = sorted_indices.reshape(-1).contiguous()
    out = torch.empty((n_out, h), dtype=out_dtype, device=dev)

    blk_c = min(max(_next_pow2(h), BLK_C_MIN), BLK_C_CAP)
    blk_t = BLK_T
    while _ceil_div(n_out, blk_t) > MAX_PROGRAMS:      # AI Core block_dim cap
        blk_t *= 2
    _unpermute_kernel[(_ceil_div(n_out, blk_t),)](
        permuted_tokens.contiguous(), idx, prb, out,
        h, top_k, n_out, rows,
        BLK_T=blk_t, BLK_C=blk_c,
        HAS_PROBS=probs is not None,
    )
    if restore_shape is not None:
        out = out.view(restore_shape)
    return out
