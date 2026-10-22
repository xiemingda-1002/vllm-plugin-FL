# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation replacing the torch_npu built-in of the same name.
# SPDX-License-Identifier: Apache-2.0
"""moe_token_permute -- torch_npu built-in replaced by a Triton implementation.

Source
------
There is **no AscendC source for this operator under csrc/ascend**: a grep for
``moe_token_permute`` / ``MoeTokenPermute`` over the whole csrc tree returns
nothing. ``torch_npu.npu_moe_token_permute`` is a built-in of torch_npu itself,
so the contract below is derived from the call sites and from the two operators
that consume the result, not from a kernel header. Every claim carries the file
and line it comes from.

Call sites (3 real uses + 1 docstring mention)
----------------------------------------------
[1] impl/moe/token_dispatcher.py L575-579  (_dispatch_preprocess)
        permutated_local_input_tokens, reversed_local_input_permutation_mapping =
            torch_npu.npu_moe_token_permute(
                tokens=hidden_states,          # [num_tokens, hidden] (viewed 2-D at L565)
                indices=topk_ids,              # [num_tokens, top_k]
                num_out_tokens=num_out_tokens, # == topk_ids.numel()  (L596)
            )
[2] impl/moe/token_dispatcher.py L676-678  (_dispatch_postprocess, quantized branch)
        dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
            dynamic_scale_after_all2all.unsqueeze(-1),      # [M, 1]
            global_input_tokens_local_experts_indices,      # [M]
        )
[3] impl/moe/token_dispatcher.py L682-684  (_dispatch_postprocess)
        global_input_tokens, reversed_global_input_permutation_mapping =
            torch_npu.npu_moe_token_permute(
                global_input_tokens,                            # [M, hidden]
                global_input_tokens_local_experts_indices,      # [M]
            )
[4] impl/moe/moe_comm_method.py L214 -- the name appears in a docstring only.

Index convention (the part that must be right)
----------------------------------------------
The second return value is consumed as ``sorted_indices`` of
``npu_moe_token_unpermute`` at three places:
  * [1] -> _combine_postprocess L700-707:
        npu_moe_token_unpermute(permuted_tokens=..., sorted_indices=
            reversed_local_input_permutation_mapping.to(torch.int32),
            probs=topk_weights, restore_shape=hidden_shape_before_permute)
  * [3] -> _combine_preprocess L691-693: npu_moe_token_unpermute(hidden_states, rev_global)
  * the structurally identical all-gather path L429-433 passes ``expanded_row_idx``
    from ``npu_moe_init_routing`` in the same slot, and impl/moe_init_routing_custom.py
    L33-35 documents that value as ``expanded_row_idx[src] = dst`` (row_idx_type=0).
  * impl/moe_token_unpermute.py L75-80 states the same for this tree: "at flat
    source position src = t*top_k + j it holds the row of the permuted matrix for
    that (token, slot) pair".
  * the call sites name the value "reversed ... permutation_mapping".
=> the return value is the **inverse permutation**::

       sorted_indices[src] = dst,   src = t*top_k + j in [0, n)
       permuted_tokens[dst] == tokens[src // top_k]
       unpermute: out[t] = sum_j permuted_tokens[sorted_indices[t, j]] * probs[t, j]

   It is NOT ``argsort(indices)`` (that would be the dst -> src direction). This
   is why the value can be substituted for ``expanded_row_idx`` one-for-one.

Semantics implemented
---------------------
::

    flat  = indices.reshape(-1)                    # n = num_tokens * top_k
    order = stable ascending sort of flat by expert id
    dst[order[d]] = d                              # src -> dst, returned
    permuted[d]   = tokens[order[d] // top_k]      # whole row, dtype-agnostic copy
    rows  = n if num_out_tokens is None else min(num_out_tokens, n)
    only dst < rows are written; the mapping is returned cut to `rows`.

Ties: equal expert ids keep **ascending source order**. That is the deterministic
convention impl/moe_init_routing_custom.py documents (its docstring L63-65) and
the one impl/fused_moe.py L115 obtains from ``torch.argsort(flat_experts,
stable=True)``.

How the sort is done (no host sync, no torch.argsort)
----------------------------------------------------
The torch_npu original sorts with the AscendC hardware Sort; a Triton kernel
cannot sort up to n = 65536 entries inside one block (UB is 192 KB), so this
uses the **stable counting sort** that impl/moe_init_routing_custom.py already
runs on this backend (its _hist_kernel / _excl_scan_kernel / _block_base_kernel
/ _route_kernel), re-implemented here so that the module stays self-contained:

  1. _perm_hist_kernel    tl.histogram of the ids per BLK-block -> HIST[nb, BINS]
  2. _perm_scan_kernel    BASE[b, e] = #(id == e in blocks < b), and OFF = the
                          exclusive prefix sum of the total histogram
  3. _perm_route_kernel   dst = OFF[e] + BASE[b, e] + #(same id earlier in block)
                          (rank_local is an exact 2-D masked reduction)
  4. _perm_scatter_kernel permuted[dst] = tokens[src // top_k]

Counting sort needs one bin per expert, so **BINS is a hard precondition**: every
id must satisfy 0 <= id < BINS. BINS = _next_pow2(expert_num) when the caller
passes ``expert_num`` (> 0, an extension keyword, not part of the torch_npu
schema), otherwise DEFAULT_EXPERT_BINS = 1024 -- which covers every MoE model in
this tree (DeepSeek-V3/V4 256, Qwen3-MoE 128, GLM-4.5 160, Kimi K2 384). The 3
call sites know the exact value (``self.num_experts`` / ``self.num_local_experts``
in token_dispatcher.py), so a wiring layer can pass it and remove the bound.

Ids outside [0, BINS) are **dropped** (documented, not silently mis-permuted):
their output row stays zero and their mapping entry stays -1, which
impl/moe_token_unpermute.py already skips (`0 <= idx < P`). The torch_npu
original instead sorts out-of-range ids to the end; that is the single
behavioural divergence and it only triggers on out-of-contract input.

num_out_tokens / padded_mode / expert_range
-------------------------------------------
``num_out_tokens`` truncates *after* sorting (like ``argsort(...)[:rows]`` in
impl/fused_moe.py L115), so the kept rows are the lowest expert ids.
``padded_mode=True`` and ``expert_range`` are not implemented: no call site in
this tree uses either and no local source pins their layout, so both raise
instead of guessing (discipline: implement the parameter or raise with the value).

Output mapping shape
--------------------
For the untruncated case the mapping keeps the shape of ``indices``, i.e.
[num_tokens, top_k] for a 2-D ``indices``, because that is what
impl/moe_token_unpermute.py consumes (it raises on a 1-D mapping, L231-236) and
what the call sites feed it. A truncated mapping is returned 1-D of length
``num_out_tokens``, since [num_tokens, top_k] no longer describes it;
``.reshape(-1)`` gives the flat form in both cases.

Block layout / pitfalls
-----------------------
All kernels use 1-D blocks and scalar row indices. impl/moe_init_routing_custom.py
documents a measured backend hazard: a 2-D block with a 2-D *masked store* drops
lanes on BiShengIR/FlagTree for non-power-of-two widths. The only 2-D expression
here is the rank reduction (no store), which is the form that operator already
runs, and the token copy is one row at a time with a 1-D column tile.
The program count is capped by MAX_PROGRAMS (the AI Core block_dim limit of
65,535 measured in _agent_logs/TRITON_OPERATOR_PITFALLS.md C1); the host doubles
the rows per program until the grid fits.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["moe_token_permute"]

#: Bin count used when the caller does not pass ``expert_num``. Every id must be
#: < this value (see the module docstring); 1024 covers every MoE model in this
#: tree, and ``expert_num`` lets a caller size the histogram exactly.
DEFAULT_EXPERT_BINS = 1024
#: Bins are a single 1-D block in _perm_scan_kernel, so the count has to be a
#: power of two that fits one block (the same 1024 cap moe_init_routing_custom.py
#: uses for its BLK_E tile).
MAX_EXPERT_BINS = 1024
#: Entries per histogram block / per routing block (matches the routing operator).
_HIST_BLK = 64
_ROUTE_BLK = 64
#: Rows per scatter program and the 1-D column tile of the token copy.
_SCATTER_BLK_R = 8
_SCATTER_BLK_C = 512
#: AI Core block_dim cap (TRITON_OPERATOR_PITFALLS.md C1).
MAX_PROGRAMS = 32_768
#: Dtypes the token copy passes through bit-exactly. Anything else is handled by
#: viewing the buffer as uint8 when it is one byte wide, and refused otherwise.
_COPY_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


@triton.jit
def _perm_hist_kernel(E, HIST, n, BINS: tl.constexpr, BLK: tl.constexpr):
    """Per-block x per-expert id counts, one bin row per program, no atomics.

    ``tl.histogram`` needs every value inside [0, BINS), so out-of-range ids are
    parked in the last bin and subtracted again: the counts of the valid ids stay
    exact, no separate masked histogram path is needed, and the store stays
    unmasked. Same construction as moe_init_routing_custom.py `_hist_kernel`.
    """
    b = tl.program_id(0)
    o = b * BLK + tl.arange(0, BLK)
    m = o < n
    e = tl.load(E + o, mask=m, other=-1).to(tl.int32)
    good = m & (e >= 0) & (e < BINS)
    ec = tl.where(good, e, BINS - 1)
    h = tl.histogram(ec, BINS)
    bad = tl.sum(tl.where(good, 0, 1), axis=0)
    h = tl.where(tl.arange(0, BINS) == BINS - 1, h - bad, h)
    tl.store(HIST + b * BINS + tl.arange(0, BINS), h)


@triton.jit
def _perm_scan_kernel(HIST, OFF, BASE, NB, BINS: tl.constexpr):
    """Single program: BASE[b, e] = #(id == e in blocks < b), OFF = exclusive scan.

    A serial loop over the NB histogram blocks, so the per-block base and the
    global exclusive prefix sum come out of one launch (the routing operator
    splits this into `_excl_scan_kernel` + `_block_base_kernel`).
    """
    e = tl.arange(0, BINS)
    run = tl.zeros([BINS], dtype=tl.int32)
    for b in range(0, NB):
        h = tl.load(HIST + b * BINS + e)
        tl.store(BASE + b * BINS + e, run)
        run = run + h
    tl.store(OFF + e, tl.cumsum(run, axis=0) - run)


@triton.jit
def _perm_route_kernel(E, OFF, BASE, DST, n, BINS: tl.constexpr, BLK: tl.constexpr):
    """Stable counting sort: dst[i] = OFF[e_i] + base[block_i, e_i] + rank_local_i.

    rank_local_i = #{ j in the same block : j < i and e_j == e_i } (2-D masked
    reduction, exact). Out-of-range ids get the sentinel -1, which no valid id can
    equal, so they never influence a valid lane's rank; their own dst is masked
    out of the store and keeps the -1 the caller pre-filled.
    """
    b = tl.program_id(0)
    o = b * BLK + tl.arange(0, BLK)
    i = tl.arange(0, BLK)
    m = o < n
    e = tl.load(E + o, mask=m, other=-1).to(tl.int32)
    good = m & (e >= 0) & (e < BINS)

    ec = tl.where(good, e, -1)
    same = (ec[:, None] == ec[None, :]) & (i[None, :] < i[:, None])
    rank = tl.sum(tl.where(same, 1, 0), axis=1)

    # Clamped index for addressing: a masked-out lane must not compute a wild
    # address (the Ascend backend can raise on address computation even when the
    # load itself is masked).
    ecl = tl.where(good, e, 0)
    off = tl.load(OFF + ecl, mask=good, other=0)
    base = tl.load(BASE + b * BINS + ecl, mask=good, other=0)
    tl.store(DST + o, off + base + rank, mask=good)


@triton.jit
def _perm_scatter_kernel(TOK, DST, OUT, W, K, n, ROWS,
                         BLK_R: tl.constexpr, BLK_C: tl.constexpr):
    """permuted[dst] = whole row tokens[src // K]; one row at a time, 1-D columns.

    Pure data movement, so the dtype is irrelevant (int8 scales and bf16 hidden
    states take the same path). `dst` is a bijection over the valid entries of a
    permutation, so there is no write conflict.
    """
    r0 = tl.program_id(0) * BLK_R
    for i in range(BLK_R):
        s = r0 + i
        if s < n:
            d = tl.load(DST + s).to(tl.int32)
            if (d >= 0) & (d < ROWS):
                tb = (s // K).to(tl.int64) * W
                ob = d.to(tl.int64) * W
                for c0 in range(0, tl.cdiv(W, BLK_C)):
                    c = c0 * BLK_C + tl.arange(0, BLK_C)
                    mc = c < W
                    v = tl.load(TOK + tb + c, mask=mc, other=0)
                    tl.store(OUT + ob + c, v, mask=mc)


def moe_token_permute(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int | None = None,
    *,
    padded_mode: bool = False,
    expert_range=None,
    expert_num: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent of ``torch_npu.npu_moe_token_permute`` (MoE group-wise reorder).

    Args:
        tokens: [num_tokens, *tail] (or 1-D [num_tokens]) tokens to reorder. Any
            dtype whose element is a plain scalar (bf16/fp16/fp32/int8/...); the
            copy is bit-exact and dtype agnostic.
        indices: [num_tokens, top_k] (or the flat 1-D form [num_tokens*top_k])
            int32/int64 expert id per (token, slot).
        num_out_tokens: optional number of rows of the result; truncation happens
            after the sort, so the kept rows are the lowest expert ids.
        padded_mode: keyword-only, default False. Raises when True (no call site
            in this tree uses the padded layout, and no local source pins it).
        expert_range: keyword-only, default None. Raises when given (same reason).
        expert_num: extension keyword, default -1 (not part of the torch_npu
            schema). Number of experts; when > 0 the counting-sort histogram is
            sized to _next_pow2(expert_num) and every id must be < expert_num.
            When <= 0 the default DEFAULT_EXPERT_BINS bins are used.

    Returns:
        (permuted_tokens, sorted_indices):
            permuted_tokens  [rows, *tail], rows = min(num_out_tokens or n, n)
            sorted_indices   int32, the **inverse** permutation: for a 2-D
                ``indices`` it is [num_tokens, top_k] with
                ``sorted_indices[t, j] == dst``, where dst is the row of
                ``permuted_tokens`` holding ``(token t, slot j)``. Entries for
                ids outside [0, bins) are -1. Truncated output is returned 1-D.
    """
    if padded_mode:
        raise NotImplementedError(
            "padded_mode=True: no call site in this tree uses the padded layout "
            "(token_dispatcher.py L575/L676/L682 all use the unpadded one) and no "
            "source in this tree pins its output shape."
        )
    if expert_range is not None:
        raise NotImplementedError(
            f"expert_range={expert_range}: the range/offset semantics of the "
            "original operator are not pinned by any source in this tree (no call "
            "site passes it), so it is refused instead of guessed."
        )
    if tokens.dim() < 1:
        raise ValueError(f"tokens must have at least 1 dim, got {tuple(tokens.shape)}")
    if indices.dim() not in (1, 2):
        raise ValueError(
            f"indices must be 1-D [num_tokens*top_k] or 2-D [num_tokens, top_k], "
            f"got {tuple(indices.shape)}"
        )
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"indices must be int32/int64, got {indices.dtype}")

    n_tok = int(tokens.shape[0])
    n = int(indices.numel())
    trailing = tuple(int(d) for d in tokens.shape[1:])
    w = 1
    for d in trailing:
        w *= d
    if n_tok == 0 or n == 0:
        mapping = torch.zeros(tuple(indices.shape), dtype=torch.int32, device=indices.device)
        return torch.zeros((0, *trailing), dtype=tokens.dtype, device=tokens.device), mapping
    if n % n_tok != 0:
        raise ValueError(
            f"indices.numel()={n} is not a multiple of tokens.shape[0]={n_tok}: "
            "top_k cannot be derived"
        )
    k = n // n_tok

    if expert_num is None or int(expert_num) <= 0:
        bins = DEFAULT_EXPERT_BINS
    else:
        bins = _next_pow2(int(expert_num))
        if bins > MAX_EXPERT_BINS:
            raise NotImplementedError(
                f"expert_num={int(expert_num)} needs {bins} histogram bins, above the "
                f"{MAX_EXPERT_BINS}-bin single-block limit of _perm_scan_kernel"
            )
    bins = max(bins, 2)

    rows = n if num_out_tokens is None else min(int(num_out_tokens), n)
    if rows <= 0:
        mapping = torch.zeros((0,), dtype=torch.int32, device=indices.device)
        return torch.zeros((0, *trailing), dtype=tokens.dtype, device=tokens.device), mapping

    dev = tokens.device
    # Token dtype -> something the copy kernel can move bit-exactly. A one-byte
    # dtype Triton has no element type for (e.g. float8_e8m0fnu) is copied as
    # uint8 and viewed back at the end; the bytes are unchanged either way.
    out_dtype = tokens.dtype
    copy_dtype = out_dtype
    tok_flat = tokens.contiguous()
    if copy_dtype not in _COPY_DTYPES:
        if copy_dtype.itemsize != 1:
            raise NotImplementedError(
                f"tokens.dtype={copy_dtype} is neither a plain loadable dtype nor a "
                "one-byte type that can be viewed as uint8"
            )
        tok_flat = tok_flat.view(torch.uint8)
        copy_dtype = torch.uint8
    tok_flat = tok_flat.reshape(-1)

    e = indices.reshape(-1).contiguous().to(torch.int32)

    nb = _ceil_div(n, _HIST_BLK)
    hist = torch.zeros(nb * bins, dtype=torch.int32, device=dev)
    off = torch.zeros(bins, dtype=torch.int32, device=dev)
    base = torch.zeros(nb * bins, dtype=torch.int32, device=dev)
    # Pre-filled with -1 so that entries outside [0, bins) keep the documented
    # "dropped" marker without a separate fill kernel.
    dst = torch.full((n,), -1, dtype=torch.int32, device=dev)

    _perm_hist_kernel[(nb,)](e, hist, n, BINS=bins, BLK=_HIST_BLK)
    _perm_scan_kernel[(1,)](hist, off, base, nb, BINS=bins)
    _perm_route_kernel[(nb,)](e, off, base, dst, n, BINS=bins, BLK=_ROUTE_BLK)

    permuted = torch.zeros((rows, *trailing), dtype=copy_dtype, device=dev)
    blk_r = _SCATTER_BLK_R
    while _ceil_div(n, blk_r) > MAX_PROGRAMS:      # AI Core program cap
        blk_r *= 2
    blk_c = min(max(_next_pow2(w), 16), _SCATTER_BLK_C)
    _perm_scatter_kernel[(_ceil_div(n, blk_r),)](
        tok_flat, dst, permuted.reshape(-1),
        w, k, n, rows,
        BLK_R=blk_r, BLK_C=blk_c,
    )
    if copy_dtype is not out_dtype:
        permuted = permuted.view(out_dtype)

    if rows == n:
        mapping = dst.reshape(tuple(indices.shape))
    else:
        mapping = dst[:rows]
    return permuted, mapping
