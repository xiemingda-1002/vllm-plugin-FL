# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""moe_init_routing_v2 -- AscendC -> Triton

Source (15 files, this module's only source of truth)
----------------------------------------------------
csrc/ascend/mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/
    moe_init_routing_v2.cpp            entry, tiling-key dispatch        L64-123
    moe_init_routing_v2_tiling.h       IO indices, attrs, tiling keys    L7-36, L268-294
    moe_v2_common.h                    flag values, MAX_EXPERT_NUM       L47-53
    moe_v2_sort_base.h                 shared sort state                 L38-60
    moe_v2_mrgsort.h / moe_v2_mrgsort_out.h   k-way merge
    moe_v2_sort_one_core.h             10001/10011  (single core sort)   L49-133
    moe_v2_sort_multi_core.h           10002/10012  (multi core + merge)
    moe_v2_expert_token_out.h          count / cumsum / before-capacity  L126-281
    moe_v2_src_to_dst_op.h             expanded_row_idx, dropless        L78-109
    moe_v2_src_to_dst_with_capacity.h  expanded_row_idx + x, drop_pad    L114-195
    moe_v2_gather_out.h                expanded_x, dropless              L88-139
    moe_v2_init_routing_fullload.h     fused path, tiling key 20000      L93-285
    moe_v2_src_to_dst_op_simt.h        arch35 SIMT variant (unused here) L78-86
    tiling_base.h                      DoTiling() argument order         L16-47

Interface, from the tiling header (L25-36); names and values verbatim, laid out in
two columns for brevity
-------------------------------------------
    INDEX_INPUT_X                = 0        ATTR_ACTIVE_ROWS             = 0
    INDEX_INPUT_EXPERT_IDX       = 1        ATTR_EXPERT_CAPACITY         = 1
    OUTOUT_EXPANDED_X            = 0        ATTR_EXPERT_NUM              = 2
    OUTOUT_EXPANDED_ROW_IDX      = 1        ATTR_DROP_PAD_MODE           = 3
    OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2
    OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3
    ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4
    ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5

Two inputs, four outputs, six attributes.  Tiling keys (L7-11): 10001 dropless /
single core, 10002 dropless / multi core, 10011 drop_pad / single core, 10012
drop_pad / multi core, 20000 fused high performance.  The key only selects the
*implementation*, never the semantics: moe_v2_init_routing_fullload.h L196-244
(count/cumsum), moe_v2_expert_token_out.h L126-281 (same, per core + atomics),
moe_v2_src_to_dst_op.h L78-109 vs moe_v2_init_routing_fullload.h L149-170 (same
expanded_row_idx value from a second sort) all agree, so one Triton path covers
all five keys.

Call sites that fix the required parameter combinations
------------------------------------------------------
[1] vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py L1145-1157,
    inside ``npu_moe_init_routing`` (L1131-1157):

        input_dtype = act_quant_type or hidden_states.dtype
        return torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            scale=scale,
            active_num=active_num,
            expert_num=expert_num,
            expert_tokens_num_type=expert_tokens_num_type,   # default 1 (L1139)
            expert_tokens_num_flag=expert_tokens_num_flag,    # default True (L1140)
            active_expert_range=active_expert_range,          # default None (L1141)
            quant_mode=quant_mode,                            # default -1 (L1142)
            x_dtype=input_dtype if input_dtype in QUANT_DTYPES else None,
        )

[2] vllm_fl/dispatch/backends/vendor/ascend/impl/moe/token_dispatcher.py L660-672,
    inside ``_dispatch_postprocess`` (quantized float8_e8m0fnu branch):

        global_input_tokens, reversed_global_input_permutation_mapping, _, routed_scale = (
            torch_npu.npu_moe_init_routing_v2(
                global_input_tokens,
                experts_indices_2d_copy,               # [M, 1]  -> k == 1
                scale=dynamic_scale_for_routing,
                active_num=experts_indices_2d_copy.shape[0],   # == M*k
                expert_num=self.num_local_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.num_local_experts],
                x_dtype=dst_type,
            )
        )

What those two sentences mean for the port:
  * ``active_num``  == the number of routed slots (N*k) at both sites, i.e. the
    untruncated case; ``expert_tokens_num_type=1`` + ``flag=True`` is COUNT, seen
    in the AscendC numbering as flag 2 (moe_v2_common.h L50, and the in-tree
    fused tiling op_host/dispatch_ffn_combine_bf16_tiling.cpp L207-210 hardcodes
    ``activeNum = M*topK``, ``dropPadMode = 0``, ``expertTokensCountOrCumsumFlag
    = 2``, ``expertTokensBeforeCapacityFlag = false``).
  * ``active_expert_range=[0, num_local_experts]`` is the degenerate range == the
    full expert set, i.e. a no-op; the 15-file kernel has no range attribute at
    all (tiling header L25-36 lists six attributes and none of them is a range).
  * site [2] additionally passes ``scale=`` and ``x_dtype=dst_type`` (a quantized
    expanded_x) and site [1] can pass ``x_dtype`` too.  Neither a ``scale`` input
    nor a quantized output exists in the 15 files (INDEX_INPUT lists exactly x and
    expert_idx; every output write in moe_v2_gather_out.h L114-133 is a plain
    dtype copy, no Cast/quant), so the quantized combination raises here.

Semantics, each with its file:line
----------------------------------
1. Sort: **ascending expert id, ties by ascending flattened source index**.
   moe_v2_sort_one_core.h L76 negates the id (key = -id: `Muls(..., -1)`), L98
   ``Sort<float, true>`` sorts descending => ascending on -id == ascending on id.
   The payload is the source index (L61 ``ArithProgression(rowIdxLocal, 0, 1)``,
   L105 ``Extract(sortedExpert, dstToSrc, ...)``), padded tail is MIN_FP32
   (L86) so padding lands last.  Multi core is the same sort per core plus a
   k-way ``MrgSort<float, true>`` (moe_v2_mrgsort.h L130-138), merging lists in
   core order, so ties stay in ascending source order across cores as well.
   The AscendC headers do not document tie order; this implementation and the
   in-tree reference (impl/fused_moe.py L115 ``torch.argsort(..., stable=True)``)
   both use ascending source order.
2. ``expanded_row_idx[src] = dst`` for dropless (moe_v2_src_to_dst_op.h L103-104:
   ``outOffset = dstToSrc[idx]; DataCopyPad(expandedRowIdx[outOffset], <dst>)``;
   the fused path writes the identical array, init_routing_fullload.h L151-170).
   In drop_pad mode it is ``expanded_row_idx[src] = e*expert_capacity + t`` or -1
   when truncated (moe_v2_src_to_dst_with_capacity.h L156-164 writes only when
   ``tokenCount < expertCapacity``; the -1 fill is moe_v2_expert_token_out.h
   L92-117 for every source row).
3. ``expanded_x`` is a whole-row copy of ``x[src // k]`` at row ``dst``
   (moe_v2_gather_out.h L99-133: ``row = initialRow / k`` indexes x,
   ``outOffset = outIndex * cols`` indexes expanded_x, ``outIndex == -1`` and
   ``outIndex >= activateRows`` are skipped).  drop_pad additionally zero-fills
   every unused capacity slot (with_capacity L128-154 and L171-195 write zeros
   for the gaps).
4. ``expert_tokens_count_or_cumsum`` (only dropless, moe_init_routing_v2.cpp L90-98
   plus tiling L278-282 which forces the flag to 0 in drop_pad mode):
   flag 1 (EXERPT_TOKENS_CUMSUM) stores an **inclusive** prefix sum, flag 2
   (EXERPT_TOKENS_COUNT) stores per-expert **counts**, flag 0 writes nothing
   (moe_v2_common.h L50-52; accumulation in moe_v2_expert_token_out.h L163-229,
   single-core equivalent in init_routing_fullload.h L214-232).  The output buffer
   is zeroed first (sort_one_core.h L170-179 / sort_multi_core.h L83-101), then
   accumulated with atomic adds, so experts with no token are 0 and, in cumsum
   mode, the tail beyond the last used expert repeats the total
   (moe_v2_expert_token_out.h L174-192, fullload L226-232).
5. ``expert_tokens_before_capacity`` (only drop_pad, tiling L278-282 forces the
   *other* flag off) is the count **before** capacity truncation
   (moe_v2_expert_token_out.h L241-248).
6. ``expert_capacity`` truncation drops the surplus slots of an expert: only the
   first ``expert_capacity`` tokens of each expert get a destination row
   (moe_v2_src_to_dst_with_capacity.h L130-154 counts per expert, L156-164 keeps
   the first ``expert_capacity``).
7. ``active_rows`` (dropless only, tiling L478-481: ``activateRows = totalLength``
   unless drop_pad, then ``activateNum``) limits the **written rows of
   expanded_x** to ``dst < active_rows``; expanded_row_idx is never truncated.

Scope of this implementation
----------------------------
* ``drop_pad_mode`` 0 (dropless) and 1 (drop_pad) are both implemented.
* ``expert_tokens_count_or_cumsum_flag`` 0 / 1 / 2 are implemented; the torch_npu
  spelling ``expert_tokens_num_flag`` + ``expert_tokens_num_type`` is accepted
  and mapped (0 -> flag 1 cumsum, 1 -> flag 2 count).  ``expert_tokens_num_type``
  == 2 (KEY_VALUE) does not exist in this kernel and raises.
* Everything this kernel cannot honour raises ``NotImplementedError`` naming the
  parameter: ``quant_mode`` != -1, ``x_dtype`` != None, a non-degenerate
  ``active_expert_range``, a ``scale`` argument (rejected by the signature, the
  kernel has no third input), an out-of-range ``drop_pad_mode`` /
  ``expert_tokens_count_or_cumsum_flag`` / ``expert_tokens_num_type``, an
  ``expert_num`` above the 4096-bin single-block histogram, and a token count
  above the AI Core program cap.
* Expert ids outside ``[0, expert_num)`` are **dropped**, the same convention as
  impl/moe_token_permute.py (its docstring L101-105) and as the in-tree reference
  impl/fused_moe.py L111-114: such a token keeps ``expanded_row_idx = -1``, is
  not copied into ``expanded_x`` and is not counted.  The AscendC kernel has no
  defined behaviour here -- it neither clamps nor validates.  A negative id
  becomes the *first* sorted id, so ``firstExpertId`` (moe_v2_expert_token_out.h
  L152-153) is negative and both copy-outs index the output with it
  (L197-202 ``expertTokensCountOrCumsumGm[this->firstExpertId]``, L243-255
  ``...BeforeCapacityGm[this->firstExpertId]``), i.e. a store before the buffer;
  an id >= expert_num pushes ``lastExpertId`` past the end and the same copies
  run past the ``expertNum``-sized buffer set up at L326-332.  The sort simply
  places such an id at an extreme (key = -id, moe_v2_sort_one_core.h L76).
  Dropping is therefore a documentation-level divergence, chosen to keep every
  index in range.  ``expanded_row_idx`` is int32 and the ids are int32 on the
  device side (``GlobalTensor<int32_t> expertIdxGm``, moe_v2_sort_base.h L38), so
  an int64 id outside the int32 range wraps exactly as it does in the original.
* CPU tensors take a documented pure-torch mirror of the same arithmetic
  (``_cpu_route`` etc., used by ``_verify_cpu.py``); every other device takes the
  Triton kernels, which transcribe that arithmetic one-to-one.
* **Not compiled and not executed on NPU.**  No CANN/BiShengIR toolchain and no
  NPU were available while writing this file; the Triton kernels are a line by
  line transcription of the CPU mirror that ``_verify_cpu.py`` checks against a
  naive reference, and nothing more is claimed for them.

Deltas against impl/moe_init_routing_custom.py (the finished sibling)
--------------------------------------------------------------------
Same skeleton (histogram -> exclusive scan -> rank -> gather), same 1-D-block
discipline (that module's docstring documents the 2-D masked-store hazard), same
"no tl.atomic_add" decision.  Differences: (a) this operator returns
``expanded_row_idx[src] = dst`` only, there is no ``row_idx_type``; (b) the
counts output here is int32 with the AscendC 0/1/2 flag numbering, not the
torch_npu 0/1/2 ``expert_tokens_num_type`` numbering with a KEY_VALUE mode;
(c) drop_pad mode and ``expert_capacity`` truncation are implemented here (the
sibling raises for drop_pad_mode=1); (d) no quantization, no ``scale``/``offset``,
no ``expanded_scale`` output; (e) the sibling raises for ``active_num`` inside
(0, N*k) -- this one implements the AscendC behaviour (write only dst <
active_rows); (f) both drop out-of-range ids, and both keep ties in ascending
source order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["moe_init_routing_v2"]

# ---------------------------------------------------------------------------
# Constants, from the AscendC source
# ---------------------------------------------------------------------------
# moe_v2_common.h L48-53
DROPLESS_MODE = 0
DROP_PAD_MODE = 1
EXERPT_TOKENS_NONE = 0
EXERPT_TOKENS_CUMSUM = 1
EXERPT_TOKENS_COUNT = 2
EXERPT_TOKENS_BEFORE_CAPACITY = 1
# moe_v2_common.h L47 -- the AscendC upper bound on expert_num.  The Triton
# histogram is one block wide, which caps this implementation lower (see below).
MAX_EXPERT_NUM_ASCENDC = 5120
#: Bins of the single-program histogram / scan must fit one block.
MAX_EXPERT_BINS = 4096
#: AI Core block_dim cap (the same 32768 measured in impl/moe_token_permute.py).
MAX_PROGRAMS = 32768
#: Tokens per histogram and routing block.  A routing program does an exact
#: [BLK, BLK] same-id comparison for the within-block rank, so this stays small.
#: Tokens per program for ``_hist_kernel`` / ``_route_kernel``.
#:
#: ``_route_kernel`` builds a ``BLK x BLK`` comparison matrix to get the exact
#: in-block rank of every token, so its UB footprint grows quadratically.  With
#: BLK=128 the BiShengHIR pipeline rejected the kernel on the first on-device build:
#:   ``ub overflow, requires 2117632 bits while 1572864 bits available!``
#: 2117632 / (128*128) == 129.2 bits per matrix cell; at BLK=64 the matrix is 4x
#: smaller and the whole kernel needs well under the 196608-byte budget.  The only
#: cost is twice as many programs (and twice as many ``_scan_kernel`` loop
#: iterations, which is a single serial program either way).
_ROUTE_BLK = 64
#: Source rows per gather program, and the 1-D column tile of the row copy.
_SCATTER_BLK_R = 8
_SCATTER_BLK_C = 512
#: Dtypes the row copy moves element-wise; anything else one byte wide is viewed
#: as uint8, and a wider unknown dtype raises.
_COPY_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _ceil_div(a: int, b: int) -> int:
    """Ceil of a/b for non-negative a, positive b."""
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (n >= 1)."""
    return max(1, 1 << (int(n) - 1).bit_length())


# ---------------------------------------------------------------------------
# Triton kernels.  Every one of them transcribes the *same* arithmetic as the CPU
# mirror further down (``_cpu_route`` / ``_cpu_run``), which is what
# ``_verify_cpu.py`` checks against an independent naive reference.
# ---------------------------------------------------------------------------
@triton.jit
def _hist_kernel(E, HIST, n, EXPERT_NUM, BINS: tl.constexpr, BLK: tl.constexpr):
    """Per-block x per-expert token counts, one bin row per program, no atomics.

    ``tl.atomic_add`` is avoided on purpose: impl/moe_init_routing_custom.py
    L119-129 measured it over-counting on this backend whenever two lanes of one
    vector add to the same expert, which corrupts every downstream group offset.
    ``tl.histogram`` needs all values inside [0, BINS), so ids outside
    [0, expert_num) are parked in the last bin and subtracted again; the store
    then stays unmasked and no separate masked path exists.
    """
    b = tl.program_id(0)
    o = b * BLK + tl.arange(0, BLK)
    m = o < n
    e = tl.load(E + o, mask=m, other=-1).to(tl.int32)
    good = m & (e >= 0) & (e < EXPERT_NUM)
    ec = tl.where(good, e, BINS - 1)
    h = tl.histogram(ec, BINS)
    bad = tl.sum(tl.where(good, 0, 1), axis=0)
    h = tl.where(tl.arange(0, BINS) == BINS - 1, h - bad, h)
    tl.store(HIST + b * BINS + tl.arange(0, BINS), h)


@triton.jit
def _scan_kernel(HIST, OFF, BASE, TOT, NB, BINS: tl.constexpr):
    """Single program: BASE[b,e] = counts of blocks < b, OFF = exclusive prefix over
    experts, TOT = per-expert totals.

    One serial loop over the NB histogram blocks.  BASE is what turns the
    per-block counting sort into a global stable one; OFF is the expert start
    offset (``expertTokensCountOrCumsum`` in cumsum mode is its inclusive
    version, built from TOT).
    """
    e = tl.arange(0, BINS)
    run = tl.zeros([BINS], dtype=tl.int32)
    for b in range(0, NB):
        h = tl.load(HIST + b * BINS + e)
        tl.store(BASE + b * BINS + e, run)
        run = run + h
    tl.store(OFF + e, tl.cumsum(run, axis=0) - run)
    tl.store(TOT + e, run)


@triton.jit
def _route_kernel(E, OFF, BASE, DST, n, EXPERT_NUM, BINS: tl.constexpr, BLK: tl.constexpr):
    """Stable counting sort: DST[src] = OFF[e] + BASE[block, e] + rank_local.

    ``rank_local`` is #{j in this block : j < i and e_j == e_i}, an exact 2-D
    masked reduction.  Out-of-range ids get the sentinel -1, which no valid id
    equals, so they never influence a valid lane's rank; their own DST stays at
    the -1 the caller pre-filled (the "dropped" marker).
    """
    b = tl.program_id(0)
    o = b * BLK + tl.arange(0, BLK)
    i = tl.arange(0, BLK)
    m = o < n
    e = tl.load(E + o, mask=m, other=-1).to(tl.int32)
    good = m & (e >= 0) & (e < EXPERT_NUM)

    ec = tl.where(good, e, -1)
    same = (ec[:, None] == ec[None, :]) & (i[None, :] < i[:, None])
    rank = tl.sum(tl.where(same, 1, 0), axis=1)

    # Clamped address: a masked-out lane must not build a wild offset.
    ecl = tl.where(good, e, 0)
    off = tl.load(OFF + ecl, mask=good, other=0)
    base = tl.load(BASE + b * BINS + ecl, mask=good, other=0)
    tl.store(DST + o, off + base + rank, mask=good)


@triton.jit
def _counts_kernel(TOT, OUT, EXPERT_NUM, BINS: tl.constexpr, MODE: tl.constexpr):
    """Single program: per-expert counts (MODE 2) or the inclusive prefix sum
    (MODE 1) of them.

    Zero beyond the last used expert falls out of the arithmetic: the buffer is
    zero-initialised and ``TOT[e] == 0`` for an unused expert, so cumsum keeps
    repeating the total exactly as moe_v2_expert_token_out.h L174-192 does.
    """
    e = tl.arange(0, BINS)
    m = e < EXPERT_NUM
    t = tl.load(TOT + e, mask=m, other=0)
    if MODE == 1:      # EXERPT_TOKENS_CUMSUM, moe_v2_common.h L51
        v = tl.cumsum(t, axis=0)
    else:              # EXERPT_TOKENS_COUNT, moe_v2_common.h L50
        v = t
    tl.store(OUT + e, v, mask=m)


@triton.jit
def _gather_kernel(X, DST, OUT, n, K, COLS, ROWS, BLK_R: tl.constexpr, BLK_C: tl.constexpr):
    """Dropless expanded_x: row ``DST[src]`` of the output is row ``src // K`` of x.

    Walks **source** rows and scatters, exactly like moe_v2_gather_out.h L101-133
    (which also derives the x row as ``token / k`` and the output row as the
    expanded_row_idx value, skipping -1 and ``>= activateRows``).  ``DST`` is a
    bijection over the kept sources, so there is no write conflict.

    1-D blocks only, scalar row index: a [BLK_D, BLK_C] 2-D block with a 2-D
    masked store silently drops lanes on BiShengIR/FlagTree for non-power-of-two
    column counts (measured and documented in impl/moe_init_routing_custom.py's
    module docstring).
    """
    r0 = tl.program_id(0) * BLK_R
    for i in range(BLK_R):
        s = r0 + i
        if s < n:
            d = tl.load(DST + s).to(tl.int32)
            if (d >= 0) & (d < ROWS):
                xb = (s // K).to(tl.int64) * COLS
                ob = d.to(tl.int64) * COLS
                for c0 in range(0, tl.cdiv(COLS, BLK_C)):
                    c = c0 * BLK_C + tl.arange(0, BLK_C)
                    mc = c < COLS
                    v = tl.load(X + xb + c, mask=mc, other=0)
                    tl.store(OUT + ob + c, v, mask=mc)


@triton.jit
def _capacity_kernel(X, E, DST, OFF, DRI, XOUT, n, K, COLS, EXPERT_NUM,
                     CAPACITY, BLK_R: tl.constexpr, BLK_C: tl.constexpr):
    """drop_pad: expanded_row_idx[src] = e*capacity + t, and expanded_x[that] = x[src//K].

    ``t = DST[src] - OFF[e]`` is the rank of the token inside its expert (the
    per-expert counter moe_v2_src_to_dst_with_capacity.h L126-165 keeps by hand
    while walking dst order).  ``t >= capacity`` means the token is dropped: no
    row is written and expanded_row_idx keeps its -1 prefill, which is the
    ``while (tokenCount < expertCapacity)`` guard at L156-164.  Unused capacity
    slots are zero, matching the zero fill at L128-154 / L171-195.
    """
    r0 = tl.program_id(0) * BLK_R
    cap = CAPACITY.to(tl.int64)
    for i in range(BLK_R):
        s = r0 + i
        if s < n:
            e = tl.load(E + s).to(tl.int32)
            d = tl.load(DST + s).to(tl.int32)
            # One branch, clamped address: a dropped source row must not build a
            # wild offset (expert 0 is always a valid OFF index).
            ok = (e >= 0) & (e < EXPERT_NUM) & (d >= 0)
            ecl = tl.where(ok, e, 0)
            off = tl.load(OFF + ecl).to(tl.int64)
            t = tl.where(ok, d.to(tl.int64) - off, cap)
            if t < cap:
                p = ecl.to(tl.int64) * cap + t
                tl.store(DRI + s, p.to(tl.int32))
                xb = (s // K).to(tl.int64) * COLS
                ob = p * COLS
                for c0 in range(0, tl.cdiv(COLS, BLK_C)):
                    c = c0 * BLK_C + tl.arange(0, BLK_C)
                    mc = c < COLS
                    v = tl.load(X + xb + c, mask=mc, other=0)
                    tl.store(XOUT + ob + c, v, mask=mc)


# ---------------------------------------------------------------------------
# Host-side arithmetic: CPU mirror of the kernels above.
#
# ``_verify_cpu.py`` drives these against a naive argsort reference.  They are
# also the code path taken for CPU tensors, so the arithmetic that the Triton
# kernels transcribe is the arithmetic that gets numerically checked; nothing
# here is dead code.
# ---------------------------------------------------------------------------
def _cpu_route(e_flat: torch.Tensor, expert_num: int, blk: int) -> tuple:
    """Mirror of _hist_kernel + _scan_kernel + _route_kernel.

    Returns ``(dst, off, tot)`` in int64.  ``dst[src]`` is -1 for a dropped
    (out-of-range) id.

    hist[b, e]  count of valid tokens with id e in block b        (_hist_kernel)
    base[b, e]  sum over b' < b of hist[b', e]                     (_scan_kernel)
    off[e]      sum over e' < e of tot[e']                         (_scan_kernel)
    dst[src]    off[e] + base[b, e] + rank_local                   (_route_kernel)
    """
    n = int(e_flat.numel())
    dev = e_flat.device
    nb = _ceil_div(n, blk)
    good = (e_flat >= 0) & (e_flat < expert_num)
    blk_of = torch.arange(n, device=dev) // blk

    hist = torch.zeros(nb * expert_num, dtype=torch.int64, device=dev)
    src = torch.nonzero(good, as_tuple=False).reshape(-1)
    if src.numel() > 0:
        flat = blk_of[src] * expert_num + e_flat[src]
        hist.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int64))
    hist = hist.view(nb, expert_num)
    tot = hist.sum(0)

    base = torch.zeros((nb, expert_num), dtype=torch.int64, device=dev)
    if nb > 1:
        base[1:] = hist.cumsum(0)[:-1]
    off = torch.zeros(expert_num, dtype=torch.int64, device=dev)
    if expert_num > 1:
        off[1:] = tot.cumsum(0)[:-1]

    # Two views of the id, exactly as the Triton kernels keep them: the sentinel
    # view (-1, no valid id can equal it) is what the rank comparison sees, the
    # clamped view (0) is only ever used to build an in-range address.
    esent = torch.where(good, e_flat, torch.full_like(e_flat, -1))
    ecl = torch.where(good, e_flat, torch.zeros_like(e_flat))
    rank = torch.zeros(n, dtype=torch.int64, device=dev)
    for b in range(nb):
        lo = b * blk
        hi = min(lo + blk, n)
        seg = esent[lo:hi]
        r = torch.arange(hi - lo, device=dev)
        same = (seg[:, None] == seg[None, :]) & (r[None, :] < r[:, None])
        rank[lo:hi] = same.sum(1).to(torch.int64)

    dst = torch.full((n,), -1, dtype=torch.int64, device=dev)
    if src.numel() > 0:
        dst[src] = off[ecl[src]] + base[blk_of[src], ecl[src]] + rank[src]
    return dst, off, tot


def _effective_rows(active_rows, total: int) -> int:
    """Number of expanded_x rows the gather may write (tiling L478-481 + L119).

    ``activateRows`` is used verbatim in dropless mode, so -1/0/``>= total`` all
    mean "no truncation" here; a value inside (0, total) truncates.
    """
    if active_rows is None or active_rows < 0:
        return total
    if active_rows == 0 or active_rows >= total:
        return total
    return int(active_rows)


def _resolve_token_flag(count_or_cumsum_flag, num_type, num_flag) -> int:
    """Fold the AscendC flag (0/1/2) and the torch_npu (type, flag) pair into one.

    moe_v2_common.h L50-52 numbers the AscendC flag 1=cumsum, 2=count, 0=none;
    the torch_npu attribute numbers the *type* 0=cumsum, 1=count, 2=key-value
    (see impl/moe_init_routing_custom.py L102-105 for that enumeration).  The two
    spellings are not interchangeable digit for digit, hence this mapping.
    """
    # Two *spellings*, not two arguments: the torch_npu spelling is the pair.
    ascendc_given = count_or_cumsum_flag is not None
    npu_given = num_type is not None or num_flag is not None
    if ascendc_given and npu_given:
        raise ValueError(
            "pass either expert_tokens_count_or_cumsum_flag (AscendC spelling) or "
            "expert_tokens_num_type/expert_tokens_num_flag (torch_npu spelling), "
            f"got flag={count_or_cumsum_flag}, type={num_type}, num_flag={num_flag}"
        )
    if ascendc_given:
        flag = int(count_or_cumsum_flag)
        if flag not in (EXERPT_TOKENS_NONE, EXERPT_TOKENS_CUMSUM, EXERPT_TOKENS_COUNT):
            raise NotImplementedError(
                f"expert_tokens_count_or_cumsum_flag={flag}: only 0 (none), 1 (cumsum) "
                "and 2 (count) exist in this kernel (moe_v2_common.h L50-52)"
            )
        return flag
    if num_type is None and num_flag is None:
        return EXERPT_TOKENS_NONE
    if num_flag is None:
        raise ValueError(
            "expert_tokens_num_type was given without expert_tokens_num_flag; the flag "
            "is the on/off switch (torch_npu default is False), so state explicitly "
            "whether the count output is wanted"
        )
    if not bool(num_flag):
        return EXERPT_TOKENS_NONE
    if num_type is None:
        raise ValueError(
            "expert_tokens_num_flag=True needs expert_tokens_num_type "
            "(0 = cumsum, 1 = count)"
        )
    if int(num_type) == 0:
        return EXERPT_TOKENS_CUMSUM
    if int(num_type) == 1:
        return EXERPT_TOKENS_COUNT
    raise NotImplementedError(
        f"expert_tokens_num_type={int(num_type)}: the KEY_VALUE layout does not exist "
        "in this kernel (moe_v2_expert_token_out.h L231-258 writes plain int32 counts, "
        "never (expert_id, count) pairs)"
    )


def _check_expert_range(active_expert_range, expert_num: int) -> None:
    """Only the degenerate range is honoured; the kernel has no range attribute."""
    if active_expert_range is None:
        return
    if isinstance(active_expert_range, torch.Tensor):
        active_expert_range = active_expert_range.tolist()
    values = [int(v) for v in active_expert_range]
    if values == [0, int(expert_num)]:
        return
    raise NotImplementedError(
        f"active_expert_range={values}: the AscendC kernel has no expert-range "
        "attribute (moe_init_routing_v2_tiling.h L25-36 lists six attributes and "
        "none of them is a range; the sort in moe_v2_sort_one_core.h L49-133 covers "
        "every id), only the no-op range [0, expert_num] can be accepted"
    )


def _cpu_run(x, expert_idx, active_rows, expert_capacity, expert_num, drop_pad_mode,
             token_flag, before_capacity_flag, k):
    """Documented CPU mirror of the whole pipeline (see the module docstring)."""
    n_rows, cols = int(x.shape[0]), int(x.shape[1])
    n = n_rows * k
    dev = x.device

    dst, off, tot = _cpu_route(expert_idx.reshape(-1).to(torch.int32), expert_num, _ROUTE_BLK)
    good = dst >= 0

    if drop_pad_mode == DROPLESS_MODE:
        rows = _effective_rows(active_rows, n)
        expanded_x = torch.zeros((n, cols), dtype=x.dtype, device=dev)
        keep = torch.nonzero(good & (dst < rows), as_tuple=False).reshape(-1)
        if keep.numel() > 0:
            expanded_x[dst[keep]] = x[keep // k]
        expanded_row_idx = dst.to(torch.int32).contiguous()
        if token_flag == EXERPT_TOKENS_NONE:
            counts = torch.zeros((0,), dtype=torch.int32, device=dev)
        else:
            counts = torch.zeros(expert_num, dtype=torch.int32, device=dev)
            counts[:] = tot.to(torch.int32)
            if token_flag == EXERPT_TOKENS_CUMSUM:
                counts = counts.cumsum(0).to(torch.int32)
        before_capacity = torch.zeros((0,), dtype=torch.int32, device=dev)
        return expanded_x, expanded_row_idx, counts, before_capacity

    # drop_pad
    expanded_row_idx = torch.full((n,), -1, dtype=torch.int32, device=dev)
    expanded_x = torch.zeros((expert_num * expert_capacity, cols), dtype=x.dtype, device=dev)
    keep = torch.nonzero(good, as_tuple=False).reshape(-1)
    if keep.numel() > 0:
        e_keep = expert_idx.reshape(-1)[keep].to(torch.int64)
        rank = dst[keep] - off[e_keep]
        inside = rank < int(expert_capacity)
        keep = keep[inside]
        e_keep = e_keep[inside]
        rank = rank[inside]
        padded = e_keep * int(expert_capacity) + rank
        expanded_row_idx[keep] = padded.to(torch.int32)
        expanded_x[padded] = x[keep // k]
    counts = torch.zeros((0,), dtype=torch.int32, device=dev)
    if before_capacity_flag:
        before_capacity = tot.to(torch.int32).contiguous()
    else:
        before_capacity = torch.zeros((0,), dtype=torch.int32, device=dev)
    return expanded_x, expanded_row_idx, counts, before_capacity


def _prepare_x(x: torch.Tensor) -> tuple:
    """Return ``(view, dtype_for_kernels, original_dtype)`` for the row copy."""
    if x.dtype in _COPY_DTYPES:
        return x, x.dtype, x.dtype
    if x.element_size() == 1:
        # e.g. float8_e8m0fnu: no Triton element type, but a byte copy is exact.
        return x.view(torch.uint8), torch.uint8, x.dtype
    raise NotImplementedError(
        f"x.dtype={x.dtype} is neither a plain loadable dtype nor one byte wide, "
        "so the row copy cannot be dtype-agnostic"
    )


def moe_init_routing_v2(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    active_rows: int | None = None,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = DROPLESS_MODE,
    expert_tokens_count_or_cumsum_flag: int | None = None,
    expert_tokens_before_capacity_flag: bool = False,
    active_num: int | None = None,
    expert_tokens_num_type: int | None = None,
    expert_tokens_num_flag: bool | None = None,
    active_expert_range=None,
    quant_mode: int | None = None,
    x_dtype=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mooncake/AscendC ``MoeInitRoutingV2``: group tokens by expert id.

    Args:
        x: ``[N, cols]`` tokens.  Any dtype that is a plain scalar; the copy is
            bit-exact for every supported dtype.
        expert_idx: ``[N, k]`` int32/int64 expert id per (token, slot).  The
            flattened index ``src = t*k + j`` is the ``expanded_row_idx`` domain.
        active_rows: AscendC ``ATTR_ACTIVE_ROWS`` (tiling header L27).  Also
            accepted as ``active_num`` (the torch_npu spelling used by both call
            sites).  -1 / 0 / None / ``>= N*k`` mean "all rows"; a value inside
            (0, N*k) writes only ``expanded_x`` rows ``< active_rows``.
        expert_capacity: AscendC ``ATTR_EXPERT_CAPACITY`` (L28).  Only used when
            ``drop_pad_mode == 1``; must be > 0 there.
        expert_num: AscendC ``ATTR_EXPERT_NUM`` (L29).  Required, > 0.
        drop_pad_mode: 0 = dropless, 1 = drop_pad (L30).
        expert_tokens_count_or_cumsum_flag: 0 = none, 1 = cumsum, 2 = count
            (moe_v2_common.h L50-52).  The torch_npu spelling
            ``expert_tokens_num_type`` (0 = cumsum, 1 = count) plus
            ``expert_tokens_num_flag`` is accepted instead.
        expert_tokens_before_capacity_flag: drop_pad only, counts before the
            capacity cut (L32).
        active_expert_range: accepted only as the no-op ``[0, expert_num]``.
        quant_mode: accepted only as -1 / None (this kernel does not quantize).
        x_dtype: accepted only as None (no quantized expanded_x in these files).

    Returns:
        ``(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum,
        expert_tokens_before_capacity)``:
            expanded_x: dropless ``[N*k, cols]`` of ``x.dtype``;
                drop_pad ``[expert_num*expert_capacity, cols]`` (``.view(
                expert_num, expert_capacity, cols)`` gives the 3-D layout).
            expanded_row_idx: ``[N*k]`` int32, ``[src] = dst`` (drop_pad: the
                padded destination, -1 when dropped).
            expert_tokens_count_or_cumsum: ``[expert_num]`` int32, empty when the
                flag is 0 or when ``drop_pad_mode == 1`` (the tiling forces the
                flag to 0 there, moe_init_routing_v2_tiling.h L278-282).
            expert_tokens_before_capacity: ``[expert_num]`` int32 when requested
                in drop_pad mode, empty otherwise (L278-282 forces it off in
                dropless mode).
    """
    # ---- attribute checks -------------------------------------------------
    if active_rows is not None and active_num is not None and int(active_rows) != int(active_num):
        raise ValueError(
            f"active_rows={int(active_rows)} and active_num={int(active_num)} are the "
            "same AscendC attribute (ATTR_ACTIVE_ROWS); pass only one"
        )
    if active_rows is None:
        active_rows = active_num
    if active_rows is not None and int(active_rows) < -1:
        raise ValueError(f"active_rows={int(active_rows)} must be >= -1")

    if quant_mode is not None and int(quant_mode) != -1:
        raise NotImplementedError(
            f"quant_mode={int(quant_mode)}: these 15 files never quantize (every write "
            "in moe_v2_gather_out.h L114-133 and moe_v2_src_to_dst_with_capacity.h "
            "L144-147 is a plain dtype copy); the quantized dispatch path lives in "
            "other kernels of the fused operator"
        )
    if x_dtype is not None:
        raise NotImplementedError(
            f"x_dtype={x_dtype}: a quantized expanded_x output does not exist in these "
            "files; only expanded_x of x.dtype is produced"
        )
    if int(drop_pad_mode) not in (DROPLESS_MODE, DROP_PAD_MODE):
        raise NotImplementedError(
            f"drop_pad_mode={int(drop_pad_mode)}: only 0 (dropless) and 1 (drop_pad) "
            "exist (moe_v2_common.h L48-49)"
        )
    if int(expert_num) <= 0:
        raise ValueError(f"expert_num={int(expert_num)} must be > 0")
    expert_num = int(expert_num)
    token_flag = _resolve_token_flag(
        expert_tokens_count_or_cumsum_flag, expert_tokens_num_type, expert_tokens_num_flag
    )
    _check_expert_range(active_expert_range, expert_num)

    if int(drop_pad_mode) == DROP_PAD_MODE:
        # tiling L278-282: the two flags are mutually exclusive, the mode decides.
        if int(expert_capacity) <= 0:
            raise ValueError(
                f"expert_capacity={int(expert_capacity)} must be > 0 when "
                "drop_pad_mode=1"
            )
        expert_capacity = int(expert_capacity)
        token_flag = EXERPT_TOKENS_NONE
        before_capacity_flag = bool(expert_tokens_before_capacity_flag)
    else:
        expert_capacity = 0
        before_capacity_flag = False

    # ---- shape checks -----------------------------------------------------
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D [N, cols], got {tuple(x.shape)}")
    if expert_idx.dim() != 2:
        raise ValueError(f"expert_idx must be 2-D [N, k], got {tuple(expert_idx.shape)}")
    if expert_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"expert_idx must be int32/int64, got {expert_idx.dtype}")
    if int(expert_idx.shape[0]) != int(x.shape[0]):
        raise ValueError(
            f"x and expert_idx disagree on dim 0: {int(x.shape[0])} vs "
            f"{int(expert_idx.shape[0])}"
        )

    n_rows = int(x.shape[0])
    cols = int(x.shape[1])
    k = int(expert_idx.shape[1])
    n = n_rows * k
    dev = x.device

    empty_i32 = torch.zeros((0,), dtype=torch.int32, device=dev)
    if n == 0 or cols == 0 or dev.type == "cpu":
        # The CPU mirror is the documented test/debug path; see _cpu_run.
        return _cpu_run(x, expert_idx, active_rows, expert_capacity, expert_num,
                        int(drop_pad_mode), token_flag, before_capacity_flag, k)

    # ---- tiling -----------------------------------------------------------
    bins = max(_next_pow2(expert_num), 2)
    if bins > MAX_EXPERT_BINS:
        raise NotImplementedError(
            f"expert_num={expert_num} needs {bins} histogram bins, above the "
            f"{MAX_EXPERT_BINS}-bin single-block limit of _scan_kernel / "
            f"_counts_kernel (the AscendC kernel allows up to "
            f"{MAX_EXPERT_NUM_ASCENDC}, moe_v2_common.h L47)"
        )
    blk = _ROUTE_BLK
    nb = _ceil_div(n, blk)
    if nb > MAX_PROGRAMS:
        raise NotImplementedError(
            f"N*k={n} needs {nb} routing programs of {blk} tokens, above the "
            f"{MAX_PROGRAMS} AI Core program cap"
        )

    x_view, copy_dtype, out_dtype = _prepare_x(x.contiguous())
    x_flat = x_view.reshape(-1)
    e_flat = expert_idx.reshape(-1).contiguous().to(torch.int32)

    hist = torch.zeros(nb * bins, dtype=torch.int32, device=dev)
    off = torch.zeros(bins, dtype=torch.int32, device=dev)
    base = torch.zeros(nb * bins, dtype=torch.int32, device=dev)
    tot = torch.zeros(bins, dtype=torch.int32, device=dev)
    # -1 prefill: the documented "dropped" marker for out-of-range ids, and the
    # drop_pad "truncated by capacity" marker; both keep a single write path.
    dst = torch.full((n,), -1, dtype=torch.int32, device=dev)

    _hist_kernel[(nb,)](e_flat, hist, n, expert_num, BINS=bins, BLK=blk)
    _scan_kernel[(1,)](hist, off, base, tot, nb, BINS=bins)

    blk_r = _SCATTER_BLK_R
    while _ceil_div(n, blk_r) > MAX_PROGRAMS:
        blk_r *= 2
    blk_c = max(16, min(_next_pow2(cols), _SCATTER_BLK_C))

    if int(drop_pad_mode) == DROPLESS_MODE:
        rows = _effective_rows(active_rows, n)
        out = torch.zeros((n, cols), dtype=copy_dtype, device=dev)
        _route_kernel[(nb,)](e_flat, off, base, dst, n, expert_num, BINS=bins, BLK=blk)
        _gather_kernel[(_ceil_div(n, blk_r),)](
            x_flat, dst, out.reshape(-1), n, k, cols, rows,
            BLK_R=blk_r, BLK_C=blk_c,
        )
        if copy_dtype is not out_dtype:
            out = out.view(out_dtype)
        expanded_row_idx = dst
        if token_flag == EXERPT_TOKENS_NONE:
            counts = empty_i32
        else:
            counts = torch.zeros(expert_num, dtype=torch.int32, device=dev)
            _counts_kernel[(1,)](tot, counts, expert_num, BINS=bins, MODE=token_flag)
        return out, expanded_row_idx, counts, empty_i32

    # ---- drop_pad ---------------------------------------------------------
    out = torch.zeros((expert_num * expert_capacity, cols), dtype=copy_dtype, device=dev)
    expanded_row_idx = torch.full((n,), -1, dtype=torch.int32, device=dev)
    _route_kernel[(nb,)](e_flat, off, base, dst, n, expert_num, BINS=bins, BLK=blk)
    _capacity_kernel[(_ceil_div(n, blk_r),)](
        x_flat, e_flat, dst, off, expanded_row_idx, out.reshape(-1), n, k, cols,
        expert_num, expert_capacity, BLK_R=blk_r, BLK_C=blk_c,
    )
    if copy_dtype is not out_dtype:
        out = out.view(out_dtype)
    if before_capacity_flag:
        before_capacity = torch.zeros(expert_num, dtype=torch.int32, device=dev)
        _counts_kernel[(1,)](tot, before_capacity, expert_num, BINS=bins,
                            MODE=EXERPT_TOKENS_COUNT)
    else:
        before_capacity = empty_i32
    return out, expanded_row_idx, empty_i32, before_capacity
