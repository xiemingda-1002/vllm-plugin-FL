# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""moe_init_routing_custom -- AscendC -> Triton

Source: csrc/ascend/moe/moe_init_routing_custom/

Interface (op_host/moe_init_routing_custom_def.cpp L22-97):
    Input : x          [N, cols]              int8/fp16/bf16/fp32   (REQUIRED)
            expert_idx [N, k]                 int32                 (REQUIRED)
            scale      [expert_num, cols] or [1] fp32               (OPTIONAL)
            offset     [1] fp32                                     (OPTIONAL)
    Output: expanded_x                     dropless: [xOutNum, cols]
            expanded_row_idx               [N*k]            int32
            expert_tokens_count_or_cumsum  see below        int64
            expanded_scale                 [xOutNum]        fp32
    Attr  : active_num(-1), expert_capacity(-1), expert_num(-1), drop_pad_mode(0),
            expert_tokens_num_type(0), expert_tokens_num_flag(false), quant_mode(-1),
            active_expert_range([], empty means [0, expert_num)), row_idx_type(0)

--------------------------------------------------------------------------
Formula / semantics source
--------------------------------------------------------------------------
[A] Main flow (tiling_key = MOE_INIT_ROUTING_CUSTOM_PERFORMANCE):
    op_kernel/moe_custom_full_load.h
    L79-89   CopyIn       -- read expert_idx (flattened N*k)
    L91-129  SortCompute  -- **stable sort by ascending expert id**
                             key = -expert_id (negated, then handed to a descending Sort
                             => ascending expert ids)
                             payload = flattened source index 0..N*k-1
                             L121 Extract splits out (sorted expert id, sorted source index)
    L131-163 ExpertCountCompute -- (expert_id, count) pairs in KEY_VALUE form
    L165-227 CopyOutDynamicQuant -- dynamic quantization + writing expanded_row_idx
        L211-224: row_idx_type==1 => expanded_row_idx[dst] = src
                  row_idx_type==0 => expanded_row_idx[src] = dst (the destination index is
                  stored at the source index)
[B] Counts / prefix sums (the three semantics of expert_tokens_count_or_cumsum):
    op_kernel/moe_custom_expert_tokens_count.h
    L24-31     CUMSUM_MODE=0, COUNT_MODE=1, KEY_VALUE_MODE=2
    L237-278   Compute: COUNT -> one count per expert;
               CUMSUM -> `expertCountOutLocal[j] = loopTokenCount` (written after the
                        accumulation) => an **inclusive prefix sum**
                        cumsum[e] = #{token : exp<=e}
               KEY_VALUE -> L338-350: keep only experts with count!=0, written in ascending
                            expert order as (expert_id, count) pairs, padded with 0 elsewhere
    L146-151   KEY_VALUE output length = min(actualExpertNum+1, expertNum) * 2
[C] Shape inference: op_host/moe_init_routing_custom_infershape.cpp
    L609-661   n = max(x.dim0, expert_idx.dim0); total = n*k;
               activeNum in {0,-1} => total, otherwise min(active_num, total)
               dropless: expanded_x = [xOutNum, cols]
               expanded_row_idx = [n*k]
               cumsum/count: KEY_VALUE => [expert_num, 2]; otherwise [expertEnd-expertStart]
               expanded_scale: NON_QUANT/DYNAMIC_QUANT and dropless => [xOutNum]
    L686-697   in quantized modes the dtype of expanded_x is int8

--------------------------------------------------------------------------
Scope of this implementation (important)
--------------------------------------------------------------------------
* **Supports drop_pad_mode=0 (dropless) + quant_mode in {-1 non-quantized, 1 dynamic
  quantization}**. drop_pad_mode=1 (droppad, output [expert_num, expert_capacity, cols]) and
  quant_mode=0 (static quantization) are out of scope for this Triton implementation and raise
  explicitly when requested.
* Stable-sort assumption: elements sharing an expert id are ordered by **ascending flattened
  source index** (the stability of the AscendC hardware Sort for equal keys is not documented;
  this implementation picks a deterministic convention that matches the fp64 reference).
* The dynamic quantization chain follows AscendC exactly (full_load.h L188-199):
      smooth = x_row * scale[expert_of_this_token, :]      # fp32
      s      = max(|smooth|) / 127
      int8   = CAST_RINT(CAST_ODD(smooth / s))             # via fp16
  AscendC's CAST_ODD(fp32 -> fp16) rounding mode differs from Triton's default RN by up to
  1 LSB, so the int8 output is compared with a **+/-1 LSB** tolerance (see the test file).
* With expert_tokens_num_flag=False, AscendC does not write expert_tokens_count_or_cumsum
  (infershape does not set its shape either); this implementation returns an all-zero tensor.
* **Backend pitfall (fixed in this version)**: a `[BLK_D, BLK_C]` 2-D block combined with a
  **2-D masked store** **silently drops some lanes** on BiShengIR/FlagTree (dropped elements
  keep the initial content of the output buffer: the non-quantized path shows a whole zeroed
  segment, the quantized path shows a missing per-row prefix), and which rows/columns are
  dropped varies with the tile configuration and the compiled artifact. It always reproduces
  for non-power-of-two column counts (cols=17/24/33/48 all FAIL) and never for power-of-two
  ones (cols=32/64/128), which is why it stayed hidden while the old tests only covered
  power-of-two widths.
  Fix: `_gather_kernel` / `_rowscale_kernel` / `_apply_quant_kernel` all use a **1-D block**
  (the row index is a scalar, the columns form a 1-D tile, and the in-row loop over columns is
  a runtime loop), which measured element-wise identical for every column count; this also
  removes the risk of a degenerate `[1, N]` block tripping the FlagTree
  `BlockPtrAnalysis.cpp:309 BlockData::divBlock` assertion.
* When active_num falls inside (0, N*k), AscendC "sorts all N*k entries first and then
  truncates the output"; to avoid semantic ambiguity this implementation raises for that
  interval (-1, 0 and >=N*k are supported).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["moe_init_routing_custom"]

# expert_tokens_num_type
CUMSUM = 0
COUNT = 1
KEY_VALUE = 2
# quant_mode
NON_QUANT = -1
STATIC_QUANT = 0
DYNAMIC_QUANT = 1
# row_idx_type
GATHER = 0
SCATTER = 1

_BLK = 64


@triton.jit
def _hist_kernel(E_FLAT, HIST, CNT, NB_E, E, n, BLK: tl.constexpr, USE_CNT: tl.constexpr):
    """Build the global histogram and the per-block x per-expert counts (atomic adds, so the
    counts are order-independent)."""
    b = tl.program_id(0)
    o = b * BLK + tl.arange(0, BLK)
    m = o < n
    e = tl.load(E_FLAT + o, mask=m, other=0).to(tl.int32)
    valid = m & (e >= 0) & (e < E)
    tl.atomic_add(HIST + e, 1, mask=valid)
    if USE_CNT:
        tl.atomic_add(CNT + b * E + e, 1, mask=valid)


@triton.jit
def _excl_scan_kernel(HIST, OFF, N_TOTAL, E, BLK_E: tl.constexpr):
    """Single program: OFF = exclusive prefix sum of HIST, with the total stored into
    N_TOTAL[0].

    NOTE: within a chunk the exclusive prefix (run + in-chunk exclusive cumsum) has to be
    computed and stored *before* run is advanced; an earlier version that stored run first and
    advanced it afterwards wrote all zeros when nchunk==1.
    """
    run = tl.zeros([BLK_E], dtype=tl.int32)
    nchunk = tl.cdiv(E, BLK_E)
    for c in range(nchunk):
        e_off = c * BLK_E + tl.arange(0, BLK_E)
        m = e_off < E
        h = tl.where(m, tl.load(HIST + e_off, mask=m, other=0), 0)
        excl = tl.cumsum(h, axis=0) - h
        tl.store(OFF + e_off, run + excl, mask=m)
        run = run + h
    tl.store(N_TOTAL + tl.arange(0, 1), tl.sum(run, axis=0))


@triton.jit
def _block_base_kernel(CNT, BASE, NB, E, BLK_E: tl.constexpr):
    """Single program: BASE[b, e] = sum_{b'<b} CNT[b', e] (serial scan over blocks)."""
    run = tl.zeros([BLK_E], dtype=tl.int32)
    nchunk = tl.cdiv(E, BLK_E)
    for b in range(NB):
        for c in range(nchunk):
            e_off = c * BLK_E + tl.arange(0, BLK_E)
            m = e_off < E
            cbb = tl.load(CNT + b * E + e_off, mask=m, other=0)
            tl.store(BASE + b * E + e_off, run, mask=m)
            run = run + cbb


@triton.jit
def _route_kernel(E_FLAT, OFF, BASE, ROWO, DE, E, K, n, BLK: tl.constexpr,
                  ROW_IDX_TYPE: tl.constexpr):
    """Stable counting sort: dst[i] = OFF[e_i] + base[block_i, e_i] + rank_local_i.

    rank_local_i = #{ j in the same block : j < i and e_j == e_i } (2-D masked reduction,
    exact).
    """
    b = tl.program_id(0)
    o = b * BLK + tl.arange(0, BLK)
    i_off = tl.arange(0, BLK)
    m = o < n
    e = tl.load(E_FLAT + o, mask=m, other=0).to(tl.int32)

    same = (e[:, None] == e[None, :]) & (i_off[None, :] < i_off[:, None])
    rank = tl.sum(tl.where(same, 1, 0), axis=1)

    off = tl.load(OFF + e, mask=m, other=0)
    base = tl.load(BASE + b * E + e, mask=m, other=0)
    dst = off + base + rank

    # destination position -> flattened source index (used later to gather expanded_x)
    if ROW_IDX_TYPE == 1:      # SCATTER: ROWO[dst] = src
        tl.store(ROWO + dst, o, mask=m)
    else:                      # GATHER:  ROWO[src] = dst
        tl.store(ROWO + o, dst, mask=m)
    tl.store(DE + dst, e, mask=m)


@triton.jit
def _dst_to_src_kernel(ROWO, SRC, n, BLK: tl.constexpr, ROW_IDX_TYPE: tl.constexpr):
    """Turn expanded_row_idx back into a dst->src mapping."""
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = o < n
    v = tl.load(ROWO + o, mask=m, other=0)
    if ROW_IDX_TYPE == 1:      # SCATTER: ROWO[dst] = src
        tl.store(SRC + o, v, mask=m)
    else:                      # GATHER:  ROWO[src] = dst
        tl.store(SRC + v, o, mask=m)


@triton.jit
def _gather_kernel(X, SRC, OUTX, TOTAL, K, COLS, BLK_R: tl.constexpr, BLK_C: tl.constexpr):
    """Non-quantized path: copy whole rows of x into expanded_x following dst->src.

    WARNING: **1-D blocks only** (the row index is a scalar, the columns are a 1-D tile): a
    ``[BLK_D, BLK_C]`` 2-D block with a 2-D masked store silently drops writes on this backend
    (see the backend pitfall in the module docstring), which always reproduces for
    non-power-of-two column counts (cols=17/24/33/48); the 1-D form is element-wise identical
    for every column count measured. A row block of BLK_R rows is processed serially inside one
    program, with an in-row loop over BLK_C column chunks (runtime loop).
    """
    r0 = tl.program_id(0) * BLK_R
    for i in range(BLK_R):
        d = r0 + i
        if d < TOTAL:
            src = tl.load(SRC + d)
            if src >= 0:
                xb = (src // K) * COLS
                ob = d * COLS
                for c0 in range(0, tl.cdiv(COLS, BLK_C)):
                    c = c0 * BLK_C + tl.arange(0, BLK_C)
                    mc = c < COLS
                    v = tl.load(X + xb + c, mask=mc, other=0.0)
                    tl.store(OUTX + ob + c, v, mask=mc)


@triton.jit
def _rowscale_kernel(X, SRC, DE, SMOOTH, ESCALE, TOTAL, K, COLS,
                     BLK_R: tl.constexpr, BLK_C: tl.constexpr,
                     HAS_SMOOTH: tl.constexpr = True):
    """smooth = x_row * scale[expert_of_row]; s = max|smooth| / 127 (one s per row).

    ``HAS_SMOOTH=False`` corresponds to the production call site ``quant_mode=1, scale=None``:
    the AscendC dynamic-quant branch (``moe_v2_fullload_dynamic_quant.h`` L238-247) only does
    ``s = ReduceMax(Abs(x_row)) / 127`` without a smooth factor.

    Uses 1-D blocks only as well (see `_gather_kernel`): the row index is a scalar, the columns
    are chunked by BLK_C for an online max, and the full-row max still completes inside one
    program (no cross-program reduction).
    """
    r0 = tl.program_id(0) * BLK_R
    for i in range(BLK_R):
        d = r0 + i
        if d < TOTAL:
            src = tl.load(SRC + d)
            if src >= 0:
                xb = (src // K) * COLS
                if HAS_SMOOTH:
                    sb = tl.load(DE + d) * COLS
                run = tl.zeros([BLK_C], dtype=tl.float32)
                for c0 in range(0, tl.cdiv(COLS, BLK_C)):
                    c = c0 * BLK_C + tl.arange(0, BLK_C)
                    mc = c < COLS
                    xv = tl.load(X + xb + c, mask=mc, other=0.0).to(tl.float32)
                    if HAS_SMOOTH:
                        sm = tl.load(SMOOTH + sb + c, mask=mc,
                                     other=0.0).to(tl.float32)
                        z = xv * sm
                    else:
                        z = xv
                    run = tl.maximum(run, tl.where(mc, tl.abs(z), 0.0))
                tl.store(ESCALE + d + tl.arange(0, 1),
                         tl.max(run, axis=0, keep_dims=True) / 127.0)


@triton.jit
def _apply_quant_kernel(X, SRC, DE, SMOOTH, ESCALE, OUTX, TOTAL, K, COLS,
                        BLK_R: tl.constexpr, BLK_C: tl.constexpr,
                        HAS_SMOOTH: tl.constexpr = True):
    """int8 = CAST_RINT(CAST_ODD(smooth / s)), via fp16 (AscendC full_load.h L197-199).

    1-D blocks (see `_gather_kernel`): a 2-D masked store drops writes for non-power-of-two
    column counts.
    """
    r0 = tl.program_id(0) * BLK_R
    for i in range(BLK_R):
        d = r0 + i
        if d < TOTAL:
            src = tl.load(SRC + d)
            if src >= 0:
                xb = (src // K) * COLS
                ob = d * COLS
                # An all-zero row has s == 0: use 1.0 as a placeholder (every element of the
                # row is 0, so the quantized result is still 0), the same treatment as in
                # rms_norm_dynamic_quant.py; otherwise 0/0=NaN would be turned into -128 by
                # `tl.maximum(NaN,-128)`, while the reference implementation gives 0.
                s = tl.maximum(tl.load(ESCALE + d + tl.arange(0, 1)), 1e-30)
                if HAS_SMOOTH:
                    sb = tl.load(DE + d) * COLS
                for c0 in range(0, tl.cdiv(COLS, BLK_C)):
                    c = c0 * BLK_C + tl.arange(0, BLK_C)
                    mc = c < COLS
                    xv = tl.load(X + xb + c, mask=mc, other=0.0).to(tl.float32)
                    if HAS_SMOOTH:
                        sm = tl.load(SMOOTH + sb + c, mask=mc,
                                     other=0.0).to(tl.float32)
                        num = xv * sm
                    else:
                        num = xv
                    q = num / s
                    f = q.to(tl.float16).to(tl.float32)
                    r = tl.where(f >= 0, tl.floor(f + 0.5), tl.ceil(f - 0.5))
                    # AscendC's CAST_RINT is a **saturating** conversion (the reference
                    # implementation also clamps to [-128,127]): without this step an all-zero
                    # row with amax=0 makes q=inf/nan and the int8 result differs from AscendC
                    # by as much as 239.
                    r = tl.minimum(tl.maximum(r, -128.0), 127.0)
                    tl.store(OUTX + ob + c, r.to(tl.int8), mask=mc)


@triton.jit
def _counts_kernel(HIST, COUT, expert_start, expert_end, expert_num,
                   BLK_E: tl.constexpr, MODE: tl.constexpr):
    """Single program: turn the histogram into one of the COUNT / CUMSUM / KEY_VALUE
    outputs."""
    run = tl.zeros([1], dtype=tl.int64)
    kv = tl.zeros([1], dtype=tl.int64)
    nchunk = tl.cdiv(expert_num, BLK_E)
    for c in range(nchunk):
        e_off = c * BLK_E + tl.arange(0, BLK_E)
        m = e_off < expert_num
        h = tl.load(HIST + e_off, mask=m, other=0).to(tl.int64)
        inrange = m & (e_off >= expert_start) & (e_off < expert_end)
        if MODE == 1:      # COUNT: one count per expert
            tl.store(COUT + (e_off - expert_start), h, mask=inrange)
        elif MODE == 0:    # CUMSUM: inclusive prefix sum, cumsum[e] = #{token : expert <= e}
            cum = tl.cumsum(tl.where(m, h, 0), axis=0) + run
            tl.store(COUT + (e_off - expert_start), cum, mask=inrange)
        else:              # KEY_VALUE(2): ascending expert order, only count != 0 is kept
            nz = inrange & (h != 0)
            pos = tl.cumsum(tl.where(nz, 1, 0), axis=0) - 1 + kv
            # One 2-D [BLK_E,2] store writes (expert_id, count) at once: two separate
            # "strided + masked" stores would trip the FlagTree FLIR
            # InterleaveStatusWithMaskOptimization assertion (InterleaveOptimization.cpp:635);
            # see inplace_partial_rotary_mul.py for details.
            col = tl.arange(0, 2)
            v2 = tl.where(col[None, :] == 0, e_off.to(tl.int64)[:, None], h[:, None])
            tl.store(COUT + pos[:, None] * 2 + col[None, :], v2, mask=nz[:, None])
        run = run + tl.sum(tl.where(m, h, 0), axis=0)
        kv = kv + tl.sum(tl.where(inrange & (h != 0), 1, 0), axis=0)


def moe_init_routing_custom(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    scale: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
    *,
    active_num: int = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = CUMSUM,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = NON_QUANT,
    active_expert_range: tuple[int, int] | list[int] = (),
    row_idx_type: int = GATHER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equivalent of the AscendC MoeInitRoutingCustom (dropless; non-quantized / dynamic
    quantization).

    Returns:
        (expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale)
    """
    assert drop_pad_mode == 0, "this Triton implementation only covers drop_pad_mode=0 (dropless)"
    assert quant_mode in (NON_QUANT, DYNAMIC_QUANT), "only quant_mode in {-1, 1} is covered"
    assert expert_num > 0, "expert_num must be provided"
    assert x.dim() == 2 and expert_idx.dim() == 2, "x/expert_idx must be 2-D"
    assert expert_idx.dtype == torch.int32, "expert_idx must be int32"

    N, cols = x.shape
    k = expert_idx.shape[1]
    n = max(N, expert_idx.shape[0])
    total = n * k
    assert active_num in (-1, 0) or active_num >= total, \
        "the truncation semantics of active_num inside (0, N*k) differ from AscendC, " \
        "so this implementation does not support it"
    assert x.shape[0] == expert_idx.shape[0], "x and expert_idx must agree on dim 0"

    if len(active_expert_range) == 2:
        expert_start, expert_end = int(active_expert_range[0]), int(active_expert_range[1])
    else:
        expert_start, expert_end = 0, expert_num
    assert 0 <= expert_start < expert_end <= expert_num

    dev = x.device
    e_flat = expert_idx.reshape(-1).contiguous()
    assert e_flat.numel() == total

    nb = ceil_div(total, _BLK)
    blk_e = max(next_pow2(min(expert_num, 1024)), 2)
    hist = torch.zeros(expert_num, dtype=torch.int32, device=dev)
    cnt = torch.zeros(nb * expert_num, dtype=torch.int32, device=dev)
    off = torch.zeros(expert_num, dtype=torch.int32, device=dev)
    base = torch.zeros(nb * expert_num, dtype=torch.int32, device=dev)
    n_total = torch.zeros(1, dtype=torch.int32, device=dev)
    rowo = torch.full((total,), -1, dtype=torch.int32, device=dev)
    de = torch.zeros(total, dtype=torch.int32, device=dev)

    _hist_kernel[(nb,)](e_flat, hist, cnt, nb, expert_num, total, BLK=_BLK, USE_CNT=True)
    _excl_scan_kernel[(1,)](hist, off, n_total, expert_num, BLK_E=blk_e)
    _block_base_kernel[(1,)](cnt, base, nb, expert_num, BLK_E=blk_e)
    _route_kernel[(nb,)](e_flat, off, base, rowo, de, expert_num, k, total,
                         BLK=_BLK, ROW_IDX_TYPE=int(row_idx_type))

    src = torch.zeros(total, dtype=torch.int32, device=dev)
    _dst_to_src_kernel[(ceil_div(total, _BLK),)](rowo, src, total, BLK=_BLK,
                                                 ROW_IDX_TYPE=int(row_idx_type))

    # The quantized/copy paths all use 1-D blocks (scalar row index + 1-D column tile):
    #   * BLK_R: each program processes BLK_R rows serially (>= 2, which keeps row-level
    #     parallelism; with 1-D blocks there is no degenerate [1, N] block any more, so
    #     BLK_D>=2 is no longer needed to avoid the FlagTree
    #     `BlockPtrAnalysis.cpp:309 BlockData::divBlock` assertion)
    #   * BLK_C: in-row column chunking (runtime loop); UB usage ~ constant x BLK_C x 32 bit
    blk_r_q = 4
    blk_c_q = max(16, min(next_pow2(cols), 1024))
    if quant_mode == DYNAMIC_QUANT:
        # scale is None -> take the AscendC "pure dynamic quantization" branch: the scale is
        # derived from the per-row amax only, with no per-expert smooth factor. Production
        # (token_dispatcher.py L401 / fused_moe_layer.py) calls it this way, which an earlier
        # implementation simply asserted on.
        has_smooth = scale is not None
        if has_smooth:
            smat = scale.contiguous()
            if smat.dim() == 1:
                smat = smat.expand(expert_num, cols).contiguous()
            assert smat.shape == (expert_num, cols), \
                f"scale must be [{expert_num},{cols}], got {tuple(smat.shape)}"
        else:
            # Placeholder tensor: the kernel does not read it when HAS_SMOOTH=False
            smat = torch.zeros(1, dtype=torch.float32, device=dev)
        xq = torch.empty((total, cols), dtype=torch.int8, device=dev)
        escale = torch.zeros(total, dtype=torch.float32, device=dev)
        _rowscale_kernel[(ceil_div(total, blk_r_q),)](
            x, src, de, smat, escale, total, k, cols, BLK_R=blk_r_q, BLK_C=blk_c_q,
            HAS_SMOOTH=has_smooth)
        _apply_quant_kernel[(ceil_div(total, blk_r_q),)](
            x, src, de, smat, escale, xq, total, k, cols, BLK_R=blk_r_q, BLK_C=blk_c_q,
            HAS_SMOOTH=has_smooth)
        expanded_x = xq
    else:
        ex = torch.zeros((total, cols), dtype=x.dtype, device=dev)
        blk_r_g = 8
        blk_c_g = max(16, min(next_pow2(cols), 512))
        _gather_kernel[(ceil_div(total, blk_r_g),)](
            x, src, ex, total, k, cols, BLK_R=blk_r_g, BLK_C=blk_c_g)
        expanded_x = ex
        escale = torch.zeros(total, dtype=torch.float32, device=dev)

    # ---- expert_tokens_count_or_cumsum ----
    if expert_tokens_num_flag:
        if expert_tokens_num_type == KEY_VALUE:
            cout = torch.zeros((expert_num, 2), dtype=torch.int64, device=dev)
            _counts_kernel[(1,)](hist, cout.reshape(-1), expert_start, expert_end, expert_num,
                                 BLK_E=blk_e, MODE=KEY_VALUE)
        else:
            cout = torch.zeros(expert_end - expert_start, dtype=torch.int64, device=dev)
            _counts_kernel[(1,)](hist, cout, expert_start, expert_end, expert_num,
                                 BLK_E=blk_e, MODE=int(expert_tokens_num_type))
    else:
        cout = torch.zeros(expert_end - expert_start, dtype=torch.int64, device=dev)

    return expanded_x, rowo, cout, escale
