# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""scatter_nd_update_v2 -- AscendC to Triton.

Source: csrc/ascend/moe/scatter_nd_update_v2/
Interface (op_host/scatter_nd_update_v2_def.cpp L27-72 + infershape L30-45 + tiling L302-363):
    Input : var      [d0, ..., d_{K-1}, *tail]   (**the AscendC operator**: updates in place,
                                                 the output shares the address of var)
            indices  [*lead, K]   int32/int64
            updates  [*lead, *tail]
    Output: var      [d0, ..., d_{K-1}, *tail]   same dtype as the input var

    NOTE **semantics of this Triton implementation (differs from the AscendC description
    above; the comments were updated rather than the logic)**:
       **By default (`out=None`) a new tensor is returned and `var` is left untouched**
       (see the end of this file:
       `y = var.contiguous().clone() if out is None else out`);
       **only when `out` is passed does the write go into the caller-provided buffer**.
       => The caller **must use the return value**; dropping it and expecting `var` to be
       updated in place does not behave as one might expect.
       (Also note: when `var` is non-contiguous, `var.contiguous()` copies it once more.)
       This is a known contract deviation from the AscendC operator.
    Attr  : strides (REQUIRED, ListInt, length = K)

Tiling derivation (scatter_nd_update_v2_tiling.cpp L306-352):
    K          = indices.shape[-1]
    indexDim_  = K
    # L126-129: the strides are used directly as linearization coefficients
    indicesMask_[k] = strides[k]
    scatterLength_  = prod(var.shape[K:])             # L325-329 (contiguous length updated by one index tuple)
    indexRow        = prod(indices.shape[:-1])        # L330-333
    totalLength     = prod(var.shape[:K])             # L316-319

Formulas (op_kernel/scatter_nd_update_linear_index.h L191-216 Compute4LinearIndex):
    linear_index[n] = sum_{k<K} indices[n, k] * strides[k]          # L208 Muls + L210 Add
(op_kernel/scatter_nd_update_no_sort.h L96-121 ProcessOneIndex):
    var_flat[linear_index[n] + s] = updates_flat[n*S + s],  s in [0, S)   # L119-121
(op_kernel/scatter_nd_update_v2.h L157-215 Compute/CopyOut, sorted path):
    First sort by linear index in descending order
    (scatter_nd_update_linear_index.h L218-253 ComputeForSort), then traverse in reverse
    and skip duplicates with `if (curIdxValue == lastProcessedIdx_) continue;` (L205-208) --
    i.e. **only one update takes effect per linear_index**.

Semantic assumption (important): the two paths (sort / no_sort) are equivalent by design to
    "for the same linear_index, **the update with the largest flat index n wins**
     (last write wins)".
    The no_sort path writes in increasing n order (L96-104), which is last-wins by
    construction; the sort path keeps, among a group of duplicates, "the entry that ends up
    last in the sorted array" (with a stable sort that is the one with the largest original n).
    Note: the AscendC Sort is unstable and only sorts within blockLength-sized blocks, so with
    duplicate indices across blocks the original implementation is itself non-deterministic.
    This implementation provides deterministic last-wins behavior.

Implementation: two stages (both pure Triton)
    1) linear index kernel: li[n] = sum_k indices[n,k]*strides[k], and at the same time use
       tl.atomic_max on LAST[li] to record the largest n for that address (the last writer);
    2) scatter kernel: only write var_flat[li[n] + s] when LAST[li[n]] == n.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["scatter_nd_update_v2"]

#: `LAST` dedup-table cache: (phys_range, device) -> int32 buffer filled with -1.
#: In production this operator is called on **every layer and every step**
#: (DeepSeek-V4: 43 layers x 2 sites), and each `torch.full((numel(var),), ...)` would
#: allocate a fresh buffer (up to tens of MB) only to drop it again. With 60GB of HBM
#: already occupied by weights this easily fails device-side allocation
#: (observed as `wait for compute device to finish failed, runtime result = 507035`).
#: Caching the buffer reduces the work to a single `fill_(-1)` and removes the repeated
#: allocations.
_LAST_CACHE: dict = {}


def _alloc_last(phys_range: int, dev) -> torch.Tensor:
    key = (int(phys_range), str(dev))
    buf = _LAST_CACHE.get(key)
    if buf is None:
        buf = torch.full((phys_range,), -1, dtype=torch.int32, device=dev)
        _LAST_CACHE[key] = buf
    else:
        buf.fill_(-1)
    return buf


@triton.jit
def _snd_linear_index_kernel(
    IDX, STRIDES, LI, LAST, N, K, LAST_N,
    BLK_N: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLK_N + tl.arange(0, BLK_N)
    rmask = rows < N

    kk = tl.arange(0, BLK_K)
    kmask = kk < K

    st = tl.load(STRIDES + kk, mask=kmask, other=0).to(tl.int32)
    idx = tl.load(
        IDX + rows[:, None] * K + kk[None, :],
        mask=rmask[:, None] & kmask[None, :], other=0,
    ).to(tl.int32)

    li = tl.sum(idx * st[None, :], axis=1).to(tl.int32)
    # Addresses must be clamped: LAST only has LAST_N elements and an out-of-range atomic
    # would corrupt HBM (in production this shows up as `PTA call acl api failed` /
    # runtime 507035 on all 8 cards at once).
    ok = rmask & (li >= 0) & (li < LAST_N)
    li_c = tl.minimum(tl.maximum(li, 0), LAST_N - 1)
    # LI holds the **raw** linear index (possibly out of range); the scatter kernel
    # re-checks it.
    tl.store(LI + rows, li, mask=rmask)
    # LAST[li] = max(LAST[li], n)  -> flat index of the last update for that address
    tl.atomic_max(LAST + li_c, rows, mask=ok)


@triton.jit
def _snd_scatter_kernel(
    LI, LAST, UPD, OUT, N, S, LAST_N, OUT_N,
    BLK_N: tl.constexpr,
    BLK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    rows = pid_n * BLK_N + tl.arange(0, BLK_N)
    rmask = rows < N

    li = tl.load(LI + rows, mask=rmask, other=-1).to(tl.int32)
    li_ok = rmask & (li >= 0) & (li < LAST_N)
    # Always use the clamped value for addressing: when li is out of range (LI may hold
    # uninitialized garbage) `OUT + li + s` computes a wild address, and even with the lane
    # masked out the Ascend backend may still raise "vector core exception" (507035) during
    # address computation.
    li_c = tl.minimum(tl.maximum(li, 0), LAST_N - 1)
    last = tl.load(LAST + li_c, mask=li_ok, other=-1)
    win = li_ok & (last == rows)

    s = pid_s * BLK_S + tl.arange(0, BLK_S)
    smask = s < S
    off = li_c[:, None] + s[None, :]
    m = win[:, None] & smask[None, :] & (off < OUT_N)

    u = tl.load(UPD + rows[:, None] * S + s[None, :], mask=m, other=0)
    tl.store(OUT + off, u, mask=m)


def scatter_nd_update_v2(
    var: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    strides,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Equivalent to the AscendC ScatterNdUpdateV2 (which updates var in place).

    Args:
        var:     [d0..d_{K-1}, *tail]
        indices: [*lead, K], int32 or int64
        updates: [*lead, *tail]
        strides: integer sequence of length K (the strides attribute)
        out:     optional output buffer pre-filled with the contents of var
                 (defaults to clone(var))

    Returns:
        the updated tensor (same dtype/shape as var)
    """
    assert indices.dim() >= 1, "indices must have at least 1 dim"
    K = int(indices.shape[-1])
    assert K >= 1, f"invalid last dim (K) of indices: {K}"
    assert len(strides) == K, f"len(strides)={len(strides)} != K={K}"
    assert var.dim() >= K, f"var has {var.dim()} dims < K={K}"

    N = 1
    for d in indices.shape[:-1]:
        N *= int(d)
    S = 1
    for d in var.shape[K:]:
        S *= int(d)

    tail = tuple(int(d) for d in var.shape[K:])
    want = tuple(int(d) for d in indices.shape[:-1]) + tail
    got = tuple(int(d) for d in updates.shape)
    if got != want:
        # Production call sites (vllm_fl's indexer_quant_scatter_part1) may flatten the
        # update into a form other than ``indices.shape[:-1] + tail``; for example with
        # indices[1,1,1] + var[...,128] they pass updates[1,128]. The AscendC version
        # addresses the raw buffer linearly and tolerates this. As long as the element
        # count matches, reshape to the canonical shape; otherwise raise.
        if int(updates.numel()) == (math.prod(want) if want else 1):
            updates = updates.reshape(want)
        else:
            raise ValueError(
                f"updates shape {got} (numel={updates.numel()}) is neither the canonical "
                f"shape {want} nor a total-element match -> cannot be interpreted safely"
            )

    st = [int(s) for s in strides]
    # Largest reachable linear-index offset + single-update length -> LAST dedup table size
    max_off = 0
    for k in range(K):
        max_off += (int(var.shape[k]) - 1) * st[k]
    phys_range = max_off + S
    assert phys_range < 2 ** 31 - 1, f"linear index range {phys_range} exceeds int32"

    y = var.contiguous().clone() if out is None else out
    y = y.contiguous() if not y.is_contiguous() else y
    if N == 0:
        return y

    dev = var.device
    idx = indices.contiguous().view(N, K)
    upd = updates.contiguous().view(N, S)
    y_flat = y.view(-1)

    # LI is filled with -1 rather than left empty: out-of-range rows are never written and
    # uninitialized garbage would make address computation land on a wild address (see the
    # comment in _snd_scatter_kernel).
    li = torch.full((N,), -1, dtype=torch.int32, device=dev)
    last = _alloc_last(phys_range, dev)
    st_t = torch.tensor(st, dtype=torch.int32, device=dev)

    BLK_N = 64
    BLK_K = max(2, next_pow2(K))
    _snd_linear_index_kernel[(ceil_div(N, BLK_N),)](
        idx, st_t, li, last, N, K, phys_range,
        BLK_N=BLK_N, BLK_K=BLK_K,
    )

    BLK_N2 = 32
    BLK_S = min(next_pow2(S), 256)
    _snd_scatter_kernel[(ceil_div(N, BLK_N2), ceil_div(S, BLK_S))](
        li, last, upd, y_flat, N, S, phys_range, int(y_flat.numel()),
        BLK_N=BLK_N2, BLK_S=BLK_S,
    )
    return y
