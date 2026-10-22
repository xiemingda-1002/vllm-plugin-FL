# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""scatter_nd_update -- call-site adapter over the existing scatter_nd_update_v2.

No kernel is added here. Two facts decide that:

1. csrc/ascend has **no** ``scatter_nd_update`` operator; the AscendC operator in
   this tree is ``csrc/ascend/moe/scatter_nd_update_v2/`` and
   impl/scatter_nd_update_v2.py already implements it (in place, with the
   ``strides`` attribute derived from the addressed tensor exactly as the C++
   wrapper does, and last-write-wins on duplicate indices -- documented in that
   file, L14-54).
2. The four ``torch_npu.npu_scatter_nd_update_`` call sites need nothing that
   implementation does not provide.

Call sites (4, all in vllm_fl/attention/ascend/sfa_v1.py)
--------------------------------------------------------
L1780-1784  npu_scatter_nd_update_(kv_cache[0].view(-1, packed_head_dim),
                                   slot_mapping_sfa.view(-1, 1),
                                   packed_kv.view(-1, packed_head_dim))
L1854-1858  npu_scatter_nd_update_(kv_cache[0].view(-1, fused_kv_no_split.shape[-1]),
                                   slot_mapping_sfa[:num_actual_tokens].view(-1, 1),
                                   fused_kv_no_split[:num_actual_tokens])
L1905-1909  npu_scatter_nd_update_(kv_cache[dsa_k_cache_idx].view(-1, k_li.shape[-1]),
                                   slot_mapping.view(-1, 1),
                                   k_li.view(-1, k_li.shape[-1]))
L1923-1927  npu_scatter_nd_update_(kv_cache[dsa_k_scale_cache_idx].view(-1, k_li_scale.shape[-1]),
                                   slot_mapping.view(-1, 1),
                                   k_li_scale.view(-1, k_li_scale.shape[-1]))

All four have the same shape contract: ``var`` is a 2-D view [num_slots,
row_width] of the paged cache, ``indices`` is ``slot_mapping.view(-1, 1)`` (so
K = 1), and ``updates`` is [num_tokens, row_width]. With K = 1 and a contiguous
``var`` the linear index of scatter_nd_update_v2 is just
``indices[:, 0] * row_width`` (sum_k indices[:, k] * strides[k], strides ==
[row_width, 1]), which is the slot address of the paged cache. The contract
matches one-for-one, so v2 is enough -- there is no semantic gap to fill.

The ``[N, 1]`` index shape is not cosmetic: impl/linearnorm/minimax_m3_qknorm_rope.py
L161-164 records it as hardware-verified -- a 1-D tensor of N slots is read as a
*single* N-dimensional coordinate, so the row id must be passed as [N, 1], which
is exactly what all four call sites do.

Negative slots (graph capture pads unused slots with -1, see the same file
L154-158) are skipped by v2: a negative entry produces a negative linear index,
which the index kernel excludes and the scatter kernel never writes. That is the
ScatterNd behaviour the call sites rely on.

Argument mapping
----------------
::

    torch_npu.npu_scatter_nd_update_(var, indices, updates)
        == scatter_nd_update_v2(var, indices, updates)

Both write into ``var`` and return the same tensor, so
``scatter_nd_update_(...)`` followed by reading ``var`` and
``y = scatter_nd_update_(...)`` behave identically.

Duplicates / out-of-range entries: the AscendC sort path is itself
non-deterministic for duplicate linear indices (scatter_nd_update_v2.py L46-54);
v2 implements a deterministic last-write-wins. v2 also clamps out-of-range
linear indices and skips them, which is the memory-safe reading of the same
contract (the AscendC body has no lower bound).
"""

from __future__ import annotations

import torch

from .scatter_nd_update_v2 import scatter_nd_update_v2

__all__ = ["scatter_nd_update"]


def scatter_nd_update(
    var: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    strides=None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-place scatter, identical to ``torch_npu.npu_scatter_nd_update_``.

    Args:
        var:     [d0, ..., d_{K-1}, *tail]; updated in place (the call sites pass
                 a 2-D view of the paged KV cache).
        indices: [*lead, K] int32/int64; the call sites pass ``slot_mapping.view(-1, 1)``.
        updates: [*lead, *tail] values to write.
        strides: optional, forwarded to scatter_nd_update_v2 (derived from ``var``
                 when omitted, which is what the C++ wrapper does).
        out:     optional explicit output buffer, forwarded unchanged.

    Returns:
        ``var`` (updated in place).
    """
    if indices.dim() < 1:
        raise ValueError(f"indices must have at least 1 dim, got {tuple(indices.shape)}")
    if var.dim() < int(indices.shape[-1]):
        raise ValueError(
            f"var has {var.dim()} dims but indices address {int(indices.shape[-1])} of them"
        )
    return scatter_nd_update_v2(var, indices, updates, strides=strides, out=out)
