# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""lightning_indexer -- Triton equivalent of `npu_lightning_indexer`

## Provenance (important, do not delete)
The non-quantized `npu_lightning_indexer` has **no AscendC implementation in the
vllm-ascend source tree**:
  - under vllm-ascend's `csrc/attention/` no `OpDef` registers the name `LightningIndexer`
    (only `FusedLightningIndexerManage` / `FusedQuantLightningIndexerManage` /
     `VllmAscendQuantLightningIndexerV2` / `LightningIndexerQuant`);
  - `csrc/attention/lightning_indexer_v2/` contains **only one** 235-line file,
    `op_kernel/arch35/vf/common/lightning_indexer_v2_vector1_base.h`, with no
    op_host/def/tiling;
  - production code `vllm_ascend/device/device_op.py` L591-592 states explicitly:
        # DSV3.2 currently has graph compilation issues when using torch_npu.npu.lightning_indexer.
        # So two branches are maintained temporarily.
        # TODO: torch.ops._C_ascend.npu_lightning_indexer needs to be removed.
  => it is provided by an **external CANN package** (or built into torch_npu), not by this
     repository's sources.

## Definition used here
`lightning_indexer` is defined as the special case of the **quantized version**
(`LightningIndexerQuant`, whose source is available) with identity scales, **reusing the already
validated quantized kernel**:

    query_dequant_scale == 1.0 , key_dequant_scale == 1.0
    => score[s1,n2,s2] = sum_g relu( query[s1,n2*gs+g] . key[s2,n2] ) * weights[s1,n2*gs+g]

With scale=1 this is **bit-identical** to the `lightning_indexer_quant` formula (source:
`lightning_indexer_quant_service_cube.h` L542-543 applies `reluPre=1` along the Cube QK path;
`..._service_vector.h` L383 multiplies by kScale only afterwards), because `x * 1.0 == x` is
exact in fp16.

## Call-site style (`vllm_ascend/device/device_op.py` L632-640)
    torch.ops._C_ascend.npu_lightning_indexer(
        query=q_li, key=kv_cache[idx], weights=weights,
        actual_seq_lengths_query=..., actual_seq_lengths_key=...,
        block_table=attn_metadata.block_table,
        layout_query="TND", layout_key="PA_BSND",
        sparse_count=2048, sparse_mode=3)
i.e. "non-quantized = quantized minus `query_dequant_scale`/`key_dequant_scale`/
`query_quant_mode`/`key_quant_mode`, everything else identical".

**Zero tolerance for the index output**: sparse_indices must match element-wise exactly
(short of `sparse_count` the slots are filled with -1).
**Tie-breaking**: AscendC leaves the order of equal scores undefined; this implementation picks
the **smaller index** (consistent with the quantized version).
"""

from __future__ import annotations

import torch

from .lightning_indexer_quant import lightning_indexer_quant

__all__ = ["lightning_indexer", "npu_lightning_indexer"]


def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
) -> torch.Tensor:
    """Equivalent of `npu_lightning_indexer` (see the module docstring for provenance).

    Args:
        query: int8  TND [T,N1,D] / BSND [B,S1,N1,D]
        key:   int8  PA_BSND [numBlocks, blockSize, N2, D] / BSND [B,S2,N2,D]
        weights: per q head, [T,N1] (TND) or [B,S1,N1]
        actual_seq_lengths_query/key, block_table, layout_*, sparse_count, sparse_mode:
            same names and meaning as the quantized version (TND needs cu_seqlens;
            PA_BSND needs block_table)
        query_quant_mode/key_quant_mode: kept for signature compatibility only; the
            non-quantized version has no quant modes (only 0 is accepted)

    Returns:
        sparse_indices int32, TND -> [T,N2,K]; BSND -> [B,S1,N2,K]
    """
    if int(query_quant_mode) != 0 or int(key_quant_mode) != 0:
        raise NotImplementedError(
            "the non-quantized lightning_indexer has no quant_mode semantics (only 0 is "
            "accepted); use lightning_indexer_quant for the quantized variant")
    if query.dtype != torch.int8 or key.dtype != torch.int8:
        raise ValueError("query/key must be int8 (same as the quantized version)")

    tnd = layout_query == "TND"
    if tnd:
        T, N1, _ = query.shape
        rows_w = T
        B = None
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND requires actual_seq_lengths_query (cu_seqlens)")
        cu = actual_seq_lengths_query.detach().to(torch.int64).cpu().flatten()
        if cu.numel() >= 2:
            B = cu.numel() - 1
    else:
        B, S1, N1, _ = query.shape
        rows_w = B * S1
        T = rows_w

    # Identity scale: 1.0 in fp16 (x * 1.0 is exactly x in fp16)
    qs = torch.ones(rows_w, N1, dtype=torch.float16, device=query.device)
    if layout_key in ("PA_BSND", "PA_BBND"):
        if block_table is None:
            raise ValueError("layout_key=PA_BSND requires block_table")
        num_blocks, block_size = key.shape[0], key.shape[1]
        ks = torch.ones(num_blocks, block_size, dtype=torch.float16, device=key.device)
    else:
        # BSND: the quantized version accepts [B,S2] or [S2]
        if B is None:
            raise ValueError("BSND key requires a known batch size")
        ks = torch.ones(B * key.shape[1], dtype=torch.float16, device=key.device).view(B, key.shape[1])

    return lightning_indexer_quant(
        query=query, key=key, weights=weights,
        query_dequant_scale=qs, key_dequant_scale=ks,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        query_quant_mode=0, key_quant_mode=0,
        layout_query=layout_query, layout_key=layout_key,
        sparse_count=sparse_count, sparse_mode=sparse_mode)


# Call-site name alias (both names are exported).
npu_lightning_indexer = lightning_indexer
