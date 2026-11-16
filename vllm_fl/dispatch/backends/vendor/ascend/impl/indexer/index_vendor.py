# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax indexer bridges to the selected operator bundle."""

from __future__ import annotations

import torch


def minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    from vllm_fl.dispatch.backends.vendor.ascend.attention_selector import (
        get_minimax_sparse_kernels,
    )

    return get_minimax_sparse_kernels().index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        max_seq_len,
        num_kv_heads,
        sm_scale,
    )


def minimax_m3_index_topk(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm_fl.dispatch.backends.vendor.ascend.attention_selector import (
        get_minimax_sparse_kernels,
    )

    return get_minimax_sparse_kernels().index_topk(
        score,
        cu_seqlens_q,
        prefix_lens,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
        out,
    )
