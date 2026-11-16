# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax indexer bridges to the selected operator bundle."""

from __future__ import annotations

import torch


def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int | None = None,
    out: torch.Tensor | None = None,
    sm_scale: float | None = None,
) -> torch.Tensor:
    from vllm_fl.dispatch.backends.vendor.ascend.attention_selector import (
        get_minimax_sparse_kernels,
    )

    return get_minimax_sparse_kernels().index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        out,
        sm_scale,
    )
