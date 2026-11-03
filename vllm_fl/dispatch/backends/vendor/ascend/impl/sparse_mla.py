# SPDX-License-Identifier: Apache-2.0
"""Native BF16 sparse MLA providers; cache lifecycle belongs to the caller."""

import torch_npu


def sparse_mla_attention(
    query,
    query_rope,
    key_cache,
    rope_cache,
    indices,
    block_table,
    query_ends,
    seq_lens,
    scale,
):
    return torch_npu.npu_sparse_flash_attention(
        query=query,
        key=key_cache,
        value=key_cache,
        query_rope=query_rope,
        key_rope=rope_cache,
        sparse_indices=indices.reshape(query.shape[0], 1, -1),
        scale_value=scale,
        block_table=block_table,
        actual_seq_lengths_query=query_ends,
        actual_seq_lengths_kv=seq_lens,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_block_size=1,
        sparse_mode=3,
        attention_mode=2,
    )[0]


def sparse_mla_indexer(
    query,
    key_cache,
    weights,
    block_table,
    query_ends,
    seq_lens,
    topk,
):
    return torch_npu.npu_lightning_indexer(
        query=query,
        key=key_cache,
        weights=weights,
        actual_seq_lengths_query=query_ends,
        actual_seq_lengths_key=seq_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=3,
    )[0]
