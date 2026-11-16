# Copyright (c) 2026 BAAI. All rights reserved.
"""Vendor adapter for the public MiniMax operator bundle.

Framework code owns metadata, cache/workspace allocation and graph lifetime.
The selected bundle calls operator APIs directly; execution errors never retry.
"""

import os

import torch
from flaggems_vllm import (
    apply_minimax_m3_rotary_ascend,
    attend_minimax_m3_sparse_ascend,
    fused_minimax_m3_qknorm_rope_kv_insert_ascend,
    insert_minimax_m3_index_cache_ascend,
    insert_minimax_m3_kv_cache_ascend,
    load_minimax_m3_ascend_native,
    minimax_m3_index_decode_ascend,
    minimax_m3_index_score_ascend,
    minimax_m3_index_topk_ascend,
    prepare_minimax_m3_gemma_weight_ascend,
)


class MiniMaxM3SparseKernels:
    cache_contract = "minimax_m3_paged_v1"
    replay_contract = "persistent_metadata_v1"

    qknorm_rope_cache = staticmethod(fused_minimax_m3_qknorm_rope_kv_insert_ascend)
    rotary = staticmethod(apply_minimax_m3_rotary_ascend)
    gemma_weight = staticmethod(prepare_minimax_m3_gemma_weight_ascend)
    index_score = staticmethod(minimax_m3_index_score_ascend)
    index_topk = staticmethod(minimax_m3_index_topk_ascend)
    index_decode = staticmethod(minimax_m3_index_decode_ascend)
    insert_main_cache = staticmethod(insert_minimax_m3_kv_cache_ascend)
    insert_index_cache = staticmethod(insert_minimax_m3_index_cache_ascend)
    attend = staticmethod(attend_minimax_m3_sparse_ascend)

    @staticmethod
    def prepare():
        """Load and check the independent vendor payload before cache writes."""
        activation = getattr(torch.ops.npu, "npu_clipped_swiglu", None)
        required = {"alpha", "limit", "bias", "interleaved"}
        schemas = getattr(activation, "_schemas", {})
        if not callable(activation) or not any(
            required.issubset({arg.name for arg in schema.arguments})
            for schema in schemas.values()
        ):
            raise RuntimeError(
                "MiniMax-M3 requires the compatible NPU clipped-SwiGLU API"
            )
        if os.environ.get("VLLM_FL_M3_NPU_SPARSE_ATTN", "1") == "1":
            load_minimax_m3_ascend_native()
            if not callable(
                getattr(
                    torch.ops._flaggems_m3_ascend, "npu_sparse_attention_score", None
                )
            ):
                raise RuntimeError(
                    "MiniMax-M3 operator bundle lacks native sparse attention"
                )

    @staticmethod
    def activation(x, *, alpha, limit, bias):
        # This is a genuine vendor API adapter, not a reference implementation.
        return torch.ops.npu.npu_clipped_swiglu(
            x,
            alpha=alpha,
            limit=limit,
            bias=bias,
            interleaved=False,
        )

    @staticmethod
    def run_attention(layer, query, index_query, output):
        # Keep the runtime-owned indexer/attend ordering and capture boundary.
        layer._run_attention(query, index_query, output)
