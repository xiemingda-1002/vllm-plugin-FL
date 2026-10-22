# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned cache leaves for the DeepSeek V4 reference path."""

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

from vllm_fl.dispatch import CachedOp

_write = CachedOp("deepseek_v4_cache_write")


def validate_cache_spec(spec):
    if not isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
        raise TypeError("DeepSeek V4 requires an MLA cache spec")
    if (
        spec.num_kv_heads != 1
        or spec.head_size <= 0
        or spec.compress_ratio not in (1, 4, 128)
        or spec.block_size <= 0
        or spec.block_size % spec.compress_ratio
    ):
        raise ValueError("Invalid DeepSeek V4 cache dimensions or compression ratio")
    if spec.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.int8):
        raise NotImplementedError("Only plain BF16/FP16/FP32/INT8 caches are supported")
    if spec.cache_dtype_str not in (
        None,
        "auto",
        "float16",
        "bfloat16",
        "float32",
        "int8",
    ):
        raise NotImplementedError(
            "Packed or quantized MLA cache layouts are not supported"
        )


class DeepseekV4CacheLayer(nn.Module, AttentionLayerBase):
    _fl_deepseek_v4_cache = True

    def __init__(self, prefix, spec, *, vllm_config=None):
        super().__init__()
        validate_cache_spec(spec)
        config = vllm_config or get_current_vllm_config()
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate cache layer: {prefix}")
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = torch.tensor([])
        context[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        from vllm_fl.attention.deepseek_v4 import DeepseekV4CacheBackend

        return DeepseekV4CacheBackend

    def store(self, values, metadata):
        slots = (
            metadata.output_slots
            if self.spec.compress_ratio > 1
            else metadata.slot_mapping
        )
        _write(values, slots, self.kv_cache)
