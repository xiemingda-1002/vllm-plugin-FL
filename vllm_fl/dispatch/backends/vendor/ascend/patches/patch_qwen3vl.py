# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is adapted from the matching vLLM-Ascend v0.24.0rc1
# ``vllm_ascend/patch/worker/patch_qwen3vl.py`` for FL's vendor-scoped runtime.
# Licensed under the Apache License, Version 2.0.
"""Current-v0.24 Qwen3-VL Ascend worker patch.

Imported solely by ``apply_ascend_patches`` after the fused MRoPE custom op is
registered.  The fallback is deliberately retained for ordinary Qwen3 rotary
embeddings.
"""
from __future__ import annotations

from functools import wraps

import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer, Qwen3VLForConditionalGeneration, pos_embed_interpolate_native,
)
from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

from vllm_fl.ascend_forward_context import _EXTRA_CTX


def _deepstack_tp_wrap(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        embeds = func(*args, **kwargs)
        if embeds is None:
            return embeds
        try:
            enabled = bool(_EXTRA_CTX.flash_comm_v1_enabled)
        except (AssertionError, AttributeError, KeyError):
            enabled = False
        if not enabled:
            return embeds
        tp_size, tp_rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        embeds.tensors = {name: value.chunk(tp_size)[tp_rank] for name, value in embeds.tensors.items()}
        return embeds
    return wrapped


def _qwen3_attention_forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    # FL does not replace MRotaryEmbedding with a distinct Ascend subclass.
    # The upstream base type therefore identifies the same MRoPE contract here;
    # ordinary RotaryEmbedding remains on the rc1 fallback below.
    if isinstance(self.rotary_emb, MRotaryEmbedding):
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos_sin = cos_sin.to(device=qkv.device, dtype=qkv.dtype)
        q, k, v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv, q_weight=self.q_norm.weight, k_weight=self.k_norm.weight,
            cos_sin=cos_sin, num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim, eps=self.q_norm.variance_epsilon,
            mrope_section=self.rotary_emb.mrope_section,
            is_interleaved=self.rotary_emb.mrope_interleaved, rope_dim=self.rotary_emb.rotary_dim)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
    output, _ = self.o_proj(self.attn(q, k, v))
    return output


def _fast_pos_embed_interpolate(self, grid_thw):
    return torch.cat([pos_embed_interpolate_native(self.pos_embed.weight, t, h, w,
        self.num_grid_per_side, self.spatial_merge_size, self.dtype) for t, h, w in grid_thw], dim=0)


def _patch_qwen3vl_moe_model_config(model_cls=Qwen3VLMoeForConditionalGeneration) -> None:
    """Preserve the dense Qwen3-VL encoder configuration invariant for MoE.

    Upstream Qwen3VLMoeForConditionalGeneration deliberately bypasses the
    dense Qwen3VLForConditionalGeneration initializer. Unlike the dense
    initializer, that path does not retain ``vllm_config.model_config`` on the
    instance. The inherited encoder CUDA graph methods access that instance
    field. This Ascend-only compatibility wrapper restores the ownership
    invariant after the original MoE initializer has completed, without
    changing construction or model math.
    """
    original_init = model_cls.__init__
    if getattr(original_init, "_fl_qwen3vl_model_config_patch", False):
        return

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "model_config"):
            self.model_config = kwargs["vllm_config"].model_config

    wrapped_init._fl_qwen3vl_model_config_patch = True
    model_cls.__init__ = wrapped_init


def apply_qwen3vl_patch() -> None:
    """Idempotently install rc1 Qwen3-VL methods on the Ascend worker only."""
    if getattr(apply_qwen3vl_patch, "_applied", False):
        return
    Qwen3Attention.forward = _qwen3_attention_forward
    Qwen3MoeAttention.forward = _qwen3_attention_forward
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = _deepstack_tp_wrap(
        Qwen3VLForConditionalGeneration._get_deepstack_input_embeds)
    Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate
    _patch_qwen3vl_moe_model_config()
    try:
        from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM
        if not hasattr(Qwen3MoeLLMForCausalLM, "start_layer"):
            Qwen3MoeLLMForCausalLM.start_layer = property(lambda self: self.model.start_layer)
        if not hasattr(Qwen3MoeLLMForCausalLM, "end_layer"):
            Qwen3MoeLLMForCausalLM.end_layer = property(lambda self: self.model.end_layer)
    except Exception:
        pass
    apply_qwen3vl_patch._applied = True
