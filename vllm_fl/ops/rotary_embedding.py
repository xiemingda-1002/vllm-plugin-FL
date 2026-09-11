# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional
import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm_fl.dispatch import CachedOp

_rotary_embedding = CachedOp("rotary_embedding")


class RotaryEmbeddingFL(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base,
            is_neox_style, dtype
        )
        if current_platform.device_type == "npu":
            from vllm_fl.dispatch.backends.vendor.ascend.impl.canonical_rotary import (
                ensure_npu_rotary_embedding_registered,
            )

            ensure_npu_rotary_embedding_registered()

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        positions = positions.flatten()

        if current_platform.device_type == "npu" and key is not None:
            return torch.ops.vllm.npu_rotary_embedding(
                positions,
                query,
                key,
                self.cos_sin_cache,
                self.head_size,
                self.rotary_dim,
                self.is_neox_style,
            )

        num_tokens = positions.shape[0]

        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        cos, sin = self.cos_sin_cache.chunk(2, dim=-1)

        q_embed, k_embed = _rotary_embedding(
            self,
            query_rot,
            key_rot,
            cos,
            sin,
            positions,
            not self.is_neox_style,  # rotary_interleaved
            True,  # inplace
        )

        if self.rotary_dim < self.head_size:
            query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
            key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
        else:
            query = q_embed.reshape(query_shape)
            key = k_embed.reshape(key_shape)

        return query, key


__all__ = ["RotaryEmbeddingFL"]
