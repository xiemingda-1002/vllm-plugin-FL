# Copyright (c) 2026 BAAI. All rights reserved.

"""Canonical functional NPU rotary op used by graph-fusion patterns."""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op

_REGISTERED = False


def npu_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the CANN rotary primitive while preserving input shapes."""
    import torch_npu

    query_shape, key_shape = query.shape, key.shape
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    if rotary_dim < head_size:
        query_by_head = query.view(num_tokens, -1, head_size)
        key_by_head = key.view(num_tokens, -1, head_size)
        query_rot = query_by_head[..., :rotary_dim]
        key_rot = key_by_head[..., :rotary_dim]
        query_pass = query_by_head[..., rotary_dim:]
        key_pass = key_by_head[..., rotary_dim:]
        query_rot = query_rot.contiguous().clone().view(num_tokens, -1)
        key_rot = key_rot.contiguous().clone().view(num_tokens, -1)
        torch_npu._npu_rotary_embedding(
            positions,
            query_rot,
            key_rot,
            rotary_dim,
            cos_sin_cache,
            is_neox_style,
        )
        query = torch.cat(
            (query_rot.view(num_tokens, -1, rotary_dim), query_pass), dim=-1
        )
        key = torch.cat(
            (key_rot.view(num_tokens, -1, rotary_dim), key_pass), dim=-1
        )
    else:
        query = query.contiguous().clone().view(num_tokens, -1)
        key = key.contiguous().clone().view(num_tokens, -1)
        torch_npu._npu_rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox_style,
        )
    return query.view(query_shape), key.view(key_shape)


def _npu_rotary_embedding_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), torch.empty_like(key)


def ensure_npu_rotary_embedding_registered() -> None:
    """Register the canonical op lazily and idempotently."""
    global _REGISTERED
    if _REGISTERED or hasattr(torch.ops.vllm, "npu_rotary_embedding"):
        _REGISTERED = True
        return
    direct_register_custom_op(
        op_name="npu_rotary_embedding",
        op_func=npu_rotary_embedding,
        fake_impl=_npu_rotary_embedding_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )
    _REGISTERED = True
