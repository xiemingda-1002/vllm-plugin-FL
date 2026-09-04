# SPDX-License-Identifier: Apache-2.0
"""Small FL-owned helpers shared by the Ascend DSA implementation."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.model_executor.models.utils import extract_layer_index

from vllm_fl.cpu_binding import AscendDeviceType, get_ascend_device_type

_ATTENTION_CALCULATION_STREAM: torch.npu.Stream | None = None


def get_ascend_option(name: str, default: Any = False) -> Any:
    """Read an optional Ascend setting from vLLM's additional config."""
    try:
        additional_config = get_current_vllm_config().additional_config
    except AssertionError:
        additional_config = None
    if not additional_config:
        return default
    return additional_config.get(name, default)


def enable_dsa_cp() -> bool:
    """DSA context parallelism is outside the initial FL migration scope."""
    return False


def olora_tp_enable() -> bool:
    """The initial FL DSA route uses the model's ordinary TP partitioning."""
    return False


def attention_calculation_stream() -> torch.npu.Stream:
    global _ATTENTION_CALCULATION_STREAM
    if _ATTENTION_CALCULATION_STREAM is None:
        _ATTENTION_CALCULATION_STREAM = torch_npu.npu.Stream()
    return _ATTENTION_CALCULATION_STREAM


def npu_stream_switch(target_stream: torch.npu.Stream, *, enabled: bool = True):
    if not enabled:
        return nullcontext()
    if target_stream is None:
        raise ValueError("target_stream must be provided when stream switching is enabled")
    return torch.npu.stream(target_stream)


def extract_dsv4_layer_index(config: Any, layer_name: str) -> int:
    layer_idx = extract_layer_index(layer_name)
    if ".mtp." in f".{layer_name}." and layer_idx < config.num_hidden_layers:
        return config.num_hidden_layers + layer_idx
    return layer_idx


def get_dsv4_compress_ratio(config: Any, layer_idx: int) -> int:
    compress_ratios = getattr(config, "compress_ratios", None)
    if compress_ratios is None or layer_idx >= len(compress_ratios):
        return 0
    return compress_ratios[layer_idx]


def get_compressed_pos_and_indices(
    num_computed_tokens: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    arrange_np: np.ndarray,
    kv_cache_groups: Any,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Build the per-cache-group compressed positions for a DSA step."""
    if num_computed_tokens.shape != num_scheduled_tokens.shape:
        raise ValueError(
            "num_computed_tokens and num_scheduled_tokens must have "
            "the same shape"
        )
    if np.any(num_computed_tokens < 0) or np.any(num_scheduled_tokens < 0):
        raise ValueError("token counts must be non-negative")

    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    positions_by_group: list[np.ndarray] = []
    request_indices_by_group: list[np.ndarray] = []
    scheduled_counts_by_group: list[np.ndarray] = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = next(iter(spec.kv_cache_specs.values()))
        compress_ratio = getattr(spec, "compress_ratio", 1)
        if compress_ratio > 1:
            historical_lengths = num_computed_tokens // compress_ratio
            total_lengths = (
                num_computed_tokens + num_scheduled_tokens
            ) // compress_ratio
        else:
            historical_lengths = num_computed_tokens
            total_lengths = num_computed_tokens + num_scheduled_tokens

        new_counts = total_lengths - historical_lengths
        prefix_offsets = np.concatenate(
            ([0], np.cumsum(new_counts[:-1]))
        )
        compressed_positions = np.arange(
            int(np.sum(new_counts)), dtype=np.int64
        ) + np.repeat(historical_lengths - prefix_offsets, new_counts)
        request_indices = np.repeat(arrange_np, new_counts)

        positions_by_group.append(compressed_positions)
        request_indices_by_group.append(request_indices)
        scheduled_counts_by_group.append(new_counts)

    return (
        positions_by_group,
        request_indices_by_group,
        scheduled_counts_by_group,
    )


def compute_dsa_slot_mappings(
    block_tables: Any,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    positions_by_group: list[np.ndarray],
    request_indices_by_group: list[np.ndarray],
) -> None:
    """Populate ordinary and compressed DSA slot-mapping buffers."""
    for group_index, block_table in enumerate(block_tables.block_tables):
        compressed_positions = positions_by_group[group_index]
        request_indices = request_indices_by_group[group_index]
        if compressed_positions.size == 0:
            continue
        block_indices = compressed_positions // block_table.block_size
        block_numbers = block_table.block_table.np[
            request_indices,
            block_indices,
        ]
        block_offsets = compressed_positions % block_table.block_size
        np.add(
            block_numbers * block_table.block_size,
            block_offsets,
            out=block_table.slot_mapping.np[: compressed_positions.size],
        )
        block_table.slot_mapping.copy_to_gpu(compressed_positions.size)


@lru_cache(maxsize=1)
def is_vllm_0202() -> bool:
    import vllm

    return vllm.__version__.split("+")[0] == "0.20.2"


__all__ = [
    "AscendDeviceType",
    "attention_calculation_stream",
    "enable_dsa_cp",
    "extract_dsv4_layer_index",
    "compute_dsa_slot_mappings",
    "get_compressed_pos_and_indices",
    "get_ascend_device_type",
    "get_ascend_option",
    "get_dsv4_compress_ratio",
    "is_vllm_0202",
    "npu_stream_switch",
    "olora_tp_enable",
]
