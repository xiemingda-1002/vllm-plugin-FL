# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from dataclasses import replace
from typing import Any

import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.worker.utils import bind_kv_cache as upstream_bind_kv_cache


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Any],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """Bind vendor cache leaves without advertising unrelated CUDA capabilities.

    vLLM 0.24 permits multiple caches per decoder layer only for CUDA-like,
    XPU and CPU platforms. Iluvatar uses the same runner cache representation
    but deliberately reports is_cuda_alike() as False. Handle its binding
    here without enabling unrelated CUDA kernels or allocator capabilities.
    """
    iluvatar = (
        getattr(current_platform, "vendor_name", None) == "iluvatar"
        and current_platform.device_type == "cuda"
    )
    ascend_leaves = (
        getattr(current_platform, "vendor_name", None) == "ascend"
        and current_platform.device_type == "npu"
        and bool(kv_caches)
        and all(
            getattr(forward_context[name], "_fl_deepseek_v4_cache", False)
            or getattr(forward_context[name], "_fl_sparse_mla_cache", False)
            for name in kv_caches
        )
    )
    if not (iluvatar or ascend_leaves):
        upstream_bind_kv_cache(
            kv_caches, forward_context, runner_kv_caches, num_attn_module
        )
        return

    assert len(runner_kv_caches) == 0
    index_to_names: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index_to_names[extract_layer_index(layer_name, num_attn_module)].append(
            layer_name
        )

    # Match the GPU runner's order, including every cache in a layer. Keep
    # references to the original tensors: graph capture and shared KV layers
    # depend on their identity, storage and layout remaining unchanged.
    for layer_index in sorted(index_to_names):
        for layer_name in index_to_names[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


class _PageSizeZeroers:
    def __init__(self, zeroers):
        self.zeroers = zeroers

    def zero_block_ids(self, block_ids):
        for zeroer in self.zeroers:
            zeroer.zero_block_ids(block_ids)


def create_kv_block_zeroer(**kwargs):
    """Use the upstream zero kernel for each real Ascend page size.

    Independent contiguous cache components need not have equal byte widths.
    Padding their pages would break the native attention input layout.
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.utils import KVBlockZeroer

    if getattr(current_platform, "vendor_name", None) != "ascend":
        return KVBlockZeroer(**kwargs)
    groups = list(kwargs["attn_groups_iter"])
    sizes = kwargs["kernel_block_sizes"]
    context = kwargs["static_forward_context"]
    excluded = kwargs.get("runner_only_attn_layers") or set()
    buckets = defaultdict(list)
    for group in groups:
        spec = group.kv_cache_spec
        if not isinstance(spec, FullAttentionSpec):
            continue
        if group.kv_cache_group_id >= len(sizes):
            continue
        kernel_bs = sizes[group.kv_cache_group_id]
        dim = group.backend.get_kv_cache_block_dim(
            kernel_bs,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=kwargs["cache_dtype"],
        )
        names_by_size = defaultdict(list)
        for name in group.layer_names:
            if name in excluded:
                continue
            cache = context[name].kv_cache
            if isinstance(cache, torch.Tensor):
                page_bytes = (
                    cache.stride(dim)
                    * cache.element_size()
                    * (spec.block_size // kernel_bs)
                )
                names_by_size[page_bytes].append(name)
        for page_bytes, names in names_by_size.items():
            buckets[page_bytes].append(replace(group, layer_names=names))
    return _PageSizeZeroers(
        [
            KVBlockZeroer(**{**kwargs, "attn_groups_iter": bucket})
            for bucket in buckets.values()
        ]
    )
