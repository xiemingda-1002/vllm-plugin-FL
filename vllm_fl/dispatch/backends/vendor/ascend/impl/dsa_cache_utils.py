# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlagOS Contributors

"""KV-cache grouping and tensor planning for DeepSeek V4 DSA."""

from collections import defaultdict

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa_cache import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)

_original_group_and_unify_kv_cache_specs = (
    kv_cache_utils.group_and_unify_kv_cache_specs
)
_original_get_kv_cache_groups_uniform_groups = (
    kv_cache_utils._get_kv_cache_groups_uniform_groups
)
_original_get_kv_cache_config_deepseek_v4 = (
    kv_cache_utils._get_kv_cache_config_deepseek_v4
)


def _uses_ascend_dsa(kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
    return any(
        isinstance(
            spec,
            (AscendMLAAttentionSpec, AscendSlidingWindowMLASpec),
        )
        and getattr(spec, "model_version", None) == "deepseek_v4"
        for spec in kv_cache_specs.values()
    )


def _group_and_unify_dsa_kv_cache_specs(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    if not _uses_ascend_dsa(kv_cache_specs):
        return _original_group_and_unify_kv_cache_specs(kv_cache_specs)

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    sliding_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_specs.items():
        if isinstance(spec, AscendSlidingWindowMLASpec):
            sliding_specs[spec.block_size][name] = spec
        elif isinstance(spec, AscendMLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    # C4 defines the canonical page-size buckets used by the physical cache
    # tensors. Keep C128 in its own logical group so compressed-token block
    # accounting remains independent.
    grouped: list[UniformTypeKVCacheSpecs] = []
    for ratio in sorted(ratio_specs, key=lambda value: (value != 4, value)):
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(ratio_specs[ratio])
        assert uniform_spec is not None
        grouped.append(uniform_spec)

    for specs in sliding_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(specs)
        assert uniform_spec is not None
        grouped.append(uniform_spec)

    assert len(grouped) >= 3, "DeepSeek V4 DSA requires C4, C128, and state-cache groups."
    return grouped


def _get_dsa_kv_cache_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    first_specs = grouped_specs[0].kv_cache_specs
    if not _uses_ascend_dsa(first_specs):
        return _original_get_kv_cache_groups_uniform_groups(grouped_specs)

    c4_spec, c128_spec = grouped_specs[:2]
    assert all(
        isinstance(spec, AscendMLAAttentionSpec)
        for spec in c4_spec.kv_cache_specs.values()
    )
    assert all(
        isinstance(spec, AscendMLAAttentionSpec)
        for spec in c128_spec.kv_cache_specs.values()
    )
    groups = [
        KVCacheGroupSpec(
            layer_names=list(c4_spec.kv_cache_specs),
            kv_cache_spec=c4_spec,
        ),
        KVCacheGroupSpec(
            layer_names=list(c128_spec.kv_cache_specs),
            kv_cache_spec=c128_spec,
        ),
    ]

    num_tuples_per_group = [
        spec.get_num_layer_tuples() for spec in grouped_specs
    ]
    num_layer_tuples = kv_cache_utils._approximate_gcd(
        num_tuples_per_group,
        lower_bound=num_tuples_per_group[0],
    )
    canonical_page_sizes = c4_spec.get_page_sizes()

    for sliding_spec in grouped_specs[2:]:
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        sliding_page_sizes = sliding_spec.get_page_sizes()
        assert max(sliding_page_sizes) <= max(canonical_page_sizes)

        size_to_candidate = {
            size: min(
                candidate
                for candidate in canonical_page_sizes
                if candidate >= size
            )
            for size in sliding_page_sizes
        }
        for layer_name, layer_spec in sliding_spec.kv_cache_specs.items():
            candidate = size_to_candidate[layer_spec.page_size_bytes]
            if layer_spec.page_size_bytes < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)

        assert len({len(layers) for layers in layers_per_size.values()}) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for index in range(num_tuple_groups):
            selected_tuples = layer_tuples[index::num_tuple_groups]
            layer_names = [
                name for layer_tuple in selected_tuples for name in layer_tuple
            ]
            specs = {
                name: sliding_spec.kv_cache_specs[name]
                for name in layer_names
            }
            uniform_spec = UniformTypeKVCacheSpecs.from_specs(specs)
            assert uniform_spec is not None
            groups.append(
                KVCacheGroupSpec(
                    layer_names=layer_names,
                    kv_cache_spec=uniform_spec,
                )
            )

    return groups


def _get_dsa_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    first_group_spec = kv_cache_groups[0].kv_cache_spec
    if not (
        isinstance(first_group_spec, UniformTypeKVCacheSpecs)
        and _uses_ascend_dsa(first_group_spec.kv_cache_specs)
    ):
        return _original_get_kv_cache_config_deepseek_v4(
            vllm_config,
            kv_cache_groups,
            available_memory,
        )

    canonical_page_sizes = sorted(first_group_spec.get_page_sizes())
    tuple_page_bytes = sum(canonical_page_sizes)
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        buckets: dict[int, list[str]] = defaultdict(list)
        for layer_name in group.layer_names:
            buckets[specs[layer_name].page_size_bytes].append(layer_name)
        bucketed.append(buckets)

    num_layer_tuples = max(
        len(layers) for buckets in bucketed for layers in buckets.values()
    )
    num_blocks = available_memory // (
        tuple_page_bytes * num_layer_tuples
    )
    num_blocks = kv_cache_utils.may_override_num_blocks(
        vllm_config,
        num_blocks,
    )

    tensors: list[KVCacheTensor] = []
    for tuple_index in range(num_layer_tuples):
        for page_size in canonical_page_sizes:
            shared_by: list[str] = []
            for buckets in bucketed:
                layers = buckets.get(page_size)
                if layers is not None and tuple_index < len(layers):
                    shared_by.append(layers[tuple_index])
            tensors.append(
                KVCacheTensor(
                    size=page_size * num_blocks,
                    shared_by=shared_by,
                )
            )
    return num_blocks, tensors


def install_dsa_cache_utils() -> None:
    """Install DSA-only cache grouping while preserving generic models."""
    current = kv_cache_utils.group_and_unify_kv_cache_specs
    if getattr(current, "_fl_dsa_cache_utils", False):
        return

    _group_and_unify_dsa_kv_cache_specs._fl_dsa_cache_utils = True
    kv_cache_utils.group_and_unify_kv_cache_specs = (
        _group_and_unify_dsa_kv_cache_specs
    )
    kv_cache_utils._get_kv_cache_groups_uniform_groups = (
        _get_dsa_kv_cache_groups
    )
    kv_cache_utils._get_kv_cache_config_deepseek_v4 = (
        _get_dsa_kv_cache_config
    )


__all__ = ["install_dsa_cache_utils"]
