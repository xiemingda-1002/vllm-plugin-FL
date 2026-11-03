# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA physical leaves must retain their widths in the KV planner."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        current_platform.device_type != "npu", reason="Requires Ascend NPU"
    ),
]


def test_glm_config_preserves_checkpoint_rope_parameters():
    from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig

    parameters = {"rope_type": "default", "rope_theta": 8_000_000}
    config = GlmMoeDsaConfig(rope_parameters=parameters)
    assert config.rope_parameters == parameters
    assert config.rope_theta == parameters["rope_theta"]
    assert config.to_dict()["rope_parameters"] == parameters


@pytest.mark.parametrize("widths", [(512, 64, 128), (64, 128, 512), (128, 512, 64)])
def test_heterogeneous_leaves_keep_layout_and_budget(widths):
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    from vllm_fl.attention.sparse_mla import SparseMLACacheSpec

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )
    specs = {
        f"model.layers.0.cache{width}": SparseMLACacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=width,
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        )
        for width in widths
    }
    groups = get_kv_cache_groups(config, specs)
    assert len(groups) == 1
    assert isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
    assert groups[0].kv_cache_spec.kv_cache_specs == specs
    budget = 4 * sum(spec.page_size_bytes for spec in specs.values())
    plan = get_kv_cache_config_from_groups(config, groups, budget)
    assert plan.num_blocks == 4
    assert sum(tensor.size for tensor in plan.kv_cache_tensors) == budget
    for tensor in plan.kv_cache_tensors:
        assert tensor.block_stride == tensor.offset == 0
        assert len(tensor.shared_by) == 1
        assert tensor.size == 4 * specs[tensor.shared_by[0]].page_size_bytes
