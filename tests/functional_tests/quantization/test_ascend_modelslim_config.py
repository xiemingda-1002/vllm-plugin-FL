# SPDX-License-Identifier: Apache-2.0
"""Explicit config checks using a Run-owned checkpoint metadata sample."""

import json
import os
from pathlib import Path

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.fixture
def config():
    from vllm_fl.quantization.ascend.config import AscendModelSlimConfig

    path = os.environ.get("FL_MODELSLIM_DESCRIPTOR")
    if not path:
        pytest.skip("Set FL_MODELSLIM_DESCRIPTOR to Run-owned metadata sample")
    return AscendModelSlimConfig(json.loads(Path(path).read_text()))


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("layers.0.attn.wq_a", "W8A8_DYNAMIC"),
        ("model.layers.0.self_attn.wq_b", "W8A8_DYNAMIC"),
        ("model.layers.0.self_attn.wo_a", "FLOAT"),
        ("model.layers.0.mlp.gate", "FLOAT"),
        ("model.layers.0.mlp.experts", "W8A8_DYNAMIC"),
        ("model.layers.0.mlp.shared_experts.gate_up_proj", "W8A8_DYNAMIC"),
        ("model.embed_tokens", "FLOAT"),
        ("lm_head", "FLOAT"),
    ],
)
def test_actual_descriptor_prefixes(config, prefix, expected):
    assert config.get_layer_quant_type(prefix) == expected


def test_method_resolution(config):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from vllm_fl.quantization.ascend.linear import AscendW8A8DynamicLinearMethod

    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    method = config.get_quant_method(layer, "model.layers.0.self_attn.wq_a")
    assert isinstance(method, AscendW8A8DynamicLinearMethod)
    assert isinstance(
        config.get_quant_method(layer, "lm_head"), UnquantizedLinearMethod
    )
    assert config.get_quant_method(torch.nn.Identity(), "irrelevant") is None


def test_packed_mixed_and_missing(config):
    key = "model.layers.0.mlp.experts.0.down_proj.weight"
    config.quant_description[key] = "FLOAT"
    with pytest.raises(ValueError, match="Mixed"):
        config.get_layer_quant_type("model.layers.0.mlp.experts")
    del config.quant_description[key]
    with pytest.raises(ValueError, match="Missing"):
        config.get_layer_quant_type("model.layers.0.mlp.experts")


def test_framework_packing_preserves_descriptor_expert_aliases(config):
    from copy import deepcopy

    from vllm.model_executor.model_loader.utils import configure_quant_config

    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM

    original = deepcopy(DeepseekV4ForCausalLM.packed_modules_mapping)
    configure_quant_config(config, DeepseekV4ForCausalLM)
    assert config.packed_modules_mapping is DeepseekV4ForCausalLM.packed_modules_mapping
    assert "experts" not in config.packed_modules_mapping
    assert config.get_layer_quant_type("model.layers.0.mlp.experts") == "W8A8_DYNAMIC"
    assert (
        config.get_layer_quant_type("model.layers.0.mlp.shared_experts.gate_up_proj")
        == "W8A8_DYNAMIC"
    )

    class AlternatePacking(DeepseekV4ForCausalLM):
        packed_modules_mapping = {"shared_pair": ["gate_proj", "up_proj"]}

    alternate = deepcopy(AlternatePacking.packed_modules_mapping)
    configure_quant_config(config, AlternatePacking)
    assert (
        config.get_layer_quant_type("model.layers.0.mlp.shared_experts.shared_pair")
        == "W8A8_DYNAMIC"
    )
    assert config.get_layer_quant_type("model.layers.0.mlp.experts") == "W8A8_DYNAMIC"
    assert AlternatePacking.packed_modules_mapping == alternate
    assert DeepseekV4ForCausalLM.packed_modules_mapping == original


def test_external_mapper_idempotence(config):
    from vllm.model_executor.models.utils import WeightsMapper

    mapper = WeightsMapper(orig_to_new_prefix={"model.": "wrapped.model."})
    config.apply_vllm_mapper(mapper)
    once = dict(config.quant_description)
    config.apply_vllm_mapper(mapper)
    assert config.quant_description == once
    assert (
        config.get_layer_quant_type("wrapped.model.layers.0.self_attn.wq_a")
        == "W8A8_DYNAMIC"
    )


@pytest.mark.parametrize(
    "description",
    [
        {"a.weight": "W8A8"},
        {"a.weight": "W4A8_DYNAMIC"},
        {"kv_cache_type": "C8"},
        {"fa_quant_type": "W8A8_DYNAMIC"},
    ],
)
def test_unsupported_types(description):
    from vllm_fl.quantization.ascend.config import AscendModelSlimConfig

    with pytest.raises(NotImplementedError):
        AscendModelSlimConfig(description)
