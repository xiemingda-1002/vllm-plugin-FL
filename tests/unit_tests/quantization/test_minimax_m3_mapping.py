# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    should_ignore_layer,
)

from vllm_fl.models.minimax_m3_registration import (
    add_fused_quantization_mapping,
    register_minimax_m3_support,
)
from vllm_fl.quantization.minimax_m3 import (
    indexer_quantization_config,
    install_indexer_quantization_mapping,
)


@pytest.mark.parametrize(
    "prefix,fused,parts",
    [
        (
            "language_model.model.layers.0.self_attn",
            "qkv_proj",
            ["q_proj", "k_proj", "v_proj"],
        ),
        (
            "language_model.model.layers.0.mlp",
            "gate_up_proj",
            ["gate_proj", "up_proj"],
        ),
        (
            "language_model.model.layers.3.block_sparse_moe.shared_experts",
            "gate_up_proj",
            ["gate_proj", "up_proj"],
        ),
    ],
)
def test_named_quantization_targets_survive_fusion(prefix, fused, parts):
    class Model:
        pass

    layer = torch.nn.Linear(4, 8, bias=False)
    targets = [f"{prefix}.{part}" for part in parts]
    name = f"{prefix}.{fused}"
    assert find_matched_target(name, layer, targets) is None
    add_fused_quantization_mapping(Model)
    assert (
        find_matched_target(name, layer, targets, Model.packed_modules_mapping)
        == targets[0]
    )


def test_mapping_preserves_vision_exclusions_and_upstream_metadata():
    class Base:
        packed_modules_mapping = {"other_proj": ["left_proj", "right_proj"]}

    class Model(Base):
        pass

    add_fused_quantization_mapping(Model)
    add_fused_quantization_mapping(Model)
    assert Base.packed_modules_mapping == {"other_proj": ["left_proj", "right_proj"]}
    assert Model.packed_modules_mapping["other_proj"] == ["left_proj", "right_proj"]
    prefix = "vision_tower.vision_model.encoder.layers.0.self_attn"
    ignored = [f"{prefix}.{part}" for part in ["q_proj", "k_proj", "v_proj"]]
    assert should_ignore_layer(
        f"{prefix}.qkv_proj", ignored, Model.packed_modules_mapping
    )


def test_existing_fusion_mapping_is_preserved():
    class Model:
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
        }

    original = Model.packed_modules_mapping
    add_fused_quantization_mapping(Model)
    assert Model.packed_modules_mapping["qkv_proj"] is original["qkv_proj"]
    assert "gate_up_proj" not in original
    completed = Model.packed_modules_mapping
    add_fused_quantization_mapping(Model)
    assert Model.packed_modules_mapping is completed


def test_sparse_indexer_rules_do_not_change_dense_or_vision_mapping():
    class Model:
        pass

    add_fused_quantization_mapping(Model)
    shared = SimpleNamespace(packed_modules_mapping=Model.packed_modules_mapping)
    sparse = indexer_quantization_config(shared)
    layer = torch.nn.Linear(4, 8, bias=False)
    sparse_name = "language_model.model.layers.3.self_attn.qkv_proj"
    parts = ["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"]
    targets = [sparse_name.rsplit(".", 1)[0] + "." + part for part in parts]
    assert (
        find_matched_target(sparse_name, layer, targets, sparse.packed_modules_mapping)
        == targets[0]
    )
    assert (
        find_matched_target(
            sparse_name, layer, targets[:3], sparse.packed_modules_mapping
        )
        is None
    )
    assert should_ignore_layer(sparse_name, targets, sparse.packed_modules_mapping)
    for partial in (targets[:3], targets[3:]):
        with pytest.raises(ValueError, match="different quantization schemes"):
            should_ignore_layer(sparse_name, partial, sparse.packed_modules_mapping)

    for prefix in (
        "language_model.model.layers.0.self_attn",
        "vision_tower.vision_model.encoder.layers.0.self_attn",
    ):
        dense_targets = [prefix + "." + part for part in parts[:3]]
        name = prefix + ".qkv_proj"
        assert (
            find_matched_target(
                name, layer, dense_targets, shared.packed_modules_mapping
            )
            == dense_targets[0]
        )
        assert should_ignore_layer(name, dense_targets, shared.packed_modules_mapping)


def test_indexer_factory_preserves_existing_oot_initialization(monkeypatch):
    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot

    initialized = []

    class MinimaxM3QKVParallelLinearWithIndexer(PluggableLayer):
        def __init__(
            self,
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            total_num_index_heads,
            index_head_size,
            bias=False,
            quant_config=None,
            prefix="",
        ):
            super().__init__()
            initialized.append(self)
            self.quant_config = quant_config

    class ExistingOOT(MinimaxM3QKVParallelLinearWithIndexer):
        pass

    monkeypatch.setitem(
        op_registry_oot, MinimaxM3QKVParallelLinearWithIndexer.__name__, ExistingOOT
    )
    owner = SimpleNamespace(
        MinimaxM3QKVParallelLinearWithIndexer=MinimaxM3QKVParallelLinearWithIndexer
    )
    install_indexer_quantization_mapping(owner)
    shared = SimpleNamespace(
        packed_modules_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    )
    layer = owner.MinimaxM3QKVParallelLinearWithIndexer(
        4096, 128, 32, 4, 4, 128, quant_config=shared
    )
    assert type(layer) is ExistingOOT
    assert initialized == [layer]
    assert len(layer.quant_config.packed_modules_mapping["qkv_proj"]) == 5
    assert len(shared.packed_modules_mapping["qkv_proj"]) == 3


@pytest.mark.parametrize("regex", [False, True])
def test_indexer_factory_preserves_layer_type_targets(regex):
    class MinimaxM3QKVParallelLinearWithIndexer:
        def __init__(
            self,
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            total_num_index_heads,
            index_head_size,
            bias=False,
            quant_config=None,
            prefix="",
        ):
            pass

    owner = SimpleNamespace(
        MinimaxM3QKVParallelLinearWithIndexer=MinimaxM3QKVParallelLinearWithIndexer
    )
    install_indexer_quantization_mapping(owner)
    layer = owner.MinimaxM3QKVParallelLinearWithIndexer(4096, 128, 32, 4, 4, 128)
    target = MinimaxM3QKVParallelLinearWithIndexer.__name__
    if regex:
        target = "re:^" + target + "$"
    assert (
        find_matched_target("model.layers.3.self_attn.qkv_proj", layer, [target])
        == target
    )


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize(
    "available",
    [
        [],
        ["MiniMaxM3SparseForCausalLM"],
        ["MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"],
    ],
)
def test_registration_is_lazy_and_platform_scoped(monkeypatch, cuda, available):
    import vllm.platforms as platforms
    from vllm.model_executor.models import ModelRegistry

    calls = []
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_cuda=lambda: cuda)
    )
    monkeypatch.setattr(ModelRegistry, "get_supported_archs", lambda: available)
    monkeypatch.setattr(
        ModelRegistry, "register_model", lambda *args: calls.append(args)
    )
    register_minimax_m3_support()
    assert calls == [
        (name, f"vllm_fl.models.minimax_m3:{name}") for name in available if cuda
    ]
