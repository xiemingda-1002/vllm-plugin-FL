# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU wiring checks for MiniMax's layer-local fused projection metadata."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[3] / "vllm_fl/quantization/minimax_m3.py"
SPEC = importlib.util.spec_from_file_location("minimax_indexer_metadata", SOURCE)
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)
QKV = ["q_proj", "k_proj", "v_proj"]
INDEXER = [*QKV, "index_q_proj", "index_k_proj"]


def test_sparse_view_preserves_shared_dense_vision_and_other_rules():
    mapping = {"qkv_proj": QKV.copy(), "gate_up_proj": ["gate_proj", "up_proj"]}
    original = SimpleNamespace(
        packed_modules_mapping=mapping, ignore=["vision"], scheme=object()
    )
    sparse = metadata.indexer_quantization_config(original)
    assert sparse is not original
    assert sparse.packed_modules_mapping is not mapping
    assert sparse.packed_modules_mapping["qkv_proj"] == INDEXER
    assert original.packed_modules_mapping["qkv_proj"] == QKV
    assert sparse.ignore is original.ignore and sparse.scheme is original.scheme
    assert sparse.packed_modules_mapping["gate_up_proj"] is mapping["gate_up_proj"]
    other = metadata.indexer_quantization_config(original)
    sparse.packed_modules_mapping["qkv_proj"].clear()
    assert original.packed_modules_mapping["qkv_proj"] == QKV
    assert other.packed_modules_mapping["qkv_proj"] == INDEXER


def test_existing_five_member_metadata_remains_intact():
    original = SimpleNamespace(packed_modules_mapping={"qkv_proj": INDEXER.copy()})
    sparse = metadata.indexer_quantization_config(original)
    assert original.packed_modules_mapping["qkv_proj"] == INDEXER
    assert (
        sparse.packed_modules_mapping["qkv_proj"]
        is not original.packed_modules_mapping["qkv_proj"]
    )
    assert metadata.indexer_quantization_config(None) is None


@pytest.mark.parametrize("mapping", [None, {"qkv_proj": ["other_proj"]}])
def test_unsupported_metadata_fails_without_mutating_config(mapping):
    original = SimpleNamespace(packed_modules_mapping=mapping)
    with pytest.raises((TypeError, ValueError)):
        metadata.indexer_quantization_config(original)
    assert original.packed_modules_mapping is mapping


@pytest.fixture
def consumer():
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
            self.quant_config = quant_config
            self.prefix = prefix
            self.layout = (
                hidden_size,
                head_size,
                total_num_heads,
                total_num_kv_heads,
                total_num_index_heads,
                index_head_size,
                bias,
            )
            if quant_config is not None:
                self.method = quant_config.get_quant_method(self, prefix)

        def forward(self, value):
            return value

        def weight_loader(self, value):
            return value

    original = MinimaxM3QKVParallelLinearWithIndexer
    return SimpleNamespace(MinimaxM3QKVParallelLinearWithIndexer=original), original


@pytest.mark.parametrize("positional", [True, False])
def test_actual_consumer_receives_local_config_before_method_selection(
    consumer, positional
):
    owner, original_class = consumer
    selected = []

    class Config:
        packed_modules_mapping = {"qkv_proj": QKV.copy()}

        def get_quant_method(self, layer, prefix):
            selected.append((self, layer, prefix))
            return self.packed_modules_mapping["qkv_proj"]

    config = Config()
    unrelated = SimpleNamespace(MinimaxM3QKVParallelLinearWithIndexer=original_class)
    metadata.install_indexer_quantization_mapping(owner)
    patched = owner.MinimaxM3QKVParallelLinearWithIndexer
    metadata.install_indexer_quantization_mapping(owner)
    assert owner.MinimaxM3QKVParallelLinearWithIndexer is patched
    args = (4096, 128, 32, 4, 4, 128)
    if positional:
        layer = patched(*args, False, config, "model.layers.3.self_attn.qkv_proj")
    else:
        layer = patched(
            *args, quant_config=config, prefix="model.layers.3.self_attn.qkv_proj"
        )
    assert layer.layout == (*args, False)
    assert layer.method == INDEXER
    assert selected == [(layer.quant_config, layer, layer.prefix)]
    assert layer.quant_config is not config
    assert config.packed_modules_mapping["qkv_proj"] == QKV
    assert type(layer) is original_class
    assert patched.__wrapped__ is original_class
    assert patched.__name__ == original_class.__name__
    assert type(layer).__name__ == original_class.__name__
    assert layer.forward.__func__ is original_class.forward
    assert layer.weight_loader.__func__ is original_class.weight_loader
    assert unrelated.MinimaxM3QKVParallelLinearWithIndexer is original_class
    assert patched(*args).quant_config is None


def test_incompatible_upstream_constructor_is_rejected_before_binding():
    class ChangedIndexer:
        def __init__(self, changed_argument):
            pass

    owner = SimpleNamespace(MinimaxM3QKVParallelLinearWithIndexer=ChangedIndexer)
    with pytest.raises(RuntimeError, match="constructor contract"):
        metadata.install_indexer_quantization_mapping(owner)
    assert owner.MinimaxM3QKVParallelLinearWithIndexer is ChangedIndexer


def test_non_cuda_model_import_does_not_load_nvidia(monkeypatch):
    root = Path(__file__).parents[3]
    for name in ("vllm_fl", "vllm_fl.models"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    upstream = ModuleType("vllm.models.minimax_m3")
    upstream.MiniMaxM3SparseForCausalLM = type("Causal", (), {})
    upstream.MiniMaxM3SparseForConditionalGeneration = type("Multimodal", (), {})
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(is_cuda=lambda: False)
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    nvidia = ModuleType("vllm.models.minimax_m3.nvidia")

    def unexpected(name):
        raise AssertionError("Non-CUDA import reached NVIDIA model: " + name)

    nvidia.__getattr__ = unexpected
    monkeypatch.setitem(sys.modules, nvidia.__name__, nvidia)
    registration_name = "vllm_fl.models.minimax_m3_registration"
    registration_spec = importlib.util.spec_from_file_location(
        registration_name, root / "vllm_fl/models/minimax_m3_registration.py"
    )
    registration = importlib.util.module_from_spec(registration_spec)
    registration_spec.loader.exec_module(registration)
    monkeypatch.setitem(sys.modules, registration_name, registration)
    model_spec = importlib.util.spec_from_file_location(
        "vllm_fl.models.minimax_m3", root / "vllm_fl/models/minimax_m3.py"
    )
    model = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model)
    assert model.MiniMaxM3SparseForCausalLM is upstream.MiniMaxM3SparseForCausalLM
    assert (
        model.MiniMaxM3SparseForConditionalGeneration
        is upstream.MiniMaxM3SparseForConditionalGeneration
    )
