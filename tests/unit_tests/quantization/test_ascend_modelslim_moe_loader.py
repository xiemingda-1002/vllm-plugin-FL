"""CPU loader contracts for the ModelSlim DeepSeek-V4 routed MoE path.

The ModelSlim config and Ascend adapter loaded here are production files.  The
vLLM package is unavailable in this checkout's CPU environment, so a small
framework surface is isolated below.  Crucially, scale/weight loading invokes
the *extracted upstream v0.24 RoutedExperts methods*, not a locally invented
loader implementation.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal, overload

import pytest
import torch

ROOT = Path(__file__).parents[3]
CONFIG_PATH = (
    ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/config.py"
)
MOE_PATH = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/moe.py"


def _upstream_routed_source() -> Path:
    """Find a baseline checkout; skip extraction outside migration CI."""
    source_root = Path(
        os.environ.get("FL_TEST_VLLM_SOURCE_ROOT", ROOT.parent / "vllm-v0.24.0")
    )
    source = source_root / "vllm/model_executor/layers/fused_moe/routed_experts.py"
    if not source.is_file():
        pytest.skip(
            "exact upstream loader extraction requires FL_TEST_VLLM_SOURCE_ROOT "
            "or the migration sibling vllm-v0.24.0 checkout"
        )
    return source


def _module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _package(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    package = _module(monkeypatch, name)
    package.__path__ = []


def _upstream_loader_class():
    """Compile the exact current-v0.24 methods needed by this contract test."""
    upstream_routed = _upstream_routed_source()
    tree = ast.parse(upstream_routed.read_text(), filename=str(upstream_routed))
    routed = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RoutedExperts"
    )
    needed = {
        "_map_global_expert_id_to_local_expert_id",
        "_get_hidden_dim",
        "_narrow_expert_data_for_padding",
        "_load_w13",
        "_load_w2",
        "_load_per_channel_weight_scale",
        "_load_model_weight_or_group_weight_scale",
        "weight_loader",
    }
    methods = [
        node
        for node in routed.body
        if isinstance(node, ast.FunctionDef) and node.name in needed
    ]
    namespace = {
        "torch": torch,
        "Literal": Literal,
        "overload": overload,
        "FusedMoeWeightScaleSupported": SimpleNamespace(
            CHANNEL=SimpleNamespace(value="channel"),
            GROUP=SimpleNamespace(value="group"),
            BLOCK=SimpleNamespace(value="block"),
            TENSOR=SimpleNamespace(value="tensor"),
        ),
    }
    compiled = compile(
        ast.Module(body=methods, type_ignores=[]), str(upstream_routed), "exec"
    )
    exec(compiled, namespace)  # noqa: S102 -- source is the checked-in upstream baseline.
    return type(
        "ExtractedUpstreamRoutedExperts", (), {name: namespace[name] for name in needed}
    )


@pytest.fixture
def modelslim_moe(monkeypatch: pytest.MonkeyPatch):
    """Load real config.py + moe.py with only vLLM/platform dependencies stubbed."""

    class QuantizationConfig(ABC):
        @classmethod
        @abstractmethod
        def get_name(cls): ...

        @classmethod
        @abstractmethod
        def get_supported_act_dtypes(cls): ...

        @classmethod
        @abstractmethod
        def get_min_capability(cls): ...

        @classmethod
        @abstractmethod
        def get_config_filenames(cls): ...

        @classmethod
        @abstractmethod
        def from_config(cls, config): ...

        @abstractmethod
        def get_quant_method(self, layer, prefix): ...

    class QuantizeMethodBase:
        pass

    class FusedMoEMethodBase(QuantizeMethodBase):
        def __init__(self, moe_config):
            self.moe_config = moe_config

    class LinearBase:
        pass

    class AttentionLayerBase:
        pass

    class MoERunner:
        pass

    class RoutedExperts(torch.nn.Module):
        pass

    class VocabParallelEmbedding:
        pass

    class UnquantizedEmbeddingMethod(QuantizeMethodBase):
        pass

    class AscendUnquantizedFusedMoEMethod(FusedMoEMethodBase):
        pass

    class WeightsMapper:
        def __init__(self, orig_to_new_prefix=None, orig_to_new_substr=None):
            self.prefix = orig_to_new_prefix or {}
            self.substr = orig_to_new_substr or {}

        def _map_name(self, name):
            for old, new in self.prefix.items():
                if name.startswith(old):
                    name = new + name[len(old) :]
            for old, new in self.substr.items():
                name = name.replace(old, new)
            return name

        def apply_dict(self, values):
            return {self._map_name(key): value for key, value in values.items()}

    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.models",
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.ascend",
        "vllm_fl.dispatch.backends.vendor.ascend.impl",
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization",
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe",
    ):
        _package(monkeypatch, name)
    sys.modules[
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization"
    ].ASCEND_QUANTIZATION_METHOD = "ascend"

    current = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, hf_config=SimpleNamespace(model_type="deepseek_v4")
        )
    )
    _module(monkeypatch, "vllm.config", get_current_vllm_config=lambda: current)
    logger = logging.getLogger("modelslim-moe-loader-test")
    logger.info_once = logger.info  # type: ignore[attr-defined]
    _module(monkeypatch, "vllm.logger", logger=logger)
    _module(
        monkeypatch,
        "vllm.model_executor.layers.attention_layer_base",
        AttentionLayerBase=AttentionLayerBase,
    )
    _module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe",
        MoERunner=MoERunner,
        RoutedExperts=RoutedExperts,
        FusedMoEMethodBase=FusedMoEMethodBase,
        FusedMoeWeightScaleSupported=SimpleNamespace(
            CHANNEL=SimpleNamespace(value="channel"),
            GROUP=SimpleNamespace(value="group"),
        ),
    )
    _module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.config",
        FusedMoEConfig=object,
    )
    _module(monkeypatch, "vllm.model_executor.layers.linear", LinearBase=LinearBase)
    _module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.base_config",
        QuantizationConfig=QuantizationConfig,
        QuantizeMethodBase=QuantizeMethodBase,
    )
    _module(
        monkeypatch,
        "vllm.model_executor.layers.vocab_parallel_embedding",
        VocabParallelEmbedding=VocabParallelEmbedding,
        UnquantizedEmbeddingMethod=UnquantizedEmbeddingMethod,
    )
    _module(
        monkeypatch, "vllm.model_executor.models.utils", WeightsMapper=WeightsMapper
    )
    _module(
        monkeypatch,
        "vllm.model_executor.utils",
        set_weight_attrs=lambda param, attrs: [
            setattr(param, key, value) for key, value in attrs.items()
        ],
    )
    _module(monkeypatch, "transformers", PretrainedConfig=object)
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.hardware",
        AscendDeviceType=SimpleNamespace(A5="A5"),
        get_ascend_device_type=lambda: (_ for _ in ()).throw(
            AssertionError("device query")
        ),
    )
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.utils",
        calc_split_factor=lambda value: value,
        get_model_file=lambda *args: None,
    )
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.linear",
        AscendLinearMethod=object,
        create_linear_scheme=lambda _: object(),
    )
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.compat",
        get_ascend_config=lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False), enable_fused_mc2=0
        ),
    )
    _module(
        monkeypatch,
        "vllm_fl.ascend_forward_context",
        _EXTRA_CTX=SimpleNamespace(moe_comm_type="allgather"),
        MoECommType=SimpleNamespace(ALLGATHER="allgather"),
    )
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.experts_selector",
        select_experts=object(),
        zero_experts_compute=object(),
    )
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.moe_runtime_args",
        build_fused_experts_input=object(),
    )
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.quant_type",
        QuantType=SimpleNamespace(NONE="none", W8A8="w8a8"),
    )
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.fused_moe",
        AscendUnquantizedFusedMoEMethod=AscendUnquantizedFusedMoEMethod,
    )

    moe_name = "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.moe"
    moe_spec = importlib.util.spec_from_file_location(moe_name, MOE_PATH)
    assert moe_spec and moe_spec.loader
    moe = importlib.util.module_from_spec(moe_spec)
    monkeypatch.setitem(sys.modules, moe_name, moe)
    moe_spec.loader.exec_module(moe)

    config_name = "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.config"
    config_spec = importlib.util.spec_from_file_location(config_name, CONFIG_PATH)
    assert config_spec and config_spec.loader
    config = importlib.util.module_from_spec(config_spec)
    monkeypatch.setitem(sys.modules, config_name, config)
    config_spec.loader.exec_module(config)
    return config, moe, RoutedExperts, AscendUnquantizedFusedMoEMethod


def _raw_expert_descriptor(value: str) -> dict[str, str]:
    return {"hc_head_fn": "FLOAT"} | {
        f"layers.0.ffn.experts.0.{shard}.weight": value for shard in ("w1", "w2", "w3")
    }


def test_mapped_dsv4_selection_precedes_allocation_and_upstream_loader(modelslim_moe):
    config_mod, moe_mod, routed_cls, _ = modelslim_moe
    config = config_mod.AscendModelSlimConfig(_raw_expert_descriptor("W8A8_DYNAMIC"))
    layer = routed_cls()
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=1), tp_rank=0, is_act_and_mul=True
    )
    layer.quant_config = config
    layer.expert_map_manager = SimpleNamespace(
        map_global_to_local=lambda expert_id: expert_id
    )
    extracted = _upstream_loader_class()
    for name in (
        "_map_global_expert_id_to_local_expert_id",
        "_get_hidden_dim",
        "_narrow_expert_data_for_padding",
        "_load_w13",
        "_load_w2",
        "_load_per_channel_weight_scale",
        "_load_model_weight_or_group_weight_scale",
        "weight_loader",
    ):
        setattr(type(layer), name, extracted.__dict__[name])

    # Raw DSV4 path must map before selection; allocation follows that selected method.
    method = config.get_quant_method(layer, "layers.0.ffn.experts")
    assert isinstance(method, moe_mod.AscendFusedMoEMethod)
    layer.quant_method = method
    method.create_weights(
        layer, 1, 4, 2, torch.bfloat16, weight_loader=layer.weight_loader
    )

    assert {name for name, _ in layer.named_parameters()} == {
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w13_weight_offset",
        "w2_weight_scale",
        "w2_weight_offset",
    }
    assert layer.w13_weight.dtype is torch.int8 and layer.w13_weight.shape == (1, 4, 4)
    assert layer.w2_weight.dtype is torch.int8 and layer.w2_weight.shape == (1, 4, 2)
    assert (
        layer.w13_weight_scale.dtype is torch.bfloat16
        and layer.w13_weight_scale.shape == (1, 4, 1)
    )
    assert (
        layer.w2_weight_scale.dtype is torch.bfloat16
        and layer.w2_weight_scale.shape == (1, 4, 1)
    )
    for name in (
        "w13_weight_scale",
        "w13_weight_offset",
        "w2_weight_scale",
        "w2_weight_offset",
    ):
        assert getattr(layer, name).quant_method == "channel"
        assert getattr(layer, name).weight_loader == layer.weight_loader

    # Exact upstream v0.24 loader methods: w1/w3 merge into w13 and w2 copies.
    w1 = torch.full((2, 4), 11, dtype=torch.int8)
    w3 = torch.full((2, 4), 33, dtype=torch.int8)
    w2 = torch.full((4, 2), 22, dtype=torch.int8)
    layer.weight_loader(layer.w13_weight, w1, "w13_weight", "w1", 0)
    layer.weight_loader(layer.w13_weight, w3, "w13_weight", "w3", 0)
    layer.weight_loader(layer.w2_weight, w2, "w2_weight", "w2", 0)
    assert torch.equal(layer.w13_weight[0, :2], w1)
    assert torch.equal(layer.w13_weight[0, 2:], w3)
    assert torch.equal(layer.w2_weight[0], w2)

    scale_1 = torch.full((2, 1), 1, dtype=torch.bfloat16)
    scale_3 = torch.full((2, 1), 3, dtype=torch.bfloat16)
    scale_2 = torch.full((4, 1), 2, dtype=torch.bfloat16)
    layer.weight_loader(layer.w13_weight_scale, scale_1, "w13_weight_scale", "w1", 0)
    layer.weight_loader(layer.w13_weight_scale, scale_3, "w13_weight_scale", "w3", 0)
    layer.weight_loader(layer.w2_weight_scale, scale_2, "w2_weight_scale", "w2", 0)
    assert torch.equal(layer.w13_weight_scale[0, :2], scale_1)
    assert torch.equal(layer.w13_weight_scale[0, 2:], scale_3)
    assert torch.equal(layer.w2_weight_scale[0], scale_2)


def test_all_float_dsv4_experts_select_unquantized_before_allocation(modelslim_moe):
    config_mod, _, routed_cls, unquantized_cls = modelslim_moe
    config = config_mod.AscendModelSlimConfig(_raw_expert_descriptor("FLOAT"))
    layer = routed_cls()
    layer.moe_config = object()

    method = config.get_quant_method(layer, "layers.0.ffn.experts")
    assert isinstance(method, unquantized_cls)
    assert not isinstance(method, type(None))
