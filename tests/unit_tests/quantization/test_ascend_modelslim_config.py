"""CPU-only contracts for the Ascend ModelSlim descriptor parser.

The production import normally brings in the selected vendor stack.  These
tests load ``config.py`` under an isolated package with a deliberately small
vLLM surface so parser behavior can be checked without FlagGems or an NPU.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from abc import ABC, abstractmethod
from builtins import __import__ as builtin_import
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/config.py"
QUANT_INIT = (
    ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/__init__.py"
)
UTILS = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/utils.py"


def _module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def isolated_config(monkeypatch: pytest.MonkeyPatch):
    """Load the real parser while replacing only its framework dependencies."""

    class QuantizationConfig(ABC):
        @classmethod
        @abstractmethod
        def get_name(cls):
            raise NotImplementedError

        @classmethod
        @abstractmethod
        def get_supported_act_dtypes(cls):
            raise NotImplementedError

        @classmethod
        @abstractmethod
        def get_min_capability(cls):
            raise NotImplementedError

        @classmethod
        @abstractmethod
        def get_config_filenames(cls):
            raise NotImplementedError

        @classmethod
        @abstractmethod
        def from_config(cls, config):
            raise NotImplementedError

        @abstractmethod
        def get_quant_method(self, layer, prefix):
            raise NotImplementedError

    class QuantizeMethodBase:
        pass

    class LinearBase:
        pass

    class AttentionLayerBase:
        pass

    class MoERunner:
        pass

    class RoutedExperts:
        pass

    class VocabParallelEmbedding:
        pass

    class UnquantizedEmbeddingMethod:
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

    vllm = _module(monkeypatch, "vllm")
    config_mod = _module(
        monkeypatch,
        "vllm.config",
        get_current_vllm_config=lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v4")
            )
        ),
    )
    vllm.config = config_mod
    logger = logging.getLogger("modelslim-test")
    logger.info_once = logger.info  # type: ignore[attr-defined]
    _module(monkeypatch, "vllm.logger", logger=logger)
    _module(monkeypatch, "vllm.model_executor")
    _module(monkeypatch, "vllm.model_executor.layers")
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
    )
    _module(monkeypatch, "vllm.model_executor.layers.linear", LinearBase=LinearBase)
    _module(monkeypatch, "vllm.model_executor.layers.quantization")
    _module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.base_config",
        QuantizationConfig=QuantizationConfig,
        QuantizeMethodBase=QuantizeMethodBase,
    )
    _module(
        monkeypatch,
        "vllm.model_executor.layers.vocab_parallel_embedding",
        UnquantizedEmbeddingMethod=UnquantizedEmbeddingMethod,
        VocabParallelEmbedding=VocabParallelEmbedding,
    )
    _module(monkeypatch, "vllm.model_executor.models")
    _module(
        monkeypatch, "vllm.model_executor.models.utils", WeightsMapper=WeightsMapper
    )
    _module(monkeypatch, "transformers", PretrainedConfig=object)

    for name in (
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.ascend",
        "vllm_fl.dispatch.backends.vendor.ascend.impl",
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization",
    ):
        package = _module(monkeypatch, name)
        package.__path__ = []
    _module(
        monkeypatch,
        "vllm_fl.platforms.ascend.hardware",
        AscendDeviceType=SimpleNamespace(A5="A5"),
        get_ascend_device_type=lambda: (_ for _ in ()).throw(
            AssertionError("device query during parser import")
        ),
    )
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.utils",
        calc_split_factor=lambda values: values,
        get_model_file=lambda *args, **kwargs: None,
    )
    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.linear",
        AscendLinearMethod=type("AscendLinearMethod", (), {}),
        create_linear_scheme=lambda _: object(),
    )
    class AscendFusedMoEMethod:
        def __init__(self, scheme, moe_config, tid2eid=None):
            self.scheme = scheme
            self.moe_config = moe_config
            self.tid2eid = tid2eid

    _module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.moe",
        AscendFusedMoEMethod=AscendFusedMoEMethod,
        create_moe_scheme=lambda quant_type: (
            object()
            if quant_type == "W8A8_DYNAMIC"
            else (_ for _ in ()).throw(NotImplementedError(quant_type))
        ),
    )
    package = sys.modules["vllm_fl.dispatch.backends.vendor.ascend.impl.quantization"]
    package.ASCEND_QUANTIZATION_METHOD = "ascend"

    forbidden = {"torch_npu", "flag_gems", "flaggems"}
    imported = []

    def guarded_import(name, *args, **kwargs):
        imported.append(name)
        if name.split(".")[0] in forbidden:
            raise AssertionError(f"forbidden import: {name}")
        return builtin_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    spec = importlib.util.spec_from_file_location(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.config", CONFIG
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert not forbidden.intersection(name.split(".")[0] for name in imported)
    module._test_classes = SimpleNamespace(
        moe_runner=MoERunner, routed_experts=RoutedExperts
    )
    return module, LinearBase, AttentionLayerBase, WeightsMapper


def test_deepseek_v4_descriptor_adaptations_and_prefix_mapping(isolated_config):
    module, _, _, _ = isolated_config
    config = module.AscendModelSlimConfig(
        {
            "hc_head_fn": "FLOAT",
            "layers.0.attn.wq.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.w1.weight": "W8A8_DYNAMIC",
            "layers.0.attn_norm.weight": "FLOAT",
            "embed.weight": "FLOAT",
            "head.weight": "FLOAT",
        }
    )
    assert (
        config.quant_description["model.layers.0.self_attn.wq.weight"] == "W8A8_DYNAMIC"
    )
    assert (
        config.quant_description["model.layers.0.mlp.gate_proj.weight"]
        == "W8A8_DYNAMIC"
    )
    # The hc-head adaptation adds the model root; the subsequent V4 prefix
    # mapper is applied to queried module prefixes, not rewritten here.
    assert config.quant_description["model.layers.0.attn_norm.weight"] == "FLOAT"
    assert config.quant_description["model.embed_tokens.weight"] == "FLOAT"
    assert config.quant_description["model.lm_head.weight"] == "FLOAT"
    assert (
        config.quant_prefix_mapper("deepseek_v4", "layers.0.attn.wq")
        == "model.layers.0.self_attn.wq"
    )
    with pytest.raises(NotImplementedError):
        config.get_min_capability()


def test_packed_float_mixing_and_missing_shards_fail(isolated_config):
    module, _, _, _ = isolated_config
    prefix = "model.layers.0.mlp.gate_up_proj"
    mapping = module.get_packed_modules_mapping("deepseek_v4")
    with pytest.raises(ValueError, match="Not all shards"):
        module.get_linear_quant_type(
            {
                "model.layers.0.mlp.gate_proj.weight": "FLOAT",
                "model.layers.0.mlp.up_proj.weight": "W8A8_DYNAMIC",
            },
            prefix,
            mapping,
        )
    with pytest.raises(KeyError):
        module.get_linear_quant_type(
            {"model.layers.0.mlp.gate_proj.weight": "FLOAT"}, prefix, mapping
        )
    all_float = {
        "model.layers.0.mlp.gate_proj.weight": "FLOAT",
        "model.layers.0.mlp.up_proj.weight": "FLOAT",
    }
    all_w8a8 = {key: "W8A8_DYNAMIC" for key in all_float}
    assert module.get_linear_quant_type(all_float, prefix, mapping) == "FLOAT"
    assert module.get_linear_quant_type(all_w8a8, prefix, mapping) == "W8A8_DYNAMIC"


def test_mapper_is_idempotent_and_unrelated_layer_returns_none(isolated_config):
    module, _, _, mapper_cls = isolated_config
    config = module.AscendModelSlimConfig({"hf.foo.weight": "FLOAT"})
    mapper = mapper_cls(orig_to_new_prefix={"hf.": "model."})
    config.apply_vllm_mapper(mapper)
    once = dict(config.quant_description)
    config.apply_vllm_mapper(mapper)
    assert config.quant_description == once == {"model.foo.weight": "FLOAT"}
    assert config.get_quant_method(object(), "anything") is None


def test_config_import_does_not_query_device_or_import_flaggems(
    isolated_config, monkeypatch
):
    module, _, _, _ = isolated_config
    # ``isolated_config`` guards imports while loading the real config module.
    # Do not assert against the process-wide module cache here: the Ascend
    # pytest environment may legitimately preload torch_npu in conftest.
    assert module.AscendModelSlimConfig.get_config_filenames() == []


def test_descriptor_loading_missing_error_and_from_config_no_overwrite(
    isolated_config, monkeypatch, tmp_path
):
    module, _, _, _ = isolated_config
    descriptor = tmp_path / "quant_model_description.json"
    descriptor.write_text('{"layers.0.attn.wq.weight": "W8A8_DYNAMIC"}')
    utils = sys.modules[
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.utils"
    ]
    monkeypatch.setattr(utils, "get_model_file", lambda *args, **kwargs: descriptor)
    config = module.AscendModelSlimConfig()
    config.maybe_update_config(str(tmp_path))
    assert config.quant_description["layers.0.attn.wq.weight"] == "W8A8_DYNAMIC"
    monkeypatch.setattr(utils, "get_model_file", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="ModelSlim Quantization Config Not Found"):
        module.AscendModelSlimConfig().maybe_update_config(str(tmp_path))
    supplied = module.AscendModelSlimConfig({"already.weight": "FLOAT"})
    supplied.maybe_update_config(str(tmp_path))
    assert supplied.quant_description == {"already.weight": "FLOAT"}


@pytest.mark.parametrize("kind", ["moe_runner", "routed_experts"])
def test_w8a8_dynamic_moe_accepts_and_other_moe_types_reject(isolated_config, kind):
    module, _, _, _ = isolated_config
    moe_cls = getattr(module._test_classes, kind)
    config = module.AscendModelSlimConfig({"model.layers.0.mlp.weight": "W8A8_DYNAMIC"})
    layer = moe_cls()
    layer.moe_config = object()
    method = config.get_quant_method(layer, "model.layers.0.mlp", tid2eid="ids")
    assert method.moe_config is layer.moe_config
    assert method.tid2eid == "ids"

    unsupported = module.AscendModelSlimConfig({"model.layers.0.mlp.weight": "W8A8"})
    with pytest.raises(NotImplementedError, match="W8A8"):
        unsupported.get_quant_method(layer, "model.layers.0.mlp")


def test_registration_is_idempotent_and_parser_choices_are_extended(monkeypatch):
    registrations = []
    _module(monkeypatch, "vllm")
    _module(monkeypatch, "vllm.model_executor")
    _module(monkeypatch, "vllm.model_executor.layers")
    registry = _module(
        monkeypatch,
        "vllm.model_executor.layers.quantization",
        register_quantization_config=lambda name: (
            lambda cls: registrations.append((name, cls))
        ),
    )
    del registry
    package_name = "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization"
    package = _module(monkeypatch, package_name)
    package.__path__ = []
    config = _module(
        monkeypatch,
        package_name + ".config",
        AscendModelSlimConfig=type("Config", (), {}),
    )
    del config
    spec = importlib.util.spec_from_file_location(package_name, QUANT_INIT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)
    spec.loader.exec_module(module)
    action = SimpleNamespace(choices=["fp8"])
    parser = SimpleNamespace(_option_string_actions={"--quantization": action})
    module.register_modelslim(parser)
    module.register_modelslim(parser)
    assert len(registrations) == 1
    assert action.choices == ["fp8", "ascend"]


def test_autodetect_recreates_quant_config_and_preserves_explicit_choice(
    monkeypatch, tmp_path
):
    """The platform hook runs after VllmConfig construction, so assignment matters."""
    calls = []
    _module(monkeypatch, "vllm")
    _module(monkeypatch, "vllm.envs", VLLM_USE_MODELSCOPE=False)
    _module(
        monkeypatch, "vllm.logger", logger=logging.getLogger("modelslim-autodetect")
    )

    class FakeVllmConfig:
        @staticmethod
        def _get_quantization_config(model_config, load_config):
            calls.append((model_config, load_config))
            return "rebuilt"

    _module(monkeypatch, "vllm.config", VllmConfig=FakeVllmConfig)
    package_name = "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization"
    package = _module(
        monkeypatch,
        package_name,
        ASCEND_QUANTIZATION_METHOD="ascend",
        MODELSLIM_CONFIG_FILENAME="quant_model_description.json",
    )
    package.__path__ = []
    spec = importlib.util.spec_from_file_location(package_name + ".utils", UTILS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    (tmp_path / "quant_model_description.json").write_text("{}")
    model_config = SimpleNamespace(
        model=str(tmp_path), revision=None, quantization=None
    )
    vllm_config = SimpleNamespace(
        model_config=model_config, load_config="load", quant_config=None
    )
    module.maybe_auto_detect_quantization(vllm_config)
    assert model_config.quantization == "ascend"
    assert vllm_config.quant_config == "rebuilt"
    assert calls == [(model_config, "load")]

    explicit = SimpleNamespace(model=str(tmp_path), revision=None, quantization="fp8")
    unchanged = SimpleNamespace(
        model_config=explicit, load_config="load", quant_config="original"
    )
    module.maybe_auto_detect_quantization(unchanged)
    assert explicit.quantization == "fp8"
    assert unchanged.quant_config == "original"
    assert len(calls) == 1
