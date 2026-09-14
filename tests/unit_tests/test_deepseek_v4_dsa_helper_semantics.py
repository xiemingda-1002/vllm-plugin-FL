import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def dsa_compat_module(monkeypatch):
    """Load the two helpers without a torch_npu or full vLLM installation."""
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))

    vllm = types.ModuleType("vllm")
    model_executor = types.ModuleType("vllm.model_executor")
    models = types.ModuleType("vllm.model_executor.models")
    utils = types.ModuleType("vllm.model_executor.models.utils")

    def extract_layer_index(layer_name: str) -> int:
        return next(int(part) for part in layer_name.split(".") if part.isdigit())

    utils.extract_layer_index = extract_layer_index
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.utils", utils)

    module_name = "vllm_fl.dispatch.backends.vendor.ascend.dsa_compat"
    for package_name in (
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.ascend",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    hardware = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.hardware"
    )
    hardware.AscendDeviceType = object
    hardware.get_ascend_device_type = lambda: None
    monkeypatch.setitem(sys.modules, hardware.__name__, hardware)

    path = (
        Path(__file__).parents[2]
        / "vllm_fl/dispatch/backends/vendor/ascend/dsa_compat.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_extract_dsv4_layer_index_accepts_model_layer_namespace(
    dsa_compat_module,
) -> None:
    config = SimpleNamespace(num_hidden_layers=61)

    assert dsa_compat_module.extract_dsv4_layer_index(config, "model.layers.0") == 0


def test_extract_dsv4_layer_index_offsets_mtp_namespace(dsa_compat_module) -> None:
    config = SimpleNamespace(num_hidden_layers=61)

    assert dsa_compat_module.extract_dsv4_layer_index(config, "mtp.0") == 61


def test_get_dsv4_compress_ratio_returns_configured_value(dsa_compat_module) -> None:
    config = SimpleNamespace(compress_ratios=[4, 128])

    assert dsa_compat_module.get_dsv4_compress_ratio(config, 1) == 128


@pytest.mark.parametrize(
    ("config", "layer_idx"),
    [
        (SimpleNamespace(), 0),
        (SimpleNamespace(compress_ratios=[4]), 1),
    ],
    ids=["missing-ratios", "mtp-ratios-out-of-range"],
)
def test_get_dsv4_compress_ratio_defaults_to_dense(
    dsa_compat_module, config, layer_idx: int
) -> None:
    assert dsa_compat_module.get_dsv4_compress_ratio(config, layer_idx) == 0
