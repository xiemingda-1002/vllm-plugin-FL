# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU startup contracts; these do not validate Ascend numerical execution."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


@pytest.fixture
def runtime():
    spec = importlib.util.spec_from_file_location(
        "minimax_ascend_contract",
        ROOT / "vllm_fl/models/minimax_m3_ascend_runtime.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForCausalLM"]),
            dtype="torch.bfloat16",
            quantization=None,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            worker_cls="auto",
            tensor_parallel_size=8,
            data_parallel_size=2,
            enable_expert_parallel=True,
        ),
        device_config=SimpleNamespace(device_type="npu"),
    )


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True)])
def test_explicit_runtime_choice(runtime, monkeypatch, value, expected):
    monkeypatch.delenv(runtime.ENV_NAME, raising=False)
    if value is not None:
        monkeypatch.setenv(runtime.ENV_NAME, value)
    assert runtime.enabled() is expected


def test_invalid_runtime_or_plugin_choice_fails(runtime, monkeypatch):
    monkeypatch.setenv(runtime.ENV_NAME, "auto")
    with pytest.raises(ValueError):
        runtime.enabled()
    monkeypatch.setenv("VLLM_PLUGINS", "fl,ascend")
    with pytest.raises(RuntimeError, match="VLLM_PLUGINS=fl"):
        runtime.require_plugin_filter()


@pytest.mark.parametrize(
    "path,value",
    [
        (("model_config", "dtype"), "torch.float16"),
        (("model_config", "quantization"), "compressed-tensors"),
        (("model_config", "hf_config", "architectures"), ["OtherModel"]),
        (("model_config", "hf_config", "quantization_config"), {"bits": 8}),
        (("cache_config", "cache_dtype"), "fp8"),
        (("parallel_config", "worker_cls"), "vllm_fl.worker.worker.WorkerFL"),
        (("parallel_config", "tensor_parallel_size"), 1),
        (("parallel_config", "data_parallel_size"), 1),
        (("parallel_config", "enable_expert_parallel"), False),
        (("device_config", "device_type"), "cuda"),
    ],
)
def test_unsupported_config_is_rejected_without_mutation(runtime, path, value):
    candidate = config()
    target = candidate
    for part in path[:-1]:
        target = getattr(target, part)
    setattr(target, path[-1], value)
    before = deepcopy(candidate)
    with pytest.raises(ValueError):
        runtime.validate_config(candidate)
    assert candidate == before


@pytest.mark.parametrize(
    "architecture",
    sorted(("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration")),
)
def test_supported_config_is_not_rewritten(runtime, architecture):
    candidate = config()
    candidate.model_config.hf_config.architectures = [architecture]
    before = deepcopy(candidate)
    runtime.validate_config(candidate)
    assert candidate == before


def test_vendor_hook_failure_propagates_without_later_hooks(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "check_dependencies", lambda: None)
    calls = []
    vendor = ModuleType("vllm_ascend")
    failure = RuntimeError("vendor startup failed")

    def start():
        calls.append("connector")
        raise failure

    vendor.register_connector = start
    for name in runtime.GENERAL_HOOKS[1:]:
        setattr(vendor, name, lambda: calls.append("later"))
    monkeypatch.setitem(sys.modules, "vllm_ascend", vendor)
    with pytest.raises(RuntimeError) as caught:
        runtime.register_general_plugins()
    assert caught.value is failure
    assert calls == ["connector"]
    with pytest.raises(RuntimeError, match="fresh process"):
        runtime.register_general_plugins()
    assert calls == ["connector"]


def test_successful_vendor_startup_is_idempotent(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "check_dependencies", lambda: None)
    calls = []
    vendor = ModuleType("vllm_ascend")
    for name in runtime.GENERAL_HOOKS:
        setattr(vendor, name, lambda name=name: calls.append(name))
    monkeypatch.setitem(sys.modules, "vllm_ascend", vendor)
    runtime.register_general_plugins()
    runtime.register_general_plugins()
    assert calls == list(runtime.GENERAL_HOOKS)


def test_missing_vendor_hook_fails_before_any_hook(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "check_dependencies", lambda: None)
    calls = []
    vendor = ModuleType("vllm_ascend")
    vendor.register_connector = lambda: calls.append("connector")
    monkeypatch.setitem(sys.modules, "vllm_ascend", vendor)
    with pytest.raises(RuntimeError, match="register_model_loader"):
        runtime.register_general_plugins()
    assert calls == []


def test_wrong_vendor_commit_is_rejected_before_import(runtime, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "fl")
    vendor = SimpleNamespace(
        version="0.24.0rc1",
        read_text=lambda _: json.dumps({"vcs_info": {"commit_id": "wrong"}}),
    )
    monkeypatch.setattr(runtime.importlib.metadata, "distribution", lambda _: vendor)
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "version",
        lambda name: "0.24.0" if name == "vllm" else "2.10.0.post2",
    )
    with pytest.raises(RuntimeError, match="commit"):
        runtime.check_dependencies()


def load_fl_entrypoint(runtime, monkeypatch):
    """Load the complete real entry module without third-party chip libraries."""
    torch = ModuleType("torch")
    torch.float4_e2m1fn_x2 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("vllm_fl.version", "vllm_fl.models", "vllm_fl.utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm_fl.utils"].get_op_config = lambda: None
    monkeypatch.setitem(
        sys.modules, "vllm_fl.models.minimax_m3_ascend_runtime", runtime
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_fl", ROOT / "vllm_fl/__init__.py", submodule_search_locations=[]
    )
    entry = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vllm_fl", entry)
    spec.loader.exec_module(entry)
    return entry


def test_real_plugin_entries_select_vendor_without_fl_hooks(runtime, monkeypatch):
    monkeypatch.setenv(runtime.ENV_NAME, "1")
    monkeypatch.setenv("VLLM_PLUGINS", "fl")
    entry = load_fl_entrypoint(runtime, monkeypatch)
    calls = []
    monkeypatch.setattr(runtime, "check_dependencies", lambda: None)
    vendor = ModuleType("vllm_ascend")
    for name in runtime.GENERAL_HOOKS:
        setattr(vendor, name, lambda name=name: calls.append(name))
    monkeypatch.setitem(sys.modules, "vllm_ascend", vendor)

    def unexpected_fl_hook():
        raise AssertionError("FL runtime must not be mixed with vendor runtime")

    monkeypatch.setattr(entry, "_init_vendor_device", unexpected_fl_hook)
    monkeypatch.setattr(entry, "_patch_ascend_torch_accelerator", unexpected_fl_hook)
    assert entry.register() == runtime.PLATFORM_PATH
    entry.register_model()
    entry.register_model()
    assert calls == list(runtime.GENERAL_HOOKS)


def test_default_real_entries_do_not_import_vendor(runtime, monkeypatch):
    monkeypatch.delenv(runtime.ENV_NAME, raising=False)
    monkeypatch.delitem(sys.modules, "vllm_ascend", raising=False)
    entry = load_fl_entrypoint(runtime, monkeypatch)

    def original_fl_path():
        raise LookupError("reached original FL path")

    monkeypatch.setattr(entry, "_init_vendor_device", original_fl_path)
    monkeypatch.setattr(entry, "_patch_ascend_torch_accelerator", original_fl_path)
    for hook in (entry.register, entry.register_model):
        with pytest.raises(LookupError, match="original FL path"):
            hook()
    assert "vllm_ascend" not in sys.modules


def test_real_platform_inherits_complete_vendor_lifecycle(runtime, monkeypatch):
    calls = []

    class NPUPlatform:
        additional_env_vars = ["VENDOR_SETTING"]

        @classmethod
        def check_and_update_config(cls, candidate):
            calls.append(candidate)

    # No real vendor runtime is executed. Distinct inherited hooks let the
    # test detect accidental replacement by FL hooks or partial delegation.
    for name in (
        "pre_register_and_update",
        "register_custom_kv_cache_specs",
        "get_compile_backend",
        "get_pass_manager_cls",
        "get_device_communicator_cls",
        "get_static_graph_wrapper_cls",
        "get_attn_backend_cls",
        "set_additional_forward_context",
        "opaque_attention_op",
        "support_hybrid_kv_cache",
        "support_static_graph_mode",
    ):
        setattr(NPUPlatform, name, classmethod(lambda cls, *args: None))
    platform = ModuleType("vllm_ascend.platform")
    platform.NPUPlatform = NPUPlatform
    utils = ModuleType("vllm_ascend.utils")
    utils.AscendDeviceType = SimpleNamespace(A3=object())
    utils.get_ascend_device_type = lambda: utils.AscendDeviceType.A3
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", platform)
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.models.minimax_m3_ascend_runtime", runtime
    )
    monkeypatch.setattr(runtime, "check_dependencies", lambda: None)
    monkeypatch.setattr(runtime, "register_general_plugins", lambda: None)
    spec = importlib.util.spec_from_file_location(
        "minimax_platform_contract", ROOT / "vllm_fl/platform_minimax_ascend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = module.MiniMaxAscendPlatform
    assert selected.__bases__ == (NPUPlatform,)
    for name in vars(NPUPlatform):
        if not name.startswith("__") and name not in {
            "check_and_update_config",
            "additional_env_vars",
        }:
            assert name not in vars(selected)
    assert selected.additional_env_vars == [
        "VENDOR_SETTING",
        runtime.ENV_NAME,
        "VLLM_PLUGINS",
    ]
    assert NPUPlatform.additional_env_vars == ["VENDOR_SETTING"]
    candidate = config()
    selected.check_and_update_config(candidate)
    assert calls == [candidate]
    candidate.model_config.dtype = "float16"
    with pytest.raises(ValueError, match="BF16"):
        selected.check_and_update_config(candidate)
    assert calls == [candidate]
