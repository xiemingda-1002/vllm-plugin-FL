# Copyright (c) 2026 BAAI. All rights reserved.

import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def _assert_vllm_ascend_not_imported() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name == "vllm_ascend" or name.startswith("vllm_ascend.")
    )
    assert imported == []


def test_model_runner_uses_vllm_024_gpu_runner_contract(monkeypatch) -> None:
    # This import contract is platform-independent, but model_runner imports
    # FL's graph module whose class body requires a recognized device. Do not
    # depend on another test having initialized a global platform first.
    import vllm.platforms

    class ImportContractPlatform(vllm.platforms.Platform):
        _enum = vllm.platforms.PlatformEnum.OOT
        device_name = "test"
        device_type = "txda"
        torch_device_fn = SimpleNamespace(Stream=object)

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        ImportContractPlatform(),
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm_fl.worker.model_runner import ModelRunnerFL

    assert issubclass(ModelRunnerFL, GPUModelRunner)
    assert ModelRunnerFL.__bases__[0] is GPUModelRunner
    assert GPUModelRunner in ModelRunnerFL.__mro__
    assert inspect.signature(ModelRunnerFL.__init__) == inspect.signature(
        GPUModelRunner.__init__
    )
    _assert_vllm_ascend_not_imported()


def test_model_runner_prewarms_only_vllm_compile_before_stock_compile() -> None:
    source = (
        Path(__file__).parents[3] / "vllm_fl" / "worker" / "model_runner.py"
    ).read_text()
    load_model = source[source.index("    def load_model(") :]
    load_model = load_model[: load_model.index("    def _setup_eagle3")]
    prepare = load_model.index("self.graph_runtime.prepare_model_compile()")
    stock_compile = load_model.index("self.model.compile(fullgraph=True, backend=backend)")
    mode_guard = load_model.rfind("CompilationMode.VLLM_COMPILE", 0, prepare)

    assert mode_guard != -1
    assert mode_guard < prepare < stock_compile
    assert "CompilationMode.NONE" not in load_model[mode_guard:prepare]


def test_fl_registration_uses_upstream_glm_without_legacy_bridge(
    monkeypatch,
) -> None:
    import vllm.platforms
    import vllm_fl
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from vllm_fl.patches import moe_sum, qwen3_5_text
    from vllm_fl.patches import glm_moe_dsa as legacy_glm

    def fail_if_called() -> None:
        raise AssertionError("legacy GLM platform patch was called")

    monkeypatch.setattr(legacy_glm, "apply_platform_patches", fail_if_called)
    monkeypatch.setattr(vllm_fl, "_patch_custom_ops", lambda: None)
    monkeypatch.setattr(vllm_fl, "_patch_ascend_torch_accelerator", lambda: None)
    monkeypatch.setattr(vllm_fl, "_patch_flash_attn_import", lambda: None)
    monkeypatch.setattr(vllm_fl, "_patch_transformers_compat", lambda: None)
    monkeypatch.setattr(vllm_fl, "_get_op_config", lambda: None)
    monkeypatch.setattr(vllm_fl, "_register_flagcx_connector", lambda: None)
    monkeypatch.setattr(vllm_fl, "register_quant_linear", lambda: None)
    monkeypatch.setattr(vllm_fl, "register_router", lambda: None)
    monkeypatch.setattr(vllm_fl, "_register_gdn_packed_decode_patch", lambda: False)
    monkeypatch.setattr(moe_sum, "patch_vllm_moe_sum", lambda: None)
    monkeypatch.setattr(qwen3_5_text, "apply_qwen3_5_text_patches", lambda: None)
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(vendor_name="ascend", device_type="npu"),
    )
    legacy_config = _CONFIG_REGISTRY.get("glm_moe_dsa")

    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    vllm_fl.register_model()

    assert "GlmMoeDsaForCausalLM" in ModelRegistry.get_supported_archs()
    assert isinstance(GlmMoeDsaForCausalLM, type)
    assert CONFIG_MAPPING["glm_moe_dsa"].__name__ == "GlmMoeDsaConfig"
    assert _CONFIG_REGISTRY.get("glm_moe_dsa") is legacy_config
    native_glm_config = CONFIG_MAPPING["glm_moe_dsa"](
        rope_parameters={"rope_type": "deepseek_yarn", "factor": 40}
    )
    assert native_glm_config.rope_parameters["rope_type"] == "deepseek_yarn"
    assert native_glm_config.rope_parameters["factor"] == 40
    assert not getattr(legacy_glm, "_fl_patched", False)
    _assert_vllm_ascend_not_imported()


def test_installed_fl_entry_points_resolve_without_vllm_ascend() -> None:
    probe = r"""
import sys
from importlib.metadata import entry_points

def load_one(group):
    matches = [ep for ep in entry_points(group=group) if ep.name == "fl"]
    assert len(matches) == 1, (group, matches)
    return matches[0].load()

platform_plugin = load_one("vllm.platform_plugins")
general_plugin = load_one("vllm.general_plugins")
assert platform_plugin() == "vllm_fl.platform.PlatformFL"
general_plugin()
assert not any(
    name == "vllm_ascend" or name.startswith("vllm_ascend.")
    for name in sys.modules
)
print("actual_platform_entry_point=pass")
print("actual_general_entry_point=pass")
print("vllm_ascend_modules=0")
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd="/tmp",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    print(completed.stdout)
    print(completed.stderr, file=sys.stderr)
    assert completed.returncode == 0
