"""Static ownership and import-isolation checks for Ascend configuration."""

import ast
import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]


def test_ascend_config_is_the_only_worker_config_owner() -> None:
    config = (_ROOT / "vllm_fl/configs/ascend.py").read_text()
    compat = (
        _ROOT
        / "vllm_fl/dispatch/backends/vendor/ascend/impl/moe/compat.py"
    ).read_text()

    assert config.splitlines().count("_WORKER_VLLM_CONFIG = None") == 1
    assert "_WORKER_VLLM_CONFIG" not in compat
    assert "from vllm_fl.configs.ascend import" in compat


def test_ascend_config_does_not_depend_on_moe_implementation() -> None:
    config = (_ROOT / "vllm_fl/configs/ascend.py").read_text()
    assert "impl.moe.compat" not in config
    assert "import torch" not in config


def _load_isolated_ascend_config(monkeypatch):
    """Load the vendor config without torch, torch_npu, or an Ascend runtime."""
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda _name: types.SimpleNamespace(
        warning_once=lambda *_args, **_kwargs: None
    )
    vllm_module = types.ModuleType("vllm")
    vllm_module.logger = logger_module
    config_module = types.ModuleType("vllm.config")
    config_module.get_current_vllm_config = lambda: (_ for _ in ()).throw(AssertionError)
    flashcomm_module = types.ModuleType("vllm_fl.ascend_flashcomm")
    flashcomm_module.enable_flashcomm1 = lambda *_args, **_kwargs: False
    flashcomm_module.shared_expert_dp_enabled_for_config = lambda _config: False
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm_fl", types.ModuleType("vllm_fl"))
    monkeypatch.setitem(sys.modules, "vllm_fl.ascend_flashcomm", flashcomm_module)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)

    spec = importlib.util.spec_from_file_location(
        "isolated_ascend_config", _ROOT / "vllm_fl/configs/ascend.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ascend_config_imports_without_torch_or_ascend_package(monkeypatch) -> None:
    """The neutral config layer needs neither torch nor the MoE package."""
    module = _load_isolated_ascend_config(monkeypatch)
    assert module.get_ascend_additional_config() == {}


@pytest.mark.parametrize("additional_config", [{}, {"recompute_scheduler_enable": False}])
def test_ascend_recompute_scheduler_defaults_to_disabled(
    monkeypatch, additional_config
) -> None:
    module = _load_isolated_ascend_config(monkeypatch)
    module.init_ascend_config(types.SimpleNamespace(additional_config=additional_config))

    assert module.get_ascend_config().recompute_scheduler_enable is False


def test_ascend_recompute_scheduler_decode_opt_in_fails_closed(monkeypatch) -> None:
    module = _load_isolated_ascend_config(monkeypatch)
    module.init_ascend_config(
        types.SimpleNamespace(
            additional_config={"recompute_scheduler_enable": True},
            kv_transfer_config=types.SimpleNamespace(kv_role="kv_consumer"),
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="FL Ascend PD decode recompute scheduling is not migrated",
    ):
        module.get_ascend_config()


def test_ascend_recompute_scheduler_producer_opt_in_is_ignored(monkeypatch) -> None:
    module = _load_isolated_ascend_config(monkeypatch)
    module.init_ascend_config(
        types.SimpleNamespace(
            additional_config={"recompute_scheduler_enable": True},
            kv_transfer_config=types.SimpleNamespace(kv_role="kv_producer"),
        )
    )

    assert module.get_ascend_config().recompute_scheduler_enable is False


@pytest.mark.parametrize("kv_role", [None, "kv_both", "other"])
def test_ascend_recompute_scheduler_invalid_role_matches_rc1(
    monkeypatch, kv_role
) -> None:
    module = _load_isolated_ascend_config(monkeypatch)
    module.init_ascend_config(
        types.SimpleNamespace(
            additional_config={"recompute_scheduler_enable": True},
            kv_transfer_config=(
                None if kv_role is None else types.SimpleNamespace(kv_role=kv_role)
            ),
        )
    )

    with pytest.raises(ValueError, match="PD-disaggregated D nodes"):
        module.get_ascend_config()


def test_moe_compat_caches_shared_expert_stream_with_a_stub(monkeypatch) -> None:
    """The stream is MoE execution state, not configuration state."""
    source = (
        _ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/moe/compat.py"
    ).read_text()
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_SHARED_EXPERTS_CALCULATION_STREAM"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "shared_experts_calculation_stream"
        )
    ]
    stream = object()
    original_import = __import__

    def import_stub(name, *args, **kwargs):
        if name == "torch_npu":
            return types.SimpleNamespace(npu=types.SimpleNamespace(Stream=lambda: stream))
        return original_import(name, *args, **kwargs)

    namespace = {"__builtins__": dict(vars(builtins), __import__=import_stub)}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "compat.py", "exec"),
        namespace,
    )
    get_stream = namespace["shared_experts_calculation_stream"]

    assert get_stream() is stream
    assert get_stream() is stream


def test_all_configuration_functions_were_moved_from_moe_compat() -> None:
    """AST-level name check prevents config state from drifting back to MoE."""
    import ast

    compat_tree = ast.parse(
        (_ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/moe/compat.py").read_text()
    )
    config_tree = ast.parse((_ROOT / "vllm_fl/configs/ascend.py").read_text())
    compat_defs = {node.name for node in compat_tree.body if isinstance(node, ast.FunctionDef)}
    config_defs = {node.name for node in config_tree.body if isinstance(node, ast.FunctionDef)}
    moved = {
        "init_ascend_config", "clear_ascend_config", "_current_vllm_config_or_none",
        "get_ascend_additional_config", "balance_scheduling_enabled", "_nested_config",
        "_reject_unsupported", "get_ascend_config", "enable_sp", "enable_sp_by_pass",
    }
    assert moved <= config_defs
    assert not moved & compat_defs


def test_compat_top_level_inventory_preserves_execution_helpers() -> None:
    """Compare the original compat inventory, allowing only config relocation."""
    import ast

    tree = ast.parse(
        (_ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/moe/compat.py").read_text()
    )
    defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    original = {
        "init_ascend_config", "clear_ascend_config", "_current_vllm_config_or_none",
        "get_ascend_additional_config", "balance_scheduling_enabled", "_nested_config",
        "_reject_unsupported", "get_ascend_config", "require_a2_bf16_allgather",
        "enable_sp", "enable_sp_by_pass", "enable_custom_op",
        "is_hierarchical_communication_enabled", "should_skip_allreduce_across_dp_group",
        "dispose_tensor", "maybe_trans_nz", "npu_stream_switch", "shared_expert_dp_enabled",
        "shared_experts_calculation_stream", "get_mc2_group", "split_tensor_along_first_dim",
        "get_moe_num_logical_experts", "VllmEplbAdaptor", "init_eplb_config",
    }
    assert original <= defined | imported
    assert "require_a2_bf16_allgather" in defined


def test_moe_layout_constant_and_runtime_state_are_preserved() -> None:
    tree = ast.parse(
        (_ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/moe/compat.py").read_text()
    )
    values = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert values["ACL_FORMAT_FRACTAL_NZ"] == 29
    assert values["_SHARED_EXPERTS_CALCULATION_STREAM"] is None
