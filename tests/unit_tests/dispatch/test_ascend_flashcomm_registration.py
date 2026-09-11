# Copyright (c) 2026 BAAI. All rights reserved.

import ast
import importlib
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_fl.ascend_flashcomm as flashcomm
from vllm_fl.dispatch.backends.vendor.ascend.impl.moe import compat


def _config(*, flashcomm1=False, pass_sp=False, enforce_eager=False):
    return SimpleNamespace(
        additional_config={
            "enable_flashcomm1": flashcomm1,
            "refresh": True,
        },
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=pass_sp)
        ),
    )


@pytest.fixture(autouse=True)
def _reset_gate(monkeypatch):
    monkeypatch.setattr(flashcomm, "_ENABLE_FLASHCOMM1", None)


def test_moe_compat_enable_sp_delegates_unified_flashcomm1_gate() -> None:
    enabled = _config(flashcomm1=True)
    disabled = _config(flashcomm1=False)

    assert compat.enable_sp(enabled)
    assert not compat.enable_sp(disabled)


def test_pass_sp_remains_separate_from_flashcomm1(monkeypatch) -> None:
    config = _config(flashcomm1=False, pass_sp=True)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)

    assert not compat.enable_sp(config)
    assert compat.enable_sp_by_pass()


def test_pass_sp_is_disabled_for_eager_model(monkeypatch) -> None:
    config = _config(pass_sp=True, enforce_eager=True)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)

    assert not compat.enable_sp_by_pass()


def test_shared_expert_dp_forces_unified_gate(monkeypatch) -> None:
    config = _config(flashcomm1=False)
    config.additional_config["enable_shared_expert_dp"] = True
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: config)

    assert compat.shared_expert_dp_enabled()
    config.additional_config["refresh"] = False
    assert compat.enable_sp(config)


def test_importing_compat_does_not_import_vllm_ascend() -> None:
    tree = ast.parse(inspect.getsource(compat))
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        getattr(node, "module", None) != "vllm_ascend"
        and not str(getattr(node, "module", "")).startswith("vllm_ascend.")
        and all(
            alias.name != "vllm_ascend"
            and not alias.name.startswith("vllm_ascend.")
            for alias in getattr(node, "names", ())
        )
        for node in imports
    )


def test_flashcomm_registration_is_inside_ascend_patch_lifecycle() -> None:
    patch = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.patch"
    )
    tree = ast.parse(inspect.getsource(patch))
    apply_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_ascend_patches"
    )
    calls = {
        node.func.id
        for node in ast.walk(apply_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "register_flashcomm_ops_and_layers" in calls

    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        "flashcomm" not in ast.unparse(node)
        and ".impl.linear" not in ast.unparse(node)
        for node in top_level_imports
    )


def test_flashcomm_registration_registers_exact_frozen_apis(
    monkeypatch,
) -> None:
    patch = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.patch"
    )
    monkeypatch.setattr(
        patch, "_flashcomm_ops_and_layers_registered", False
    )
    custom_op = importlib.import_module("vllm.model_executor.custom_op")
    register_oot = Mock()
    monkeypatch.setattr(custom_op.PluggableLayer, "register_oot", register_oot)

    ensure_flashcomm = Mock()
    ensure_linear = Mock()
    module_prefix = "vllm_fl.dispatch.backends.vendor.ascend.impl"
    flashcomm_module = SimpleNamespace(
        ensure_ascend_flashcomm_custom_ops_registered=ensure_flashcomm
    )
    linear_classes = {
        "AscendQKVParallelLinear": type("AscendQKVParallelLinear", (), {}),
        "AscendMergedColumnParallelLinear": type(
            "AscendMergedColumnParallelLinear", (), {}
        ),
        "AscendColumnParallelLinear": type(
            "AscendColumnParallelLinear", (), {}
        ),
        "AscendRowParallelLinear": type("AscendRowParallelLinear", (), {}),
        "AscendReplicatedLinear": type("AscendReplicatedLinear", (), {}),
        "ensure_ascend_linear_custom_ops_registered": ensure_linear,
    }
    linear_module = SimpleNamespace(**linear_classes)
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.flashcomm_custom_ops",
        flashcomm_module,
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.linear",
        linear_module,
    )

    patch.register_flashcomm_ops_and_layers()
    patch.register_flashcomm_ops_and_layers()

    ensure_flashcomm.assert_called_once_with()
    ensure_linear.assert_called_once_with()
    registrations = {
        call.kwargs["name"]: call.kwargs["_decorated_layer_cls"]
        for call in register_oot.call_args_list
    }
    assert registrations == {
        "QKVParallelLinear": linear_classes["AscendQKVParallelLinear"],
        "MergedColumnParallelLinear": linear_classes[
            "AscendMergedColumnParallelLinear"
        ],
        "ColumnParallelLinear": linear_classes[
            "AscendColumnParallelLinear"
        ],
        "RowParallelLinear": linear_classes["AscendRowParallelLinear"],
        "ReplicatedLinear": linear_classes["AscendReplicatedLinear"],
    }
