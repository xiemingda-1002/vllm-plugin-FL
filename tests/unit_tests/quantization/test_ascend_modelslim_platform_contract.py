# Copyright (c) 2026 BAAI. All rights reserved.

"""Exercise platform registration without loading a hardware backend."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

PLATFORM = Path(__file__).parents[3] / "vllm_fl" / "platform.py"
PACKAGE = "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization"


def _method(name):
    tree = ast.parse(PLATFORM.read_text())
    platform = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    method = next(
        node
        for node in platform.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    return method


@pytest.mark.parametrize("device", ["npu", "cuda", "musa"])
def test_registration_is_vendor_scoped(monkeypatch, device):
    register = Mock()
    parts = PACKAGE.split(".")
    parent = None
    for i, name in enumerate(parts):
        full_name = ".".join(parts[: i + 1])
        module = ModuleType(full_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, full_name, module)
        if parent is not None:
            setattr(parent, name, module)
        parent = module
    parent.register_modelslim = register
    method = _method("pre_register_and_update")
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(PLATFORM), "exec"),
        namespace,
    )
    parser = object()
    namespace[method.name](
        SimpleNamespace(device_name=device, vendor_name=device), parser
    )
    if device == "npu":
        register.assert_called_once_with(parser)
    else:
        register.assert_not_called()


def test_discovery_call_is_under_npu_and_model_guards():
    method = _method("check_and_update_config")
    parents = {}
    for node in ast.walk(method):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "maybe_auto_detect_quantization"
    ]
    assert len(calls) == 1
    guards = []
    node = calls[0]
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            guards.append(ast.unparse(node.test))
    assert "cls.device_type == 'npu'" in guards
    assert "model_config is not None" in guards
