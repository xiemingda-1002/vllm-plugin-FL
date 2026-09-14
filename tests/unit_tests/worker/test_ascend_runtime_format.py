# Copyright (c) 2026 BAAI. All rights reserved.

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[3]


def test_internal_format_setup_is_lazy_and_idempotent(monkeypatch):
    path = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/runtime.py"
    spec = importlib.util.spec_from_file_location("_ascend_runtime", path)
    module = importlib.util.module_from_spec(spec)
    # A None module entry makes any import fail during module inspection.
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    spec.loader.exec_module(module)
    options = []

    class WriteOnlyConfig:
        def __setattr__(self, name, value):
            assert name == "allow_internal_format"
            options.append({"ALLOW_INTERNAL_FORMAT": "enable" if value else "disable"})

    config = WriteOnlyConfig()
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu=SimpleNamespace(config=config))
    )
    module.configure_native_runtime()
    assert options == [{"ALLOW_INTERNAL_FORMAT": "enable"}]
    module.configure_native_runtime()
    assert options == [{"ALLOW_INTERNAL_FORMAT": "enable"}] * 2


def test_worker_runtime_setup_is_npu_only_after_device_selection():
    source = (ROOT / "vllm_fl/worker/worker.py").read_text()
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WorkerFL"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "init_device"
    )
    parents = {
        child: parent
        for parent in ast.walk(method)
        for child in ast.iter_child_nodes(parent)
    }
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)]
    setup = next(n for n in calls if ast.unparse(n.func) == "configure_native_runtime")
    select = next(
        n for n in calls if ast.unparse(n.func) == "current_platform.set_device"
    )
    check = next(n for n in calls if ast.unparse(n.func) == "check_ascend_device_type")
    assert select.lineno < check.lineno < setup.lineno
    node = setup
    guards = []
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            guards.append(ast.unparse(node.test))
    assert "current_platform.device_type == 'npu'" in guards
