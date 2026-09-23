"""Exercise Ascend initialization ordering without importing device libraries."""

import ast
from pathlib import Path
import types

import pytest


@pytest.mark.parametrize(
    "environment",
    [
        {"VLLM_FL_PLATFORM": "ascend"},
        {"VLLM_VENDOR": "ascend"},
        {"VLLM_FL_PLATFORM": "cuda", "VLLM_VENDOR": "ascend"},
        {},
    ],
)
def test_package_import_does_not_require_runtime_vendor_environment(environment):
    source = Path(__file__).parents[2] / "vllm_fl" / "__init__.py"
    tree = ast.parse(source.read_text())
    events = []

    def import_stub(name, *args, **kwargs):
        if name == "":
            return types.SimpleNamespace(version=types.SimpleNamespace())
        assert name == "vllm_fl.utils"
        events.append("utils")
        return types.SimpleNamespace(get_op_config=None)

    # Execute the actual initialization segment between the torch compatibility
    # shim and the first utility import, using isolated imports/environment.
    start = next(
        i for i, node in enumerate(tree.body)
        if isinstance(node, ast.Delete)
    ) + 1
    end = next(
        i for i, node in enumerate(tree.body)
        if isinstance(node, ast.ImportFrom) and node.module == "vllm_fl.utils"
    ) + 1
    segment = ast.Module(body=tree.body[start:end], type_ignores=[])
    exec(compile(segment, str(source), "exec"), {
        "os": types.SimpleNamespace(environ=environment),
        "__builtins__": {"__import__": import_stub},
    })
    assert events == ["utils"]


@pytest.mark.parametrize(
    "device_type,expected",
    [("npu", ["bootstrap", "enable"]), ("cuda", []), ("musa", [])],
)
def test_platform_kernel_import_owns_ascend_bootstrap(device_type, expected):
    source = Path(__file__).parents[2] / "vllm_fl" / "platform.py"
    tree = ast.parse(source.read_text())
    platform_cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    method = next(
        node for node in platform_cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "import_kernels"
    )
    ascend_guard = next(node for node in method.body if isinstance(node, ast.If))
    events = []

    def import_stub(name, *args, **kwargs):
        assert name == "vllm_fl.ascend_custom_ops"
        return types.SimpleNamespace(
            bootstrap_custom_op_env=lambda: events.append("bootstrap"),
            enable_custom_op=lambda: events.append("enable"),
        )

    exec(
        compile(
            ast.Module(body=[ascend_guard], type_ignores=[]),
            str(source),
            "exec",
        ),
        {
            "cls": types.SimpleNamespace(device_type=device_type),
            "__builtins__": {"__import__": import_stub},
        },
    )
    assert events == expected


@pytest.mark.parametrize("initial", [None, "0", "1"])
def test_ascend_graph_policy_overrides_environment_and_cached_flag(monkeypatch, initial):
    import os
    import sys

    if initial is None:
        monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", initial)
    envs = types.ModuleType("vllm.envs")
    envs.VLLM_USE_BREAKABLE_CUDAGRAPH = True
    vllm = types.ModuleType("vllm")
    vllm.envs = envs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    source, guard = _graph_policy_guard()
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(source), "exec"), {
        "vendor_name": "ascend", "device_type": "npu", "os": os,
    })
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
    assert envs.VLLM_USE_BREAKABLE_CUDAGRAPH is False


@pytest.mark.parametrize("vendor,device,expected", [
    ("ascend", "npu", ["disable"]),
    ("nvidia", "cuda", []),
    ("musa", "musa", []),
    ("other", "npu", []),
])
def test_platform_import_graph_policy_is_ascend_scoped(vendor, device, expected):
    source, guard = _graph_policy_guard()
    calls = []
    envs = types.SimpleNamespace(VLLM_USE_BREAKABLE_CUDAGRAPH=True)
    environment = {}
    def import_stub(name, *args, **kwargs):
        assert name == "vllm.envs"
        calls.append("disable")
        return types.SimpleNamespace(envs=envs)
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(source), "exec"), {
        "vendor_name": vendor, "device_type": device,
        "os": types.SimpleNamespace(environ=environment),
        "__builtins__": {"__import__": import_stub},
    })
    assert calls == expected
    assert environment == ({"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"} if expected else {})
    assert envs.VLLM_USE_BREAKABLE_CUDAGRAPH is (not bool(expected))


def _graph_policy_guard():
    source = Path(__file__).parents[2] / "vllm_fl/platform.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlatformFL")
    guard = next(n for n in cls.body if isinstance(n, ast.If) and any(
        isinstance(child, ast.Import) and any(alias.name == "vllm.envs" for alias in child.names)
        for child in ast.walk(n)))
    return source, guard
