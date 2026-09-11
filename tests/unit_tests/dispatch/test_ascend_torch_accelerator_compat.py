import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PATCH_SOURCE = (
    REPO_ROOT
    / "vllm_fl/dispatch/backends/vendor/ascend/patches/patch_torch_accelerator.py"
)
PLATFORM_SOURCE = REPO_ROOT / "vllm_fl/platform.py"
PLUGIN_SOURCE = REPO_ROOT / "vllm_fl/__init__.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location(
        "fl_test_patch_torch_accelerator", PATCH_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_torch():
    calls: list[tuple[str, tuple, dict]] = []

    def api(name, result=None):
        def invoke(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return invoke

    npu = SimpleNamespace(
        empty_cache=api("empty_cache"),
        memory_stats=api(
            "memory_stats", {"allocated_bytes.all.peak": 11}
        ),
        memory_reserved=api("memory_reserved", 7),
        reset_peak_memory_stats=api("reset_peak_memory_stats"),
        mem_get_info=api("mem_get_info", (100, 200)),
    )
    accelerator = SimpleNamespace()
    return SimpleNamespace(npu=npu, accelerator=accelerator), calls


def test_patch_maps_all_five_memory_apis_and_is_idempotent() -> None:
    module = _load_patch_module()
    fake_torch, calls = _fake_torch()
    module.torch = fake_torch

    module.patch_torch_accelerator()
    first_bindings = {
        target: getattr(fake_torch.accelerator, target)
        for target in module._NPU_MEMORY_API_MAP
    }
    module.patch_torch_accelerator()

    for target, source in module._NPU_MEMORY_API_MAP.items():
        assert getattr(fake_torch.accelerator, target) is getattr(
            fake_torch.npu, source
        )
        assert getattr(fake_torch.accelerator, target) is first_bindings[target]

    fake_torch.accelerator.empty_cache()
    fake_torch.accelerator.memory_stats("npu:0")
    fake_torch.accelerator.memory_reserved("npu:0")
    fake_torch.accelerator.reset_peak_memory_stats("npu:0")
    assert fake_torch.accelerator.get_memory_info("npu:0") == (100, 200)
    assert [name for name, _, _ in calls] == [
        "empty_cache",
        "memory_stats",
        "memory_reserved",
        "reset_peak_memory_stats",
        "mem_get_info",
    ]


@pytest.mark.parametrize(
    ("torch_namespace", "message"),
    [
        (SimpleNamespace(accelerator=SimpleNamespace()), "requires torch.npu"),
        (SimpleNamespace(npu=SimpleNamespace()), "requires torch.accelerator"),
    ],
)
def test_patch_fails_closed_for_incomplete_torch_runtime(
    torch_namespace, message
) -> None:
    module = _load_patch_module()
    module.torch = torch_namespace

    with pytest.raises(RuntimeError, match=message):
        module.patch_torch_accelerator()


def test_patch_fails_closed_before_partial_assignment() -> None:
    module = _load_patch_module()
    fake_torch, _ = _fake_torch()
    del fake_torch.npu.memory_reserved
    sentinel = object()
    for target in module._NPU_MEMORY_API_MAP:
        setattr(fake_torch.accelerator, target, sentinel)
    module.torch = fake_torch

    with pytest.raises(RuntimeError, match="memory_reserved"):
        module.patch_torch_accelerator()
    for target in module._NPU_MEMORY_API_MAP:
        assert getattr(fake_torch.accelerator, target) is sentinel


def test_pre_register_keeps_only_its_ascend_package_import() -> None:
    tree = ast.parse(PLATFORM_SOURCE.read_text(encoding="utf-8"))
    platform = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    method = next(
        node
        for node in platform.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "pre_register_and_update"
    )
    method.decorator_list = []
    namespace: dict[str, object] = {}
    ast.fix_missing_locations(method)
    exec(
        compile(
            ast.Module([method], type_ignores=[]),
            str(PLATFORM_SOURCE),
            "exec",
        ),
        namespace,
    )

    cls = SimpleNamespace(
        device_type="cuda", device_name="cuda", vendor_name="nvidia"
    )
    namespace["pre_register_and_update"](cls)
    source = ast.unparse(method)
    assert "patch_torch_accelerator" not in source
    assert "vllm_fl.dispatch.backends.vendor.ascend" in source


def _run_plugin_helper(platform, patch_callback) -> list[str]:
    tree = ast.parse(PLUGIN_SOURCE.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_ascend_torch_accelerator"
    )
    helper.decorator_list = []
    imported: list[str] = []

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        if name == "vllm.platforms":
            return SimpleNamespace(current_platform=platform)
        if name.endswith("patch_torch_accelerator"):
            return SimpleNamespace(patch_torch_accelerator=patch_callback)
        raise AssertionError(f"unexpected import from helper: {name}")

    namespace = {"__builtins__": {"__import__": fake_import}}
    ast.fix_missing_locations(helper)
    exec(
        compile(
            ast.Module([helper], type_ignores=[]),
            str(PLUGIN_SOURCE),
            "exec",
        ),
        namespace,
    )
    namespace["_patch_ascend_torch_accelerator"]()
    return imported


def test_register_model_installs_shim_for_ascend_before_other_hooks() -> None:
    calls: list[str] = []
    platform = SimpleNamespace(vendor_name="ascend", device_type="npu")

    imported = _run_plugin_helper(
        platform, lambda: calls.append("patch_torch_accelerator")
    )

    assert calls == ["patch_torch_accelerator"]
    assert imported == [
        "vllm.platforms",
        "vllm_fl.dispatch.backends.vendor.ascend.patches.patch_torch_accelerator",
    ]


@pytest.mark.parametrize(
    ("vendor_name", "device_type"),
    [("nvidia", "cuda"), ("other_npu", "npu"), ("ascend", "cuda")],
)
def test_register_model_does_not_patch_non_ascend_platforms(
    vendor_name, device_type
) -> None:
    sentinel_apis = {
        name: object() for name in _load_patch_module()._NPU_MEMORY_API_MAP
    }
    accelerator = SimpleNamespace(**sentinel_apis)
    before = {
        name: getattr(accelerator, name) for name in sentinel_apis
    }
    calls: list[str] = []
    platform = SimpleNamespace(
        vendor_name=vendor_name, device_type=device_type
    )

    imported = _run_plugin_helper(
        platform, lambda: calls.append("patch_torch_accelerator")
    )

    assert calls == []
    assert not any(name.endswith("patch_torch_accelerator") for name in imported)
    assert {
        name: getattr(accelerator, name) for name in sentinel_apis
    } == before


def test_register_model_calls_memory_shim_before_other_hooks() -> None:
    tree = ast.parse(PLUGIN_SOURCE.read_text(encoding="utf-8"))
    register_model = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "register_model"
    )
    first_statement = register_model.body[1]
    assert isinstance(first_statement, ast.Expr)
    assert isinstance(first_statement.value, ast.Call)
    assert isinstance(first_statement.value.func, ast.Name)
    assert first_statement.value.func.id == "_patch_ascend_torch_accelerator"


def test_worker_loads_general_plugins_before_worker_construction() -> None:
    worker_base = pytest.importorskip("vllm.v1.worker.worker_base")
    import inspect
    import textwrap

    source = textwrap.dedent(
        inspect.getsource(worker_base.WorkerWrapperBase.init_worker)
    )
    function = ast.parse(source).body[0]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    load_plugins = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "load_general_plugins"
    )
    construct_worker = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "worker_class"
    )
    assert load_plugins.lineno < construct_worker.lineno


def test_vllm_memory_snapshot_uses_redirected_npu_apis(monkeypatch) -> None:
    mem_utils = pytest.importorskip("vllm.utils.mem_utils")
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "accelerator"):
        pytest.skip("matched torch.accelerator API is unavailable")

    module = _load_patch_module()
    fake_torch, calls = _fake_torch()
    monkeypatch.setattr(torch, "npu", fake_torch.npu, raising=False)
    for target in module._NPU_MEMORY_API_MAP:
        monkeypatch.setattr(
            torch.accelerator,
            target,
            lambda *args, **kwargs: None,
            raising=False,
        )
    monkeypatch.setattr(
        torch.accelerator, module._PATCH_MARKER, False, raising=False
    )
    module.torch = torch
    module.patch_torch_accelerator()

    monkeypatch.setattr(
        mem_utils,
        "current_platform",
        SimpleNamespace(
            mem_get_info=lambda device: (100, 200),
            is_integrated_gpu=lambda device_index: False,
            device_name="NPU",
        ),
    )
    snapshot = mem_utils.MemorySnapshot(device=torch.device("npu:0"))

    assert snapshot.torch_peak == 11
    assert snapshot.free_memory == 100
    assert snapshot.total_memory == 200
    assert snapshot.torch_memory == 7
    assert snapshot.non_torch_memory == 93
    assert [name for name, _, _ in calls] == [
        "memory_stats",
        "memory_reserved",
    ]
