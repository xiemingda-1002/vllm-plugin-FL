import ast
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[3] / "vllm_fl/worker/worker.py"
)


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def test_worker_uses_current_vllm_memory_profiler() -> None:
    tree = _tree()
    local_classes = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    assert "MemorySnapshot" not in local_classes
    assert "MemoryProfilingResult" not in local_classes

    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm.utils.mem_utils"
        for alias in node.names
    }
    assert {"MemorySnapshot", "memory_profiling"} <= imports


def test_memory_baseline_order_is_platform_specific() -> None:
    tree = _tree()
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL"
    )
    init_device = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_device"
    )
    calls = [node for node in ast.walk(init_device) if isinstance(node, ast.Call)]
    snapshot_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "_take_initial_memory_snapshot"
    ]
    distributed = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "init_worker_distributed_environment"
    )
    seed = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "set_random_seed"
    )

    assert len(snapshot_calls) == 2
    assert min(call.lineno for call in snapshot_calls) < distributed.lineno
    assert distributed.lineno < seed.lineno
    assert seed.lineno < max(call.lineno for call in snapshot_calls)

    snapshot_method = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_take_initial_memory_snapshot"
    )
    snapshot = next(
        node
        for node in ast.walk(snapshot_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MemorySnapshot"
    )
    assert any(keyword.arg == "device" for keyword in snapshot.keywords)
