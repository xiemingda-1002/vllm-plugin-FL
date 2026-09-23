"""CPU-only source-contract tests; never import device-aware FL packages."""

import ast
import logging
import sys
import threading
import weakref
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

# Resolve the package source before fixtures replace platform modules. This
# supports editable vLLM installs without importing its device-aware package.
_VLLM_SOURCE = Path(find_spec("vllm").origin).parent


@pytest.fixture
def subject(monkeypatch):
    source = Path(__file__).parents[3] / "vllm_fl/dispatch/__init__.py"
    names = {"CachedOp", "_requires_compile_prewarm", "prewarm_cached_ops"}
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == names
    module = ModuleType("_cached_op_isolation_subject")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    manager = SimpleNamespace(
        policy_epoch=0,
        impl=SimpleNamespace(impl_id="test.add", fn=lambda x: x + 1),
        calls=[],
    )
    manager._resolve_impl = lambda name: manager.impl
    manager._record_first_use = lambda *args: manager.calls.append("first-use")
    manager._mark_failed_impl = lambda *args: manager.calls.append("failed")
    manager.call = lambda name, x: x + 10
    module.__dict__.update(
        torch=torch,
        _CACHED_OPS=weakref.WeakSet(),
        _CACHED_OPS_LOCK=threading.RLock(),
        _OP_FAST_PATH_ENABLED=True,
        _UNAVAILABLE_OP_ERROR="No available implementation for op=",
        _logger=logging.getLogger(__name__),
        get_default_manager=lambda: manager,
        get_policy_epoch=lambda: 0,
        get_policy=lambda: SimpleNamespace(strict=False),
        is_dump_enabled=lambda: False,
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
        module.__dict__,
    )
    # Only the platform identity is simulated. The compile guard and cached
    # dispatch methods above are executed verbatim from the product source.
    platform = SimpleNamespace(vendor_name="nvidia", device_type="cuda")
    root = ModuleType("vllm")
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = platform
    root.platforms = platforms
    monkeypatch.setitem(sys.modules, "vllm", root)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    torch._dynamo.reset()
    yield module, manager, platform
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "vendor,device",
    [
        ("nvidia", "cuda"),
        ("mthreads", "musa"),
        ("other", "npu"),
        ("ascend", "cuda"),
    ],
)
@pytest.mark.parametrize("fast_path", [True, False])
def test_non_ascend_compile_keeps_original_dispatch(subject, vendor, device, fast_path):
    module, _, platform = subject
    platform.vendor_name, platform.device_type = vendor, device
    module._OP_FAST_PATH_ENABLED = fast_path
    cached = module.CachedOp("test")
    sample = torch.tensor([2.0])
    expected = torch.tensor([3.0 if fast_path else 12.0])
    assert torch.equal(cached(sample), expected)
    assert cached._prepared_for_compile is False
    compiled = torch.compile(lambda x: cached(x), fullgraph=True, backend="eager")
    assert torch.equal(compiled(sample), expected)


@pytest.mark.parametrize("warm_eager", [False, True])
def test_ascend_still_rejects_missing_compile_prewarm(subject, warm_eager):
    module, _, platform = subject
    platform.vendor_name, platform.device_type = "ascend", "npu"
    cached = module.CachedOp("test")
    sample = torch.tensor([2.0])
    if warm_eager:
        assert torch.equal(cached(sample), sample + 1)
    compiled = torch.compile(lambda x: cached(x), fullgraph=True, backend="eager")
    with pytest.raises(Exception, match="was not prepared before compilation"):
        compiled(sample)


def test_ascend_prepared_trace_never_enters_manager(subject):
    module, _, platform = subject
    platform.vendor_name, platform.device_type = "ascend", "npu"
    cached = module.CachedOp("test")
    assert module.prewarm_cached_ops() == 1

    def forbidden(*args):
        raise AssertionError("tracing entered the dispatch manager")

    module.get_default_manager = forbidden
    module.get_policy_epoch = forbidden
    compiled = torch.compile(lambda x: cached(x), fullgraph=True, backend="eager")
    sample = torch.tensor([2.0])
    assert torch.equal(compiled(sample), sample + 1)


def test_ascend_still_rejects_disabled_fast_path(subject):
    module, _, platform = subject
    platform.vendor_name, platform.device_type = "ascend", "npu"
    module._OP_FAST_PATH_ENABLED = False
    cached = module.CachedOp("test")
    with pytest.raises(RuntimeError, match="requires VLLM_FL_OP_FAST_PATH=1"):
        module.prewarm_cached_ops()
    compiled = torch.compile(lambda x: cached(x), fullgraph=True, backend="eager")
    with pytest.raises(Exception, match="FAST_PATH disabled"):
        compiled(torch.tensor([2.0]))


def test_eager_does_not_resolve_platform(subject, monkeypatch):
    module, _, _ = subject

    def forbidden():
        raise AssertionError("eager execution queried the compile platform gate")

    monkeypatch.setattr(module, "_requires_compile_prewarm", forbidden)
    assert module.CachedOp("test")(4) == 5


@pytest.mark.parametrize("vendor,device", [("nvidia", "cuda"), ("ascend", "npu")])
def test_compile_with_upstream_lazy_platform_lookup(subject, vendor, device):
    module, _, platform = subject
    platform.vendor_name, platform.device_type = vendor, device
    platforms = sys.modules["vllm.platforms"]
    del platforms.current_platform
    platforms._current_platform = platform
    source = _VLLM_SOURCE / "platforms/__init__.py"
    tree = ast.parse(Path(source).read_text())
    lookup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    )
    exec(
        compile(ast.Module(body=[lookup], type_ignores=[]), str(source), "exec"),
        platforms.__dict__,
    )
    cached = module.CachedOp("test")
    sample = torch.tensor([2.0])
    assert torch.equal(cached(sample), sample + 1)
    if vendor == "ascend":
        cached.prepare_for_compile()
    compiled = torch.compile(lambda x: cached(x), fullgraph=True, backend="eager")
    assert torch.equal(compiled(sample), sample + 1)


@pytest.mark.parametrize("vendor,device", [("nvidia", "cuda"), ("ascend", "npu")])
def test_eager_policy_invalidation_and_fallback_preserved(subject, vendor, device):
    module, manager, platform = subject
    platform.vendor_name, platform.device_type = vendor, device
    cached = module.CachedOp("test")
    assert cached(1) == 2
    manager.impl = SimpleNamespace(impl_id="test.next", fn=lambda x: x + 2)
    manager.policy_epoch += 1
    assert cached(1) == 3

    def fail(x):
        raise RuntimeError("implementation failed")

    manager.impl = SimpleNamespace(impl_id="test.fail", fn=fail)
    manager.policy_epoch += 1
    assert cached(1) == 11
    assert cached._use_manager_call
    assert "failed" in manager.calls
