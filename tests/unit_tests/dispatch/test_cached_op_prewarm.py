# Copyright (c) 2026 BAAI. All rights reserved.

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch


class FakeManager:
    def __init__(self, implementations):
        self.policy_epoch = 0
        self.implementations = implementations
        self.resolve_calls = []
        self.first_use_calls = []
        self.failed = []
        self.fallback_calls = 0

    def _resolve_impl(self, op_name):
        self.resolve_calls.append(op_name)
        implementation = self.implementations.get(op_name)
        if implementation is None:
            raise RuntimeError(
                f"No available implementation for op='{op_name}'. Registered: []"
            )
        if isinstance(implementation, Exception):
            raise implementation
        return implementation

    def _record_first_use(self, op_name, implementation):
        self.first_use_calls.append((op_name, implementation.impl_id))

    def _mark_failed_impl(self, op_name, impl_id):
        self.failed.append((op_name, impl_id))

    def call(self, op_name, *args, **kwargs):
        self.fallback_calls += 1
        return 99


@pytest.fixture
def isolated_cached_ops(monkeypatch):
    import vllm_fl.dispatch as dispatch

    registry = weakref.WeakSet()
    monkeypatch.setattr(dispatch, "_CACHED_OPS", registry)
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(dispatch, "is_dump_enabled", lambda: False)
    monkeypatch.setattr(dispatch, "get_policy_epoch", lambda: 0)
    return dispatch, registry


def _impl(impl_id, fn):
    return SimpleNamespace(impl_id=impl_id, fn=fn)


def test_prewarm_resolves_and_records_first_use_without_execution(
    monkeypatch, isolated_cached_ops
) -> None:
    dispatch, _ = isolated_cached_ops
    executions = []
    manager = FakeManager({"op": _impl("vendor.op", lambda x: executions.append(x) or x)})
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    cached_op = dispatch.CachedOp("op")

    assert dispatch.prewarm_cached_ops() == 1
    assert manager.resolve_calls == ["op"]
    assert manager.first_use_calls == []
    assert executions == []
    assert cached_op(7) == 7
    assert manager.resolve_calls == ["op"]
    assert manager.first_use_calls == [("op", "vendor.op")]
    assert executions == [7]


def test_prewarm_keeps_only_unavailable_optional_op_lazy(
    monkeypatch, isolated_cached_ops
) -> None:
    dispatch, _ = isolated_cached_ops
    manager = FakeManager({"required": _impl("vendor.required", lambda x: x + 1)})
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    required = dispatch.CachedOp("required")
    optional = dispatch.CachedOp("optional")

    assert dispatch.prewarm_cached_ops() == 1
    assert required._prepared_for_compile is True
    assert optional._impl is None
    assert optional._prepared_for_compile is False

    manager.implementations["optional"] = _impl("vendor.optional", lambda x: x * 2)
    assert optional(3) == 6


def test_prewarm_propagates_initialization_and_configuration_errors(
    monkeypatch, isolated_cached_ops
) -> None:
    dispatch, _ = isolated_cached_ops
    manager = FakeManager({"op": RuntimeError("plugin initialization failed")})
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    cached_op = dispatch.CachedOp("op")

    with pytest.raises(RuntimeError, match="plugin initialization failed"):
        dispatch.prewarm_cached_ops()
    assert cached_op._impl is None


@pytest.mark.parametrize(
    "message",
    [
        "No implementation available for op='op' under strict policy. Candidates: []",
        "No implementation selected for op='op'. Candidates: [], Order: []",
    ],
)
def test_prewarm_propagates_missing_or_policy_selection_errors(
    monkeypatch, isolated_cached_ops, message
) -> None:
    dispatch, _ = isolated_cached_ops
    manager = FakeManager({"op": RuntimeError(message)})
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    cached_op = dispatch.CachedOp("op")

    with pytest.raises(RuntimeError, match="No implementation"):
        dispatch.prewarm_cached_ops()
    assert cached_op._impl is None


def test_registry_does_not_keep_cached_op_alive(isolated_cached_ops) -> None:
    dispatch, registry = isolated_cached_ops
    cached_op = dispatch.CachedOp("temporary")
    ref = weakref.ref(cached_op)
    assert list(registry) == [cached_op]

    del cached_op
    gc.collect()

    assert ref() is None
    assert list(registry) == []


def test_eager_policy_invalidation_and_fallback_are_preserved(
    monkeypatch, isolated_cached_ops
) -> None:
    dispatch, _ = isolated_cached_ops
    manager = FakeManager({"op": _impl("first", lambda x: x + 1)})
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    monkeypatch.setattr(dispatch, "get_policy", lambda: SimpleNamespace(strict=False))
    cached_op = dispatch.CachedOp("op")
    assert dispatch.prewarm_cached_ops() == 1
    assert cached_op(1) == 2

    manager.policy_epoch += 1
    manager.implementations["op"] = _impl(
        "failing", lambda x: (_ for _ in ()).throw(RuntimeError("failed"))
    )
    assert cached_op(1) == 99
    assert manager.resolve_calls == ["op", "op"]
    assert manager.failed == [("op", "failing")]
    assert manager.fallback_calls == 1
    assert cached_op._prepared_for_compile is False


def test_fullgraph_uses_only_prewarmed_tensor_implementation(
    monkeypatch, isolated_cached_ops
) -> None:
    dispatch, _ = isolated_cached_ops
    manager = FakeManager({"op": _impl("tensor.add", lambda x: x + 1)})
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    cached_op = dispatch.CachedOp("op")
    assert dispatch.prewarm_cached_ops() == 1

    def forbidden(*args, **kwargs):
        raise AssertionError("dispatch manager state was touched during tracing")

    monkeypatch.setattr(dispatch, "get_default_manager", forbidden)
    monkeypatch.setattr(dispatch, "get_policy_epoch", forbidden)
    monkeypatch.setattr(dispatch, "get_policy", forbidden)
    monkeypatch.setattr(dispatch, "is_dump_enabled", forbidden)
    monkeypatch.setattr(dispatch._logger, "debug", forbidden)
    compiled = torch.compile(lambda x: cached_op(x), fullgraph=True, backend="eager")

    assert torch.equal(compiled(torch.tensor([2])), torch.tensor([3]))


def test_fullgraph_fails_closed_without_prewarm_and_never_falls_back(
    monkeypatch, isolated_cached_ops
) -> None:
    dispatch, _ = isolated_cached_ops
    manager = FakeManager({"op": _impl("unused", lambda x: x)})
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    cached_op = dispatch.CachedOp("op")
    compiled = torch.compile(lambda x: cached_op(x), fullgraph=True, backend="eager")

    with pytest.raises(Exception, match="was not prepared before compilation"):
        compiled(torch.tensor([1]))
    assert manager.resolve_calls == []
    assert manager.fallback_calls == 0


def test_fullgraph_impl_error_does_not_enter_eager_fallback(
    monkeypatch, isolated_cached_ops
) -> None:
    dispatch, _ = isolated_cached_ops

    def fail_impl(x):
        raise RuntimeError("implementation failed while tracing")

    manager = FakeManager({"op": _impl("failing", fail_impl)})
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    cached_op = dispatch.CachedOp("op")
    assert dispatch.prewarm_cached_ops() == 1
    compiled = torch.compile(lambda x: cached_op(x), fullgraph=True, backend="eager")

    with pytest.raises(Exception, match="implementation failed while tracing"):
        compiled(torch.tensor([1]))
    assert manager.failed == []
    assert manager.fallback_calls == 0


def test_disabled_fast_path_rejects_compile_prewarm(
    monkeypatch, isolated_cached_ops
) -> None:
    dispatch, _ = isolated_cached_ops
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", False)
    cached_op = dispatch.CachedOp("op")

    with pytest.raises(RuntimeError, match="VLLM_FL_OP_FAST_PATH=1"):
        dispatch.prewarm_cached_ops()
    assert cached_op._impl is None
