# Copyright (c) 2026 BAAI. All rights reserved.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode


def test_graph_class_resolution_is_lazy_and_device_scoped(monkeypatch) -> None:
    import vllm_fl.compilation.graph_runtime as graph_runtime
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    graph_class = type("FakeGraph", (), {})
    monkeypatch.setattr(
        torch,
        "fake_device",
        SimpleNamespace(FakeGraph=graph_class),
        raising=False,
    )

    runtime = GraphRuntimeController(device_type="fake_device")
    assert runtime._device_type == "fake_device"

    with pytest.raises(NotImplementedError, match="not supported"):
        runtime.resolve_graph_class()

    monkeypatch.setitem(graph_runtime._GRAPH_CLASS_NAMES, "fake_device", "FakeGraph")
    assert runtime.resolve_graph_class() is graph_class


def test_static_input_bindings_reject_reallocated_tensor() -> None:
    from vllm_fl.compilation.graph_runtime import StaticInputBindings

    tensor = torch.zeros(4)
    bindings = StaticInputBindings.from_args((tensor, "ignored"))
    bindings.validate((tensor, "ignored"))

    with pytest.raises(AssertionError, match="changed during replay"):
        bindings.validate((torch.zeros(4), "ignored"))


def test_model_compile_prewarm_is_npu_only(monkeypatch) -> None:
    import vllm_fl.dispatch as dispatch
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    calls = []
    monkeypatch.setattr(
        dispatch, "prewarm_cached_ops", lambda: calls.append("prewarm") or 4
    )

    assert GraphRuntimeController(device_type="cuda").prepare_model_compile() == 0
    assert calls == []
    assert GraphRuntimeController(device_type="npu").prepare_model_compile() == 4
    assert calls == ["prewarm"]


def test_npu_capture_flag_is_reset_when_capture_fails() -> None:
    from vllm_fl.compilation.graph_runtime import GraphPhase, GraphRuntimeController

    runtime = GraphRuntimeController(device_type="npu")
    forward_context = SimpleNamespace(
        capturing=False,
        attn_metadata={},
        batch_descriptor=SimpleNamespace(num_tokens=1),
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        with runtime.capture_scope(forward_context):
            assert forward_context.capturing is True
            raise RuntimeError("capture failed")

    assert forward_context.capturing is False
    assert runtime.phase is GraphPhase.IDLE


def test_npu_dsa_policy_skips_attention_task_lifecycle(monkeypatch) -> None:
    import vllm_fl.compilation.graph_params as params
    import vllm_fl.attention.ascend.attention as attention
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    calls = []
    monkeypatch.setattr(params, "prepare_graph_params", lambda size: calls.append(("prepare", size)))
    monkeypatch.setattr(params, "weak_ref_workspace", lambda size: calls.append(("weakref", size)))
    monkeypatch.setattr(attention.AscendAttentionBackendImpl, "update_graph_params",
                        lambda *args: calls.append("update"))
    runtime = GraphRuntimeController(
        device_type="npu",
        vllm_config=object(),
        update_attention_tasks=False,
        platform=SimpleNamespace(
            device_type="npu",
            torch_device_fn=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(
                    synchronize=lambda: calls.append("sync")
                )
            ),
        ),
    )
    context = SimpleNamespace(
        capturing=False, batch_descriptor=SimpleNamespace(num_tokens=2)
    )

    with runtime.capture_scope(context):
        assert context.capturing is True
    with runtime.replay_scope(context):
        calls.append("replay")
    assert context.capturing is False
    assert calls == ["sync", "replay"]


def test_graph_params_are_per_shape_and_fail_closed_on_duplicate_capture() -> None:
    from vllm_fl.compilation.graph_params import (
        clear_graph_params,
        get_graph_params,
        prepare_graph_params,
    )

    clear_graph_params()
    params = prepare_graph_params(1)
    prepare_graph_params(4)
    assert set(params.attn_params) == {1, 4}
    params.attn_params[1].append(object())

    with pytest.raises(RuntimeError, match="already contain captured state"):
        prepare_graph_params(1)

    clear_graph_params()
    assert get_graph_params().attn_params == {}


def test_npu_replay_enqueues_graph_before_updating_attention_tasks() -> None:
    from vllm_fl.compilation.graph_params import clear_graph_params
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    clear_graph_params()
    calls: list[object] = []
    update_stream = object()
    config = object()
    current_stream = SimpleNamespace(
        synchronize=lambda: calls.append("current-stream-synchronize")
    )
    platform = SimpleNamespace(
        device_type="npu",
        torch_device_fn=SimpleNamespace(
            current_stream=lambda: current_stream,
            Stream=lambda: calls.append("create-update-stream") or update_stream,
        ),
    )
    class FakeImpl:
        @staticmethod
        def update_graph_params(stream, context, size, passed_config):
            calls.append(
                ("update", stream, context, size, passed_config)
            )

    runtime = GraphRuntimeController(
        device_type="npu",
        platform=platform,
        vllm_config=config,
    )
    runtime.set_attention_impls((FakeImpl,))
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=1))

    with runtime.replay_scope(context):
        calls.append("graph-replay")
    with runtime.replay_scope(context):
        calls.append("graph-replay")

    assert calls == [
        "current-stream-synchronize",
        "graph-replay",
        "create-update-stream",
        ("update", update_stream, context, 1, config),
        "current-stream-synchronize",
        "graph-replay",
        ("update", update_stream, context, 1, config),
    ]


def test_npu_replay_dispatches_sfa_noop_through_active_backend() -> None:
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    calls: list[object] = []

    class FakeSFAImpl:
        @staticmethod
        def update_graph_params(stream, context, size, passed_config):
            calls.append(("sfa-update", stream, context, size, passed_config))

    update_stream = object()
    config = object()
    runtime = GraphRuntimeController(
        device_type="npu",
        vllm_config=config,
        platform=SimpleNamespace(
            device_type="npu",
            torch_device_fn=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(
                    synchronize=lambda: calls.append("sync")
                ),
                Stream=lambda: update_stream,
            ),
        ),
    )
    runtime.set_attention_impls((FakeSFAImpl,))
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=1))

    with runtime.replay_scope(context):
        calls.append("graph-replay")

    assert calls == [
        "sync",
        "graph-replay",
        ("sfa-update", update_stream, context, 1, config),
    ]


def test_npu_replay_accepts_cache_only_backend_set() -> None:
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    runtime = GraphRuntimeController(
        device_type="npu",
        vllm_config=object(),
        platform=SimpleNamespace(
            device_type="npu",
            torch_device_fn=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
                Stream=object,
            ),
        ),
    )
    runtime.set_attention_impls(())
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=1))

    with runtime.replay_scope(context):
        pass
    assert runtime._update_stream is None


def test_attention_impl_resolver_skips_cache_only_and_deduplicates() -> None:
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    class CacheOnlyBackend(AttentionBackend):
        pass

    class ActiveImpl:
        update_graph_params = staticmethod(lambda *args: None)

    class ActiveBackend(AttentionBackend):
        get_impl_cls = staticmethod(lambda: ActiveImpl)

    attn_groups = [
        [SimpleNamespace(backend=CacheOnlyBackend)],
        [
            SimpleNamespace(backend=ActiveBackend),
            SimpleNamespace(backend=ActiveBackend),
        ],
    ]

    assert GraphRuntimeController.resolve_attention_impls(attn_groups) == (
        ActiveImpl,
    )


def test_attention_impl_resolver_accepts_no_concrete_backend() -> None:
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    class CacheOnlyBackend(AttentionBackend):
        pass

    attn_groups = [[SimpleNamespace(backend=CacheOnlyBackend)]]

    assert GraphRuntimeController.resolve_attention_impls(attn_groups) == ()


def test_attention_impl_resolver_preserves_unique_impl_order() -> None:
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    class FirstImpl:
        update_graph_params = staticmethod(lambda *args: None)

    class SecondImpl:
        update_graph_params = staticmethod(lambda *args: None)

    class FirstBackend(AttentionBackend):
        get_impl_cls = staticmethod(lambda: FirstImpl)

    class FirstAliasBackend(AttentionBackend):
        get_impl_cls = staticmethod(lambda: FirstImpl)

    class SecondBackend(AttentionBackend):
        get_impl_cls = staticmethod(lambda: SecondImpl)

    attn_groups = [[
        SimpleNamespace(backend=FirstBackend),
        SimpleNamespace(backend=SecondBackend),
        SimpleNamespace(backend=FirstAliasBackend),
    ]]

    assert GraphRuntimeController.resolve_attention_impls(attn_groups) == (
        FirstImpl,
        SecondImpl,
    )


def test_attention_impl_resolver_fails_closed_for_invalid_concrete_impl() -> None:
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    class InvalidImpl:
        pass

    class InvalidBackend(AttentionBackend):
        get_impl_cls = staticmethod(lambda: InvalidImpl)

    attn_groups = [[SimpleNamespace(backend=InvalidBackend)]]

    with pytest.raises(RuntimeError, match="InvalidBackend -> InvalidImpl"):
        GraphRuntimeController.resolve_attention_impls(attn_groups)


def test_attention_group_binding_skips_backend_resolution_when_disabled() -> None:
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    class MissingUpdater:
        pass

    class MissingUpdaterBackend(AttentionBackend):
        get_impl_cls = staticmethod(lambda: MissingUpdater)

    class RaisingBackend(AttentionBackend):
        @staticmethod
        def get_impl_cls():
            raise AssertionError("disabled attention tasks must not resolve backend")

    runtime = GraphRuntimeController(update_attention_tasks=False)
    assert runtime.bind_attention_groups([[
        SimpleNamespace(backend=MissingUpdaterBackend),
        SimpleNamespace(backend=RaisingBackend),
    ]]) == ()
    assert runtime._attention_impls == ()


def test_attention_group_binding_keeps_enabled_path_fail_closed() -> None:
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    class InvalidImpl:
        pass

    class InvalidBackend(AttentionBackend):
        get_impl_cls = staticmethod(lambda: InvalidImpl)

    runtime = GraphRuntimeController(update_attention_tasks=True)
    with pytest.raises(RuntimeError, match="InvalidBackend -> InvalidImpl"):
        runtime.bind_attention_groups([[SimpleNamespace(backend=InvalidBackend)]])
    assert runtime._attention_impls is None


def test_npu_replay_requires_runner_config() -> None:
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    runtime = GraphRuntimeController(
        device_type="npu",
        platform=SimpleNamespace(
            device_type="npu",
            torch_device_fn=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(synchronize=lambda: None)
            ),
        ),
    )
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=1))

    with pytest.raises(RuntimeError, match="active VllmConfig"):
        with runtime.replay_scope(context):
            pass


def test_npu_replay_failure_does_not_update_attention_tasks(monkeypatch) -> None:
    import vllm_fl.attention.ascend.attention as attention
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    monkeypatch.setattr(
        attention.AscendAttentionBackendImpl,
        "update_graph_params",
        lambda *args: pytest.fail("failed replay must not update graph tasks"),
    )
    runtime = GraphRuntimeController(
        device_type="npu",
        vllm_config=object(),
        platform=SimpleNamespace(
            device_type="npu",
            torch_device_fn=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(synchronize=lambda: None)
            ),
        ),
    )
    context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=1))

    with pytest.raises(RuntimeError, match="replay failed"):
        with runtime.replay_scope(context):
            raise RuntimeError("replay failed")


@pytest.mark.parametrize(
    ("device_type", "dist_backend"),
    [("npu", "hccl"), ("musa", "nccl"), ("cuda", "flagcx")],
)
def test_local_graph_capture_uses_isolated_device_stream(
    monkeypatch, device_type: str, dist_backend: str
) -> None:
    import vllm_fl.compilation.graph_runtime as graph_runtime

    events: list[object] = []
    current_stream = object()

    class CaptureStream:
        def __init__(self, *, device: torch.device) -> None:
            events.append(("create", device))

        def wait_stream(self, stream: object) -> None:
            events.append(("wait", stream))

    @contextmanager
    def stream_context(stream: CaptureStream):
        events.append(("enter", stream))
        try:
            yield
        finally:
            events.append(("exit", stream))

    platform = SimpleNamespace(
        device_type=device_type,
        dist_backend=dist_backend,
        torch_device_fn=SimpleNamespace(
            Stream=CaptureStream,
            current_stream=lambda: current_stream,
            stream=stream_context,
        ),
    )
    monkeypatch.setattr(graph_runtime, "current_platform", platform)

    @contextmanager
    def default_capture(*, device):
        pytest.fail(f"default capture must not be called for {device_type}")
        yield

    device = torch.device("cpu")
    capture = graph_runtime.get_graph_capture(default_capture)
    with capture(device) as context:
        events.append(("body", context.stream))

    capture_stream = context.stream
    assert events == [
        ("create", device),
        ("wait", current_stream),
        ("enter", capture_stream),
        ("body", capture_stream),
        ("exit", capture_stream),
    ]


def test_npu_graph_capture_restores_stream_context_after_error(monkeypatch) -> None:
    import vllm_fl.compilation.graph_runtime as graph_runtime

    events: list[str] = []

    class CaptureStream:
        def __init__(self, *, device: torch.device) -> None:
            pass

        def wait_stream(self, stream: object) -> None:
            pass

    @contextmanager
    def stream_context(stream: CaptureStream):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    platform = SimpleNamespace(
        device_type="npu",
        dist_backend="hccl",
        torch_device_fn=SimpleNamespace(
            Stream=CaptureStream,
            current_stream=object,
            stream=stream_context,
        ),
    )
    monkeypatch.setattr(graph_runtime, "current_platform", platform)

    @contextmanager
    def default_capture(*, device):
        pytest.fail("NPU must not delegate graph capture")
        yield

    with pytest.raises(RuntimeError, match="capture failed"):
        with graph_runtime.get_graph_capture(default_capture)(torch.device("cpu")):
            events.append("body")
            raise RuntimeError("capture failed")

    assert events == ["enter", "body", "exit"]


def test_cuda_graph_capture_delegates_to_upstream(monkeypatch) -> None:
    import vllm_fl.compilation.graph_runtime as graph_runtime

    events: list[object] = []
    platform = SimpleNamespace(device_type="cuda", dist_backend="nccl")
    monkeypatch.setattr(graph_runtime, "current_platform", platform)

    @contextmanager
    def default_capture(*, device):
        events.append(("enter", device))
        try:
            yield "upstream-context"
        finally:
            events.append(("exit", device))

    device = torch.device("cpu")
    with graph_runtime.get_graph_capture(default_capture)(device) as context:
        events.append(("body", context))

    assert events == [
        ("enter", device),
        ("body", "upstream-context"),
        ("exit", device),
    ]


def test_graph_wrapper_owns_capture_replay_and_attention_metadata(
    monkeypatch,
) -> None:
    import vllm_fl.compilation.graph as graph_module
    from vllm_fl.compilation.graph import GraphOptions, GraphWrapper
    from vllm_fl.compilation.graph_runtime import GraphPhase, GraphRuntimeController

    events: list[str] = []
    attention_metadata = {"layer": object()}
    forward_context = SimpleNamespace(
        batch_descriptor=object(),
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        attn_metadata=attention_metadata,
    )

    class RecordingRuntime(GraphRuntimeController):
        def on_capture_begin(self, context):
            events.append(
                f"capture-begin:{self.phase.name}:"
                f"{context.attn_metadata is attention_metadata}"
            )

        def on_capture_end(self, context):
            events.append(
                f"capture-end:{self.phase.name}:"
                f"{context.attn_metadata is attention_metadata}"
            )

        def before_replay(self, context):
            events.append(
                f"replay-begin:{self.phase.name}:"
                f"{context.attn_metadata is attention_metadata}"
            )

        def after_replay(self, context):
            events.append(
                f"replay-end:{self.phase.name}:"
                f"{context.attn_metadata is attention_metadata}"
            )

    class FakeGraph:
        def replay(self) -> None:
            events.append(f"replay:{runtime.phase.name}")

    @contextmanager
    def graph_context(graph, pool=None):
        events.append(f"graph-enter:{runtime.phase.name}")
        yield
        events.append(f"graph-exit:{runtime.phase.name}")

    platform = SimpleNamespace(
        device_type="cuda",
        get_global_graph_pool=lambda: None,
        graph_pool_handle=lambda: None,
        torch_device_fn=SimpleNamespace(graph=graph_context),
    )
    runtime = RecordingRuntime(device_type="cuda", platform=platform)
    monkeypatch.setattr(runtime, "create_graph", FakeGraph)
    monkeypatch.setattr(graph_module, "current_platform", platform)
    monkeypatch.setattr(graph_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(graph_module, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        graph_module, "validate_cudagraph_capturing_enabled", lambda: None
    )
    monkeypatch.setattr(graph_module, "set_graph_pool_id", lambda pool: None)

    def runnable(tensor):
        events.append(f"forward:{runtime.phase.name}")
        return tensor + 1

    wrapper = GraphWrapper(
        runnable,
        SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_copy_inputs=False)
        ),
        CUDAGraphMode.FULL,
        GraphOptions(weak_ref_output=False),
        graph_runtime=runtime,
    )
    tensor = torch.zeros(1)

    captured = wrapper(tensor)
    entry = wrapper.concrete_graph_entries[forward_context.batch_descriptor]
    replayed = wrapper(tensor)

    assert torch.equal(captured, torch.ones(1))
    assert torch.equal(replayed, torch.ones(1))
    assert entry.input_addresses == [tensor.data_ptr()]
    assert events == [
        "capture-begin:CAPTURING:True",
        "graph-enter:CAPTURING",
        "forward:CAPTURING",
        "graph-exit:CAPTURING",
        "capture-end:CAPTURING:True",
        "replay-begin:REPLAYING:True",
        "replay:REPLAYING",
        "replay-end:REPLAYING:True",
    ]
    assert runtime.phase is GraphPhase.IDLE


def test_graph_wrapper_eager_passthrough_without_forward_context(monkeypatch) -> None:
    import vllm_fl.compilation.graph as graph_module
    from vllm_fl.compilation.graph import GraphWrapper
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    runtime = GraphRuntimeController(device_type="cuda")
    monkeypatch.setattr(graph_module, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(
        graph_module.current_platform, "get_global_graph_pool", lambda: None
    )
    monkeypatch.setattr(
        runtime,
        "create_graph",
        lambda: pytest.fail("eager passthrough must not create a graph"),
    )
    calls: list[int] = []
    wrapper = GraphWrapper(
        lambda value: calls.append(value) or value + 1,
        SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_copy_inputs=False)
        ),
        CUDAGraphMode.FULL,
        graph_runtime=runtime,
    )

    assert wrapper(3) == 4
    assert calls == [3]


def test_runner_controller_preserves_full_breakable_and_ubatch_policy() -> None:
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    calls: list[tuple] = []

    class FullWrapper:
        _all_instances = []

        def __init__(self, model, config, runtime_mode, graph_runtime):
            calls.append(("full", model, runtime_mode))

    class BreakableWrapper:
        _all_instances = []

        def __init__(self, model, config):
            self.model = model
            calls.append(("breakable", model))

    class UBatchWrapper:
        def __init__(self, model, config, runtime_mode, device):
            calls.append(("ubatch", model, runtime_mode, device))

    runtime = GraphRuntimeController(
        full_graph_wrapper_type=FullWrapper,
        breakable_graph_wrapper_type=BreakableWrapper,
        ubatch_wrapper_type=UBatchWrapper,
    )
    config = SimpleNamespace()
    drafter = SimpleNamespace(model="draft")

    breakable = runtime.wrap_model(
        "model",
        config,
        cudagraph_mode=CUDAGraphMode.FULL,
        use_ubatching=False,
        device=torch.device("cpu"),
        drafter=drafter,
        breakable_enabled=True,
    )
    assert isinstance(breakable, BreakableWrapper)
    assert isinstance(drafter.model, BreakableWrapper)

    full = runtime.wrap_model(
        "model",
        config,
        cudagraph_mode=CUDAGraphMode.FULL,
        use_ubatching=False,
        device=torch.device("cpu"),
        drafter=None,
        breakable_enabled=False,
    )
    assert isinstance(full, FullWrapper)

    runtime.wrap_model(
        "model",
        config,
        cudagraph_mode=CUDAGraphMode.NONE,
        use_ubatching=True,
        device=torch.device("cpu"),
        drafter=None,
        breakable_enabled=False,
    )
    assert ("ubatch", "model", CUDAGraphMode.NONE, torch.device("cpu")) in calls


def test_decoder_wrapper_registry_handles_full_and_breakable() -> None:
    from vllm_fl.compilation.graph_runtime import GraphRuntimeController

    class WrapperType:
        def __init__(self, pool):
            self.graph_pool = pool

    full = WrapperType("full-pool")
    breakable = WrapperType("breakable-pool")

    class FullRegistry:
        _all_instances = [full]
        clears = 0

        @classmethod
        def clear_all_graphs(cls):
            cls.clears += 1

    class BreakableRegistry:
        _all_instances = [breakable]
        clears = 0

        @classmethod
        def clear_all_graphs(cls):
            cls.clears += 1

    runtime = GraphRuntimeController(
        full_graph_wrapper_type=FullRegistry,
        breakable_graph_wrapper_type=BreakableRegistry,
    )

    pools = runtime.set_decoder_graph_pool("profile-pool")
    assert full.graph_pool == "profile-pool"
    assert breakable.graph_pool == "profile-pool"

    runtime.clear_decoder_graphs()
    runtime.restore_decoder_graph_pools(pools)
    assert FullRegistry.clears == 1
    assert BreakableRegistry.clears == 1
    assert full.graph_pool == "full-pool"
    assert breakable.graph_pool == "breakable-pool"
