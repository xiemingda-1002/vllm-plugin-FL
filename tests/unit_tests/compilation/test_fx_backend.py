# Copyright (c) 2026 BAAI. All rights reserved.

"""Contracts for the optional normalized FX fusion adapter.

These sources are suitable for a dependency-complete test environment; keeping
CPU tensors here does not establish Ascend kernel or model correctness.
"""

import inspect
import operator
import sys
from types import SimpleNamespace

import pytest
import torch


class RecordingManager:
    def __init__(self, passes=None):
        self.passes = [object()] if passes is None else passes
        self.graphs = []

    def __call__(self, graph):
        self.graphs.append(graph)


def _compile(graph, inputs, manager, monkeypatch, *, pre_grad=None):
    from vllm.config.compilation import Range

    import vllm_fl.compilation.fx_backend as backend

    # Replace this module's reference; the real platform has a read-only
    # pass_key property and may be shared by unrelated compilation code.
    monkeypatch.setattr(
        backend, "current_platform", SimpleNamespace(pass_key="test_fusion_manager")
    )
    compile_range = Range(1, 1)
    compiler_config = {"test_fusion_manager": manager}
    if pre_grad is not None:
        compiler_config["pre_grad_custom_pass"] = pre_grad
    return backend.FXFusionAdaptor().compile(
        graph, inputs, compiler_config, compile_range
    )


@pytest.mark.parametrize("tuple_output", [False, True])
def test_aot_normalizes_graph_and_preserves_output_container(
    monkeypatch, tuple_output
) -> None:
    monkeypatch.setitem(sys.modules, "npugraph_ex", None)
    fn = (lambda value: (value + 1,)) if tuple_output else (lambda value: value + 1)
    graph = torch.fx.symbolic_trace(fn)
    manager = RecordingManager()

    compiled, handle = _compile(graph, [torch.zeros(3)], manager, monkeypatch)
    result = compiled(torch.zeros(3))

    assert handle is None
    assert isinstance(result, tuple) == tuple_output
    torch.testing.assert_close(result[0] if tuple_output else result, torch.ones(3))
    assert len(manager.graphs) == 1
    assert manager.graphs[0] is not graph.graph
    assert any(
        node.target == torch.ops.aten.add.Tensor for node in manager.graphs[0].nodes
    )


def test_aot_preserves_input_mutation_and_multiple_outputs(monkeypatch) -> None:
    def mutate(value):
        value.add_(2)
        return value, value * 3

    original = torch.fx.symbolic_trace(mutate)
    original_code = original.code
    inputs = torch.zeros(3)
    manager = RecordingManager()
    compiled, handle = _compile(original, [inputs.clone()], manager, monkeypatch)
    result = compiled(inputs)

    assert handle is None
    assert isinstance(result, tuple) and len(result) == 2
    torch.testing.assert_close(inputs, torch.full((3,), 2.0))
    torch.testing.assert_close(result[0], inputs)
    torch.testing.assert_close(result[1], torch.full((3,), 6.0))
    assert original.code == original_code
    assert len(manager.graphs) == 1


def test_no_active_passes_reuses_native_eager_compile(monkeypatch) -> None:
    from vllm.compilation.compiler_interface import EagerAdaptor

    import vllm_fl.compilation.fx_backend as backend

    graph = torch.fx.symbolic_trace(lambda value: value + 1)
    inputs = [torch.zeros(1)]
    calls = []
    sentinel = object()

    def eager_compile(self, *args, **kwargs):
        calls.append((args, kwargs))
        return sentinel, None

    monkeypatch.setattr(EagerAdaptor, "compile", eager_compile)
    monkeypatch.setattr(
        backend,
        "aot_autograd",
        lambda **kwargs: pytest.fail("AOT must be bypassed without active passes"),
    )
    result = _compile(
        graph,
        inputs,
        RecordingManager(passes=[]),
        monkeypatch,
        pre_grad=lambda graph: pytest.fail(
            "Empty passes must retain native eager behavior"
        ),
    )

    assert result == (sentinel, None)
    assert len(calls) == 1
    assert calls[0][0][0] is graph
    assert calls[0][0][1] is inputs


def test_missing_manager_fails_explicitly(monkeypatch) -> None:
    import vllm_fl.compilation.fx_backend as backend

    graph = torch.fx.symbolic_trace(lambda value: value + 1)
    monkeypatch.setattr(
        backend,
        "aot_autograd",
        lambda **kwargs: pytest.fail("Missing manager must fail before AOT"),
    )
    monkeypatch.setattr(
        backend, "current_platform", SimpleNamespace(pass_key="test_fusion_manager")
    )
    with pytest.raises(KeyError, match="test_fusion_manager"):
        backend.FXFusionAdaptor().compile(graph, [torch.zeros(1)], {}, None)


def test_pre_grad_runs_on_private_graph_before_aot_and_fusion(monkeypatch) -> None:
    from vllm.compilation.passes.inductor_pass import get_pass_context
    from vllm.config.compilation import Range

    graph = torch.fx.symbolic_trace(lambda value: (value + 1,))
    original_code = graph.code
    events = []
    private_graphs = []

    def pre_grad(private_graph):
        assert get_pass_context().compile_range == Range(1, 1)
        assert private_graph is not graph.graph
        private_graphs.append(private_graph)
        events.append("pre_grad")
        for node in private_graph.nodes:
            if node.target is operator.add:
                node.target = operator.sub

    class Manager(RecordingManager):
        def __call__(self, normalized_graph):
            assert get_pass_context().compile_range == Range(1, 1)
            assert normalized_graph is not private_graphs[0]
            assert any(
                node.target == torch.ops.aten.sub.Tensor
                for node in normalized_graph.nodes
            )
            events.append("fusion")
            super().__call__(normalized_graph)

    compiled, handle = _compile(
        graph, [torch.zeros(3)], Manager(), monkeypatch, pre_grad=pre_grad
    )

    assert events == ["pre_grad", "fusion"]
    assert handle is None
    assert graph.code == original_code
    assert any(node.target is operator.add for node in graph.graph.nodes)
    torch.testing.assert_close(compiled(torch.zeros(3))[0], -torch.ones(3))


def test_pre_grad_error_propagates_before_aot_and_preserves_original(
    monkeypatch,
) -> None:
    import vllm_fl.compilation.fx_backend as backend

    graph = torch.fx.symbolic_trace(lambda value: (value + 1,))
    original_code = graph.code
    manager = RecordingManager()

    def pre_grad(private_graph):
        assert private_graph is not graph.graph
        raise RuntimeError("pre-grad failure")

    monkeypatch.setattr(
        backend,
        "aot_autograd",
        lambda **kwargs: pytest.fail("Pre-grad error must propagate before AOT"),
    )
    with pytest.raises(RuntimeError, match="pre-grad failure"):
        _compile(graph, [torch.zeros(1)], manager, monkeypatch, pre_grad=pre_grad)

    assert manager.graphs == []
    assert graph.code == original_code


def test_adapter_rewraps_examples_owned_by_another_fake_mode(monkeypatch) -> None:
    import torch._guards

    import vllm_fl.compilation.fx_backend as backend

    replacement = torch.ones(1)
    current_mode = SimpleNamespace(from_tensor=lambda value: replacement)
    monkeypatch.setattr(torch._guards, "detect_fake_mode", lambda: current_mode)
    monkeypatch.setattr(backend, "graph_returns_tuple", lambda graph: True)
    received = []

    def aot_autograd(**kwargs):
        def compile_graph(graph, inputs):
            received.append(inputs)
            return graph

        return compile_graph

    monkeypatch.setattr(backend, "aot_autograd", aot_autograd)
    original = torch.zeros(1)
    original.fake_mode = object()
    graph = torch.fx.symbolic_trace(lambda value: (value + 1,))

    _compile(graph, [original], RecordingManager(), monkeypatch)

    assert received[0][0] is replacement


def test_real_pattern_matcher_rewrites_normalized_graph_without_nge(
    monkeypatch,
) -> None:
    from torch._inductor.pattern_matcher import PatternMatcherPass

    from vllm_fl.compilation.passes.base_pattern import BasePattern

    monkeypatch.setitem(sys.modules, "npugraph_ex", None)

    class MulsAddPattern(BasePattern):
        # Use ordinary ATen replacement operations so this checks the real
        # registration/matcher path independently of Ascend kernel dispatch.
        def get_inputs(self):
            return [torch.ones(3), torch.zeros(3)]

        def get_pattern(self):
            return lambda x, y: x * 2.0 + y

        def get_replacement(self):
            return lambda x, y: torch.add(y, x, alpha=2.0)

    matcher = PatternMatcherPass(pass_name="test_thin_fx_muls_add")
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float32))
    MulsAddPattern(config).register(matcher)

    class MatcherManager(RecordingManager):
        def __init__(self):
            super().__init__(passes=[matcher])
            self.matches = 0

        def __call__(self, graph):
            super().__call__(graph)
            self.matches += matcher.apply(graph)

    manager = MatcherManager()
    graph = torch.fx.symbolic_trace(lambda x, y: (x * 2.0 + y,))
    compiled, handle = _compile(
        graph, [torch.ones(3), torch.zeros(3)], manager, monkeypatch
    )

    assert handle is None
    assert manager.matches == 1
    assert not any(
        node.target == torch.ops.aten.mul.Tensor for node in manager.graphs[0].nodes
    )
    torch.testing.assert_close(
        compiled(torch.ones(3), torch.ones(3))[0], torch.full((3,), 3.0)
    )


def test_vllm_make_compiler_resolves_thin_adapter(monkeypatch) -> None:
    import vllm.compilation.backends as backends

    from vllm_fl.compilation.fx_backend import FXFusionAdaptor

    monkeypatch.setattr(
        backends.current_platform,
        "get_compile_backend",
        lambda: "vllm_fl.compilation.fx_backend.FXFusionAdaptor",
    )
    monkeypatch.setattr(
        backends,
        "InductorAdaptor",
        lambda: pytest.fail("Unexpected native compilation"),
    )
    compiler = backends.make_compiler(SimpleNamespace(backend="fl_fx_fusion"))
    assert isinstance(compiler, FXFusionAdaptor)


def test_adapter_inherits_eager_cache_contract_without_device_compiler() -> None:
    from vllm.compilation.compiler_interface import EagerAdaptor

    import vllm_fl.compilation.fx_backend as backend

    assert issubclass(backend.FXFusionAdaptor, EagerAdaptor)
    assert backend.FXFusionAdaptor.initialize_cache is EagerAdaptor.initialize_cache
    assert backend.FXFusionAdaptor.load is EagerAdaptor.load
    source = inspect.getsource(backend)
    assert "import npugraph_ex" not in source
    assert "import vllm_ascend" not in source
    assert "get_npu_backend" not in source
