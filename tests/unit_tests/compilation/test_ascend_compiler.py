# Copyright (c) 2026 BAAI. All rights reserved.

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _vllm_config(**options):
    return SimpleNamespace(
        additional_config={"ascend_compilation_config": options}
    )


def _install_fake_npu_stack(monkeypatch, events, *, options_api="new"):
    torch_npu = ModuleType("torch_npu")
    torch_npu.__version__ = "test-version"
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    npugraph_ex = ModuleType("npugraph_ex")

    class CompilerConfig:
        pass

    npugraph_ex.CompilerConfig = CompilerConfig

    def backend(graph, example_inputs):
        events.append(("backend", graph, example_inputs))
        return graph.forward

    def get_npu_backend(*, compiler_config):
        events.append(("compiler_config", compiler_config))
        return backend

    npugraph_ex.get_npu_backend = get_npu_backend
    configs = ModuleType("npugraph_ex.configs")
    configs.__path__ = []
    compiler_config_module = ModuleType("npugraph_ex.configs.compiler_config")

    def process_kwargs_options(config, kwargs):
        events.append(("options", config, kwargs))

    if options_api == "new":
        compiler_config_module._process_kwargs_options = process_kwargs_options
    legacy_config_module = ModuleType("npugraph_ex.configs.npugraphex_config")
    legacy_config_module._process_kwargs_options = process_kwargs_options
    monkeypatch.setitem(sys.modules, "npugraph_ex", npugraph_ex)
    monkeypatch.setitem(sys.modules, "npugraph_ex.configs", configs)
    monkeypatch.setitem(
        sys.modules, "npugraph_ex.configs.compiler_config", compiler_config_module
    )
    monkeypatch.setitem(
        sys.modules,
        "npugraph_ex.configs.npugraphex_config",
        legacy_config_module,
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_compile_mode=lambda **kwargs: events.append(("compile_mode", kwargs))
        ),
        raising=False,
    )


@pytest.mark.parametrize("options_api", ["new", "legacy"])
def test_compiler_uses_npugraph_options_without_inductor(
    monkeypatch, options_api
) -> None:
    import torch._inductor
    import vllm_fl.compilation.compiler_interface as compiler_module

    events = []
    _install_fake_npu_stack(monkeypatch, events, options_api=options_api)
    monkeypatch.setattr(compiler_module, "graph_returns_tuple", lambda graph: True)
    monkeypatch.setattr(
        torch._inductor,
        "standalone_compile",
        lambda *args, **kwargs: pytest.fail("Inductor must not be called"),
        raising=False,
    )

    graph = torch.fx.symbolic_trace(lambda value: (value + 1,))
    compiler = compiler_module.AscendCompiler()
    compiler.compute_hash(
        _vllm_config(enable_npugraph_ex=True, enable_static_kernel=False)
    )
    compiled, handle = compiler.compile(graph, [torch.zeros(1)], {}, None)

    assert handle is None
    assert torch.equal(compiled(torch.zeros(1))[0], torch.ones(1))
    assert events[0] == ("compile_mode", {"jit_compile": False})
    assert events[1][0] == "options"
    assert events[1][2] == {
        "options": {
            "force_eager": True,
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
        }
    }
    assert events[2][0] == "compiler_config"
    assert events[3][0] == "backend"


def test_compiler_wraps_non_tuple_graph(monkeypatch) -> None:
    import vllm_fl.compilation.compiler_interface as compiler_module

    events = []
    _install_fake_npu_stack(monkeypatch, events)
    monkeypatch.setattr(compiler_module, "graph_returns_tuple", lambda graph: False)

    def wrap(graph, example_inputs, backend):
        events.append(("tuple_wrapper", graph, example_inputs))
        return backend(graph, example_inputs)

    monkeypatch.setattr(compiler_module, "make_graph_return_tuple", wrap)
    graph = torch.fx.symbolic_trace(lambda value: value + 1)
    compiler = compiler_module.AscendCompiler()
    compiler.compute_hash(_vllm_config())
    compiled, _ = compiler.compile(graph, [torch.zeros(1)], {}, None)

    assert torch.equal(compiled(torch.zeros(1)), torch.ones(1))
    assert any(event[0] == "tuple_wrapper" for event in events)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"enable_static_kernel": True}, "does not yet support enable_static_kernel"),
    ],
)
def test_unsupported_options_fail_closed(monkeypatch, options, message) -> None:
    import vllm_fl.compilation.compiler_interface as compiler_module

    torch_npu = ModuleType("torch_npu")
    torch_npu.__version__ = "test-version"
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    compiler = compiler_module.AscendCompiler()
    compiler.compute_hash(_vllm_config(**options))
    graph = torch.fx.symbolic_trace(lambda value: value + 1)

    with pytest.raises(NotImplementedError, match=message):
        compiler.compile(graph, [torch.zeros(1)], {}, None)


def test_npugraph_disabled_uses_fusion_pass_compiler(monkeypatch) -> None:
    import vllm_fl.compilation.compiler_interface as compiler_module

    torch_npu = ModuleType("torch_npu")
    torch_npu.__version__ = "test-version"
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "npugraph_ex", None)

    events = []

    class FusionPassManager:
        def __call__(self, graph):
            events.append(("fusion_pass", graph))

    def compile_fx(graph, example_inputs, inner_compile, decompositions):
        events.append(("compile_fx", graph, example_inputs, decompositions))
        compiled_graph = inner_compile(graph, example_inputs)
        return compiled_graph.forward

    monkeypatch.setattr(compiler_module, "_compile_fx", compile_fx)
    graph = torch.fx.symbolic_trace(lambda value: value + 1)
    original_graph = graph
    compiler = compiler_module.AscendCompiler()
    compiler.compute_hash(_vllm_config(enable_npugraph_ex=False))

    compiled, handle = compiler.compile(
        graph,
        [torch.zeros(1)],
        {compiler_module.COMPILATION_PASS_KEY: FusionPassManager()},
        None,
    )

    assert handle is None
    assert torch.equal(compiled(torch.zeros(1)), torch.ones(1))
    assert events[0][0] == "compile_fx"
    assert events[0][1] is not original_graph
    assert events[1] == ("fusion_pass", events[0][1].graph)


def test_npugraph_disabled_aot_fusion_path_executes(monkeypatch) -> None:
    from vllm.compilation.passes.inductor_pass import pass_context
    from vllm.config.compilation import Range

    import vllm_fl.compilation.compiler_interface as compiler_module

    torch_npu = ModuleType("torch_npu")
    torch_npu.__version__ = "test-version"
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "npugraph_ex", None)

    events = []

    class FusionPassManager:
        def __call__(self, graph):
            events.append(graph)

    graph = torch.fx.symbolic_trace(lambda value: (value + 1,))
    compiler = compiler_module.AscendCompiler()
    compiler.compute_hash(_vllm_config(enable_npugraph_ex=False))
    with pass_context(Range(1, 1)):
        compiled, handle = compiler.compile(
            graph,
            [torch.zeros(1)],
            {compiler_module.COMPILATION_PASS_KEY: FusionPassManager()},
            Range(1, 1),
        )

    assert handle is None
    assert torch.equal(compiled(torch.zeros(1))[0], torch.ones(1))
    assert len(events) == 1


def test_npugraph_disabled_requires_fusion_manager(monkeypatch) -> None:
    import vllm_fl.compilation.compiler_interface as compiler_module

    torch_npu = ModuleType("torch_npu")
    torch_npu.__version__ = "test-version"
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    def compile_fx(graph, example_inputs, inner_compile, decompositions):
        del decompositions
        return inner_compile(graph, example_inputs)

    monkeypatch.setattr(compiler_module, "_compile_fx", compile_fx)
    graph = torch.fx.symbolic_trace(lambda value: (value + 1,))
    compiler = compiler_module.AscendCompiler()
    compiler.compute_hash(_vllm_config(enable_npugraph_ex=False))

    with pytest.raises(RuntimeError, match="graph_fusion_manager"):
        compiler.compile(graph, [torch.zeros(1)], {}, None)


def test_compiler_source_has_no_vllm_ascend_dependency() -> None:
    import vllm_fl.compilation.compiler_interface as compiler_module

    assert "vllm_ascend" not in inspect.getsource(compiler_module)
    assert not any(
        name == "vllm_ascend" or name.startswith("vllm_ascend.")
        for name in sys.modules
    )


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"unknown": True}, ValueError, "Unsupported FL Ascend compiler options"),
        (
            {"enable_npugraph_ex": 1},
            TypeError,
            "enable_npugraph_ex must be a bool",
        ),
    ],
)
def test_compiler_options_reject_unknown_and_non_bool(options, error, message) -> None:
    from vllm_fl.compilation.compiler_interface import AscendCompilerOptions

    with pytest.raises(error, match=message):
        AscendCompilerOptions.from_vllm_config(_vllm_config(**options))


def test_fusion_flags_are_strict_and_part_of_compiler_hash(monkeypatch) -> None:
    from vllm_fl.compilation.compiler_interface import (
        AscendCompiler,
        AscendCompilerOptions,
    )

    with pytest.raises(TypeError, match="fuse_qknorm_rope must be a bool"):
        AscendCompilerOptions.from_vllm_config(
            _vllm_config(fuse_qknorm_rope=1)
        )

    torch_npu = ModuleType("torch_npu")
    torch_npu.__version__ = "test-version"
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    enabled = AscendCompiler().compute_hash(
        _vllm_config(fuse_muls_add=True)
    )
    disabled = AscendCompiler().compute_hash(
        _vllm_config(fuse_muls_add=False)
    )
    assert enabled != disabled


def test_compiler_rewraps_tensor_from_a_different_fake_mode(monkeypatch) -> None:
    import torch._guards
    import vllm_fl.compilation.compiler_interface as compiler_module

    events = []
    _install_fake_npu_stack(monkeypatch, events)
    monkeypatch.setattr(compiler_module, "graph_returns_tuple", lambda graph: True)
    replacement = torch.ones(1)

    class CurrentFakeMode:
        def from_tensor(self, tensor):
            events.append(("rewrap", tensor))
            return replacement

    current_mode = CurrentFakeMode()
    monkeypatch.setattr(torch._guards, "detect_fake_mode", lambda: current_mode)
    original = torch.zeros(1)
    original.fake_mode = object()
    graph = torch.fx.symbolic_trace(lambda value: (value + 1,))
    compiler = compiler_module.AscendCompiler()
    compiler.compute_hash(_vllm_config())
    compiler.compile(graph, [original], {}, None)

    backend_event = next(event for event in events if event[0] == "backend")
    assert backend_event[2] == [replacement]
    assert any(event[0] == "rewrap" for event in events)


def test_vllm_make_compiler_constructs_fl_compiler_without_inductor(
    monkeypatch,
) -> None:
    import vllm.compilation.backends as backends
    from vllm_fl.compilation.compiler_interface import AscendCompiler

    qualname = "vllm_fl.compilation.compiler_interface.AscendCompiler"
    monkeypatch.setattr(
        backends.current_platform, "get_compile_backend", lambda: qualname
    )
    monkeypatch.setattr(
        backends,
        "InductorAdaptor",
        lambda: pytest.fail("InductorAdaptor must not be constructed"),
    )
    monkeypatch.setattr(
        backends,
        "InductorStandaloneAdaptor",
        lambda *args, **kwargs: pytest.fail(
            "InductorStandaloneAdaptor must not be constructed"
        ),
    )

    compiler = backends.make_compiler(SimpleNamespace(backend=qualname))
    assert isinstance(compiler, AscendCompiler)
