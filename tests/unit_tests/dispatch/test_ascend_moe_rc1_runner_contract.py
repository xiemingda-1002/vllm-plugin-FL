# Copyright (c) 2026 BAAI. All rights reserved.

import __future__
import ast
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
MOE = (
    ROOT
    / "vllm_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "ascend"
    / "impl"
    / "moe"
)


def _source(name: str) -> str:
    return (MOE / name).read_text(encoding="utf-8")


def _exec_source_definitions(source, names, scope):
    """Execute product functions, injecting only their runtime dependencies."""
    class RemoveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

    tree = ast.parse(source)
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in definitions} == set(names)
    module = ast.Module(body=definitions, type_ignores=[])
    module = ast.fix_missing_locations(RemoveImports().visit(module))
    # Annotation evaluation is disabled independently of the import transformer.
    exec(compile(module, "actual_ascend_moe_contract", "exec", flags=__future__.annotations.compiler_flag), scope)


def _registry_scope():
    context = (ROOT / "vllm_fl" / "ascend_forward_context.py").read_text(encoding="utf-8")
    scope = {"Enum": Enum, "_MoECommMethods": {}, "constructor_calls": []}
    _exec_source_definitions(context, ["MoECommType"], scope)
    for name in ("AllGatherCommImpl", "MC2CommImpl", "AlltoAllCommImpl", "FusedMC2CommImpl"):
        def constructor(config, name=name):
            scope["constructor_calls"].append((name, config))
            return SimpleNamespace(implementation=name, config=config)
        scope[name] = constructor
    _exec_source_definitions(
        _source("moe_comm_method.py"),
        ["setup_moe_comm_method", "get_moe_comm_method", "is_moe_comm_method_available"],
        scope,
    )
    return scope


@pytest.mark.parametrize("ep_size", [1, 2, 4, 16, 32, 64])
def test_actual_registry_setup_and_lookup(ep_size):
    scope = _registry_scope()
    config = SimpleNamespace(ep_size=ep_size)
    scope["setup_moe_comm_method"](config)
    comm_type = scope["MoECommType"]
    expected = {comm_type.ALLGATHER} if ep_size == 1 else set(comm_type)
    assert set(scope["_MoECommMethods"]) == expected
    assert len(scope["constructor_calls"]) == len(expected)
    assert all(config is called_config for _, called_config in scope["constructor_calls"])
    for method in comm_type:
        assert scope["is_moe_comm_method_available"](method) == (method in expected)
        if method in expected:
            assert scope["get_moe_comm_method"](method) is scope["_MoECommMethods"][method]
        else:
            with pytest.raises(NotImplementedError, match=method.name):
                scope["get_moe_comm_method"](method)
    assert scope["get_moe_comm_method"](None) is None
    assert not scope["is_moe_comm_method_available"](None)


def test_actual_registry_reinitialization_clears_stale_ep_transports():
    scope = _registry_scope()
    scope["setup_moe_comm_method"](SimpleNamespace(ep_size=16))
    scope["constructor_calls"].clear()
    config = SimpleNamespace(ep_size=1)
    scope["setup_moe_comm_method"](config)
    assert scope["constructor_calls"] == [("AllGatherCommImpl", config)]
    assert set(scope["_MoECommMethods"]) == {scope["MoECommType"].ALLGATHER}
    for name in ("MC2", "FUSED_MC2", "ALLTOALL"):
        method = scope["MoECommType"][name]
        assert not scope["is_moe_comm_method_available"](method)
        with pytest.raises(NotImplementedError, match=name):
            scope["get_moe_comm_method"](method)


@pytest.mark.parametrize("device_name", ["A2", "A3"])
@pytest.mark.parametrize("ep_size", [1, 2, 4, 16, 32, 64])
@pytest.mark.parametrize("enable_fused_mc2", [0, 1])
@pytest.mark.parametrize("num_tokens,capacity", [(1, 8), (8, 8), (9, 8), (1, None)])
def test_actual_selector_resolves_registered_transport(
    device_name, ep_size, enable_fused_mc2, num_tokens, capacity
):
    scope = _registry_scope()
    config = SimpleNamespace(
        ep_size=ep_size,
        additional_config={},
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True, world_size_across_dp=ep_size,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(get_num_experts=lambda: 128),
    )
    scope["setup_moe_comm_method"](config)
    device_type = Enum("AscendDeviceType", ["A2", "A3", "A5", "_310P"])
    scope.update(
        AscendDeviceType=device_type,
        _is_moe_model=lambda config: True,
        _active_ep_world_size=lambda: ep_size,
        _get_ascend_device_type=lambda: device_type[device_name],
        get_mc2_tokens_capacity=lambda: capacity,
        get_ascend_config=lambda: SimpleNamespace(enable_fused_mc2=enable_fused_mc2),
    )
    context = (ROOT / "vllm_fl" / "ascend_forward_context.py").read_text(encoding="utf-8")
    _exec_source_definitions(
        context,
        ["_ep_world_size", "_select_a2_moe_comm_method", "_select_a3_moe_comm_method",
         "_large_prefill_allgather_enabled", "select_moe_comm_method"],
        scope,
    )
    selected = scope["select_moe_comm_method"](num_tokens, config)
    comm_type = scope["MoECommType"]
    if ep_size == 1:
        expected = comm_type.ALLGATHER
    elif device_name == "A2":
        expected = (
            comm_type.MC2
            if ep_size >= 16 and capacity is not None and num_tokens <= capacity
            else comm_type.ALLGATHER
        )
    elif enable_fused_mc2 and ep_size <= 32 and capacity is not None:
        expected = comm_type.FUSED_MC2
    elif capacity is not None and num_tokens <= capacity:
        expected = comm_type.MC2
    else:
        expected = comm_type.ALLTOALL
    assert selected == expected
    assert scope["is_moe_comm_method_available"](selected)
    assert scope["get_moe_comm_method"](selected).config is config

    # The same real selector must bypass device transport selection for these boundaries.
    config.parallel_config.enable_expert_parallel = False
    assert scope["select_moe_comm_method"](num_tokens, config) == comm_type.ALLGATHER
    scope["_is_moe_model"] = lambda config: False
    assert scope["select_moe_comm_method"](num_tokens, config) is None
    assert scope["get_moe_comm_method"](None) is None


def test_runner_initializes_groups_before_registry_setup():
    tree = ast.parse(_source("fused_moe.py"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMoERunner")
    constructor = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    setup = next(
        node for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "setup_moe_comm_method"
    )
    assert ast.unparse(setup.args[0]) == "self.moe_config"
    for name in ("get_tp_group", "get_dp_group", "get_ep_group"):
        group_call = next(
            node for node in ast.walk(constructor)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        )
        assert group_call.lineno < setup.lineno


@pytest.mark.parametrize("quant_name", ["NONE", "W8A8", "W4A8"])
@pytest.mark.parametrize("device_name", ["A2", "A3"])
def test_fused_mc2_packaged_execution_gate(quant_name, device_name):
    # Execute the actual constructor guard without importing an NPU runtime.
    tree = ast.parse(_source("fused_moe.py"))
    guard = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "enable_fused_mc2"
    )
    body = [node for node in guard.body if not isinstance(node, ast.ImportFrom)]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))

    class QuantType(Enum):
        NONE = auto()
        W8A8 = auto()
        W4A8 = auto()

    class AscendDeviceType(Enum):
        A2 = auto()
        A3 = auto()

    scope = {
        "self": SimpleNamespace(quant_type=QuantType[quant_name]),
        "QuantType": QuantType,
        "AscendDeviceType": AscendDeviceType,
        "get_ascend_device_type": lambda: AscendDeviceType[device_name],
    }
    code = compile(module, "fused_mc2_constructor_guard", "exec")
    if device_name == "A3" and quant_name in ("NONE", "W8A8"):
        exec(code, scope)
    else:
        with pytest.raises(NotImplementedError):
            exec(code, scope)


def test_rc1_moe_package_has_no_vllm_ascend_runtime_dependency() -> None:
    for path in MOE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from vllm_ascend" not in source
        assert "import vllm_ascend" not in source


def test_runner_lifecycle_and_ordinary_comm_registry_are_complete() -> None:
    runner = _source("fused_moe.py")
    assert "moe_comm_method.prepare(" in runner
    assert "self._quant_method.apply(" in runner
    assert "moe_comm_method.finalize(" in runner
    assert "setup_moe_comm_method(self.moe_config)" in runner

    comm = _source("moe_comm_method.py")
    assert "TokenDispatcherWithAllGather" in comm
    assert "PrepareAndFinalizeWithAllGather" in comm
    assert (
        "_MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)"
        in comm
    )
    assert "_MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)" in comm
    assert (
        "_MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)"
        in comm
    )
    assert (
        "_MoECommMethods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)"
        in comm
    )

    forward_context = (ROOT / "vllm_fl" / "ascend_forward_context.py").read_text(
        encoding="utf-8"
    )
    assert "if device_type is AscendDeviceType.A2:" in forward_context
    assert "return _select_a2_moe_comm_method" in forward_context
    assert "if device_type is AscendDeviceType.A3:" in forward_context
    assert "selected = _select_a3_moe_comm_method" in forward_context
    assert "return selected" in forward_context


def test_rc1_weight_conversion_is_idempotent_and_loader_safe() -> None:
    runner = _source("fused_moe.py")
    assert 'getattr(layer, _WEIGHTS_PROCESSED_ATTR, False)' in runner
    assert 'replace_parameter(layer, "w13_weight", w13_data)' in runner
    assert 'replace_parameter(layer, "w2_weight", w2_data)' in runner
    assert "setattr(layer, _WEIGHTS_PROCESSED_ATTR, True)" in runner


def test_unmigrated_execution_modes_fail_closed() -> None:
    runner = _source("fused_moe.py")
    for feature in (
        "EPLB",
    ):
        assert feature in runner

    modelslim_moe = (
        ROOT
        / "vllm_fl"
        / "dispatch"
        / "backends"
        / "vendor"
        / "ascend"
        / "impl"
        / "quantization"
        / "moe.py"
    ).read_text(encoding="utf-8")
    assert 'quant_type.upper() == "W8A8_DYNAMIC"' in modelslim_moe
    assert "supports only ALLGATHER communication" not in modelslim_moe
    assert "scale_from_float_to_int64" in modelslim_moe

    moe_mlp = _source("moe_mlp.py")
    assert "elif HAS_TRITON:" in moe_mlp
    assert "from vllm_fl.dispatch.backends.vendor.ascend.impl.triton.activation.swiglu_quant import" in moe_mlp
    assert "quantized Triton SwiGLU is not migrated" not in moe_mlp

    device_operator = (
        ROOT
        / "vllm_fl"
        / "dispatch"
        / "backends"
        / "vendor"
        / "ascend"
        / "impl"
        / "device_operator.py"
    ).read_text(encoding="utf-8")
    # The current rc1 A2 implementation forwards W8A8 routing's quant_mode to
    # the native operator.  A local BF16-only guard would reject DeepSeek's
    # quantized path before the operator can produce its dynamic scales.
    assert "quant_mode=quant_mode" in device_operator
    assert "does not support quantized routing" not in device_operator
    assert "MXFP MoE quantization is only supported on Ascend A5" in device_operator


def test_shared_expert_dp_is_config_derived_and_propagated_to_prepare() -> None:
    runner = _source("fused_moe.py")
    assert "self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp" in runner
    assert "enable_shared_expert_dp=self.enable_shared_expert_dp" in runner

    config = (ROOT / "vllm_fl" / "configs" / "ascend.py").read_text(
        encoding="utf-8"
    )
    assert "shared_expert_dp_enabled_for_config(vllm_config)" in config
    assert "enable_sp(" in config
