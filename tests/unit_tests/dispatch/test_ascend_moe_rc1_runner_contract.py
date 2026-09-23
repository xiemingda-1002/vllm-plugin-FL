# Copyright (c) 2026 BAAI. All rights reserved.

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
    assert "return _select_a3_moe_comm_method" in forward_context


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
