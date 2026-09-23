from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
NATIVE_ROOT = ROOT / "csrc/ascend/moe/moe_init_routing_custom"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_current_rc1_native_routing_build_and_dispatch_closure() -> None:
    payload = {path.relative_to(NATIVE_ROOT) for path in NATIVE_ROOT.rglob("*") if path.is_file()}
    assert len(payload) == 38
    assert Path("moe_init_routing_custom_torch_adpt.h") in payload
    assert Path("op_host/aclnn_moe_init_routing_custom.cpp") in payload
    assert Path("op_host/moe_init_routing_custom_infershape.cpp") in payload
    assert Path("op_kernel/moe_init_routing_custom.cpp") in payload

    binding = _text("csrc/ascend/torch_binding.cpp")
    assert binding.count('ops.impl("npu_moe_init_routing_custom"') == 2
    assert "TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)" in binding
    assert "int[2] active_expert_range=[]" in binding
    assert "at::empty_symint" in binding
    assert "moe_init_routing_custom" in _text("csrc/ascend/build_opp.sh")


def test_ascend_device_operator_exposes_rc1_moe_boundary_without_runtime_dependency() -> None:
    source = _text(
        "vllm_fl/dispatch/backends/vendor/ascend/impl/device_operator.py"
    )
    for method in (
        "npu_moe_init_routing",
        "npu_moe_token_unpermute",
        "maybe_normalize_mxfp_scale_layout",
        "npu_dynamic_quant",
        "npu_grouped_matmul_swiglu_quant",
        "npu_grouped_matmul_gmm2",
    ):
        assert f"def {method}(" in source
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert not any(name.startswith("vllm_ascend") for name in imported_modules)
    assert "npu_moe_init_routing_custom" in source


def test_ascend_method_matches_current_vllm_modular_contract() -> None:
    source = _text(
        "vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe_layer.py"
    )
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "apply"
    )
    assert [argument.arg for argument in method.args.args] == [
        "self",
        "layer",
        "x",
        "topk_weights",
        "topk_ids",
        "shared_experts",
        "shared_experts_input",
    ]
    assert "npu_moe_init_routing_custom" in source
    assert "npu_grouped_matmul" in source
    assert "npu_swiglu" in source
    assert "npu_moe_token_unpermute" in source
    assert "torch.abs(expanded_row_idx)" in source
    assert 'getattr(layer, "expert_map", None)' in source
    assert "first_expert_idx = self.moe.ep_rank * layer.local_num_experts" in source
    assert "topk_weights = topk_weights * valid.to(topk_weights.dtype)" in source
    assert "flat_experts[valid]" not in source
    assert "vllm_ascend" not in source


def test_ascend_weight_postprocess_is_idempotent_and_preserves_loader() -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe_layer import (
        AscendUnquantizedFusedMoEMethod,
    )

    method = object.__new__(AscendUnquantizedFusedMoEMethod)
    object.__setattr__(method, "_maybe_pad_weight", lambda weight: weight)

    routed_experts = torch.nn.Module()
    w13 = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    w2 = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    routed_experts.register_parameter(
        "w13_weight", torch.nn.Parameter(w13.clone(), requires_grad=False)
    )
    routed_experts.register_parameter(
        "w2_weight", torch.nn.Parameter(w2.clone(), requires_grad=False)
    )
    weight_loader = object()
    routed_experts.w13_weight.weight_loader = weight_loader
    routed_experts.w2_weight.weight_loader = weight_loader

    method.process_weights_after_loading(routed_experts)
    assert torch.equal(routed_experts.w13_weight, w13.transpose(1, 2))
    assert torch.equal(routed_experts.w2_weight, w2.transpose(1, 2))
    assert routed_experts.w13_weight.weight_loader is weight_loader
    assert routed_experts.w2_weight.weight_loader is weight_loader
    w13_parameter = routed_experts.w13_weight
    w2_parameter = routed_experts.w2_weight
    w13_data_ptr = routed_experts.w13_weight.data_ptr()
    w2_data_ptr = routed_experts.w2_weight.data_ptr()

    method.process_weights_after_loading(routed_experts)
    assert routed_experts.w13_weight is w13_parameter
    assert routed_experts.w2_weight is w2_parameter
    assert routed_experts.w13_weight.data_ptr() == w13_data_ptr
    assert routed_experts.w2_weight.data_ptr() == w2_data_ptr


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_eplb": True},
        {"pcp_size": 2},
        {"has_bias": True},
        {"is_lora_enabled": True},
    ],
)
def test_unsupported_parallel_and_feature_combinations_fail_closed(overrides) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe_layer import (
        _validate_supported_config,
    )

    values = {
        "use_ep": False,
        "ep_size": 1,
        "dp_size": 1,
        "pcp_size": 1,
        "is_sequence_parallel": False,
        "has_bias": False,
        "is_lora_enabled": False,
        "enable_eplb": False,
    }
    values.update(overrides)
    values["moe_parallel_config"] = SimpleNamespace(
        enable_eplb=values.pop("enable_eplb")
    )
    with pytest.raises(NotImplementedError):
        _validate_supported_config(SimpleNamespace(**values))


@pytest.mark.parametrize(
    ("use_ep", "ep_size", "dp_size", "is_sequence_parallel"),
    [
        (False, 1, 1, False),
        (True, 2, 1, False),
        (True, 4, 2, False),
        (True, 4, 2, True),
    ],
)
def test_supported_tp_and_dp_ep_parallel_configs(
    use_ep: bool,
    ep_size: int,
    dp_size: int,
    is_sequence_parallel: bool,
) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe_layer import (
        _validate_supported_config,
    )

    _validate_supported_config(
        SimpleNamespace(
            use_ep=use_ep,
            ep_size=ep_size,
            dp_size=dp_size,
            pcp_size=1,
            is_sequence_parallel=is_sequence_parallel,
            has_bias=False,
            is_lora_enabled=False,
            moe_parallel_config=SimpleNamespace(enable_eplb=False),
        )
    )


@pytest.mark.parametrize(
    ("use_ep", "ep_size", "dp_size"),
    [
        (False, 2, 1),
        (False, 1, 2),
    ],
)
def test_inconsistent_parallel_configs_fail_closed(
    use_ep: bool, ep_size: int, dp_size: int
) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe_layer import (
        _validate_supported_config,
    )

    with pytest.raises(NotImplementedError):
        _validate_supported_config(
            SimpleNamespace(
                use_ep=use_ep,
                ep_size=ep_size,
                dp_size=dp_size,
                pcp_size=1,
                is_sequence_parallel=False,
                has_bias=False,
                is_lora_enabled=False,
                moe_parallel_config=SimpleNamespace(enable_eplb=False),
            )
        )


def test_generic_registration_tail_does_not_overwrite_ascend_factory() -> None:
    source = _text("vllm_fl/ops/custom_ops.py")
    assert 'if not is_ascend and "fused_moe" not in (blacklist or []):' in source
