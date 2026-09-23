"""CPU-only contracts for the rc1 ModelSlim W8A8_DYNAMIC MoE closure."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[3]
MODULE = "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.moe"


def _package(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType(name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, name, package)


@pytest.fixture
def moe_module(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.ascend",
        "vllm_fl.dispatch.backends.vendor.ascend.impl",
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization",
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe",
        "vllm_fl.configs",
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
    ):
        _package(name, monkeypatch)

    current = types.SimpleNamespace(
        model_config=types.SimpleNamespace(dtype=torch.bfloat16), additional_config={}
    )
    config = types.ModuleType("vllm.config")
    config.get_current_vllm_config = lambda: current
    monkeypatch.setitem(sys.modules, "vllm.config", config)

    fused = types.ModuleType("vllm.model_executor.layers.fused_moe")

    class FusedMoEMethodBase:
        def __init__(self, moe_config):
            self.moe_config = moe_config

    fused.FusedMoEMethodBase = FusedMoEMethodBase
    fused.FusedMoeWeightScaleSupported = types.SimpleNamespace(
        CHANNEL=types.SimpleNamespace(value="channel"),
        GROUP=types.SimpleNamespace(value="group"),
    )
    monkeypatch.setitem(sys.modules, fused.__name__, fused)
    fused_config = types.ModuleType("vllm.model_executor.layers.fused_moe.config")
    fused_config.FusedMoEConfig = object
    monkeypatch.setitem(sys.modules, fused_config.__name__, fused_config)
    model_utils = types.ModuleType("vllm.model_executor.utils")
    model_utils.set_weight_attrs = lambda param, attrs: [
        setattr(param, k, v) for k, v in attrs.items()
    ]
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)

    forward = types.ModuleType("vllm_fl.ascend_forward_context")
    forward.MoECommType = types.SimpleNamespace(
        ALLGATHER="allgather", MC2="mc2", ALLTOALL="alltoall", FUSED_MC2="fused_mc2"
    )
    forward._EXTRA_CTX = types.SimpleNamespace(
        moe_comm_type="allgather", moe_comm_method=None
    )
    monkeypatch.setitem(sys.modules, forward.__name__, forward)
    ascend_config = types.ModuleType("vllm_fl.configs.ascend")
    ascend_config.get_ascend_config = lambda: types.SimpleNamespace(
        eplb_config=types.SimpleNamespace(dynamic_eplb=False), enable_fused_mc2=0
    )
    monkeypatch.setitem(sys.modules, ascend_config.__name__, ascend_config)
    selector = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.experts_selector"
    )
    selector.select_experts = lambda **kwargs: (
        torch.tensor([[0.25]], dtype=torch.float32),
        torch.tensor([[1]], dtype=torch.int32),
    )
    selector.zero_experts_compute = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("unexpected")
    )
    monkeypatch.setitem(sys.modules, selector.__name__, selector)
    runtime = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.moe_runtime_args"
    )
    runtime.build_fused_experts_input = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    quant_type = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.quant_type"
    )
    quant_type.QuantType = types.SimpleNamespace(NONE="none", W8A8="w8a8")
    monkeypatch.setitem(sys.modules, quant_type.__name__, quant_type)

    spec = importlib.util.spec_from_file_location(
        MODULE,
        ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/moe.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE, module)
    spec.loader.exec_module(module)
    return module, current, forward


def test_dynamic_moe_descriptor_and_channel_metadata(moe_module):
    module, _, _ = moe_module
    scheme = module.AscendW8A8DynamicFusedMoEMethod()
    method = module.AscendFusedMoEMethod(scheme, object(), tid2eid="ids")
    layer = torch.nn.Module()
    method.create_weights(layer, 3, 4, 2, torch.bfloat16, weight_loader="loader")

    assert layer.w13_weight.shape == (3, 4, 4)
    assert layer.w2_weight.shape == (3, 4, 2)
    assert layer.w13_weight_scale.shape == (3, 4, 1)
    assert layer.w2_weight_scale.shape == (3, 4, 1)
    assert layer.w13_weight_scale.quant_method == "channel"
    assert method.is_monolithic is False
    assert method.maybe_make_prepare_finalize() is None
    assert method.tid2eid == "ids"


def test_adapter_forwards_complete_runtime_contract(moe_module):
    module, _, _ = moe_module
    captured = {}

    class Scheme:
        def apply(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                routed_out=torch.ones(1), before_dispatch_evt="event"
            )

    result = module.AscendFusedMoEMethod(Scheme(), object(), tid2eid="tid-map").apply(
        object(),
        torch.ones(1, 2),
        torch.ones(1, 3),
        2,
        True,
        expert_map=torch.tensor([0]),
        topk_group=1,
        num_expert_group=2,
        scoring_func="sigmoid",
        routed_scaling_factor=0.5,
        e_score_correction_bias=torch.ones(3),
        is_prefill=False,
        enable_force_load_balance=True,
        log2phy=torch.tensor([0]),
        global_redundant_expert_num=1,
        pertoken_scale=torch.ones(1),
        activation="swiglu",
        apply_router_weight_on_input=True,
        mc2_mask=torch.tensor([True]),
    )
    assert result.before_dispatch_evt == "event"
    assert captured["tid2eid"] == "tid-map"
    assert captured["is_prefill"] is False
    assert captured["enable_force_load_balance"] is True
    assert captured["apply_router_weight_on_input"] is True
    assert captured["global_redundant_expert_num"] == 1


def test_dynamic_moe_postload_transposes_nz_and_retains_scales(moe_module, monkeypatch):
    module, _, _ = moe_module
    casts = []
    monkeypatch.setattr(
        module,
        "_torch_npu",
        lambda: types.SimpleNamespace(
            npu_format_cast=lambda t, fmt: casts.append((t.shape, fmt)) or t
        ),
    )
    layer = types.SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.empty(2, 6, 4, dtype=torch.int8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.empty(2, 4, 3, dtype=torch.int8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(torch.ones(2, 6, 1), requires_grad=False),
        w13_weight_offset=torch.nn.Parameter(torch.zeros(2, 6, 1), requires_grad=False),
        w2_weight_scale=torch.nn.Parameter(torch.ones(2, 4, 1), requires_grad=False),
        w2_weight_offset=torch.nn.Parameter(torch.zeros(2, 4, 1), requires_grad=False),
    )
    module.AscendW8A8DynamicFusedMoEMethod().process_weights_after_loading(layer)
    assert casts == [((2, 4, 6), 29), ((2, 3, 4), 29)]
    assert layer.w13_weight_scale.shape == (2, 6)
    assert layer.w2_weight_scale.shape == (2, 4)
    assert layer.w13_weight_scale_fp32.dtype is torch.float32


def test_dynamic_moe_fused_mc2_preserves_fp32_scale_bits(moe_module) -> None:
    module, _, _ = moe_module
    scale = torch.tensor([1.0, -0.5, 0.0], dtype=torch.float32)

    packed = module.scale_from_float_to_int64(scale)

    assert packed.dtype is torch.int64
    assert packed.tolist() == [0x3F800000, -1090519040, 0]


def test_dynamic_moe_apply_preserves_runner_result_and_runtime_arguments(
    moe_module, monkeypatch
):
    module, _, forward = moe_module
    captured = {}

    def select(**kwargs):
        captured["routing"] = kwargs
        return torch.tensor([[0.25]]), torch.tensor([[1]], dtype=torch.int32)

    module.select_experts = select
    output = types.SimpleNamespace(
        routed_out=torch.ones(1, 2), before_dispatch_evt="dispatch"
    )

    def fused_experts(fused_experts_input):
        captured["input"] = fused_experts_input
        return output

    forward._EXTRA_CTX.moe_comm_method = types.SimpleNamespace(
        fused_experts=fused_experts
    )
    layer = types.SimpleNamespace(
        moe_config=types.SimpleNamespace(num_logical_experts=2),
        n_shared_experts=0,
        w13_weight=torch.empty(1),
        w2_weight=torch.empty(1),
        w13_weight_scale_fp32=torch.ones(1),
        w2_weight_scale=torch.ones(1),
        swiglu_limit=0.0,
    )
    result = module.AscendW8A8DynamicFusedMoEMethod().apply(
        layer,
        torch.ones(1, 2),
        torch.ones(1, 2),
        1,
        True,
        num_experts=99,
        expert_map=torch.tensor([0]),
        log2phy=torch.tensor([0]),
        pertoken_scale=torch.ones(1),
        activation="silu",
        apply_router_weight_on_input=True,
        mc2_mask=torch.tensor([True]),
        tid2eid=torch.tensor([1]),
    )
    assert result is output
    assert result.before_dispatch_evt == "dispatch"
    assert captured["routing"]["num_experts"] == 2
    assert captured["routing"]["tid2eid"].item() == 1
    assert captured["input"]["pertoken_scale"].shape == (1,)
    assert captured["input"]["w1_scale"] == [layer.w13_weight_scale_fp32]


@pytest.mark.parametrize("comm_type", ["mc2", "alltoall"])
def test_dynamic_moe_accepts_ordinary_rc1_communication_methods(moe_module, comm_type):
    module, _, forward = moe_module
    received = {}
    forward._EXTRA_CTX.moe_comm_type = comm_type

    def fused_experts(fused_experts_input):
        received["input"] = fused_experts_input
        return types.SimpleNamespace(routed_out=torch.ones(1, 2))

    forward._EXTRA_CTX.moe_comm_method = types.SimpleNamespace(
        fused_experts=fused_experts
    )
    layer = types.SimpleNamespace(
        moe_config=types.SimpleNamespace(num_logical_experts=2),
        n_shared_experts=0,
        w13_weight=torch.empty(1),
        w2_weight=torch.empty(1),
        w13_weight_scale_fp32=torch.ones(1),
        w2_weight_scale=torch.ones(1),
        swiglu_limit=10.0,
    )
    result = module.AscendW8A8DynamicFusedMoEMethod().apply(
        layer, torch.ones(1, 2), torch.ones(1, 2), 1, True, num_experts=2,
        mc2_mask=torch.tensor([True]),
    )
    assert result.routed_out.shape == (1, 2)
    assert received["input"]["swiglu_limit"] == 10.0


def test_dynamic_moe_constructor_rejects_dynamic_eplb(moe_module, monkeypatch):
    module, _, _ = moe_module
    config = types.SimpleNamespace(
        eplb_config=types.SimpleNamespace(dynamic_eplb=True), enable_fused_mc2=0
    )
    monkeypatch.setattr(module, "get_ascend_config", lambda: config)
    with pytest.raises(NotImplementedError, match="EPLB"):
        module.AscendW8A8DynamicFusedMoEMethod()


def test_dynamic_moe_constructor_accepts_fused_mc2_w8a8(moe_module, monkeypatch):
    module, _, _ = moe_module
    config = types.SimpleNamespace(
        eplb_config=types.SimpleNamespace(dynamic_eplb=False), enable_fused_mc2=1
    )
    monkeypatch.setattr(module, "get_ascend_config", lambda: config)

    assert module.AscendW8A8DynamicFusedMoEMethod().quant_type == "w8a8"
