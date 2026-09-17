# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU contracts for the isolated rc1 Ascend ModelSlim linear closure."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[3]
MODULE = "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.linear"


def _package(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType(name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, name, package)


@pytest.fixture
def linear_module(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.ascend",
        "vllm_fl.dispatch.backends.vendor.ascend.impl",
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe",
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization",
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.utils",
    ):
        _package(name, monkeypatch)

    current = types.SimpleNamespace(
        additional_config={}, model_config=types.SimpleNamespace()
    )
    config = types.ModuleType("vllm.config")
    config.get_current_vllm_config = lambda: current
    monkeypatch.setitem(sys.modules, "vllm.config", config)

    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda _name: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger)

    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: 1
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)

    layers_linear = types.ModuleType("vllm.model_executor.layers.linear")

    class LinearMethodBase:
        pass

    class RowParallelLinear(torch.nn.Module):
        pass

    layers_linear.LinearMethodBase = LinearMethodBase
    layers_linear.RowParallelLinear = RowParallelLinear
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", layers_linear)

    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.PerTensorScaleParameter = lambda data, weight_loader: torch.nn.Parameter(
        data, requires_grad=False
    )
    monkeypatch.setitem(sys.modules, "vllm.model_executor.parameter", parameter)

    model_utils = types.ModuleType("vllm.model_executor.utils")

    def set_weight_attrs(param, attrs):
        for key, value in attrs.items():
            setattr(param, key, value)

    model_utils.set_weight_attrs = set_weight_attrs
    monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", model_utils)

    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    registrations = []
    torch_utils.direct_register_custom_op = lambda **kwargs: registrations.append(
        kwargs
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)

    hardware = types.ModuleType("vllm_fl.dispatch.backends.vendor.ascend.hardware")
    hardware.AscendDeviceType = types.SimpleNamespace(_310P=object())
    hardware.get_ascend_device_type = lambda: object()
    monkeypatch.setitem(sys.modules, hardware.__name__, hardware)

    dsa_compat = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.dsa_compat"
    )
    dsa_compat.enable_dsa_cp = lambda: bool(
        current.additional_config.get("enable_dsa_cp", False)
        and current.additional_config.get("enable_flashcomm1", False)
        and hasattr(
            getattr(current.model_config, "hf_text_config", None), "index_topk"
        )
    )
    monkeypatch.setitem(sys.modules, dsa_compat.__name__, dsa_compat)

    flashcomm = types.ModuleType("vllm_fl.ascend_flashcomm")
    flashcomm.enable_flashcomm1 = lambda *_args, **_kwargs: False
    flashcomm.shared_expert_dp_enabled_for_config = lambda *_args, **_kwargs: False
    monkeypatch.setitem(sys.modules, flashcomm.__name__, flashcomm)

    compat_spec = importlib.util.spec_from_file_location(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.compat",
        ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/moe/compat.py",
    )
    assert compat_spec is not None and compat_spec.loader is not None
    compat_module = importlib.util.module_from_spec(compat_spec)
    monkeypatch.setitem(sys.modules, compat_module.__name__, compat_module)
    compat_spec.loader.exec_module(compat_module)

    utils_spec = importlib.util.spec_from_file_location(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.linear_utils",
        ROOT
        / "vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/linear_utils.py",
    )
    assert utils_spec is not None and utils_spec.loader is not None
    utils_module = importlib.util.module_from_spec(utils_spec)
    monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
    utils_spec.loader.exec_module(utils_module)

    spec = importlib.util.spec_from_file_location(
        MODULE,
        ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/linear.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE, module)
    spec.loader.exec_module(module)
    return module, utils_module, current, RowParallelLinear, registrations


def test_static_descriptor_preserves_rc1_deq_scale_dtype(linear_module):
    module, _, _, _, registrations = linear_module
    scheme = module.create_linear_scheme("W8A8")
    module.create_linear_scheme("W8A8")

    assert registrations[0]["op_name"] == "quantize"
    assert registrations[0]["dispatch_key"] == "PrivateUse1"
    assert len(registrations) == 1
    assert scheme.get_weight(3, 5, torch.bfloat16)["weight"].dtype is torch.int8
    assert (
        scheme.get_perchannel_param(5, torch.bfloat16)["deq_scale"].dtype
        is torch.float32
    )
    assert (
        scheme.get_perchannel_param(5, torch.float16)["deq_scale"].dtype is torch.int64
    )


def test_adapter_uses_pertensor_scale_parameter_and_weight_loader(linear_module):
    module, _, _, _, _ = linear_module

    class Layer(torch.nn.Module):
        pass

    layer = Layer()
    loader = object()
    method = module.AscendLinearMethod(module.create_linear_scheme("W8A8"))
    method.create_weights(
        layer,
        input_size_per_partition=4,
        output_partition_sizes=[3, 2],
        input_size=4,
        output_size=5,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )

    assert layer.weight.shape == (5, 4)
    assert layer.input_scale.weight_loader is loader
    assert layer.input_scale.ignore_warning is True
    assert layer.quant_bias.shape == (5,)


def test_static_apply_uses_tp_rank_and_compressed_tensors_bias(
    linear_module, monkeypatch: pytest.MonkeyPatch
):
    module, _, _, _, _ = linear_module
    calls = {}

    class TorchNpu:
        @staticmethod
        def npu_quant_matmul(x, weight, deq_scale, *, bias, output_dtype):
            calls.update(
                x=x,
                weight=weight,
                deq_scale=deq_scale,
                bias=bias,
                output_dtype=output_dtype,
            )
            return torch.ones(1, dtype=output_dtype)

    monkeypatch.setattr(module, "_torch_npu", lambda: TorchNpu)
    layer = types.SimpleNamespace(
        quant_bias=torch.tensor([3], dtype=torch.int32),
        ascend_quant_method="compressed-tensors",
        weight=torch.tensor([[1]], dtype=torch.int8),
        deq_scale=torch.tensor([1.0]),
        params_dtype=torch.bfloat16,
    )
    bias = torch.tensor([7.0])
    result = module.AscendW8A8LinearMethod().apply(
        layer, torch.tensor([[1]], dtype=torch.int8), bias=bias, tp_rank=1
    )

    assert result.dtype is torch.bfloat16
    assert calls["bias"] is bias


def test_flashcomm2_and_unknown_scheme_fail_explicitly(linear_module):
    module, _, current, row_parallel, _ = linear_module

    class RowLayer(row_parallel):
        def __init__(self):
            super().__init__()
            self.prefix = "model.layers.0.self_attn.o_proj"

    current.additional_config = {"enable_flashcomm2_parallel_size": 2}
    method = module.AscendLinearMethod(module.create_linear_scheme("W8A8"))
    with pytest.raises(NotImplementedError, match="FlashComm2"):
        method.apply(RowLayer(), torch.empty(1, 1))
    with pytest.raises(NotImplementedError, match="W8A8_DYNAMIC"):
        module.create_linear_scheme("W4A8")


def test_flashcomm2_config_lifecycle_and_precedence(linear_module, monkeypatch):
    module, _, current, row_parallel, _ = linear_module
    compat = sys.modules[
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.compat"
    ]
    worker = types.SimpleNamespace(additional_config={})
    active = types.SimpleNamespace(
        additional_config={"enable_flashcomm2_parallel_size": 0}
    )
    state = {"config": active}
    monkeypatch.setattr(
        sys.modules["vllm.config"],
        "get_current_vllm_config",
        lambda: state["config"],
    )
    monkeypatch.setenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", "2")

    class RowLayer(row_parallel):
        prefix = "model.layers.0.self_attn.o_proj"

    try:
        compat.init_ascend_config(worker)
        # A live context is authoritative even when the worker/env request it.
        assert module._flashcomm2_requested() is False

        # Warmup after context exit uses the worker owner, then the environment.
        state["config"] = None
        monkeypatch.setattr(
            sys.modules["vllm.config"],
            "get_current_vllm_config",
            lambda: (_ for _ in ()).throw(AssertionError("no current config")),
        )
        assert module._flashcomm2_requested() is True
        with pytest.raises(NotImplementedError, match="FlashComm2"):
            module.AscendLinearMethod(module.create_linear_scheme("W8A8")).apply(
                RowLayer(), torch.empty(1, 1)
            )

        compat.clear_ascend_config()
        assert module._flashcomm2_requested() is True
    finally:
        compat.clear_ascend_config()


@pytest.mark.parametrize(
    "additional_config, environment, message",
    [
        ({"enable_flashcomm2_parallel_size": "2"}, None, "must be an integer"),
        ({}, "not-an-int", "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE must be an integer"),
    ],
)
def test_flashcomm2_rejects_malformed_values(
    linear_module, monkeypatch, additional_config, environment, message
):
    module, _, current, _, _ = linear_module
    current.additional_config = additional_config
    if environment is not None:
        monkeypatch.setenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", environment)
    with pytest.raises(ValueError, match=message):
        module._flashcomm2_requested()


def test_static_apply_uses_quant_bias_only_on_tp_rank_zero(
    linear_module, monkeypatch: pytest.MonkeyPatch
):
    module, _, _, _, _ = linear_module
    captured_biases = []

    class TorchNpu:
        @staticmethod
        def npu_quant_matmul(x, weight, deq_scale, *, bias, output_dtype):
            captured_biases.append(bias)
            return torch.empty(1, dtype=output_dtype)

    monkeypatch.setattr(module, "_torch_npu", lambda: TorchNpu)
    layer = types.SimpleNamespace(
        quant_bias=torch.tensor([2], dtype=torch.int32),
        weight=torch.tensor([[1]], dtype=torch.int8),
        deq_scale=torch.tensor([1.0]),
        params_dtype=torch.bfloat16,
    )
    scheme = module.AscendW8A8LinearMethod()
    x = torch.tensor([[1]], dtype=torch.int8)
    scheme.apply(layer, x, tp_rank=0)
    scheme.apply(layer, x, tp_rank=1)

    assert captured_biases == [layer.quant_bias, None]


@pytest.mark.parametrize(
    ("x", "pertoken_scale_shape", "expected_shape", "expected_scale_shape"),
    [
        (torch.empty(2, 3), (2,), (2, 4), (2,)),
        (torch.empty(2, 1, 3), (2, 1), (2, 1, 4), (2,)),
    ],
)
def test_dynamic_apply_preserves_rc1_singleton_scale_contract(
    linear_module,
    monkeypatch: pytest.MonkeyPatch,
    x: torch.Tensor,
    pertoken_scale_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    expected_scale_shape: tuple[int, ...],
):
    module, _, _, _, _ = linear_module
    calls = []

    class TorchNpu:
        @staticmethod
        def npu_dynamic_quant(input_x, *, dst_type):
            return torch.ones_like(input_x, dtype=torch.int8), torch.ones(
                pertoken_scale_shape, dtype=torch.float32
            )

        @staticmethod
        def npu_quant_matmul(
            quantized_x, weight, weight_scale, *, pertoken_scale, bias, output_dtype
        ):
            calls.append((quantized_x.shape, pertoken_scale.shape, bias, output_dtype))
            return torch.ones(quantized_x.shape[0], 4, dtype=output_dtype)

    monkeypatch.setattr(module, "_torch_npu", lambda: TorchNpu)
    layer = types.SimpleNamespace(
        weight=torch.empty(3, 4, dtype=torch.int8),
        weight_scale=torch.ones(4),
    )
    bias = torch.ones(4)
    output = module.AscendW8A8DynamicLinearMethod().apply(layer, x, bias=bias)

    assert output.shape == expected_shape
    assert calls == [((2, 3), expected_scale_shape, bias, x.dtype)]


def test_dynamic_postload_transposes_flattens_and_records_fp32_scale(
    linear_module, monkeypatch: pytest.MonkeyPatch
):
    module, _, _, _, _ = linear_module
    converted = []
    monkeypatch.setattr(
        module, "maybe_trans_nz", lambda tensor: converted.append(tensor) or tensor
    )
    layer = types.SimpleNamespace(
        prefix="model.layers.0.mlp.down_proj",
        weight=torch.nn.Parameter(
            torch.empty(4, 3, dtype=torch.int8), requires_grad=False
        ),
        weight_scale=torch.nn.Parameter(torch.ones(4, 1), requires_grad=False),
        weight_offset=torch.nn.Parameter(torch.zeros(4, 1), requires_grad=False),
    )

    module.AscendW8A8DynamicLinearMethod().process_weights_after_loading(layer)

    assert converted[0].shape == (3, 4)
    assert layer.weight.shape == (3, 4)
    assert layer.weight_scale.shape == (4,)
    assert layer.weight_offset.shape == (4,)
    assert layer.weight_scale_fp32.dtype is torch.float32


def test_dsa_cp_small_weight_uses_normal_dynamic_layout_when_sp_enabled(linear_module):
    module, _, current, _, _ = linear_module
    current.additional_config = {"enable_dsa_cp": True, "enable_flashcomm1": True}
    current.model_config = types.SimpleNamespace(
        hf_text_config=types.SimpleNamespace(index_topk=1)
    )
    layer = types.SimpleNamespace(
        prefix="model.layers.0.mlp.down_proj",
        weight=torch.nn.Parameter(
            torch.empty(4, 3, dtype=torch.int8), requires_grad=False
        ),
        weight_scale=torch.nn.Parameter(torch.ones(4, 1), requires_grad=False),
        weight_offset=torch.nn.Parameter(torch.zeros(4, 1), requires_grad=False),
    )
    # This CPU contract checks layout selection, not torch_npu format casting.
    module.maybe_trans_nz = lambda tensor: tensor

    module.AscendW8A8DynamicLinearMethod().process_weights_after_loading(layer)

    assert layer.weight.shape == (3, 4)
    assert layer.weight_scale.shape == (4,)
    assert layer.weight_offset.shape == (4,)


def test_dsa_cp_large_wq_b_uses_rc1_two_chunk_layout(linear_module, monkeypatch):
    module, _, current, _, _ = linear_module
    current.additional_config = {"enable_dsa_cp": True, "enable_flashcomm1": True}
    current.model_config = types.SimpleNamespace(
        hf_text_config=types.SimpleNamespace(index_topk=1)
    )
    monkeypatch.setattr(module, "maybe_trans_nz", lambda tensor: tensor)
    layer = types.SimpleNamespace(
        prefix="model.layers.0.self_attn.wq_b",
        weight=torch.nn.Parameter(
            torch.empty(65535 * 2, 2, dtype=torch.int8), requires_grad=False
        ),
        weight_scale=torch.nn.Parameter(
            torch.ones(65535 * 2, 1), requires_grad=False
        ),
        weight_offset=torch.nn.Parameter(
            torch.zeros(65535 * 2, 1), requires_grad=False
        ),
    )

    module.AscendW8A8DynamicLinearMethod().process_weights_after_loading(layer)

    assert layer._chunk_size == 65535
    assert layer.weight_1.shape == (2, 65535)
    assert layer.weight_2.shape == (2, 65535)
    assert layer.weight_1_scale.shape == (65535,)
    assert layer.weight_2_scale.shape == (65535,)
    assert not hasattr(layer, "weight")


def test_nz_policy_uses_config_before_env_and_skips_float_and_meta(
    linear_module, monkeypatch: pytest.MonkeyPatch
):
    _, utils_module, current, _, _ = linear_module
    format_calls = []
    torch_npu = types.SimpleNamespace(
        npu_format_cast=lambda tensor, fmt: format_calls.append((tensor, fmt)) or tensor
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_NZ", "1")
    int8_weight = torch.empty(2, 2, dtype=torch.int8)
    current.additional_config = {"weight_nz_mode": 0}
    assert not utils_module._should_trans_nz(int8_weight)
    assert utils_module.maybe_trans_nz(int8_weight) is int8_weight
    assert format_calls == []

    current.additional_config = {"weight_nz_mode": 1}
    assert utils_module._should_trans_nz(int8_weight)
    assert utils_module.maybe_trans_nz(int8_weight) is int8_weight
    assert format_calls == [(int8_weight, 29)]
    assert not utils_module._should_trans_nz(torch.empty(2, 2, dtype=torch.bfloat16))

    current.additional_config = {"weight_nz_mode": 2}
    assert utils_module.maybe_trans_nz(int8_weight) is int8_weight
    assert format_calls == [(int8_weight, 29), (int8_weight, 29)]
    assert utils_module._should_trans_nz(torch.empty(2, 2, dtype=torch.bfloat16))
    assert not utils_module._should_trans_nz(torch.empty(2, 2, dtype=torch.float32))
    assert not utils_module._should_trans_nz(
        torch.empty(2, 2, dtype=torch.int8, device="meta")
    )


def test_quantize_registration_rejects_foreign_operator(linear_module, monkeypatch):
    _, utils_module, _, _, _ = linear_module
    utils_module._QUANTIZE_REGISTERED = False
    monkeypatch.setattr(
        utils_module.torch,
        "ops",
        types.SimpleNamespace(vllm=types.SimpleNamespace(quantize=object())),
    )

    with pytest.raises(RuntimeError, match="already exists"):
        utils_module.register_quantize()
