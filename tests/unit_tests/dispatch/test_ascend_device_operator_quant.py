from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]
DEVICE_OPERATOR = (
    ROOT
    / "vllm_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "ascend"
    / "impl"
    / "device_operator.py"
)


def _load_device_operator():
    spec = importlib.util.spec_from_file_location(
        "fl_test_ascend_device_operator", DEVICE_OPERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceOperator


def test_a2_dynamic_quant_matches_current_rc1(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    quantized = object()
    generated_scale = object()

    def npu_dynamic_quant(hidden_states, *, dst_type):
        calls.append((hidden_states, dst_type))
        return quantized, generated_scale

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_dynamic_quant=npu_dynamic_quant),
    )
    device_operator = _load_device_operator()
    hidden_states = object()

    result = device_operator.npu_dynamic_quant(
        hidden_states, act_quant_type=torch.int8
    )
    assert result == (quantized, generated_scale)
    assert calls == [(hidden_states, torch.int8)]

    supplied_scale = object()
    result = device_operator.npu_dynamic_quant(hidden_states, supplied_scale)
    assert result == (hidden_states, supplied_scale)
    assert len(calls) == 1

    with pytest.raises(RuntimeError, match="only supported on Ascend A5"):
        device_operator.npu_dynamic_quant(
            hidden_states, use_mxfp_quant=True
        )


def test_a2_quant_gmm2_forwards_current_rc1_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    expected = object()

    def npu_grouped_matmul(**kwargs):
        calls.append(kwargs)
        return [expected]

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_grouped_matmul=npu_grouped_matmul),
    )
    device_operator = _load_device_operator()
    hidden_states = torch.empty(2, 4, dtype=torch.int8)
    weight = torch.empty(2, 4, 4, dtype=torch.int8)
    weight_scale = torch.empty(2, 4, dtype=torch.float32)
    per_token_scale = torch.empty(2, dtype=torch.float32)
    group_list = torch.tensor([1, 1], dtype=torch.int64)

    result = device_operator.npu_grouped_matmul_gmm2(
        hidden_states=hidden_states,
        weight=weight,
        weight_scale=weight_scale,
        per_token_scale=per_token_scale,
        group_list=group_list,
        group_list_type=1,
        input_dtype=torch.bfloat16,
        act_quant_type=torch.int8,
        weight_quant_type=torch.int8,
        scale_type=torch.float32,
        per_token_scale_type=torch.float32,
    )

    assert result is expected
    assert calls == [
        {
            "x": [hidden_states],
            "weight": weight,
            "scale": weight_scale,
            "bias": None,
            "per_token_scale": [per_token_scale],
            "split_item": 2,
            "group_list_type": 1,
            "group_type": 0,
            "group_list": group_list,
            "output_dtype": torch.float32,
        }
    ]

    with pytest.raises(RuntimeError, match="only supported on Ascend A5"):
        device_operator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=weight,
            weight_scale=weight_scale,
            per_token_scale=per_token_scale,
            group_list=group_list,
            group_list_type=1,
            input_dtype=torch.bfloat16,
            act_quant_type=torch.int8,
            weight_quant_type=torch.int8,
            scale_type=torch.float32,
            per_token_scale_type=torch.float32,
            use_mxfp_quant=True,
        )
