# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.library import infer_schema

from vllm_fl.ascend_forward_context import MoECommType


def test_moe_reduce_custom_op_registration_is_idempotent(monkeypatch) -> None:
    ops = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe_custom_ops"
    )
    register = Mock()
    monkeypatch.setattr(ops, "direct_register_custom_op", register)
    monkeypatch.setattr(ops, "_REGISTERED", False)
    monkeypatch.setattr(
        ops,
        "torch",
        SimpleNamespace(
            Tensor=torch.Tensor,
            ops=SimpleNamespace(vllm=SimpleNamespace()),
        ),
    )

    ops.ensure_ascend_moe_custom_ops_registered()
    ops.ensure_ascend_moe_custom_ops_registered()

    register.assert_called_once_with(
        op_name="maybe_all_reduce_tensor_model_parallel",
        op_func=ops._maybe_all_reduce_tensor_model_parallel_impl,
        fake_impl=ops._maybe_all_reduce_tensor_model_parallel_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )
    assert infer_schema(
        ops._maybe_all_reduce_tensor_model_parallel_impl, mutates_args=[]
    ) == "(Tensor final_hidden_states) -> Tensor"


@pytest.mark.parametrize(
    "comm_type",
    [MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2],
)
def test_moe_reduce_is_identity_when_comm_already_reduced(
    monkeypatch, comm_type
) -> None:
    ops = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe_custom_ops"
    )
    all_reduce = Mock()
    monkeypatch.setattr(ops, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        ops,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=comm_type,
            flash_comm_v1_enabled=False,
        ),
    )
    value = object()

    assert ops._maybe_all_reduce_tensor_model_parallel_impl(value) is value
    all_reduce.assert_not_called()


def test_moe_reduce_is_identity_for_flash_comm_v1(monkeypatch) -> None:
    ops = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe_custom_ops"
    )
    all_reduce = Mock()
    monkeypatch.setattr(ops, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        ops,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=MoECommType.ALLGATHER,
            flash_comm_v1_enabled=True,
        ),
    )
    value = object()

    assert ops._maybe_all_reduce_tensor_model_parallel_impl(value) is value
    all_reduce.assert_not_called()


def test_moe_reduce_calls_tp_all_reduce_for_allgather(monkeypatch) -> None:
    ops = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe_custom_ops"
    )
    reduced = object()
    all_reduce = Mock(return_value=reduced)
    monkeypatch.setattr(ops, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        ops,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=MoECommType.ALLGATHER,
            flash_comm_v1_enabled=False,
        ),
    )
    value = object()

    assert ops._maybe_all_reduce_tensor_model_parallel_impl(value) is reduced
    all_reduce.assert_called_once_with(value)


def test_moe_reduce_fake_is_identity() -> None:
    ops = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe_custom_ops"
    )
    value = object()
    assert ops._maybe_all_reduce_tensor_model_parallel_fake(value) is value


def test_registration_is_ascend_patch_scoped() -> None:
    patch_source = (
        Path(__file__).parents[3]
        / "vllm_fl"
        / "dispatch"
        / "backends"
        / "vendor"
        / "ascend"
        / "patch.py"
    ).read_text(encoding="utf-8")
    function_start = patch_source.index("def apply_ascend_patches")
    next_function = patch_source.index("\ndef ", function_start + 1)
    apply_source = patch_source[function_start:next_function]

    assert "ensure_ascend_moe_custom_ops_registered" in apply_source
    assert "moe_custom_ops" not in patch_source[:function_start]

    # Importing the generic custom-op module must not pull in this vendor op.
    vendor_module = (
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe_custom_ops"
    )
    sys.modules.pop(vendor_module, None)
    importlib.reload(importlib.import_module("vllm_fl.ops.custom_ops"))
    assert vendor_module not in sys.modules


def test_registration_signature_matches_current_api() -> None:
    ops = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.moe_custom_ops"
    )
    signature = inspect.signature(
        ops._maybe_all_reduce_tensor_model_parallel_impl
    )
    assert list(signature.parameters) == ["final_hidden_states"]
    assert signature.return_annotation == "torch.Tensor"
