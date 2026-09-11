# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.library import infer_schema


def _stub_module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture(autouse=True, scope="module")
def _python_only_flag_gems_stub():
    if "flag_gems" in sys.modules:
        yield
        return
    names = [
        "flag_gems",
        "flag_gems.runtime",
        "flag_gems.runtime.backend",
        "flag_gems.runtime.backend.device",
    ]
    saved = {name: sys.modules.get(name) for name in names}
    backend = _stub_module("flag_gems.runtime.backend")
    runtime = _stub_module("flag_gems.runtime", backend=backend)
    sys.modules["flag_gems"] = _stub_module("flag_gems")
    sys.modules["flag_gems.runtime"] = runtime
    sys.modules["flag_gems.runtime.backend"] = backend
    sys.modules["flag_gems.runtime.backend.device"] = _stub_module(
        "flag_gems.runtime.backend.device", DeviceDetector=object
    )
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_generic_layernorm_does_not_import_ascend_vendor() -> None:
    vendor_module = "vllm_fl.dispatch.backends.vendor.ascend.impl.layernorm"
    sys.modules.pop(vendor_module, None)

    generic = importlib.reload(importlib.import_module("vllm_fl.ops.layernorm"))
    custom_ops = importlib.reload(importlib.import_module("vllm_fl.ops.custom_ops"))

    assert generic.__all__ == ["RMSNormFL"]
    assert not hasattr(generic, "GemmaRMSNormFL")
    assert not hasattr(generic, "RMSNormGatedFL")
    assert "gemma_rms_norm" not in custom_ops.OOT_OPS
    assert "rms_norm_gated" not in custom_ops.OOT_OPS
    assert vendor_module not in sys.modules


def test_gated_rmsnorm_custom_op_registration_contract(monkeypatch) -> None:
    layernorm = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.layernorm"
    )
    register = Mock()
    monkeypatch.setattr(layernorm, "direct_register_custom_op", register)
    monkeypatch.setattr(layernorm, "_REGISTERED", False)
    monkeypatch.setattr(
        layernorm,
        "torch",
        SimpleNamespace(
            Tensor=torch.Tensor,
            ops=SimpleNamespace(vllm=SimpleNamespace()),
        ),
    )

    layernorm.ensure_ascend_rms_norm_gated_registered()
    layernorm.ensure_ascend_rms_norm_gated_registered()

    register.assert_called_once_with(
        op_name="ascend_rms_norm_gated",
        op_func=layernorm._ascend_rms_norm_gated_impl,
        fake_impl=layernorm._ascend_rms_norm_gated_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )
    signature = inspect.signature(layernorm._ascend_rms_norm_gated_impl)
    assert list(signature.parameters) == [
        "x",
        "z",
        "weight",
        "eps",
        "group_size",
        "norm_before_gate",
    ]
    assert signature.return_annotation == "torch.Tensor"
    assert infer_schema(
        layernorm._ascend_rms_norm_gated_impl, mutates_args=[]
    ) == (
        "(Tensor x, Tensor z, Tensor weight, float eps, SymInt group_size, "
        "bool norm_before_gate) -> Tensor"
    )


def test_gated_rmsnorm_rejects_foreign_operator_collision(monkeypatch) -> None:
    layernorm = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.layernorm"
    )
    register = Mock()
    monkeypatch.setattr(layernorm, "direct_register_custom_op", register)
    monkeypatch.setattr(layernorm, "_REGISTERED", False)
    monkeypatch.setattr(
        layernorm,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(
                vllm=SimpleNamespace(ascend_rms_norm_gated=object())
            )
        ),
    )

    try:
        layernorm.ensure_ascend_rms_norm_gated_registered()
    except RuntimeError as error:
        assert "already exists before FL Ascend registration" in str(error)
    else:
        raise AssertionError("foreign operator collision was silently accepted")
    register.assert_not_called()


def test_patch_op_cls_registers_layernorm_only_in_ascend_vendor(
    monkeypatch,
) -> None:
    patch = importlib.import_module("vllm_fl.dispatch.backends.vendor.ascend.patch")
    monkeypatch.setattr(patch, "_op_classes_patched", False)
    custom_op = importlib.import_module("vllm.model_executor.custom_op")
    register_custom = Mock()
    register_pluggable = Mock()
    monkeypatch.setattr(custom_op.CustomOp, "register_oot", register_custom)
    monkeypatch.setattr(
        custom_op.PluggableLayer, "register_oot", register_pluggable
    )

    class AscendGemmaRMSNorm:
        pass

    class AscendRMSNorm:
        pass

    class AscendRMSNormGated:
        pass

    ensure_registered = Mock()
    module_prefix = "vllm_fl.dispatch.backends.vendor.ascend.impl"
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.gdn",
        _stub_module(
            f"{module_prefix}.gdn", AscendGatedDeltaNetAttention=type("GDN", (), {})
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.layernorm",
        _stub_module(
            f"{module_prefix}.layernorm",
            AscendRMSNorm=AscendRMSNorm,
            AscendGemmaRMSNorm=AscendGemmaRMSNorm,
            AscendRMSNormGated=AscendRMSNormGated,
            ensure_ascend_rms_norm_gated_registered=ensure_registered,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.mm_encoder_attention",
        _stub_module(
            f"{module_prefix}.mm_encoder_attention",
            AscendMMEncoderAttention=type("MM", (), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{module_prefix}.vocab_parallel_embedding",
        _stub_module(
            f"{module_prefix}.vocab_parallel_embedding",
            AscendParallelLMHead=type("Head", (), {}),
            AscendVocabParallelEmbedding=type("Embedding", (), {}),
        ),
    )

    patch.patch_op_cls()
    patch.patch_op_cls()

    ensure_registered.assert_called_once_with()
    registrations = {
        call.kwargs["name"]: call.kwargs["_decorated_op_cls"]
        for call in register_custom.call_args_list
    }
    assert registrations["GemmaRMSNorm"] is AscendGemmaRMSNorm
    assert registrations["RMSNorm"] is AscendRMSNorm
    assert registrations["RMSNormGated"] is AscendRMSNormGated
    assert register_custom.call_count == 5
    assert register_pluggable.call_count == 2


def test_gated_rmsnorm_real_and_fake_contract(monkeypatch) -> None:
    layernorm = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.layernorm"
    )
    calls = {}

    def fake_kernel(x, weight, bias, eps, **kwargs):
        calls.update(
            x=x,
            weight=weight,
            bias=bias,
            eps=eps,
            kwargs=kwargs,
        )
        return x + 1, None, None

    monkeypatch.setattr(layernorm, "layer_norm_fwd_npu", fake_kernel)
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    z = torch.randn_like(x)
    weight = torch.randn(4)

    output = layernorm._ascend_rms_norm_gated_impl(
        x, z, weight, 1e-5, -1, True
    )
    fake = layernorm._ascend_rms_norm_gated_fake(
        x, z, weight, 1e-5, -1, True
    )

    assert output.shape == x.shape
    assert calls["x"].shape == (6, 4)
    assert calls["kwargs"]["z"].shape == (6, 4)
    assert calls["kwargs"]["group_size"] is None
    assert calls["kwargs"]["norm_before_gate"] is True
    assert calls["kwargs"]["is_rms_norm"] is True
    assert fake.shape == x.shape
    assert fake.dtype == x.dtype
    assert fake.device == x.device


def test_gated_rmsnorm_rejects_same_numel_different_gate_shape() -> None:
    layernorm = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.layernorm"
    )
    x = torch.randn(2, 3, 4)
    z = torch.randn(3, 2, 4)

    with pytest.raises(ValueError, match="gate shape"):
        layernorm._ascend_rms_norm_gated_impl(
            x, z, torch.ones(4), 1e-5, -1, True
        )


def test_gated_rmsnorm_forward_uses_opaque_op_only_for_silu(monkeypatch) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config

    layernorm = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.layernorm"
    )
    native = Mock(return_value=torch.tensor([1.0]))
    opaque = Mock(return_value=torch.tensor([2.0]))
    monkeypatch.setattr(layernorm.RMSNormGated, "forward_native", native)

    with set_current_vllm_config(VllmConfig()):
        norm = layernorm.AscendRMSNormGated(4, activation="silu")
    x = torch.randn(2, 4)
    z = torch.randn_like(x)
    torch_proxy = SimpleNamespace(
        Tensor=torch.Tensor,
        ops=SimpleNamespace(
            vllm=SimpleNamespace(ascend_rms_norm_gated=opaque)
        ),
    )
    monkeypatch.setattr(layernorm, "torch", torch_proxy)

    assert norm.forward_oot(x, z).item() == 2.0
    opaque.assert_called_once()
    native.assert_not_called()

    norm.activation = "sigmoid"
    assert norm.forward_oot(x, z).item() == 1.0
    norm.activation = "swish"
    assert norm.forward_oot(x, None).item() == 1.0
    assert native.call_count == 2
