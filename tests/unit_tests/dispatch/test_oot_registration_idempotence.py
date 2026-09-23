# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture()
def isolated_oot_registry():
    from vllm.model_executor.custom_op import op_registry_oot

    saved = dict(op_registry_oot)
    op_registry_oot.clear()
    try:
        yield op_registry_oot
    finally:
        op_registry_oot.clear()
        op_registry_oot.update(saved)


def _configure_registration(monkeypatch, *, vendor_name: str, device_type: str):
    import vllm.platforms
    import vllm_fl.ops.custom_ops as custom_ops
    import vllm_fl.utils as utils

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(vendor_name=vendor_name, device_type=device_type),
    )
    monkeypatch.setattr(utils, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(utils, "get_oot_blacklist", lambda: [])
    monkeypatch.setattr(utils, "get_oot_whitelist", lambda: None)
    monkeypatch.setattr(custom_ops, "_patch_fused_moe_factory", lambda: None)
    monkeypatch.setattr(custom_ops, "_patch_unquantized_moe_oracle", lambda: None)
    return custom_ops


def test_ascend_vendor_keeps_rmsnorm_and_generic_operators_register(
    monkeypatch, isolated_oot_registry
) -> None:
    from vllm.model_executor.custom_op import CustomOp
    from vllm_fl.ops.activation import SiluAndMulFL

    custom_ops = _configure_registration(
        monkeypatch, vendor_name="ascend", device_type="npu"
    )
    import vllm_fl.dispatch.backends.vendor.ascend.patch as ascend_patch

    class AscendRMSNorm:
        pass

    def apply_vendor_patch():
        if "RMSNorm" not in isolated_oot_registry:
            CustomOp.register_oot(
                _decorated_op_cls=AscendRMSNorm, name="RMSNorm"
            )

    monkeypatch.setattr(ascend_patch, "apply_ascend_patches", apply_vendor_patch)

    custom_ops.register_oot_ops(whitelist=["rms_norm", "silu_and_mul"])
    custom_ops.register_oot_ops(whitelist=["rms_norm", "silu_and_mul"])

    assert isolated_oot_registry["RMSNorm"] is AscendRMSNorm
    assert isolated_oot_registry["SiluAndMul"] is SiluAndMulFL


def test_generic_exact_class_registration_is_idempotent(
    monkeypatch, isolated_oot_registry
) -> None:
    from vllm_fl.ops.activation import SiluAndMulFL

    custom_ops = _configure_registration(
        monkeypatch, vendor_name="cuda", device_type="cuda"
    )

    custom_ops.register_oot_ops(whitelist=["silu_and_mul"])
    custom_ops.register_oot_ops(whitelist=["silu_and_mul"])

    assert isolated_oot_registry == {"SiluAndMul": SiluAndMulFL}


def test_generic_foreign_owner_conflict_remains_fatal(
    monkeypatch, isolated_oot_registry
) -> None:
    custom_ops = _configure_registration(
        monkeypatch, vendor_name="cuda", device_type="cuda"
    )

    class ForeignSilu:
        pass

    isolated_oot_registry["SiluAndMul"] = ForeignSilu

    with pytest.raises(RuntimeError, match="already registered by") as error:
        custom_ops.register_oot_ops(whitelist=["silu_and_mul"])

    assert "refusing to replace" in str(error.value)
    assert isolated_oot_registry["SiluAndMul"] is ForeignSilu


def test_cross_custom_op_pluggable_layer_conflict_remains_fatal(
    isolated_oot_registry,
) -> None:
    from vllm.model_executor.custom_op import CustomOp, PluggableLayer
    from vllm_fl.ops.custom_ops import _register_oot_once

    class FirstOwner(CustomOp):
        pass

    class SecondOwner(PluggableLayer):
        pass

    _register_oot_once(FirstOwner, "SharedName")
    with pytest.raises(RuntimeError, match="SharedName"):
        _register_oot_once(SecondOwner, "SharedName")

    assert isolated_oot_registry["SharedName"] is FirstOwner


def test_ascend_vendor_collision_is_not_hidden(
    monkeypatch, isolated_oot_registry
) -> None:
    from vllm.model_executor.custom_op import CustomOp

    custom_ops = _configure_registration(
        monkeypatch, vendor_name="ascend", device_type="npu"
    )
    import vllm_fl.dispatch.backends.vendor.ascend.patch as ascend_patch

    class ForeignRMSNorm:
        pass

    class AscendRMSNorm:
        pass

    isolated_oot_registry["RMSNorm"] = ForeignRMSNorm

    def apply_vendor_patch():
        CustomOp.register_oot(
            _decorated_op_cls=AscendRMSNorm, name="RMSNorm"
        )

    monkeypatch.setattr(ascend_patch, "apply_ascend_patches", apply_vendor_patch)

    with pytest.raises(AssertionError):
        custom_ops.register_oot_ops(whitelist=["rms_norm"])

    assert isolated_oot_registry["RMSNorm"] is ForeignRMSNorm


@pytest.mark.parametrize(
    ("vendor_name", "device_type"),
    [("cuda", "cuda"), ("other", "npu")],
)
def test_non_ascend_platforms_keep_generic_rmsnorm(
    monkeypatch, isolated_oot_registry, vendor_name, device_type
) -> None:
    from vllm_fl.ops.layernorm import RMSNormFL

    custom_ops = _configure_registration(
        monkeypatch, vendor_name=vendor_name, device_type=device_type
    )
    import vllm_fl.dispatch.backends.vendor.ascend.patch as ascend_patch

    monkeypatch.setattr(
        ascend_patch,
        "apply_ascend_patches",
        lambda: pytest.fail("non-Ascend platform called Ascend lifecycle"),
    )
    custom_ops.register_oot_ops(whitelist=["rms_norm"])

    assert isolated_oot_registry["RMSNorm"] is RMSNormFL


def test_oot_disabled_still_applies_ascend_vendor_lifecycle(
    monkeypatch, isolated_oot_registry
) -> None:
    custom_ops = _configure_registration(
        monkeypatch, vendor_name="ascend", device_type="npu"
    )
    import vllm_fl.dispatch.backends.vendor.ascend.patch as ascend_patch
    import vllm_fl.utils as utils

    calls = []
    monkeypatch.setattr(utils, "is_oot_enabled", lambda: False)
    monkeypatch.setattr(
        ascend_patch,
        "apply_ascend_patches",
        lambda: calls.append("ascend"),
    )

    custom_ops.register_oot_ops(whitelist=["rms_norm"])

    assert calls == ["ascend"]
    assert isolated_oot_registry == {}


def test_ptpu_routes_only_to_sunrise_lifecycle(
    monkeypatch, isolated_oot_registry
) -> None:
    custom_ops = _configure_registration(
        monkeypatch, vendor_name="sunrise", device_type="ptpu"
    )
    import vllm_fl.dispatch.backends.vendor.ascend.patch as ascend_patch
    import vllm_fl.dispatch.backends.vendor.sunrise.patch as sunrise_patch

    calls = []
    monkeypatch.setattr(
        ascend_patch,
        "apply_ascend_patches",
        lambda: pytest.fail("PTPU platform called Ascend lifecycle"),
    )
    monkeypatch.setattr(
        sunrise_patch,
        "apply_sunrise_patches",
        lambda: calls.append("sunrise"),
    )

    custom_ops.register_oot_ops(whitelist=["rms_norm"])

    assert calls == ["sunrise"]
    assert "RMSNorm" in isolated_oot_registry


def test_ascend_vendor_owned_set_matches_current_generic_overlap() -> None:
    from vllm_fl.ops.custom_ops import (
        OOT_OPS,
        _ASCEND_VENDOR_OWNED_REGISTRATIONS,
    )

    ascend_registered_names = {
        "QKVParallelLinear",
        "MergedColumnParallelLinear",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "ReplicatedLinear",
        "MMEncoderAttention",
        "GatedDeltaNetAttention",
        "RMSNorm",
        "GemmaRMSNorm",
        "RMSNormGated",
        "VocabParallelEmbedding",
        "ParallelLMHead",
    }
    generic_registered_names = {
        registration_name for _, registration_name in OOT_OPS.values()
    }

    assert (
        generic_registered_names & ascend_registered_names
        == _ASCEND_VENDOR_OWNED_REGISTRATIONS
        == {"RMSNorm"}
    )
