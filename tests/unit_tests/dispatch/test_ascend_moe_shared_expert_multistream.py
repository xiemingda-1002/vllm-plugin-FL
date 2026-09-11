# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[3]
MOE = ROOT / "vllm_fl" / "dispatch" / "backends" / "vendor" / "ascend" / "impl" / "moe"


def _source(name: str) -> str:
    return (MOE / name).read_text(encoding="utf-8")


def _load_compat():
    # Load this narrow facade without importing the plugin package initializer:
    # this CPU unit module intentionally has no FlagGems/NPU dependency.
    package_names = (
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.ascend",
    )
    modules = {name: _package(name) for name in package_names}
    modules["vllm_fl.ascend_flashcomm"] = SimpleNamespace(
        enable_flashcomm1=lambda *_args, **_kwargs: False
    )
    modules["vllm_fl.dispatch.backends.vendor.ascend.hardware"] = SimpleNamespace(
        AscendDeviceType=Enum("AscendDeviceType", "A2 A3 A5 _310P"),
        get_ascend_device_type=lambda: None,
    )
    path = MOE / "compat.py"
    spec = importlib.util.spec_from_file_location("_fl_ascend_moe_compat_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_modules = {name: sys.modules.get(name) for name in modules}
    try:
        sys.modules.update(modules)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def _package(name: str):
    module = type(sys)(name)
    module.__path__ = []
    return module


@pytest.fixture(autouse=True)
def _isolate_fused_mc2_environment(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_FUSED_MC2", raising=False)


def test_shared_expert_stream_is_created_lazily_once(monkeypatch) -> None:
    compat = _load_compat()
    stream = object()
    stream_factory = Mock(return_value=stream)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(Stream=stream_factory)),
    )

    assert compat._SHARED_EXPERTS_CALCULATION_STREAM is None
    assert compat.shared_experts_calculation_stream() is stream
    assert compat.shared_experts_calculation_stream() is stream
    stream_factory.assert_called_once_with()


def test_stream_switch_is_noop_when_disabled() -> None:
    compat = _load_compat()
    with compat.npu_stream_switch(None, enabled=False) as result:
        assert result is None


def test_stream_switch_uses_npu_context_when_enabled(monkeypatch) -> None:
    compat = _load_compat()
    target_stream = object()
    marker = object()
    stream_context = Mock(return_value=nullcontext(marker))
    monkeypatch.setattr(
        compat,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(stream=stream_context)),
    )

    with compat.npu_stream_switch(target_stream, enabled=True) as result:
        assert result is marker
    stream_context.assert_called_once_with(target_stream)


def test_multistream_activation_matches_current_rc1_gate() -> None:
    runner = _source("fused_moe.py")
    assert (
        "self.multistream_overlap_shared_expert = (\n"
        "            ascend_config.multistream_overlap_shared_expert\n"
        "            and shared_experts is not None\n"
        "        )"
    ) in runner
    assert (
        "if ascend_config.multistream_overlap_shared_expert:\n"
        "            raise NotImplementedError"
    ) not in runner


def test_post_load_validation_wrap_matches_current_rc1_order() -> None:
    runner = _source("fused_moe.py")
    setup = runner.index("setup_moe_comm_method(self.moe_config)")
    capture = runner.index("original_process_weights = (", setup)
    original = runner.index(
        "result = original_process_weights(*args, **kwargs)", capture
    )
    validation = runner.index("self._validate_shared_expert_consistency()", original)
    result_return = runner.index("return result", validation)
    assert setup < capture < original < validation < result_return
    assert "@wraps(original_process_weights)" in runner[capture:result_return]


def test_multistream_forward_retains_stream_and_event_lifecycle() -> None:
    runner = _source("fused_moe.py")
    start = runner.index("def _forward_shared_experts(")
    end = runner.index("def shared_forward_impl(", start)
    forward = runner[start:end]

    assert (
        "with npu_stream_switch(shared_experts_calculation_stream(), "
        "enabled=self.multistream_overlap_shared_expert):"
    ) in forward
    assert "wait_event(fused_moe_evts.before_routed_experts)" in forward
    assert "maybe_wait_event(fused_moe_evts.before_dispatch)" in forward
    assert "maybe_wait_event(fused_moe_evts.before_combine)" in forward
    assert "wait_stream(shared_experts_calculation_stream())" in forward
    assert "self._shared_experts_part1(hidden_states)" in forward
    assert "self._shared_experts_part2(hidden_states, part1_out)" in forward


def test_unmigrated_neighbor_modes_still_fail_closed() -> None:
    runner = _source("fused_moe.py")
    for rejection in (
        "fused MC2 is not migrated",
        "shared-expert DP is not migrated",
        "EPLB is not migrated",
    ):
        assert rejection in runner


def test_moe_compat_uses_rc1_nested_defaults(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(compat, "_additional_config", lambda: {})

    config = compat.get_ascend_config()

    assert config.eplb_config.dynamic_eplb is False
    assert config.eplb_config.expert_heat_collection_interval == 600
    assert config.eplb_config.algorithm_execution_interval == 50
    assert config.eplb_config.eplb_policy_type == 2
    assert config.eplb_config.expert_map_record_path is None
    assert config.ascend_compilation_config.enable_static_kernel is False
    assert config.ascend_fusion_config.fusion_ops_gmmswigluquant is False


@pytest.mark.parametrize(
    ("additional_config", "message"),
    [
        ({"eplb_config": {"dynamic_eplb": True}}, "EPLB"),
        ({"eplb_config": {"expert_map_record_path": "record.json"}}, "EPLB"),
        ({"eplb_config": {"expert_map_record_path": ""}}, "EPLB"),
        ({"eplb_config": {"expert_map_path": "map.json"}}, "EPLB"),
        ({"eplb_config": {"num_redundant_experts": 1}}, "EPLB"),
        ({"enable_fused_mc2": 1}, "fused MC2"),
        ({"enable_shared_expert_dp": True}, "shared-expert DP"),
        ({"mix_placement": True}, "mixed shared-expert placement"),
        ({"enable_mc2_hierarchy_comm": True}, "MC2 hierarchy communication"),
        (
            {"ascend_compilation_config": {"enable_static_kernel": True}},
            "static kernel generation",
        ),
        (
            {"ascend_fusion_config": {"fusion_ops_gmmswigluquant": True}},
            "gmmswigluquant fusion",
        ),
    ],
)
def test_moe_compat_rejects_known_nested_unsupported_opt_ins(
    monkeypatch, additional_config, message
) -> None:
    compat = _load_compat()
    monkeypatch.setattr(compat, "_additional_config", lambda: additional_config)

    with pytest.raises(NotImplementedError, match=message):
        compat.get_ascend_config()


def test_moe_compat_preserves_dormant_nested_eplb_tuning(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(
        compat,
        "_additional_config",
        lambda: {
            "mega_moe_max_tokens": 8192,
            "eplb_config": {
                "eplb_policy_type": 3,
                "expert_heat_collection_interval": 10,
            },
        },
    )

    config = compat.get_ascend_config()

    assert config.mega_moe_max_tokens == 8192
    assert config.eplb_config.eplb_policy_type == 3
    assert config.eplb_config.expert_heat_collection_interval == 10


def test_moe_compat_rejects_legacy_flat_eplb_key(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(compat, "_additional_config", lambda: {"dynamic_eplb": False})

    with pytest.raises(
        ValueError, match=r"additional_config\.eplb_config\.dynamic_eplb"
    ):
        compat.get_ascend_config()


def test_moe_compat_rejects_fused_mc2_environment_opt_in(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(compat, "_additional_config", lambda: {})
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "1")

    with pytest.raises(NotImplementedError, match="fused MC2 communication"):
        compat.get_ascend_config()


def test_moe_compat_config_overrides_fused_mc2_environment(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(compat, "_additional_config", lambda: {"enable_fused_mc2": 0})
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "1")

    assert compat.get_ascend_config().enable_fused_mc2 == 0


def test_moe_compat_preserves_shared_expert_overlap_opt_in(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(
        compat,
        "_additional_config",
        lambda: {"multistream_overlap_shared_expert": True},
    )

    assert compat.get_ascend_config().multistream_overlap_shared_expert is True


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"expert_heat_collection_interval": "10"}, TypeError),
        ({"algorithm_execution_interval": -1}, ValueError),
        ({"num_redundant_experts": -1}, ValueError),
        ({"eplb_policy_type": 4}, ValueError),
        ({"eplb_heat_collection_stage": "invalid"}, ValueError),
        ({"unknown_key": True}, ValueError),
    ],
)
def test_invalid_eplb_values_follow_rc1_validation(monkeypatch, settings, error):
    compat = _load_compat()
    monkeypatch.setattr(compat, "_additional_config", lambda: {"eplb_config": settings})
    with pytest.raises(error):
        compat.get_ascend_config()


@pytest.mark.parametrize(
    "name", ["eplb_config", "ascend_compilation_config", "ascend_fusion_config"]
)
def test_malformed_nested_configuration_is_not_ignored(monkeypatch, name):
    compat = _load_compat()
    monkeypatch.setattr(compat, "_additional_config", lambda: {name: []})
    with pytest.raises(TypeError, match="must be a mapping"):
        compat.get_ascend_config()


@pytest.mark.parametrize("value", [0, -1, "8192", 1.5])
def test_invalid_mega_capacity_is_rejected(monkeypatch, value):
    compat = _load_compat()
    monkeypatch.setattr(
        compat, "_additional_config", lambda: {"mega_moe_max_tokens": value}
    )
    with pytest.raises(ValueError, match="positive integer"):
        compat.get_ascend_config()


def test_malformed_current_config_does_not_fall_back_to_defaults(monkeypatch):
    compat = _load_compat()
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(
            get_current_vllm_config=lambda: SimpleNamespace(additional_config=[])
        ),
    )
    with pytest.raises(TypeError, match="additional_config must be a mapping"):
        compat.get_ascend_config()


def test_no_current_context_keeps_supported_defaults(monkeypatch):
    compat = _load_compat()

    def no_context():
        raise AssertionError("No current VllmConfig")

    monkeypatch.setitem(
        sys.modules, "vllm.config", SimpleNamespace(get_current_vllm_config=no_context)
    )
    assert compat.get_ascend_config().enable_fused_mc2 == 0
