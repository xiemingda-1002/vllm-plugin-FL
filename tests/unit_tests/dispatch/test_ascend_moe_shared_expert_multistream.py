# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


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


def _load_compat():
    path = MOE / "compat.py"
    spec = importlib.util.spec_from_file_location("_fl_ascend_moe_compat_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    capture = runner.index(
        "original_process_weights = (", setup
    )
    original = runner.index("result = original_process_weights(*args, **kwargs)", capture)
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
