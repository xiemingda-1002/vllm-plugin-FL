import inspect
from types import SimpleNamespace

import pytest

from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.compat import (
    balance_scheduling_enabled,
)
from vllm_fl.scheduling.ascend_balance import BalanceScheduler


def _config(additional_config=None):
    return SimpleNamespace(additional_config=additional_config)


def test_balance_schedule_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_BALANCE_SCHEDULING", raising=False)
    assert balance_scheduling_enabled(_config()) is False


def test_balance_schedule_defaults_off_and_config_overrides_env(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_BALANCE_SCHEDULING", "1")
    assert balance_scheduling_enabled(_config()) is True
    assert balance_scheduling_enabled(_config({"enable_balance_scheduling": False})) is False
    assert balance_scheduling_enabled(_config({"enable_balance_scheduling": True})) is True


def test_balance_schedule_rejects_invalid_environment_value(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_BALANCE_SCHEDULING", "yes")
    with pytest.raises(ValueError, match="must be an integer"):
        balance_scheduling_enabled(_config())


def test_balance_scheduler_uses_rc1_balance_flag_semantics():
    schedule_source = inspect.getsource(BalanceScheduler.schedule)

    # Keep the rc1 execution semantics rather than a queue-substitution
    # shortcut: every WAITING-loop iteration freezes admission when any DP
    # rank reached max_num_running_reqs in the preceding gathered snapshot.
    assert "max(t.item() for t in self.balance_queue) == self.max_num_running_reqs" in schedule_source
    assert "if request_queue is None:" in schedule_source
    assert "return super().schedule(throttle_prefills)" in schedule_source
