"""CPU-only contracts for the Ascend rc1 torch-NPU profiler wrapper."""

from types import SimpleNamespace

import pytest

from vllm_fl.dispatch.backends.vendor.ascend.profiler import (
    TorchNPUProfilerWrapper,
)
from vllm_fl.dispatch.backends.vendor.ascend.profiler import (
    torch_npu_profiler as profiler_module,
)
from vllm_fl.worker import worker as worker_module


def _config(**overrides):
    values = {
        "profiler": "torch",
        "torch_profiler_dir": "/tmp/trace",
        "torch_profiler_with_memory": True,
        "torch_profiler_with_stack": True,
        "delay_iterations": 0,
        "max_iterations": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_torch_npu(monkeypatch):
    calls = {}

    class Session:
        def __init__(self):
            self.starts = 0
            self.stops = 0

        def start(self):
            self.starts += 1

        def stop(self):
            self.stops += 1

    session = Session()

    def experimental_config(**kwargs):
        calls["experimental"] = kwargs
        return kwargs

    def handler(directory, worker_name):
        calls["handler"] = (directory, worker_name)
        return "handler-result"

    def profile(**kwargs):
        calls["profile"] = kwargs
        return session

    fake_profiler = SimpleNamespace(
        _ExperimentalConfig=experimental_config,
        ExportType=SimpleNamespace(Text="text"),
        ProfilerLevel=SimpleNamespace(Level1="level1"),
        AiCMetrics=SimpleNamespace(PipeUtilization="pipe"),
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        tensorboard_trace_handler=handler,
        profile=profile,
    )
    monkeypatch.setattr(profiler_module, "torch_npu", SimpleNamespace(profiler=fake_profiler))
    return calls, session


def test_npu_profiler_matches_rc1_arguments_and_worker_lifecycle(monkeypatch) -> None:
    calls, session = _fake_torch_npu(monkeypatch)
    wrapper = TorchNPUProfilerWrapper(_config(), "replay_dp2_tp3", {})

    assert calls["profile"]["activities"] == ["cpu", "npu"]
    assert calls["profile"]["with_stack"] is False
    assert calls["profile"]["profile_memory"] is True
    assert calls["profile"]["with_modules"] is True
    assert calls["handler"] == ("/tmp/trace", "replay_dp2_tp3")
    assert calls["experimental"] == {
        "export_type": "text",
        "profiler_level": "level1",
        "msprof_tx": False,
        "aic_metrics": "pipe",
        "l2_cache": False,
        "op_attr": False,
        "data_simplification": True,
        "record_op_args": False,
        "gc_detect_threshold": None,
    }

    wrapper.start()
    wrapper.step()
    wrapper.stop()
    assert (session.starts, session.stops) == (1, 1)


def test_npu_profiler_uses_worker_profiler_delay_and_max_iterations(monkeypatch) -> None:
    _, session = _fake_torch_npu(monkeypatch)
    wrapper = TorchNPUProfilerWrapper(
        _config(delay_iterations=1, max_iterations=1), "rank0", {}
    )

    wrapper.start()
    assert session.starts == 0
    wrapper.step()
    assert session.starts == 1
    wrapper.step()
    assert session.stops == 1


@pytest.mark.parametrize("env_value,expected", [("0", False), ("1", True)])
def test_msmonitor_env_default_matches_rc1(monkeypatch, env_value, expected) -> None:
    monkeypatch.setenv("MSMONITOR_USE_DAEMON", env_value)
    assert profiler_module.resolve_msmonitor_use_daemon() is expected


def test_msmonitor_explicit_config_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("MSMONITOR_USE_DAEMON", "1")
    assert profiler_module.resolve_msmonitor_use_daemon({"msmonitor_use_daemon": False}) is False
    monkeypatch.setenv("MSMONITOR_USE_DAEMON", "0")
    assert profiler_module.resolve_msmonitor_use_daemon({"msmonitor_use_daemon": True}) is True


@pytest.mark.parametrize(
    "config,additional_config,error",
    [
        (_config(profiler="cuda"), {}, "Unrecognized profiler"),
        (_config(torch_profiler_dir=""), {}, "torch_profiler_dir cannot be empty"),
        (_config(), {"msmonitor_use_daemon": True}, "cannot be both enabled"),
    ],
)
def test_npu_profiler_rejects_invalid_or_conflicting_configuration(
    monkeypatch, config, additional_config, error
) -> None:
    _fake_torch_npu(monkeypatch)
    with pytest.raises(RuntimeError, match=error):
        TorchNPUProfilerWrapper(config, "rank0", additional_config)


def test_worker_profile_selects_ascend_wrapper_with_rank_trace_name(monkeypatch) -> None:
    captured = {}

    class FakeNPUWrapper:
        def __init__(self, config, trace_name, additional_config):
            captured.update(config=config, trace_name=trace_name, additional_config=additional_config)
            self.starts = 0

        def start(self):
            self.starts += 1

    import vllm.distributed.utils as distributed_utils
    import vllm_fl.dispatch.backends.vendor.ascend.profiler as profiler_package

    monkeypatch.setattr(
        worker_module,
        "current_platform",
        SimpleNamespace(device_type="npu", vendor_name="ascend"),
    )
    monkeypatch.setattr(
        distributed_utils,
        "get_worker_rank_suffix",
        lambda global_rank: f"dp2_tp3_rank{global_rank}",
    )
    monkeypatch.setattr(profiler_package, "TorchNPUProfilerWrapper", FakeNPUWrapper)
    worker = object.__new__(worker_module.WorkerFL)
    worker.rank = 11
    worker.local_rank = 3
    worker.profiler = None
    worker.profiler_config = _config()
    worker.vllm_config = SimpleNamespace(additional_config={"msmonitor_use_daemon": False})

    worker.profile(profile_prefix="replay")

    assert captured["trace_name"] == "replay_dp2_tp3_rank11"
    assert captured["additional_config"] == {"msmonitor_use_daemon": False}
    assert worker.profiler.starts == 1


def test_worker_profile_keeps_generic_wrapper_for_non_ascend(monkeypatch) -> None:
    captured = {}

    class FakeTorchWrapper:
        def __init__(self, config, **kwargs):
            captured.update(config=config, **kwargs)
            self.starts = 0

        def start(self):
            self.starts += 1

    import vllm.distributed.utils as distributed_utils

    monkeypatch.setattr(
        worker_module,
        "current_platform",
        SimpleNamespace(device_type="cuda", vendor_name="nvidia"),
    )
    monkeypatch.setattr(distributed_utils, "get_worker_rank_suffix", lambda **_kwargs: "rank1")
    monkeypatch.setattr(worker_module, "TorchProfilerWrapper", FakeTorchWrapper)
    worker = object.__new__(worker_module.WorkerFL)
    worker.rank = 1
    worker.local_rank = 1
    worker.profiler = None
    worker.profiler_config = _config()
    worker.vllm_config = SimpleNamespace(additional_config={})

    worker.profile()

    assert captured["worker_name"] == "rank1"
    assert captured["activities"] == ["CPU", "CUDA"]
    assert worker.profiler.starts == 1
