# Copyright (c) 2026 BAAI. All rights reserved.

import inspect
from types import SimpleNamespace

from vllm.config import CUDAGraphMode

from vllm_fl import cpu_binding
from vllm_fl.platform import PlatformFL, _configure_ascend_cpu_binding
from vllm_fl.worker import worker as worker_module


def _config(*, additional_config=None, **parallel_overrides):
    parallel = {
        "numa_bind": False,
        "numa_bind_nodes": None,
        "numa_bind_cpus": None,
    }
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        additional_config=additional_config,
        parallel_config=SimpleNamespace(**parallel),
    )


def test_ascend_cpu_binding_is_enabled_by_default() -> None:
    config = _config()

    _configure_ascend_cpu_binding(config)

    assert config.additional_config == {"enable_cpu_binding": True}


def test_ascend_cpu_binding_converts_upstream_numa_options() -> None:
    config = _config(
        numa_bind=True,
        numa_bind_nodes=[0, 1],
        numa_bind_cpus=["0-15"],
    )

    _configure_ascend_cpu_binding(config)

    assert config.additional_config["enable_cpu_binding"] is True
    assert config.parallel_config.numa_bind is False
    assert config.parallel_config.numa_bind_nodes is None
    assert config.parallel_config.numa_bind_cpus is None


def test_ascend_cpu_binding_preserves_explicit_disable_with_numa_bind() -> None:
    config = _config(
        additional_config={"enable_cpu_binding": False},
        numa_bind=True,
    )

    _configure_ascend_cpu_binding(config)

    assert config.additional_config["enable_cpu_binding"] is False
    assert config.parallel_config.numa_bind is False


def test_platform_does_not_materialize_cpu_binding_for_other_vendors(
    monkeypatch,
) -> None:
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls="auto",
            all2all_backend="",
            data_parallel_size=1,
            numa_bind=False,
            numa_bind_nodes=None,
            numa_bind_cpus=None,
        ),
        model_config=None,
        cache_config=None,
        attention_config=None,
        compilation_config=SimpleNamespace(
            compile_sizes=[],
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        additional_config=None,
    )

    PlatformFL.check_and_update_config(config)

    assert config.additional_config is None


def _worker(additional_config=None, local_rank=2):
    return SimpleNamespace(
        local_rank=local_rank,
        vllm_config=SimpleNamespace(additional_config=additional_config),
    )


def test_ascend_worker_invokes_cpu_binding_by_default(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="npu")
    )
    monkeypatch.setattr(cpu_binding, "bind_cpus", calls.append)

    worker_module._maybe_bind_ascend_worker_cpus(_worker())

    assert calls == [2]


def test_ascend_worker_honors_explicit_cpu_binding_disable(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="npu")
    )
    monkeypatch.setattr(cpu_binding, "bind_cpus", calls.append)

    worker_module._maybe_bind_ascend_worker_cpus(
        _worker({"enable_cpu_binding": False})
    )

    assert calls == []


def test_non_ascend_worker_never_invokes_cpu_binding(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="cuda")
    )
    monkeypatch.setattr(cpu_binding, "bind_cpus", calls.append)

    worker_module._maybe_bind_ascend_worker_cpus(_worker())

    assert calls == []


def test_ascend_worker_binding_failure_is_nonfatal(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="npu")
    )

    def fail(_rank):
        raise RuntimeError("affinity unavailable")

    monkeypatch.setattr(cpu_binding, "bind_cpus", fail)

    worker_module._maybe_bind_ascend_worker_cpus(_worker(local_rank=7))

    assert "Bind cpus failed in rank7" in caplog.text


def test_ascend_worker_binds_after_sampler_warmup_before_seed_reset() -> None:
    source = inspect.getsource(worker_module.WorkerFL.compile_or_warm_up_model)

    assert source.index("_dummy_sampler_run") < source.index(
        "_maybe_bind_ascend_worker_cpus"
    ) < source.rindex("set_random_seed")
