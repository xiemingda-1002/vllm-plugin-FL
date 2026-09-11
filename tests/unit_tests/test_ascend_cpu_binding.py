# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_fl import cpu_binding


def test_ascend_cpu_binding_has_no_vllm_ascend_dependency() -> None:
    source = Path(cpu_binding.__file__).read_text(encoding="utf-8")

    assert "vllm_ascend" not in source


@pytest.mark.parametrize(
    ("soc_version", "expected"),
    [
        (220, cpu_binding.AscendDeviceType.A2),
        (250, cpu_binding.AscendDeviceType.A3),
        (200, cpu_binding.AscendDeviceType._310P),
        (260, cpu_binding.AscendDeviceType.A5),
    ],
)
def test_ascend_cpu_binding_resolves_runtime_soc(
    monkeypatch, soc_version, expected
) -> None:
    torch_npu = SimpleNamespace(
        npu=SimpleNamespace(get_soc_version=lambda: soc_version)
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    assert cpu_binding.get_ascend_device_type() is expected


def test_ascend_cpu_binding_rejects_unknown_runtime_soc(monkeypatch) -> None:
    torch_npu = SimpleNamespace(npu=SimpleNamespace(get_soc_version=lambda: 999))
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    with pytest.raises(RuntimeError, match="Can not support soc_version"):
        cpu_binding.get_ascend_device_type()


def _cpu_alloc() -> cpu_binding.CpuAlloc:
    alloc = object.__new__(cpu_binding.CpuAlloc)
    alloc.rank_id = 0
    alloc.device_info = SimpleNamespace(
        running_npu_list=[1],
        all_logic_npus=[0, 1],
        allowed_cpus=list(range(12)),
        npu_affinity={0: [0, 1], 1: [2, 3]},
        total_logic_npus=2,
    )
    alloc.cpu_node = {}
    alloc.numa_to_cpu_map = defaultdict(list)
    alloc.npu_cpu_pool = {}
    alloc.assign_main = {}
    alloc.assign_acl = {}
    alloc.assign_rel = {}
    alloc.uvb_cpu_pool = []
    return alloc


def test_ascend_cpu_binding_uses_rc1_generation_policies(monkeypatch) -> None:
    alloc = _cpu_alloc()

    monkeypatch.setattr(
        cpu_binding,
        "get_ascend_device_type",
        lambda: cpu_binding.AscendDeviceType.A2,
    )
    assert alloc._binding_mode() == cpu_binding.TOPO_AFFINITY_MODE
    assert alloc._reserve_irq_cpus() is True

    monkeypatch.setattr(
        cpu_binding,
        "get_ascend_device_type",
        lambda: cpu_binding.AscendDeviceType.A3,
    )
    assert alloc._binding_mode() == cpu_binding.GLOBAL_SLICE_MODE

    monkeypatch.setattr(
        cpu_binding,
        "get_ascend_device_type",
        lambda: cpu_binding.AscendDeviceType.A5,
    )
    assert alloc._binding_mode() == cpu_binding.TOPO_AFFINITY_MODE
    assert alloc._reserve_irq_cpus() is False


def test_ascend_cpu_binding_global_slice_uses_global_logical_ids(
    monkeypatch,
) -> None:
    alloc = _cpu_alloc()
    monkeypatch.setattr(
        cpu_binding,
        "get_ascend_device_type",
        lambda: cpu_binding.AscendDeviceType.A3,
    )

    alloc.build_global_slice_cpu_pool()

    assert alloc.npu_cpu_pool == {1: [6, 7, 8, 9, 10, 11]}


def test_ascend_cpu_binding_skips_non_arm_without_probing_devices(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cpu_binding, "is_arm_cpu", lambda: False)

    class UnexpectedCpuAlloc:
        def __init__(self, _rank_id):
            raise AssertionError("DeviceInfo must not be built on non-ARM hosts")

    monkeypatch.setattr(cpu_binding, "CpuAlloc", UnexpectedCpuAlloc)
    cpu_binding.bind_cpus(3)
