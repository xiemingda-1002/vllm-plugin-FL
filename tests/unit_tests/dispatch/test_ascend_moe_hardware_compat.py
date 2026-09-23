# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[3]
HARDWARE = (
    ROOT / "vllm_fl" / "platforms" / "ascend" / "hardware.py"
)
COMPAT = (
    ROOT
    / "vllm_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "ascend"
    / "impl"
    / "moe"
    / "compat.py"
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_hardware():
    return _load_module(HARDWARE, "_fl_ascend_hardware_test")


def test_build_soc_metadata_is_resolved_from_the_fl_package_root(tmp_path) -> None:
    """The moved helper must still locate wheel metadata, without an NPU probe."""
    package_root = tmp_path / "vllm_fl"
    metadata = package_root / "_cann_ops_custom" / "FL_SOC_VERSION"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("ascend910_9392\n", encoding="utf-8")
    module_path = package_root / "platforms" / "ascend" / "hardware.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(HARDWARE.read_text(encoding="utf-8"), encoding="utf-8")

    hardware = _load_module(module_path, "_fl_ascend_hardware_metadata_test")

    assert hardware._build_soc_version() == "ascend910_9392"


@pytest.mark.parametrize(
    ("build_soc", "expected_name"),
    [("ascend910b1", "A2"), ("ascend910_9392", "A3")],
)
def test_moe_device_type_comes_from_build_metadata_without_npu_probe(
    monkeypatch, build_soc, expected_name
) -> None:
    hardware = _load_hardware()
    monkeypatch.setattr(hardware, "_build_soc_version", lambda: build_soc)
    runtime_probe = Mock(side_effect=AssertionError("runtime must not be queried"))
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(get_soc_version=runtime_probe)),
    )
    hardware._ascend_device_type = None

    assert hardware.get_ascend_device_type().name == expected_name
    runtime_probe.assert_not_called()


def test_moe_device_type_rejects_unsupported_build_soc(monkeypatch) -> None:
    hardware = _load_hardware()
    monkeypatch.setattr(hardware, "_build_soc_version", lambda: "ascend999")
    hardware._ascend_device_type = None

    with pytest.raises(RuntimeError, match="build SOC_VERSION"):
        hardware.get_ascend_device_type()


def test_moe_runtime_check_preserves_build_runtime_mismatch(monkeypatch) -> None:
    hardware = _load_hardware()
    monkeypatch.setattr(hardware, "_build_soc_version", lambda: "ascend910b1")
    hardware._ascend_device_type = None
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(get_soc_version=lambda: 250)),
    )

    with pytest.raises(AssertionError, match="does not match"):
        hardware.check_ascend_device_type()


@pytest.mark.parametrize(("runtime_soc", "expected_name"), [(220, "A2"), (250, "A3")])
def test_moe_runtime_detection_supports_a2_and_a3_without_build_metadata(
    monkeypatch, runtime_soc, expected_name
) -> None:
    hardware = _load_hardware()
    monkeypatch.setattr(hardware, "_build_soc_version", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(get_soc_version=lambda: runtime_soc)),
    )
    hardware._ascend_device_type = None

    hardware.check_ascend_device_type()

    assert hardware.get_ascend_device_type().name == expected_name


@pytest.mark.parametrize("runtime_soc", [219, 226, 249, 256, 261])
def test_moe_runtime_detection_rejects_unknown_soc(monkeypatch, runtime_soc) -> None:
    hardware = _load_hardware()
    monkeypatch.setattr(hardware, "_build_soc_version", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(get_soc_version=lambda: runtime_soc)),
    )

    with pytest.raises(RuntimeError, match="Can not support soc_version"):
        hardware.check_ascend_device_type()


@pytest.mark.parametrize(
    ("build_soc", "runtime_soc", "expected"),
    [
        ("ascend910b1", 220, "A2"),
        ("ascend910b4", 225, "A2"),
        ("ascend910_9392", 250, "A3"),
        ("ascend910_9381", 255, "A3"),
    ],
)
def test_matching_native_package_and_runtime(
    monkeypatch, build_soc, runtime_soc, expected
):
    hardware = _load_hardware()
    monkeypatch.setattr(hardware, "_build_soc_version", lambda: build_soc)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(get_soc_version=lambda: runtime_soc)),
    )
    hardware.check_ascend_device_type()
    assert hardware.get_ascend_device_type().name == expected


def test_missing_metadata_does_not_guess_device_before_worker_init(monkeypatch):
    hardware = _load_hardware()
    monkeypatch.setattr(hardware, "_build_soc_version", lambda: None)
    with pytest.raises(RuntimeError, match="before worker device initialization"):
        hardware.get_ascend_device_type()


def test_moe_skip_allreduce_keeps_rc1_signature_but_fails_closed() -> None:
    compat = COMPAT.read_text(encoding="utf-8")

    start = compat.index("def should_skip_allreduce_across_dp_group(")
    end = compat.index("def dispose_tensor", start)
    function = compat[start:end]
    assert "is_draft_model: bool = False" in function
    assert "return False" in function


def test_worker_validates_only_after_selecting_an_ascend_device() -> None:
    worker = (ROOT / "vllm_fl" / "worker" / "worker.py").read_text(encoding="utf-8")
    set_device = worker.index("current_platform.set_device(self.device)")
    hardware_gate = worker.index(
        'if current_platform.device_type == "npu":', set_device
    )
    check = worker.index("check_ascend_device_type()", hardware_gate)
    dtype_check = worker.index("current_platform.check_if_supports_dtype", check)

    assert set_device < hardware_gate < check < dtype_check
