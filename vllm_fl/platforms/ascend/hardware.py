"""FL-owned Ascend build/runtime generation checks."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class AscendDeviceType(Enum):
    A2 = "A2"
    A3 = "A3"
    A5 = "A5"
    _310P = "310P"


_ascend_device_type: AscendDeviceType | None = None


def _device_type_from_build_soc(soc_version: str) -> AscendDeviceType:
    normalized = soc_version.strip().lower()
    if "310p" in normalized:
        return AscendDeviceType._310P
    if normalized.startswith(("910b", "ascend910b")):
        return AscendDeviceType.A2
    if normalized.startswith(("910c", "ascend910c", "910_93", "ascend910_93")):
        return AscendDeviceType.A3
    if normalized.startswith(("950", "ascend950")):
        return AscendDeviceType.A5
    raise RuntimeError(f"Can not support build SOC_VERSION: {soc_version}.")


def _build_soc_version() -> str | None:
    """Read existing FL wheel metadata without loading the NPU runtime."""
    package_dir = next(
        parent
        for parent in Path(__file__).resolve().parents
        if parent.name == "vllm_fl"
    )
    metadata = package_dir / "_cann_ops_custom" / "FL_SOC_VERSION"
    if metadata.is_file():
        return metadata.read_text(encoding="utf-8").strip()
    return None


def _runtime_device_type() -> AscendDeviceType:
    import torch_npu

    soc_version = torch_npu.npu.get_soc_version()
    if 220 <= soc_version <= 225:
        return AscendDeviceType.A2
    if 250 <= soc_version <= 255:
        return AscendDeviceType.A3
    if 200 <= soc_version <= 205:
        return AscendDeviceType._310P
    if soc_version == 260:
        return AscendDeviceType.A5
    raise RuntimeError(f"Can not support soc_version: {soc_version}.")


def get_ascend_device_type() -> AscendDeviceType:
    """Return build generation without an import-time torch-npu query."""
    global _ascend_device_type
    if _ascend_device_type is None:
        build_soc_version = _build_soc_version()
        if build_soc_version is None:
            raise RuntimeError(
                "FL Ascend device type metadata is unavailable before worker device "
                "initialization. Call check_ascend_device_type after set_device."
            )
        _ascend_device_type = _device_type_from_build_soc(build_soc_version)
    return _ascend_device_type


def check_ascend_device_type() -> None:
    """Validate the selected NPU after the worker calls ``set_device``."""
    global _ascend_device_type
    runtime_device_type = _runtime_device_type()
    build_soc_version = _build_soc_version()
    if build_soc_version is None:
        _ascend_device_type = runtime_device_type
        return

    build_device_type = get_ascend_device_type()
    assert build_device_type == runtime_device_type, (
        f"Current device type: {runtime_device_type} does not match the installed "
        f"version's device type: {build_device_type}, please check your installation package."
    )
