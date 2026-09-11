"""Load the Ascend custom operators built and packaged by FL itself."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_ENABLED: bool | None = None
_SOC_FAMILIES = {
    "ascend910b": ("910b", "ascend910b"),
    "ascend910_93": ("910c", "ascend910_93"),
}


def _soc_family(value: str) -> str:
    normalized = value.strip().lower()
    for family, prefixes in _SOC_FAMILIES.items():
        if any(normalized.startswith(prefix) for prefix in prefixes):
            return family
    return normalized


def _prepend_path(name: str, path: Path) -> None:
    if not path.exists():
        return
    current = [item for item in os.environ.get(name, "").split(":") if item]
    value = str(path)
    if value not in current:
        current.insert(0, value)
        os.environ[name] = ":".join(current)


def _opp_root(package_dir: Path) -> Path | None:
    root = package_dir / "_cann_ops_custom"
    candidates = (
        root / "vendors" / "custom_transformer",
        root / "custom_transformer",
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def bootstrap_custom_op_env() -> None:
    """Expose FL's packaged OPP before torch_npu initializes custom ops."""

    opp = _opp_root(Path(__file__).resolve().parent)
    if opp is None:
        return
    _prepend_path("ASCEND_CUSTOM_OPP_PATH", opp)
    _prepend_path("LD_LIBRARY_PATH", opp / "op_api" / "lib")


def enable_custom_op() -> bool:
    """Register FL's local ``_C_ascend`` dispatcher library once."""

    global _ENABLED
    if _ENABLED is not None:
        return _ENABLED

    package_dir = Path(__file__).resolve().parent
    opp = _opp_root(package_dir)
    extension = next(iter(sorted(package_dir.glob("_C_ascend*.so"))), None)
    if extension is None or opp is None:
        logger.debug(
            "FL Ascend native payload is not installed for SOC_VERSION=%s",
            os.environ.get("SOC_VERSION", "ascend910_93"),
        )
        _ENABLED = False
        return False

    expected = _soc_family(os.environ.get("SOC_VERSION", "ascend910_93"))
    build_soc_file = package_dir / "_cann_ops_custom" / "FL_SOC_VERSION"
    if build_soc_file.is_file():
        built = _soc_family(build_soc_file.read_text(encoding="utf-8").strip())
        if built != expected:
            raise RuntimeError(
                f"FL Ascend wheel was built for {built}, runtime requests {expected}"
            )

    bootstrap_custom_op_env()
    opapi = opp / "op_api" / "lib" / "libcust_opapi.so"
    try:
        if opapi.is_file():
            torch.ops.load_library(str(opapi))
        torch.ops.load_library(str(extension))
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to load FL-owned Ascend native operators from {extension}"
        ) from exc
    _ENABLED = True
    return True
