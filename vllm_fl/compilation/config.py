# Copyright (c) 2026 BAAI. All rights reserved.

"""Options for Ascend's optional FX fusions, independent of graph capture."""

from __future__ import annotations

from typing import Final

COMPILATION_PASS_KEY: Final = "graph_fusion_manager"

ASCEND_COMPILATION_DEFAULTS: Final[dict[str, bool]] = {
    "enable_npugraph_ex": False,
    "enable_static_kernel": False,
    "fuse_norm_quant": True,
    "fuse_qknorm_rope": True,
    "fuse_allreduce_rms": False,
    "fuse_muls_add": True,
}


def ascend_compilation_defaults() -> dict[str, bool]:
    """Return an isolated copy suitable for insertion into VllmConfig."""
    return dict(ASCEND_COMPILATION_DEFAULTS)


def validate_ascend_compilation_options(raw: dict) -> dict[str, bool]:
    """Validate compatibility keys without silently accepting retired backends."""
    if not isinstance(raw, dict):
        raise TypeError("additional_config.ascend_compilation_config must be a dict")
    unknown = sorted(set(raw) - set(ASCEND_COMPILATION_DEFAULTS))
    if unknown:
        raise ValueError(f"Unsupported FL Ascend compiler options: {unknown}")
    options = ascend_compilation_defaults()
    for name, value in raw.items():
        if type(value) is not bool:
            raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        options[name] = value
    for name in ("enable_npugraph_ex", "enable_static_kernel", "fuse_allreduce_rms"):
        if options[name]:
            raise NotImplementedError(
                f"FL Ascend FX fusion does not support {name}=True"
            )
    return options
