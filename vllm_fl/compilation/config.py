# Copyright (c) 2026 BAAI. All rights reserved.

"""Authoritative configuration for FL's Ascend compilation closure."""

from __future__ import annotations

from typing import Final


COMPILATION_PASS_KEY: Final = "graph_fusion_manager"

ASCEND_COMPILATION_DEFAULTS: Final[dict[str, bool]] = {
    "enable_npugraph_ex": True,
    "enable_static_kernel": False,
    "fuse_norm_quant": True,
    "fuse_qknorm_rope": True,
    "fuse_allreduce_rms": False,
    "fuse_muls_add": True,
}


def ascend_compilation_defaults() -> dict[str, bool]:
    """Return an isolated copy suitable for insertion into VllmConfig."""
    return dict(ASCEND_COMPILATION_DEFAULTS)
