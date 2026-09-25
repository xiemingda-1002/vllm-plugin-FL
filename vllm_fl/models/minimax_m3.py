# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-M3 entry point for vllm-plugin-FL on Ascend.

vLLM 0.24 already ships a complete MiniMax-M3 implementation, so unlike the
Qwen3.5 shim in this directory there is no missing upstream support to add.
What Ascend needs is platform behaviour plus a compile-capable model class:
see :mod:`vllm_fl.dispatch.backends.vendor.ascend.patches.patch_minimax_m3`,
which installs the injections and registers these architectures.

The classes are re-exported here so the registry can resolve them lazily by
``vllm_fl.models.minimax_m3:<Class>``, matching how the other migrated runtimes
are exposed.
"""

from vllm_fl.models.minimax_m3_ascend import (
    AscendMiniMaxM3Model,
    AscendMiniMaxM3SparseForCausalLM,
)

__all__ = [
    "AscendMiniMaxM3Model",
    "AscendMiniMaxM3SparseForCausalLM",
]
