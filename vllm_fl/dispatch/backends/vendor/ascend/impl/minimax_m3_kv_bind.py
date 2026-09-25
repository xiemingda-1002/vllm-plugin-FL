# Copyright (c) 2026 BAAI. All rights reserved.
"""Ascend-compatible ``bind_kv_cache``.

Upstream limitation
-------------------
``vllm.v1.worker.utils.bind_kv_cache`` refuses to run on any platform that is
not CUDA-alike, XPU or CPU when a single decoder layer owns more than one
attention layer:

```
for layer_index in sorted(index2name.keys()):
    layer_names = index2name[layer_index]
    if len(layer_names) > 1:
        if (is_cuda_alike() or is_xpu() or is_cpu()):
            pass
        else:
            raise NotImplementedError          # <-- hit on NPU
```

MiniMax-M3 always hits this: every sparse decoder layer registers **two** caches
— the main paged K/V cache and the lightning-indexer key cache — so
``index2name[layer_index]`` holds two names.

The guard exists only because upstream could not vouch for other backends; the
binding work itself is platform independent (append each cache to
``runner_kv_caches`` and attach it to its layer in the forward context). vllm-ascend
replaces the function for exactly this reason, documenting it as
"``bind_kv_cache`` func will raise an exception when current_platform is npu".

This module installs the same removal of that guard, without touching the rest
of the upstream behaviour.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)

_installed = False


def _bind_kv_cache_ascend(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, object],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """Platform-neutral re-implementation of upstream ``bind_kv_cache``.

    Identical to upstream except that a decoder layer holding several attention
    layers is supported on every platform instead of raising on non-CUDA ones.
    """
    from vllm.model_executor.models.utils import extract_layer_index

    assert len(runner_kv_caches) == 0, (
        "runner_kv_caches must be empty before binding, got "
        f"{len(runner_kv_caches)} entries"
    )

    index2name: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        # Several attention layers per decoder block (MiniMax-M3: main + indexer).
        # Every one of them owns its own cache entry, so simply bind them all.
        for layer_name in index2name[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


def install_bind_kv_cache_patch() -> bool:
    """Replace ``bind_kv_cache`` with the platform-neutral version. Idempotent."""
    global _installed
    if _installed:
        return True

    try:
        import vllm.v1.worker.utils as utils_mod
    except ImportError:
        logger.warning("MiniMax-M3: vllm.v1.worker.utils not importable; skipping")
        return False

    original = getattr(utils_mod, "bind_kv_cache", None)
    if original is None:
        logger.warning("MiniMax-M3: bind_kv_cache not found; skipping")
        return False
    if getattr(original, "_fl_ascend_replacement", False):
        _installed = True
        return True

    _bind_kv_cache_ascend._fl_ascend_replacement = True  # type: ignore[attr-defined]
    _bind_kv_cache_ascend._fl_ascend_original = original  # type: ignore[attr-defined]
    utils_mod.bind_kv_cache = _bind_kv_cache_ascend

    # gpu_model_runner (and anything else) imported the symbol directly, so
    # rebind the already-materialised references too.
    import sys

    rebased = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("vllm."):
            continue
        if getattr(module, "bind_kv_cache", None) is original:
            setattr(module, "bind_kv_cache", _bind_kv_cache_ascend)
            rebased.append(name)

    logger.info(
        "MiniMax-M3: patched bind_kv_cache for multi-attention layers on OOT "
        "(rebound %d module(s)); upstream raises NotImplementedError on NPU",
        len(rebased),
    )
    _installed = True
    return True


__all__ = ["install_bind_kv_cache_patch"]
