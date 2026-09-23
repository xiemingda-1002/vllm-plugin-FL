"""Ascend multi-cache binding from rc1's patch_qwen3_next_mtp.

SFA has main MLA and indexer caches in the same transformer layer. Bind
both without the upstream GPU/CPU-only multi-cache platform assertion.
This helper is selected by the FL runner, not patched into upstream globals.
"""

from collections import defaultdict
from typing import Any

from vllm.model_executor.models.utils import extract_layer_index


def bind_sfa_kv_cache(
    kv_caches: dict[str, Any],
    forward_context: dict[str, Any],
    runner_kv_caches: list[Any],
    num_attn_module: int = 1,
) -> None:
    assert not runner_kv_caches
    index2name: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)
    for layer_index in sorted(index2name):
        for layer_name in index2name[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache
