# Copyright (c) 2026 BAAI. All rights reserved.
"""Layer-local quantization metadata for MiniMax's five-way QKV projection."""

from collections.abc import Mapping
from copy import copy
from functools import wraps
from inspect import signature

_QKV = ("q_proj", "k_proj", "v_proj")
_INDEXER_QKV = (*_QKV, "index_q_proj", "index_k_proj")


def indexer_quantization_config(quant_config):
    """Keep dense/vision rules intact while matching every sparse QKV shard."""
    if quant_config is None:
        return None
    mapping = getattr(quant_config, "packed_modules_mapping", {})
    if not isinstance(mapping, Mapping):
        raise TypeError("MiniMax fused projection metadata must be a mapping")
    parts = mapping.get("qkv_proj")
    if parts is not None and tuple(parts) not in (_QKV, _INDEXER_QKV):
        raise ValueError("Unsupported MiniMax indexer QKV projection mapping")
    local_config = copy(quant_config)
    local_config.packed_modules_mapping = {
        **mapping,
        "qkv_proj": list(_INDEXER_QKV),
    }
    return local_config


def install_indexer_quantization_mapping(model_module) -> None:
    """Adapt only MiniMax's consumer; leave the global linear class unchanged."""
    original = model_module.MinimaxM3QKVParallelLinearWithIndexer
    if getattr(original, "_fl_indexer_mapping", False):
        return

    def indexer_linear(
        hidden_size,
        head_size,
        total_num_heads,
        total_num_kv_heads,
        total_num_index_heads,
        index_head_size,
        bias=False,
        quant_config=None,
        prefix="",
    ):
        return original(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            total_num_index_heads=total_num_index_heads,
            index_head_size=index_head_size,
            bias=bias,
            quant_config=indexer_quantization_config(quant_config),
            prefix=prefix,
        )

    # Delegate to the original class factory, including any existing OOT
    # replacement. Returned layer identities and type-name targets stay native.
    expected = signature(indexer_linear).parameters
    actual = dict(signature(original.__init__).parameters)
    actual.pop("self", None)
    if list(actual) != list(expected) or any(
        (actual[name].kind, actual[name].default) != (parameter.kind, parameter.default)
        for name, parameter in expected.items()
    ):
        raise RuntimeError("Unsupported MiniMax indexer linear constructor contract")
    indexer_linear = wraps(original, updated=())(indexer_linear)
    indexer_linear._fl_indexer_mapping = True
    model_module.MinimaxM3QKVParallelLinearWithIndexer = indexer_linear
