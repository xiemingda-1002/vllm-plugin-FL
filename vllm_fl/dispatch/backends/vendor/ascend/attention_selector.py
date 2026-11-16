# SPDX-License-Identifier: Apache-2.0
"""Select coherent sparse attention providers before cache or graph creation."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend


@cache
def get_minimax_sparse_kernels():
    """Pin one stateful M3 provider before caches and graphs are created.

    The paged-v1 ABI includes explicitly selected plane/block-major main K/V,
    key-only index cache, padded negative slots, and in-place output writes.
    All cache, indexer and attend operations belong to the same provider.
    Selection can fall back before execution; stateful methods must not.
    """
    from vllm_fl.dispatch import call_op

    kernels = call_op("minimax_m3_sparse_kernels")
    if not isinstance(kernels, type):
        raise TypeError("MiniMax sparse provider must select a kernel bundle class")
    if getattr(kernels, "cache_contract", None) != "minimax_m3_paged_v1":
        raise ValueError("MiniMax provider must preserve minimax_m3_paged_v1 caches")
    if getattr(kernels, "replay_contract", None) != "persistent_metadata_v1":
        raise ValueError("MiniMax provider must preserve persistent metadata replay")
    for method in (
        "insert_main_cache", "insert_index_cache", "run_attention", "attend",
        "qknorm_rope_cache", "rotary", "gemma_weight", "activation",
        "index_score", "index_topk", "index_decode",
    ):
        if not callable(getattr(kernels, method, None)):
            raise TypeError(f"MiniMax sparse provider must supply {method}")
    return kernels


def native_dsa_backend() -> type[AttentionBackend]:
    """Default provider; import the native backend only when selected."""
    from .impl.attention.dsa_v1 import AscendDSABackend

    return AscendDSABackend


def native_sfa_backend() -> type[AttentionBackend]:
    """Import the native SFA implementation only when selected."""
    from .impl.attention.sfa_v1 import AscendSFABackend

    return AscendSFABackend


@cache
def get_sfa_backend() -> type[AttentionBackend]:
    """Pin the main SFA and its cache-only indexer as one provider.

    Providers preserve the existing tuple cache and metadata ABI. Register
    providers and set policy before first use; cache or graph creation makes
    changing that selection unsafe without a fresh process.
    """
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_fl.dispatch import call_op

    backend = call_op("sparse_mla_attention_backend")
    if not isinstance(backend, type) or not issubclass(backend, AttentionBackend):
        raise TypeError("SFA provider must select an AttentionBackend class")
    if getattr(backend, "ascend_cache_contract", None) != "mla_tuple_v1":
        raise ValueError("SFA provider must preserve the mla_tuple_v1 cache contract")
    if getattr(backend.get_builder_cls(), "ascend_metadata_protocol", None) != "sfa_v1":
        raise ValueError("SFA provider must preserve the sfa_v1 metadata protocol")
    if getattr(backend.get_impl_cls(), "ascend_mla_cache_layout", None) != "mla_tuple_v1":
        raise ValueError("SFA implementation must preserve the mla_tuple_v1 cache layout")
    get_indexer = getattr(backend, "get_indexer_backend_cls", None)
    if not callable(get_indexer):
        raise TypeError("SFA provider must supply its cache-only indexer backend")
    indexer = get_indexer()
    if not isinstance(indexer, type) or not issubclass(indexer, AttentionBackend):
        raise TypeError("SFA indexer provider must select an AttentionBackend class")
    return backend


@cache
def get_dsa_backend() -> type[AttentionBackend]:
    """Pin one provider for the layer and its auxiliary caches in this process.

    The current planner, specs and binding implement the nonpacked V4 cache
    contract. A replacement must preserve it, including metadata build and
    graph update semantics. Different layouts need their own complete cache
    integration, not just a new attention kernel. Install providers and policy
    before first use; changing them requires a fresh process.
    """
    from vllm.v1.attention.backend import AttentionBackend

    from vllm_fl.dispatch import call_op

    backend = call_op("deepseek_v4_attention_backend")
    if not isinstance(backend, type) or not issubclass(backend, AttentionBackend):
        raise TypeError("DSA provider must select an AttentionBackend class")
    if getattr(backend, "ascend_cache_contract", None) != "deepseek_v4_nonpacked_v1":
        raise ValueError(
            "DSA provider must preserve the deepseek_v4_nonpacked_v1 cache contract"
        )
    builder = backend.get_builder_cls()
    if getattr(builder, "ascend_metadata_protocol", None) not in ("dsa_v1", "dsa_cp_v1"):
        raise ValueError(
            "DSA provider must declare a supported DSA metadata protocol"
        )
    return backend
