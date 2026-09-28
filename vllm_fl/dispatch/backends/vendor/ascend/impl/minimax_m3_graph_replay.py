# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-M3 attention backends: graph-replay capability conformance.

Why this exists
---------------
The platform runtime resolves the replay-time graph task updaters from the
active attention backends and fails closed: any backend that returns a
*concrete* implementation (i.e. overrides ``get_impl_cls``) must expose
``update_graph_params``. Backends that are purely cache carriers are expected
to inherit the abstract ``get_impl_cls`` and are skipped.

Upstream MiniMax-M3 violates that classification for two backends, because
its model deliberately merges attention wiring into the layer:

* layers 3..59 use ``MiniMaxM3SparseBackend`` -> ``MiniMaxM3SparseImpl``;
* each sparse layer's index cache uses ``MiniMaxM3IndexerBackend`` ->
  ``MiniMaxM3IndexerImpl``.

Both return a concrete class but neither needs replay-time rebinding, and
neither is a cache-only stub either. So the runtime's capability probe rejects
them at startup -- even under ``cudagraph_mode=NONE``, because the resolver
runs during runner setup regardless of the graph mode.

Why a no-op updater is the correct answer, not a workaround
-----------------------------------------------------------
On Ascend the sparse attend is served by this repository's fused
``npu_sparse_attention_score`` (see ``attention/ascend/sparse_attn_m3.py``) and
the indexer writes top-k block ids into a shared persistent buffer. Both read
only vLLM-owned, persistent, in-place-updated metadata: block tables, sequence
lengths, query lengths and the top-k buffer keep the same storage across
steps, and the graph-parameter registry therefore holds no record for them
(unlike the regular Ascend PagedAttention / FIA path, which must re-issue its
workspace handles per shape). Under ``FULL_DECODE_ONLY`` the previous runtime
never rebound these either -- it only updated the regular attention impl --
which is the behaviour verified for accuracy and performance on 910C.

Installing an explicit no-op therefore *documents* that contract and keeps the
replay loop identical, while satisfying the runtime's fail-closed probe.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_installed = False


def _noop_update_graph_params(
    update_stream,
    forward_context,
    num_tokens: int,
    vllm_config,
) -> None:
    """M3 sparse attend and indexer need no replay-time task rebinding."""


def install_minimax_m3_graph_conformance() -> bool:
    """Declare the M3 sparse/indexer impls as replay-safe (no-op updater).

    Idempotent. Returns ``True`` when the upstream M3 impls were found and the
    contract was declared. Never overrides an updater a class already defines.
    """
    global _installed
    if _installed:
        return True

    try:
        from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerImpl
        from vllm.models.minimax_m3.common.sparse_attention import (
            MiniMaxM3SparseImpl,
        )
    except ImportError:
        logger.debug(
            "MiniMax-M3 graph conformance: upstream M3 impls not importable; skip"
        )
        return False

    installed: list[str] = []
    for impl_cls in (MiniMaxM3SparseImpl, MiniMaxM3IndexerImpl):
        if callable(getattr(impl_cls, "update_graph_params", None)):
            continue
        impl_cls.update_graph_params = staticmethod(_noop_update_graph_params)
        installed.append(impl_cls.__name__)

    if installed:
        logger.info(
            "MiniMax-M3: declared replay-safe (no-op graph update) for %s",
            ", ".join(installed),
        )
    _installed = True
    return True


__all__ = ["install_minimax_m3_graph_conformance"]
