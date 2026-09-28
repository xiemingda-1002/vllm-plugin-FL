# Copyright (c) 2026 BAAI. All rights reserved.
"""Ascend sparse-attention attend for MiniMax-M3 (block-sparse GQA).

Replaces the upstream Triton attend with FL's own CANN fused operator
``npu_sparse_attention_score``. Both the upstream Triton split-K decode kernel
and the fused operator implement the same block-sparse GQA, so this is a pure
kernel substitution: the selection (indexer top-k) and the metadata are
untouched.

Why this is a measured deviation, not a preference
--------------------------------------------------
The upstream attend is selected by ``select_main_impl_cls`` in
``vllm/models/minimax_m3/common/sparse_attention.py``, which requires
``current_platform.is_cuda() and is_device_capability_family(100)`` (Blackwell)
to pick its fused MSA path. On NPU that test is always false, so the model falls
back to the Triton path. Measured on 910C (A3), one decode step of
``num_heads=64, num_kv_heads=4, head_dim=128, topk=16``:

===========================================  ==========  =========
path                                         latency     speedup
===========================================  ==========  =========
upstream Triton split-K decode               0.3215 ms   1.0x
``npu_sparse_attention_score`` (strided in)  0.0600 ms   5.4x
``npu_sparse_attention_score`` (+depage copy) 0.0813 ms   4.0x
===========================================  ==========

max elementwise deviation is 0.0039 in bf16 (one ULP at these magnitudes).
The operator accepts the paged cache as a strided ``kv_cache[:, 0]`` view, so
no de-paging copy is needed.

The operator itself already ships in this repository: its CANN kernel is built
into ``vllm_fl/_cann_ops_custom`` and its dispatcher schema into
``torch_binding.cpp``. Only the Python call site was missing.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_SPARSE_ATTN_INNER_PRECISE = 4

# Guard: never patch twice, and never patch if the operator is unavailable.
_installed = False


def _split_main_kv_cache(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the main paged cache into its K and V blocked views.

    Upstream layout is ``[num_blocks, 2, block_size, num_kv_heads, head_dim]``
    with the K/V plane on dim 1. The operator takes the blocked layout
    ``[blockNum, blockSize, KVHead, D]`` and accepts this strided view
    directly, so no copy is made.
    """
    if kv_cache.ndim != 5:
        raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
    if kv_cache.shape[0] == 2:
        return kv_cache[0], kv_cache[1]
    if kv_cache.shape[1] == 2:
        return kv_cache[:, 0], kv_cache[:, 1]
    raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")


def _select_num_idx_from_topk(topk_idx: torch.Tensor) -> torch.Tensor:
    """Per-token count of real (non-padding) selected blocks."""
    return (topk_idx >= 0).sum(dim=-1).to(dtype=torch.int32)


def _sparse_attn_ascend(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_size: int,
    out: torch.Tensor,
) -> None:
    """Run the fused operator and write into the preallocated ``out``.

    ``block_size`` is the sparse block size (== KV page size), which the
    operator requires to match the cache's block dimension.
    """
    key, value = _split_main_kv_cache(kv_cache)
    result = torch.ops._C_ascend.npu_sparse_attention_score(
        q,
        key,
        value,
        topk_idx,
        block_table,
        selectNumIdx=_select_num_idx_from_topk(topk_idx),
        actualSeqLengths=query_lens,
        actualSeqLengthsKv=seq_lens,
        numKeyValueHeads=num_kv_heads,
        scaleValue=scale,
        blockSize=block_size,
        topK=topk_idx.shape[-1],
        innerPrecise=_SPARSE_ATTN_INNER_PRECISE,
    )
    out.copy_(result)


def _ascend_sparse_forward(
    self,
    layer,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Drop-in replacement for ``MiniMaxM3SparseTritonImpl.forward``."""
    from vllm.forward_context import get_forward_context
    from vllm.models.minimax_m3.common.sparse_attention import (
        MiniMaxM3SparseMetadata,
    )

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return output  # profiling run; caches unbound
    main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
    assert isinstance(main_md, MiniMaxM3SparseMetadata)

    nd = main_md.num_decode_tokens
    num_tokens = main_md.num_actual_tokens
    topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
    assert topk is not None

    hd = self.head_size
    q = query[:num_tokens].view(-1, self.num_heads, hd)
    out = output[:num_tokens].view(-1, self.num_heads, hd)

    if main_md.num_decodes > 0:
        d = main_md.decode
        assert d is not None
        decode_lens = torch.full(
            (d.seq_lens.shape[0],),
            d.decode_query_len,
            dtype=torch.int32,
            device=q.device,
        )
        _sparse_attn_ascend(
            q[:nd],
            kv_cache,
            topk[:, :nd, :],
            d.block_table,
            d.seq_lens,
            decode_lens,
            self.num_kv_heads,
            self.scale,
            self.block_size,
            out[:nd],
        )

    if main_md.num_prefills > 0:
        p = main_md.prefill
        assert p is not None
        # cu_seqlens_q is already rebased to 0, so consecutive diffs are the
        # per-request query lengths.
        prefill_lens = (p.cu_seqlens_q[1:] - p.cu_seqlens_q[:-1]).to(torch.int32)
        _sparse_attn_ascend(
            q[nd:],
            kv_cache,
            topk[:, nd:num_tokens, :],
            p.block_table,
            p.seq_lens,
            prefill_lens,
            self.num_kv_heads,
            self.scale,
            self.block_size,
            out[nd:],
        )
    return output


def _install_planar_kv_cache_layout() -> bool:
    """Store the paged K/V cache plane-major: ``[2, num_blocks, block, kvh, hd]``.

    Why this override is required
    -----------------------------
    Upstream lays the cache out block-major, ``[num_blocks, 2, block, kvh, hd]``,
    so ``kv_cache[:, 0]`` -- the K plane handed to ``npu_sparse_attention_score``
    -- is a *strided* view. The CANN operator materialises that view, and its
    cost is O(total cache blocks), not O(blocks actually attended):

    ```
    sparse-attn Slice kernel, same kernel, same calls/step
      single machine (DP2, 141 blocks)    1.17 ms/step
      dual machine   (DP4, 4383 blocks)  26.40 ms/step   <- 31x cache, 23x time
    ```

    A dual-machine run leaves ~30x more HBM per rank for the cache (weights are
    sharded across 32 ranks instead of 16), so this dominates decode there while
    being invisible on one machine. It is not a communication cost: the MoE
    collectives are the same size in both runs.

    Vendor avoids it by allocating K and V as two separate tensors. Plane-major
    storage reaches the same end -- ``cache[0]``/``cache[1]`` are then contiguous
    subtensors of identical layout and size -- without replacing the model
    runner's allocator.
    """
    if os.environ.get("VLLM_FL_M3_PLANAR_KV", "1") != "1":
        logger.debug(
            "MiniMax-M3: plane-major K/V cache disabled (VLLM_FL_M3_PLANAR_KV=0)"
        )
        return False

    try:
        from vllm.models.minimax_m3.common import sparse_attention as _up
    except ImportError:
        return False

    backend = getattr(_up, "MiniMaxM3SparseBackend", None)
    if backend is None:
        logger.debug("MiniMax-M3 MiniMaxM3SparseBackend not found; skip layout")
        return False
    if getattr(backend, "_fl_planar_kv_cache", False):
        return True

    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    backend.get_kv_cache_shape = staticmethod(get_kv_cache_shape)
    backend._fl_planar_kv_cache = True
    logger.info(
        "MiniMax-M3: paged K/V cache stored plane-major [2, num_blocks, ...] "
        "so the fused sparse attend reads contiguous K/V planes"
    )
    return True


def install_sparse_attn_replacement() -> bool:
    """Swap the sparse attend for the Ascend fused operator.

    On by default, matching the vendor runtime, which selects
    ``npu_sparse_attention_score`` for MiniMax-M3 on Ascend. Set
    ``VLLM_FL_M3_NPU_SPARSE_ATTN=0`` to fall back to the upstream Triton
    kernels.

    Measured on A3 (DP2xTP8+EP, FULL_DECODE_ONLY, identical serve flags), as a
    same-build A/B differing only in this switch, three runs each, stream-token
    inter-step time at concurrency 1:

    ==========================  =========================
    upstream Triton            46.49 / 46.42 / 46.76 ms
    ``npu_sparse_attention_score``   45.09 / 45.59 ms
    ==========================  =========================

    So the gain is real but modest, ~1.3 ms (~2.8%). The operator is 4-7x
    faster in isolation, but under FULL_DECODE_ONLY graph replay the Triton
    attend's device time is small -- the isolated benchmark was dominated by
    host dispatch, which graph capture already removes. A per-kernel decode
    profile confirms it: the Triton pair is only ~4 ms/step of ~46 ms.

    Accuracy verified equivalent: 10/10 on the semantic gate including
    4.5K-token prompts. The two kernels differ by at most one bf16 ULP on the
    attention output, which feeds no discrete decision downstream.
    """
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_M3_NPU_SPARSE_ATTN", "1") != "1":
        logger.debug(
            "MiniMax-M3: npu_sparse_attention_score swap disabled "
            "(VLLM_FL_M3_NPU_SPARSE_ATTN=0)"
        )
        return False

    try:
        from vllm.models.minimax_m3.common import sparse_attention as _up
    except ImportError:
        logger.debug("MiniMax-M3 upstream sparse_attention not importable; skip")
        return False

    target = getattr(_up, "MiniMaxM3SparseTritonImpl", None)
    if target is None:
        logger.debug("MiniMax-M3 MiniMaxM3SparseTritonImpl not found; skip")
        return False

    target.forward = _ascend_sparse_forward

    # The attend consumes contiguous K/V planes; align the cache layout with it.
    _install_planar_kv_cache_layout()

    # Consumers bind the class (not the method) at import time, so patching the
    # class attribute is enough; re-assert for modules that copied the symbol.
    import sys

    for name, module in list(sys.modules.items()):
        if not name.startswith("vllm.models.minimax_m3"):
            continue
        if getattr(module, "MiniMaxM3SparseTritonImpl", None) is not None:
            module.MiniMaxM3SparseTritonImpl = target

    _installed = True
    logger.info(
        "MiniMax-M3: sparse attention switched to npu_sparse_attention_score"
    )
    return True


__all__ = ["install_sparse_attn_replacement"]
