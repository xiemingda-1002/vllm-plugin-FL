# Copyright (c) 2026 BAAI. All rights reserved.
"""Ascend top-k selection for the MiniMax-M3 lightning indexer.

The single measured deviation from upstream
-------------------------------------------
Upstream ``common/ops/index_topk.py`` selects sparse blocks with a **bitonic
sort** written in Triton (``_compare_and_swap`` / ``_bitonic_merge``,
``tl.standard._log2``, ``tl.core.get_int_dtype`` — 17 references). Measured on
A3, that kernel does not finish compiling within 28 minutes
(``bishengir-compile`` stays busy indefinitely), whereas the *score* kernels in
the same file compile in ~4 s and run correctly. This is the only reason we
deviate, and it is the same conclusion ``vllm-ascend`` reached when it moved
these paths to ``torch.topk``.

Reused vs replaced
------------------

=================================  =========================================
upstream symbol                    Ascend treatment
=================================  =========================================
``minimax_m3_index_score``         **reused as-is** (returns scores, no sort)
``_index_block_score_kernel``      **reused**
``_decode_index_score_kernel``     **reused** (called here with 1 chunk)
``minimax_m3_index_topk``          **replaced** (bitonic -> ``torch.topk``)
``minimax_m3_index_decode``        **replaced** (bitonic -> ``torch.topk``)
=================================  =========================================

Selected semantics, transcribed from the upstream kernels
---------------------------------------------------------
Prefill (``_topk_index_kernel``, ``MASK_INIT=False`` / ``MASK_LOCAL=False``):

* ``valid_blocks = (prefix_len + q_index + block_size) // block_size``
  (``sample_interval == 1``);
* blocks ``>= valid_blocks`` are invalid and are emitted as ``-1``;
* init blocks (``blk < init_blocks``) are forced to ``1e30``; local blocks
  (``blk >= max(0, valid_blocks - local_blocks)``) are forced to ``1e29`` and
  win over init blocks (the two ``tl.where`` are applied init-then-local);
* the emitted index is the **0-based block id**, ``-1`` for invalid.

Decode (``_decode_index_score_kernel``) bakes the same forcing into the scores
(``where(is_local, 1e29, where(is_init, 1e30, score))`` — local preferred), with

* ``num_blocks_q = ceil(kv_len / 128)``, ``kv_len = max(query_pos + 1, 0)``,
  ``query_pos = seq_len - decode_query_len + q_offset``;
* only the ordering matters (the score scale is dropped).

.. warning::
   Not yet validated numerically: the A3 devices were occupied while this was
   written. Validating against the upstream reference
   (``tests/kernels/...`` / ``reference/vllm_cp``) is the first item of the
   MiniMax-M3 test plan.
"""

from __future__ import annotations

import logging

import torch
import triton

logger = logging.getLogger(__name__)

# Must match upstream ``index_topk.SPARSE_BLOCK_SIZE``.
SPARSE_BLOCK_SIZE = 128


def _topk_from_scores(
    score: torch.Tensor,      # [num_heads, total_q, max_block] fp32, scratch
    valid_blocks: torch.Tensor,  # [total_q] int32
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Force init/local blocks in, then select ``topk`` per query token.

    Mirrors the upstream kernels (``MASK_INIT=False`` / ``MASK_LOCAL=False``):
    init blocks are forced to ``1e30`` and local blocks to ``1e29``, so init
    blocks rank **above** local blocks, and both rank above any real score.
    Forcing applies only to visible blocks; invisible blocks become ``-inf``
    and can never be selected, and tail positions are emitted as ``-1``.

    Note: the relative order of blocks with *equal* forced scores (e.g. two init
    blocks) is not defined here — ``torch.topk`` does not guarantee index order
    for ties, whereas the upstream bitonic network had its own tie behaviour.
    The selected *set* is identical, and the downstream sparse attention sums
    over the selected blocks, so the order does not affect results.
    """
    num_heads, total_q, max_block = score.shape
    device = score.device
    pos = torch.arange(max_block, device=device)

    visible = pos.unsqueeze(0) < valid_blocks.unsqueeze(1)          # [Q, B]
    is_init = pos < init_blocks                                     # [B]
    local_start = torch.clamp(valid_blocks - local_blocks, min=0)   # [Q]
    is_local = pos.unsqueeze(0) >= local_start.unsqueeze(1)         # [Q, B]

    forced = torch.where(
        is_local, torch.full_like(valid_blocks, 1e29, dtype=torch.float32).unsqueeze(1),
        torch.where(
            is_init.unsqueeze(0),
            torch.full_like(valid_blocks, 1e30, dtype=torch.float32).unsqueeze(1),
            torch.zeros_like(valid_blocks, dtype=torch.float32).unsqueeze(1),
        ),
    )  # [Q, B]
    forced = torch.where(visible, forced, torch.full_like(forced, float("-inf")))

    # Only override where something is forced; otherwise keep the raw score.
    needs_force = visible & (is_init.unsqueeze(0) | is_local)
    score = torch.where(
        needs_force.unsqueeze(0), forced.unsqueeze(0).expand_as(score), score
    )
    score = score.masked_fill(~visible.unsqueeze(0), float("-inf"))

    k = min(topk, max_block)
    topk_idx = torch.topk(score, k=k, dim=-1, largest=True, sorted=True).indices
    if k < topk:
        pad = torch.full(
            (num_heads, total_q, topk - k), -1, dtype=torch.int32, device=device
        )
        topk_idx = torch.cat((topk_idx.to(torch.int32), pad), dim=-1)

    # Invisible blocks carry -inf and sort last, but if fewer than `topk`
    # blocks are visible they still appear in the result, so mask by *selected
    # index* rather than by the [Q, max_block] visibility matrix. The -1 pads
    # added above stay -1 because -1 < valid_blocks.
    within_valid = topk_idx < valid_blocks.unsqueeze(0).unsqueeze(-1)  # [1,Q,1]->[H,Q,k]
    topk_idx = torch.where(
        within_valid, topk_idx, torch.full_like(topk_idx, -1)
    )
    return topk_idx.to(torch.int32)


def _decode_visible_blocks(
    seq_lens: torch.Tensor,
    decode_query_len: int,
    total_q: int,
    max_block: int,
) -> torch.Tensor:
    """Visible block count per query token: ``ceil(kv_len / SPARSE_BLOCK_SIZE)``.

    This is the same quantity the reused score kernel computes as
    ``num_blocks_q``, so the two agree by construction; it is needed again here
    to mask block ids that are not yet reachable.
    """
    device = seq_lens.device
    if decode_query_len == 1:
        # Hot path: one query token per request, so query_pos == seq_len - 1 and
        # kv_len == seq_len. Avoids the div/mod that decode_query_len > 1 needs.
        kv_len = seq_lens.to(torch.long)
    else:
        ids = torch.arange(total_q, device=device)
        req = ids // decode_query_len
        kv_len = (
            seq_lens[req].to(torch.long) - decode_query_len
            + (ids - req * decode_query_len) + 1
        ).clamp(min=0)
    return (
        (kv_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    ).clamp(max=max_block).to(torch.int32)


def _decode_topk_from_scores(
    score: torch.Tensor,
    valid_blocks: torch.Tensor,
    topk: int,
    max_block: int,
) -> torch.Tensor:
    """Select block ids for decode, assuming the score is already final.

    The reused upstream decode score kernel already applies the init/local
    forcing *and* leaves ``-inf`` wherever no score was stored (``score`` is
    initialised to ``-inf``), so this path needs no forcing of its own -- it
    only has to emit ``-1`` for slots that are not visible yet.

    That assumption is why :func:`_topk_from_scores` (the general form, used by
    the prefill path where the score kernel does *not* force) is not reused
    here: its ``[Q, B]`` visibility/forcing arithmetic costs ~25 small device
    ops per call, which at 57 layers per decode step dominated the indexer.
    """
    k = min(topk, max_block)
    topk_idx = torch.topk(
        score[:, :, :max_block], k=k, dim=-1, largest=True, sorted=True
    ).indices.to(torch.int32)
    if k < topk:
        topk_idx = torch.cat(
            (
                topk_idx,
                torch.full(
                    (*topk_idx.shape[:-1], topk - k),
                    -1,
                    dtype=torch.int32,
                    device=topk_idx.device,
                ),
            ),
            dim=-1,
        )
    # Fewer blocks may be visible than `topk`, in which case -inf slots are
    # selected too; emit -1 for anything beyond the visible count.
    return topk_idx.masked_fill_(
        topk_idx >= valid_blocks.unsqueeze(0).unsqueeze(-1), -1
    )


def _query_token_layout(
    cu_seqlens_q: torch.Tensor, total_q: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(request_index_per_token, q_offset_within_request)``."""
    device = cu_seqlens_q.device
    batch = cu_seqlens_q.shape[0] - 1
    lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.long)
    req = torch.repeat_interleave(
        torch.arange(batch, device=device), lengths
    )[:total_q]
    q_off = torch.arange(total_q, device=device) - cu_seqlens_q[req].to(torch.long)
    return req, q_off


def minimax_m3_index_topk(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ascend replacement for the prefill index top-k (see module docstring)."""
    num_heads, total_q, max_block = score.shape
    req, q_off = _query_token_layout(cu_seqlens_q, total_q)
    valid = torch.div(
        prefix_lens[req].to(torch.long) + q_off + SPARSE_BLOCK_SIZE,
        SPARSE_BLOCK_SIZE,
        rounding_mode="floor",
    ).clamp(max=max_block).to(torch.int32)

    result = _topk_from_scores(
        score.to(torch.float32), valid, topk, init_blocks, local_blocks
    )
    if out is not None:
        out[:, :total_q, :].copy_(result)
        return out
    return result


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ascend replacement for the decode index top-k (see module docstring).

    Reuses the upstream score kernel with a single chunk (``num_kv_chunks=1`` is
    legal: one CTA covers every block, which is what ``chunk_size_blocks =
    ceil(num_blocks / 1)`` yields), then selects with ``torch.topk``.
    """
    from vllm.models.minimax_m3.common.ops import index_topk as _up

    total_q, num_idx_heads, head_dim = idx_q.shape
    if num_idx_heads != num_kv_heads:
        raise ValueError(
            "M3 expects num_idx_heads == num_kv_heads, got "
            f"{num_idx_heads} vs {num_kv_heads}"
        )
    total_q_expected = seq_lens.shape[0] * decode_query_len
    if total_q != total_q_expected:
        raise ValueError(
            f"total_q {total_q} != batch {seq_lens.shape[0]} * "
            f"decode_query_len {decode_query_len}"
        )

    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    # Keep the last dim 16-divisible as upstream does, to avoid Triton
    # recompiles when the block count changes.
    #
    # -inf, not zero: the score kernel only writes blocks that are visible, so
    # every unwritten slot keeps this initial value. Real scores are dot
    # products and are routinely negative, so a zero-filled slot would outrank
    # them -- and once top-k has picked such a slot the block id is masked away
    # below, silently dropping a block that should have been attended to. This
    # only bites once `max_block` exceeds `topk` with few visible blocks, which
    # is exactly the long-context early-decode case.
    score_block_stride = ((max_block + 15) // 16) * 16
    score = torch.full(
        (num_idx_heads, total_q, score_block_stride),
        float("-inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )
    # Split the block range across CTAs. Each CTA writes a disjoint block range
    # into the shared score buffer, so no merge step is needed and the count
    # depends only on shape constants (cudagraph-safe).
    #
    # The chunk count is capped by the block count, NOT just by a CTA target:
    # upstream's shape-constant default (TARGET_GRID // batch, capped at 256)
    # assumes CUDA, where launching extra CTAs is nearly free. On Ascend with
    # max_seq_len 10240 the block count is only 80, so that formula produced 256
    # chunks -- 176 of which hit `chunk_start >= chunk_end` and exit immediately,
    # while still paying launch cost. Measured per decode step (DP2xTP8+EP, C=1):
    # this kernel cost 4.09 ms with 256 chunks versus 0.32 ms in vLLM-Ascend,
    # which splits the same work over ~16. Keeping >= 4 blocks per chunk lands on
    # that same region.
    decode_ctas = max(1, seq_lens.shape[0])
    target_chunks = min(512 // decode_ctas, max(1, max_block // 4))
    num_kv_chunks = 1 << (max(1, target_chunks).bit_length() - 1)
    _up._decode_index_score_kernel[(seq_lens.shape[0], num_kv_chunks)](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        init_blocks,
        local_blocks,
        decode_query_len,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=triton.next_power_of_2(max_decode_query_len),
        num_kv_chunks=num_kv_chunks,
        USE_PDL=False,
    )

    # Per-token visible block count, matching the score kernel's num_blocks_q.
    valid = _decode_visible_blocks(
        seq_lens, decode_query_len, total_q, max_block
    )

    result = _decode_topk_from_scores(score, valid, topk, max_block)
    if out is not None:
        out[:, :total_q, :].copy_(result)
        return out
    return result


def install_index_topk_replacement() -> bool:
    """Patch upstream ``index_topk`` to use the Ascend top-k. Idempotent."""
    try:
        from vllm.models.minimax_m3.common.ops import index_topk as _up
    except ImportError:
        logger.warning(
            "MiniMax-M3: upstream index_topk not importable; skipping top-k patch"
        )
        return False

    if getattr(_up, "_fl_ascend_topk", False):
        return True

    # Decode uses the vendor kernel split (index_decode_vendor) -- the upstream
    # score kernel blows up once a DP rank serves more than one request.
    #
    # Prefill likewise uses the vendor scalar kernels (index_vendor). Measured
    # on the dual-machine A3 rig at C=64 (unique prompts), the local
    # ``torch.topk`` over the upstream score tensor dominated prefill: FL's
    # batch-process speedup on 64x1k prompts was 5.8x against the vendor's
    # 13.8x, showing up as +66% TTFT. The vendor prefill kernels are linear in
    # the block count and remove that nonlinearity.
    from vllm_fl.dispatch.backends.vendor.ascend.impl.indexer.index_decode_vendor import (
        minimax_m3_index_decode as _decode_vendor,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.indexer.index_vendor import (
        minimax_m3_index_score as _score_vendor,
        minimax_m3_index_topk as _topk_vendor,
    )

    originals = {
        "minimax_m3_index_topk": (_up.minimax_m3_index_topk, _topk_vendor),
        "minimax_m3_index_decode": (_up.minimax_m3_index_decode, _decode_vendor),
        "minimax_m3_index_score": (_up.minimax_m3_index_score, _score_vendor),
    }
    for name, (_, replacement) in originals.items():
        setattr(_up, name, replacement)
    _up._fl_ascend_topk = True  # type: ignore[attr-defined]

    # Patching the defining module is not enough: consumers do
    # ``from ...index_topk import minimax_m3_index_decode`` and therefore hold
    # their own reference, which keeps calling the upstream bitonic kernels.
    # Those cannot run on Ascend at all — the top-k merge kernel exceeds the
    # unified buffer (measured: "ub overflow, requires 2753792 bits while
    # 1572864 bits available"), so leaving them bound aborts graph capture.
    import sys

    rebound = []
    for mod_name, module in list(sys.modules.items()):
        if module is None or not mod_name.startswith("vllm.models.minimax_m3"):
            continue
        for name, (original, replacement) in originals.items():
            if getattr(module, name, None) is original:
                setattr(module, name, replacement)
                rebound.append(f"{mod_name}.{name}")

    logger.info(
        "MiniMax-M3: indexer switched to the vendor kernel chain "
        "(prefill score/top-k + decode split); rebound %d consumer "
        "reference(s). Measured reason: upstream's bitonic kernels cannot run "
        "on Ascend (UB overflow / impractically long compile), and the "
        "torch.topk-over-upstream-score fallback made prefill top-k nonlinear "
        "in block count (+66%% TTFT at C=64)",
        len(rebound),
    )
    if rebound:
        logger.debug("top-k rebound in: %s", ", ".join(sorted(rebound)))
    return True


__all__ = [
    "minimax_m3_index_topk",
    "minimax_m3_index_decode",
    "install_index_topk_replacement",
]
