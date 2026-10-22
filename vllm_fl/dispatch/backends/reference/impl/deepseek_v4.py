# SPDX-License-Identifier: Apache-2.0
"""Torch reference contracts for continuous paged DeepSeek V4 caches.

Projection, page allocation, query metadata and output-row scheduling belong to
the caller. These providers own numerical work and cache writes only; they do
not depend on an Ascend extension, runner or process group.
"""

import torch


def _paged_read(cache, table, requests, positions):
    """Gather [rows, slots, dim], masking padding before any arithmetic."""
    if cache.ndim != 3 or table.ndim != 2:
        raise ValueError(
            "Expected [pages, block, dim] cache and [requests, blocks] table"
        )
    if positions.ndim != 2 or requests.shape != positions.shape[:1]:
        raise ValueError("Expected one request ID per positions row")
    block = cache.shape[1]
    logical = positions.div(block, rounding_mode="floor")
    valid = (
        (requests[:, None] >= 0)
        & (requests[:, None] < table.shape[0])
        & (positions >= 0)
        & (logical < table.shape[1])
    )
    pages = table[
        requests.long().clamp(0, table.shape[0] - 1)[:, None],
        logical.long().clamp(0, table.shape[1] - 1),
    ].long()
    valid = valid & (pages >= 0) & (pages < cache.shape[0])
    values = cache[
        pages.clamp(0, cache.shape[0] - 1), positions.long().clamp_min(0) % block
    ]
    return torch.where(valid[..., None], values, 0), valid


def _lengths(lengths, requests):
    return lengths[requests.long().clamp(0, lengths.shape[0] - 1)]


def deepseek_v4_save_state(
    kv, gate, ape, positions, slot_mapping, state_cache, compress_ratio
):
    """Save unique valid slots as FP32 [kv, gate + APE]; negative slots are padding."""
    if compress_ratio not in (4, 128):
        raise ValueError("Compression ratio must be 4 or 128")
    if state_cache.dtype != torch.float32 or state_cache.ndim != 3:
        raise ValueError("Compressor state must be FP32 [pages, block, packed dim]")
    if kv.ndim != 2 or gate.shape != kv.shape:
        raise ValueError(
            "KV and gate projections must have matching [tokens, width] shape"
        )
    if kv.dtype != torch.float32 or gate.dtype != torch.float32:
        raise ValueError("Compressor projections must retain FP32 accumulation outputs")
    if ape.shape != (compress_ratio, kv.shape[-1]):
        raise ValueError("APE shape must match compression ratio and projection width")
    if positions.shape != kv.shape[:1] or slot_mapping.shape != positions.shape:
        raise ValueError("Expected one position and slot per projected token")
    if state_cache.shape[-1] != 2 * kv.shape[-1]:
        raise ValueError("State width must pack both KV and gate projections")
    if kv.shape[0] == 0:
        return
    packed = torch.cat(
        [kv.float(), gate.float() + ape[positions.long() % compress_ratio].float()], -1
    )
    valid = slot_mapping >= 0
    slots = slot_mapping.long().clamp_min(0)
    packed = torch.where(valid[:, None], packed, 0)
    flat = state_cache.reshape(-1, state_cache.shape[-1])
    # A masked index_copy to slot zero races with a real slot-zero write.
    # Accumulating only nonzero valid contributions avoids that padding race,
    # including NaNs in padded projections, without data-dependent shapes.
    updates = torch.zeros_like(flat).index_add_(0, slots, packed)
    touched = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.int32)
    touched.index_add_(0, slots, valid.to(torch.int32))
    state_cache.copy_(
        torch.where(touched[:, None] > 0, updates, flat).reshape(state_cache.shape)
    )


def deepseek_v4_compress(
    state_cache,
    block_table,
    request_ids,
    positions,
    rms_weight,
    cos_sin_cache,
    compress_ratio=4,
    rms_eps=1e-6,
    output_dtype=torch.bfloat16,
):
    """Compress completed groups at original end positions; padded rows are zero.

    C4 mixes the previous group's first half with the current group's second
    half. C128 has no overlap. Softmax is per channel over source tokens; RMSNorm
    and adjacent-pair RoPE precede the single output cast. This is continuous
    paged state, not a cyclic state-cache implementation.
    """
    if compress_ratio not in (4, 128):
        raise ValueError("Compression ratio must be 4 or 128")
    dim = rms_weight.numel()
    coff = 2 if compress_ratio == 4 else 1
    if state_cache.dtype != torch.float32 or state_cache.shape[-1] != 2 * coff * dim:
        raise ValueError("Unexpected compressor state dtype or packed width")
    rope_dim = cos_sin_cache.shape[-1]
    if rope_dim % 2 or rope_dim > dim:
        raise ValueError("RoPE width must be even and no larger than the head")
    if positions.shape[0] == 0:
        return state_cache.new_empty((0, dim), dtype=output_dtype)
    total = coff * compress_ratio
    source = (
        positions[:, None] - total + 1 + torch.arange(total, device=positions.device)
    )
    raw, valid = _paged_read(state_cache, block_table, request_ids, source)
    raw = raw.reshape(positions.shape[0], total, 2, coff, dim)
    part = torch.arange(total, device=positions.device) // compress_ratio
    part = part[None, :, None, None].expand(positions.shape[0], total, 1, dim)
    kv = raw[:, :, 0].gather(2, part).squeeze(2)
    scores = raw[:, :, 1].gather(2, part).squeeze(2)
    rope_position = (
        positions.div(compress_ratio, rounding_mode="floor") * compress_ratio
    )
    active = (
        (positions >= 0)
        & (request_ids >= 0)
        & ((positions + 1) % compress_ratio == 0)
        & (rope_position < cos_sin_cache.shape[0])
    )
    scores = scores.masked_fill(~valid[..., None], -torch.inf)
    has_source = valid.any(-1) & active
    scores = torch.where(has_source[:, None, None], scores, 0)
    compressed = (kv * scores.softmax(dim=1)).sum(dim=1)
    normed = compressed * torch.rsqrt(
        compressed.square().mean(-1, keepdim=True) + rms_eps
    )
    normed = normed * rms_weight.float()
    cache_pos = rope_position.long().clamp(0, cos_sin_cache.shape[0] - 1)
    cos, sin = cos_sin_cache[cache_pos].float().chunk(2, -1)
    nope, rope = normed.split([dim - rope_dim, rope_dim], -1)
    even, odd = rope[:, 0::2], rope[:, 1::2]
    rotated = torch.stack([even * cos - odd * sin, odd * cos + even * sin], -1)
    result = torch.cat([nope, rotated.flatten(1)], -1)
    return torch.where(has_source[:, None], result, 0).to(output_dtype)


def deepseek_v4_indexer(
    query,
    key_cache,
    weights,
    block_table,
    request_ids,
    positions,
    key_lengths,
    topk,
    compress_ratio=4,
    query_scale=None,
    key_scale_cache=None,
):
    """C4 weighted ReLU index scores and causal logical compact-row IDs.

    Weights include model-side projection/head scaling. Quantized Q/K scales are
    dequantization scales, not additional attention scaling. C128 attends every
    completed compact row and does not use this indexer.
    """
    if compress_ratio != 4 or topk < 1:
        raise ValueError("The compressed indexer requires C4 and positive topk")
    if query.ndim != 3 or weights.shape != query.shape[:2]:
        raise ValueError(
            "Expected [tokens, heads, dim] queries and [tokens, heads] weights"
        )
    if query.dtype == torch.int8 and query_scale is None:
        raise ValueError("INT8 queries require per-token/head scales")
    if key_cache.dtype == torch.int8 and key_scale_cache is None:
        raise ValueError("INT8 keys require per-row scales")
    if query.shape[0] == 0:
        return torch.empty((0, topk), dtype=torch.int32, device=query.device)
    capacity = block_table.shape[1] * key_cache.shape[1]
    logical = torch.arange(capacity, device=query.device)[None].expand(
        query.shape[0], -1
    )
    keys, valid = _paged_read(key_cache, block_table, request_ids, logical)
    q = query.float()
    if query_scale is not None:
        q = q * query_scale.reshape(*query.shape[:2], 1).float()
    if key_scale_cache is not None:
        scales, scale_valid = _paged_read(
            key_scale_cache, block_table, request_ids, logical
        )
        keys = keys.float() * scales.float()
        valid = valid & scale_valid
    logits = torch.einsum("thd,tkd->thk", q, keys.float()).relu()
    scores = (logits * weights.float()[:, :, None]).sum(1)
    limit = torch.minimum(
        (positions + 1) // compress_ratio, _lengths(key_lengths, request_ids)
    )
    valid = valid & (positions[:, None] >= 0) & (logical < limit[:, None])
    scores = scores.masked_fill(~valid, -torch.inf)
    if topk > capacity:
        scores = torch.cat(
            [scores, scores.new_full((query.shape[0], topk - capacity), -torch.inf)], -1
        )
    values, indices = scores.topk(topk, dim=-1)
    return torch.where(torch.isfinite(values), indices, -1).to(torch.int32)


def deepseek_v4_sparse_attention(
    query,
    swa_cache,
    compact_cache,
    swa_block_table,
    compact_block_table,
    request_ids,
    positions,
    swa_lengths,
    compact_lengths,
    compact_indices,
    window_size,
    compress_ratio=4,
    scale=None,
    attn_sink=None,
    value_dim=None,
):
    """Joint SWA/compact softmax with learned zero-value sink logits.

    Compact rows overlapping the SWA window remain visible. Bounds are per query
    and request, so chunked-prefill queries cannot see later completed groups.
    Passing no compact cache selects the SWA-only path used by uncompressed layers.
    """
    if window_size < 1 or compress_ratio not in (0, 4, 128):
        raise ValueError(
            "Expected positive SWA window and compression ratio 0, 4 or 128"
        )
    dim = query.shape[-1]
    value_dim = dim if value_dim is None else value_dim
    if not 0 < value_dim <= dim:
        raise ValueError("Value width must be within the shared KV width")
    if query.shape[0] == 0:
        return query.new_empty((0, query.shape[1], value_dim))
    source = (
        positions[:, None]
        - window_size
        + 1
        + torch.arange(window_size, device=query.device)
    )
    kv, valid = _paged_read(swa_cache, swa_block_table, request_ids, source)
    valid = valid & (source < _lengths(swa_lengths, request_ids)[:, None])
    if compact_cache is not None:
        if compress_ratio == 0 or compact_indices is None:
            raise ValueError(
                "Compact attention requires compressed indices and a ratio"
            )
        compact, compact_valid = _paged_read(
            compact_cache, compact_block_table, request_ids, compact_indices
        )
        limit = torch.minimum(
            (positions + 1) // compress_ratio, _lengths(compact_lengths, request_ids)
        )
        compact_valid = compact_valid & (compact_indices < limit[:, None])
        kv = torch.cat([kv, compact], dim=1)
        valid = torch.cat([valid, compact_valid], dim=1)
    valid = valid & (positions[:, None] >= 0)
    # Sanitize invalid physical rows before matmul: zero probability * NaN is NaN.
    kv = torch.where(valid[..., None], kv, 0).float()
    logits = torch.einsum("thd,tkd->thk", query.float(), kv)
    logits = (logits * (dim**-0.5 if scale is None else scale)).masked_fill(
        ~valid[:, None], -torch.inf
    )
    denominator = torch.logsumexp(logits, dim=-1)
    if attn_sink is not None:
        denominator = torch.logaddexp(denominator, attn_sink.float()[None])
    denominator = torch.where(torch.isneginf(denominator), torch.inf, denominator)
    probabilities = torch.exp(logits - denominator[..., None])
    result = torch.einsum("thk,tkd->thd", probabilities, kv[..., :value_dim])
    active = (positions >= 0) & (request_ids >= 0)
    return torch.where(active[:, None, None], result, 0).to(query.dtype)
