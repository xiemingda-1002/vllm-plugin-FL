# SPDX-License-Identifier: Apache-2.0
"""Portable paged-cache writes for backends without a native provider."""

import torch


def paged_cache_write(values, slots, cache):
    """Write unique valid slots; negative slots are padding, never slot zero."""
    if (
        cache.ndim != 3
        or values.ndim != 2
        or values.shape[1] != cache.shape[-1]
        or slots.shape != values.shape[:1]
    ):
        raise ValueError(
            "Expected matching [tokens, dim], [tokens], [pages, block, dim]"
        )
    if cache.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.int8):
        raise ValueError("Unsupported cache storage dtype")
    if values.shape[0] == 0:
        return
    if any(size == 0 for size in cache.shape):
        raise ValueError("Cannot write to an empty cache")
    num_slots = cache.shape[0] * cache.shape[1]
    valid = (slots >= 0) & (slots < num_slots)
    indices = slots.long().clamp(0, num_slots - 1)
    # Address the original storage so padded/noncontiguous cache views retain
    # their strides. This view allocates no cache-sized workspace.
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(cache.shape, cache.stride())
    )
    storage = cache.as_strided((span,), (1,))
    channels = torch.arange(cache.shape[-1], device=cache.device) * cache.stride(2)
    for token in range(values.shape[0]):
        slot = indices[token]
        addresses = (
            slot.div(cache.shape[1], rounding_mode="floor") * cache.stride(0)
            + (slot % cache.shape[1]) * cache.stride(1)
            + channels
        )
        previous = torch.index_select(storage, 0, addresses)
        update = torch.where(valid[token], values[token].to(cache.dtype), previous)
        # One row per call avoids duplicate-index writes when padding aliases
        # a valid slot, including slot zero. Scratch is bounded by row width.
        storage.index_put_((addresses,), update)
