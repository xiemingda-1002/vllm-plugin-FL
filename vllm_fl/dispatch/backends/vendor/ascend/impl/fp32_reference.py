# Copyright (c) 2026 BAAI. All rights reserved.

import torch
import torch_npu  # noqa: F401

_observer = None  # Installed only by the optional validation worker extension.


def fp32_paged_attention(
    query,
    key,
    value,
    key_cache,
    value_cache,
    metadata,
    num_heads,
    num_kv_heads,
    head_size,
    scale,
    output,
):
    tensors = [query, key_cache, value_cache, output]
    tensors.extend(t for t in [key, value] if t is not None)
    if any(t.dtype != torch.float32 or t.device.type != "npu" for t in tensors):
        raise RuntimeError("Strict FP32 requires NPU float32 Q/K/V, cache and output")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise NotImplementedError("FP32 adapter requires contiguous ND KV layout")
    count = int(metadata.num_actual_tokens)
    ends, lengths = list(metadata.actual_seq_lengths_q), list(metadata.seq_lens_list)
    if not lengths or len(lengths) != len(ends) or ends[-1] != count:
        raise RuntimeError("Invalid packed sequence metadata")
    if any(end <= start for start, end in zip([0] + ends[:-1], ends)):
        raise RuntimeError("Empty or unordered query sequence")
    keys_flat = key_cache.view(-1, num_kv_heads, head_size)
    values_flat = value_cache.view(-1, num_kv_heads, head_size)
    if key is not None:
        slots = metadata.slot_mapping[:count].to(device=query.device, dtype=torch.int64)
        keys_flat.index_copy_(
            0, slots, key.reshape(-1, num_kv_heads, head_size)[:count]
        )
        values_flat.index_copy_(
            0, slots, value.reshape(-1, num_kv_heads, head_size)[:count]
        )
    output.zero_()
    start = 0
    for sequence, (end, length) in enumerate(zip(ends, lengths)):
        end, length = int(end), int(length)
        queries = end - start
        if queries > length:
            raise RuntimeError("Query is longer than its sequence")
        block_size = key_cache.shape[1]
        positions = torch.arange(length, device=query.device, dtype=torch.int64)
        table = metadata.block_tables[sequence].to(torch.int64)
        physical = (
            table.index_select(0, positions // block_size) * block_size
            + positions % block_size
        )
        cached_k = keys_flat.index_select(0, physical)
        cached_v = values_flat.index_select(0, physical)
        repeat = num_heads // num_kv_heads
        q = query[start:end].reshape(queries, num_heads, head_size).transpose(0, 1)
        k = cached_k.transpose(0, 1).repeat_interleave(repeat, dim=0)
        v = cached_v.transpose(0, 1).repeat_interleave(repeat, dim=0)
        query_positions = (
            torch.arange(queries, device=query.device, dtype=torch.int64)
            + length
            - queries
        )
        causal_mask = positions[None, :] > query_positions[:, None]
        with torch.autocast(device_type="npu", enabled=False):
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            scores = scores.masked_fill(causal_mask[None], float("-inf"))
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
            attended = torch.matmul(probabilities, v)
        output.reshape(-1, num_heads, head_size)[start:end].copy_(
            attended.transpose(0, 1)
        )
        start = end
    if _observer is not None:
        _observer(tensors + [scores, probabilities, attended], metadata)
    return output


def torch_rotary(
    positions, query, key, cos_sin_cache, head_size, rotary_dim, is_neox_style
):
    count = positions.numel()
    cache = cos_sin_cache.index_select(0, positions.flatten().long())
    cos, sin = cache.chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    results = []
    for tensor in (query, key):
        heads = tensor.reshape(count, -1, head_size)
        rotated, passthrough = heads[..., :rotary_dim], heads[..., rotary_dim:]
        if is_neox_style:
            first, second = rotated.chunk(2, dim=-1)
            result = torch.cat(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            )
        else:
            first, second = rotated[..., ::2], rotated[..., 1::2]
            result = torch.stack(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            ).flatten(-2)
        results.append(torch.cat((result, passthrough), dim=-1).reshape(tensor.shape))
    return tuple(results)


def torch_slot_mapping(self, num_reqs, query_start_loc, positions):
    if self.pcp_world_size * self.dcp_world_size != 1:
        raise NotImplementedError("FP32 reference slot mapping requires CP=1")
    starts = query_start_loc[: num_reqs + 1].long()
    indices = torch.arange(positions.numel(), device=positions.device)
    valid = indices < starts[-1]
    rows = torch.bucketize(indices, starts[1:], right=True).clamp(max=num_reqs - 1)
    blocks = positions.long() // self.block_size
    table = self.block_table.gpu
    slots = table[rows, blocks].long() * self.block_size + positions % self.block_size
    self.slot_mapping.gpu.fill_(-1)
    self.slot_mapping.gpu[: positions.numel()].copy_(torch.where(valid, slots, -1))
