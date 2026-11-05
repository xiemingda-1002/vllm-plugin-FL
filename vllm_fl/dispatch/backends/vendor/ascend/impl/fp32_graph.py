# Copyright (c) 2026 BAAI. All rights reserved.

"""Owner-scoped FP32 decode graphs with stable device metadata buffers."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

_observer = None  # Optional validation hook; no global runtime counters.

_OWNER = ContextVar("fl_fp32_graph_owner", default=None)


@contextmanager
def graph_owner(wrapper):
    """Make the capturing wrapper available to attention custom operators."""
    token = _OWNER.set(wrapper)
    try:
        yield
    finally:
        _OWNER.reset(token)


def _tensor_geometry(tensor):
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.data_ptr(),
    )


@dataclass
class DecodeState:
    signature: tuple
    limit: int
    lengths: torch.Tensor
    slots: torch.Tensor
    table: torch.Tensor

    def update(self, meta):
        n = self.lengths.shape[0]
        self.lengths.copy_(meta.seq_lens[:n])
        self.slots.copy_(meta.slot_mapping[:n])
        self.table.copy_(meta.block_tables[:n])


def prepare_graph(wrapper, ctx):
    """Invalidate graphs before replacing metadata or changed cache geometry."""
    metas = ctx.attn_metadata
    if not isinstance(metas, dict) or ctx.batch_descriptor is None:
        return False
    n = ctx.batch_descriptor.num_tokens
    if any(
        meta is None
        or int(meta.num_actual_tokens) != n
        or len(meta.seq_lens_list) != n
        or any(int(end) != i + 1 for i, end in enumerate(meta.actual_seq_lengths_q))
        for meta in metas.values()
    ):
        return False
    limit = wrapper.vllm_config.model_config.max_model_len
    specifications = {}
    for name, meta in metas.items():
        layer = ctx.no_compile_layers.get(name)
        cache = getattr(layer, "kv_cache", None)
        if not isinstance(cache, torch.Tensor) or cache.ndim != 5:
            return False
        block_size = cache.shape[2]
        if meta.block_tables.shape[1] * block_size < limit:
            raise ValueError("FP32 graph block table cannot cover max_model_len")
        signature = (
            id(layer),
            limit,
            _tensor_geometry(cache),
            meta.block_tables.device,
            meta.block_tables.dtype,
            tuple(meta.block_tables[:n].shape),
            meta.seq_lens.device,
            meta.slot_mapping.device,
        )
        specifications[name] = (signature, meta)
    states = wrapper.__dict__.get("_fp32_decode_states", {})
    batch_states = states.get(ctx.batch_descriptor, {})
    if set(batch_states) != set(specifications) or any(
        batch_states[name].signature != specification[0]
        for name, specification in specifications.items()
        if name in batch_states
    ):
        # Never resize a tensor held by a captured graph. Drop graph entries
        # first, then allocate a separate state with the new geometry.
        wrapper.concrete_graph_entries.pop(ctx.batch_descriptor, None)
        states.pop(ctx.batch_descriptor, None)
        states = wrapper.__dict__.setdefault("_fp32_decode_states", {})
        batch_states = {}
        for name, (signature, meta) in specifications.items():
            batch_states[name] = DecodeState(
                signature,
                limit,
                torch.empty_like(
                    meta.seq_lens[:n],
                    device=ctx.no_compile_layers[name].kv_cache.device,
                    dtype=torch.int32,
                ),
                torch.empty_like(
                    meta.slot_mapping[:n],
                    device=ctx.no_compile_layers[name].kv_cache.device,
                    dtype=torch.int64,
                ),
                torch.empty_like(
                    meta.block_tables[:n],
                    device=ctx.no_compile_layers[name].kv_cache.device,
                    dtype=torch.int64,
                ),
            )
        states[ctx.batch_descriptor] = batch_states
    for name, (_, meta) in specifications.items():
        batch_states[name].update(meta)
    return True


def full_attention(layer, model, query, key, value, cache, meta, output):
    from vllm.forward_context import get_forward_context

    owner = _OWNER.get()
    if owner is None:
        raise RuntimeError("FP32 FULL attention requires a graph owner")
    ctx = get_forward_context()
    state = owner._fp32_decode_states[ctx.batch_descriptor][layer.layer_name]
    n = query.shape[0]
    kc, vc = cache[0], cache[1]
    model.key_cache, model.value_cache = kc, vc
    tensors = (query, key, value, kc, vc, output)
    if any(t.dtype != torch.float32 or t.device != query.device for t in tensors):
        raise ValueError("FP32 FULL attention requires matching FP32 tensors")
    if not kc.is_contiguous() or not vc.is_contiguous():
        raise NotImplementedError("FP32 FULL requires contiguous ND KV cache")
    with torch.autocast(device_type="npu", enabled=False):
        kflat = kc.view(-1, model.num_kv_heads, model.head_size)
        vflat = vc.view_as(kflat)
        slots = state.slots.clamp_min(0)
        kflat.index_copy_(0, slots, key.view(n, model.num_kv_heads, model.head_size))
        vflat.index_copy_(0, slots, value.view(n, model.num_kv_heads, model.head_size))
        positions = torch.arange(state.limit, device=query.device, dtype=torch.int64)
        repeat = model.num_heads // model.num_kv_heads
        for i in range(n):
            physical = (
                state.table[i].index_select(0, positions // kc.shape[1]) * kc.shape[1]
                + positions % kc.shape[1]
            )
            k = (
                kflat.index_select(0, physical)
                .transpose(0, 1)
                .repeat_interleave(repeat, 0)
            )
            v = (
                vflat.index_select(0, physical)
                .transpose(0, 1)
                .repeat_interleave(repeat, 0)
            )
            q = query[i].reshape(model.num_heads, 1, model.head_size)
            scores = torch.matmul(q, k.transpose(-1, -2)) * model.scale
            scores = scores.masked_fill(
                (positions >= state.lengths[i].clamp_min(1))[None, None], -torch.inf
            )
            result = torch.matmul(torch.softmax(scores, -1, dtype=torch.float32), v)
            output[i].copy_(result.reshape_as(output[i]))
    if _observer is not None:
        _observer(tensors + (scores, result), meta, full=True)
    return output
