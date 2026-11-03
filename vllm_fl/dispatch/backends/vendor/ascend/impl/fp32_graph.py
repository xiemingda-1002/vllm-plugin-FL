"""Experimental strict FP32 full decode graph; fixed device metadata buffers."""
import torch
from vllm.forward_context import get_forward_context
from vllm.config.compilation import CUDAGraphMode

STATES = {}
COUNTS = {'full_decode_replays': 0, 'full_attention_captures': 0}
LIMIT = 4096

def update(state, meta):
    n = state['lengths'].shape[0]
    state['lengths'].copy_(meta.seq_lens[:n])
    state['slots'].copy_(meta.slot_mapping[:n])
    table = meta.block_tables[:n]
    state['table'].zero_()
    width = min(table.shape[1], state['table'].shape[1])
    state['table'][:, :width].copy_(table[:, :width])

def full_attention(layer, model, query, key, value, cache, meta, output):
    n = query.shape[0]
    name = layer.layer_name
    index = (name, n)
    if index not in STATES:
        state = {'lengths': torch.ones(n, device=query.device, dtype=torch.int32),
                 'slots': torch.zeros(n, device=query.device, dtype=torch.int64),
                 'table': torch.zeros((n, LIMIT // cache[0].shape[1]), device=query.device, dtype=torch.int64)}
        update(state, meta)
        STATES[index] = state
    state = STATES[index]
    kc, vc = cache[0], cache[1]
    model.key_cache, model.value_cache = kc, vc
    kflat = kc.view(-1, model.num_kv_heads, model.head_size)
    vflat = vc.view_as(kflat)
    slots = state['slots'].clamp_min(0)
    kflat.index_copy_(0, slots, key.view(n, model.num_kv_heads, model.head_size))
    vflat.index_copy_(0, slots, value.view(n, model.num_kv_heads, model.head_size))
    positions = torch.arange(LIMIT, device=query.device, dtype=torch.int64)
    repeat = model.num_heads // model.num_kv_heads
    # Static loop: n is a capture shape, never a runtime sequence length.
    for i in range(n):
        physical = state['table'][i].index_select(0, positions // kc.shape[1]) * kc.shape[1] + positions % kc.shape[1]
        k = kflat.index_select(0, physical).transpose(0, 1).repeat_interleave(repeat, 0)
        v = vflat.index_select(0, physical).transpose(0, 1).repeat_interleave(repeat, 0)
        q = query[i].reshape(model.num_heads, 1, model.head_size)
        scores = torch.matmul(q, k.transpose(-1, -2)) * model.scale
        scores = scores.masked_fill((positions >= state['lengths'][i].clamp_min(1))[None, None], -torch.inf)
        result = torch.matmul(torch.softmax(scores, -1, dtype=torch.float32), v)
        output[i].copy_(result.reshape_as(output[i]))
    COUNTS['full_attention_captures'] += 1
    return output


def prepare_graph(wrapper, ctx):
    global LIMIT
    LIMIT = wrapper.vllm_config.model_config.max_model_len
    metas = ctx.attn_metadata
    n = ctx.batch_descriptor.num_tokens
    if not isinstance(metas, dict) or any(m is not None and
        (int(m.num_actual_tokens) != n or len(m.seq_lens_list) != n)
        for m in metas.values()):
        return False
    for name, meta in metas.items():
        if meta is None:
            continue
        index = (name, n)
        if index not in STATES:
            device = torch.device('npu', torch.npu.current_device())
            STATES[index] = {'lengths': torch.ones(n, device=device, dtype=torch.int32),
                'slots': torch.zeros(n, device=device, dtype=torch.int64),
                'table': torch.zeros((n, (LIMIT + 127) // 128), device=device, dtype=torch.int64)}
        update(STATES[index], meta)
    entry = wrapper.concrete_graph_entries.get(ctx.batch_descriptor)
    if entry is not None and entry.graph is not None:
        COUNTS['full_decode_replays'] += 1
    return True
