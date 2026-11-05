# Copyright (c) 2026 BAAI. All rights reserved.
"""NPU component evidence: owner lifetime, actual replay and outer autocast."""
import importlib.util
import json
import os
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

ROOT = Path(os.environ["FL_SOURCE_DIR"])
IMPL = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, IMPL / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


g = load("fp32_graph_boundary", "fp32_graph.py")
ref = load("fp32_reference_boundary", "fp32_reference.py")
torch.npu.set_device(0)
torch.npu.matmul.allow_hf32 = False
torch.manual_seed(1105)
results = []


def case(limit, block_size, outer_dtype):
    n, heads, dim = 2, 2, 16
    width = (limit + block_size - 1) // block_size
    cache = torch.randn(2, width * n, block_size, heads, dim, device="npu")
    layer = NS(layer_name="same_layer", kv_cache=cache)
    model = NS(num_heads=heads, num_kv_heads=heads, head_size=dim, scale=dim**-0.5)
    table = torch.arange(width * n, device="npu").reshape(n, width).int()
    lengths = torch.tensor([7, 13], device="cpu", dtype=torch.int32)
    slots = table[:, 0].long() * block_size + lengths.to("npu") - 1
    meta = NS(
        num_actual_tokens=n,
        seq_lens_list=[7, 13],
        seq_lens=lengths,
        slot_mapping=slots,
        block_tables=table,
        actual_seq_lengths_q=[1, 2],
    )
    wrapper = NS(
        vllm_config=NS(model_config=NS(max_model_len=limit)), concrete_graph_entries={}
    )
    descriptor = namedtuple("Descriptor", "num_tokens")(n)
    ctx = NS(
        attn_metadata={"same_layer": meta},
        batch_descriptor=descriptor,
        no_compile_layers={"same_layer": layer},
    )
    q, k, v = [torch.randn(n, heads, dim, device="npu") for _ in range(3)]
    output = torch.empty_like(q)
    assert g.prepare_graph(wrapper, ctx)
    state = wrapper._fp32_decode_states[(n,)]["same_layer"]
    with patch(
        "vllm.forward_context.get_forward_context", return_value=ctx
    ), g.graph_owner(wrapper):
        for _ in range(3):
            with torch.autocast("npu", dtype=outer_dtype):
                g.full_attention(layer, model, q, k, v, cache, meta, output)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph), torch.autocast("npu", dtype=outer_dtype):
            g.full_attention(layer, model, q, k, v, cache, meta, output)
        errors = []
        for step in range(3):
            q.copy_(torch.randn_like(q))
            k.copy_(torch.randn_like(k))
            v.copy_(torch.randn_like(v))
            lengths.add_(1)
            slots.copy_(table[:, 0].long() * block_size + lengths.to("npu") - 1)
            meta.seq_lens_list = [8 + step, 14 + step]
            g.prepare_graph(wrapper, ctx)
            graph.replay()
            torch.npu.synchronize()
            expected = torch.empty_like(output)
            eager_cache = cache.clone()
            with torch.autocast("npu", dtype=outer_dtype):
                ref.fp32_paged_attention(
                    q,
                    k,
                    v,
                    eager_cache[0],
                    eager_cache[1],
                    meta,
                    heads,
                    heads,
                    dim,
                    model.scale,
                    expected,
                )
            error = (output - expected).abs().max().item()
            assert torch.allclose(output, expected, atol=2e-5, rtol=2e-5), error
            errors.append(error)
        # Same owner with a changed cache address must evict the captured entry.
        wrapper.concrete_graph_entries[(n,)] = NS(graph=graph)
        layer.kv_cache = cache.clone()
        assert g.prepare_graph(wrapper, ctx)
        assert (n,) not in wrapper.concrete_graph_entries
        assert wrapper._fp32_decode_states[(n,)]["same_layer"] is not state
        wrapper.vllm_config.model_config.max_model_len = limit + block_size
        layer.kv_cache = torch.randn(
            2, n * (width + 1), block_size, heads, dim, device="npu"
        )
        meta.block_tables = (
            torch.arange(n * (width + 1), device="npu").reshape(n, width + 1).int()
        )
        assert g.prepare_graph(wrapper, ctx)
        rebuilt = wrapper._fp32_decode_states[(n,)]["same_layer"]
        assert rebuilt.limit == limit + block_size
        assert rebuilt.table.shape[1] == width + 1
        assert state.limit == limit and state.table.shape[1] == width
    results.append(
        dict(
            limit=limit,
            block_size=block_size,
            outer_autocast=str(outer_dtype),
            actual_graph_replays=3,
            max_abs_errors=errors,
            cache_replacement_invalidates=True,
            hf32=False,
        )
    )
    from vllm_fl.compilation.graph import GraphWrapper

    GraphWrapper.clear_graphs(wrapper)
    assert not wrapper._fp32_decode_states
    assert not wrapper.concrete_graph_entries
    del graph, wrapper, cache
    torch.npu.empty_cache()


for limit, block_size in [(4096, 128), (8192, 128), (4097, 64)]:
    for dtype in [torch.bfloat16, torch.float16]:
        case(limit, block_size, dtype)

from vllm.config.compilation import CUDAGraphMode

from vllm_fl.dispatch.backends.vendor.ascend.impl.attention.attention import (
    AscendAttentionBackendImpl,
    AttentionType,
)

rejected = []
for kind in ["TP", "PCP", "DCP", "noncausal", "cross_attention"]:
    impl = object.__new__(AscendAttentionBackendImpl)
    parallel = NS(
        tensor_parallel_size=2 if kind == "TP" else 1,
        prefill_context_parallel_size=2 if kind == "PCP" else 1,
        decode_context_parallel_size=2 if kind == "DCP" else 1,
    )
    impl.vllm_config = NS(parallel_config=parallel)
    impl.sliding_window = None
    impl.attn_type = (
        AttentionType.ENCODER_DECODER
        if kind == "cross_attention"
        else AttentionType.DECODER
    )
    query = torch.zeros(1, 2, 16, device="npu")
    for full in [False, True]:
        try:
            context = NS(
                cudagraph_runtime_mode=(
                    CUDAGraphMode.FULL if full else CUDAGraphMode.NONE
                )
            )
            with patch(
                "vllm.forward_context.get_forward_context", return_value=context
            ):
                impl.forward(
                    NS(),
                    query,
                    query,
                    query,
                    None,
                    NS(causal=kind != "noncausal"),
                    query.clone(),
                )
        except NotImplementedError:
            rejected.append(dict(case=kind, full=full))
        else:
            raise AssertionError("Unsupported configuration accepted: " + kind)
results.append(dict(unsupported_configurations=rejected))
print("BOUNDARY_RESULT=" + json.dumps(results))
Path(os.environ["EVIDENCE_DIR"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["EVIDENCE_DIR"]) / "fp32_boundaries.json").write_text(
    json.dumps(results, indent=2)
)
