"""Focused contracts for the Ascend Qwen3-VL visual closure."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[3]
ENCODER_GRAPH = ROOT / "vllm_fl/compilation/encoder_acl_graph.py"
QWEN_PATCH = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3vl.py"


def _encoder_graph_module():
    pytest.importorskip("torch_npu")
    spec = importlib.util.spec_from_file_location("fl_qwen3vl_encoder_graph", ENCODER_GRAPH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_encoder_fia_lengths_preserve_eager_and_graph_contracts():
    module = _encoder_graph_module()
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    assert module.maybe_compute_actual_seq_lengths(cu, 7, 7) == ([3, 7], [3, 7])
    # Graph replay filters padded endpoints and restores the capture budget.
    padded = torch.tensor([0, 3, 7, 7, 99], dtype=torch.int32)
    assert module.maybe_compute_actual_seq_lengths(padded, 7, 14, cudagraph_mm_encoder=True) == (
        [3, 7], [6, 14]
    )


def test_encoder_forward_context_is_cleared_after_graph_scope():
    module = _encoder_graph_module()
    with module.set_encoder_forward_context(16, True):
        assert module.get_encoder_forward_context().token_budget == 16
        assert module.get_encoder_forward_context().capturing
    assert module.get_encoder_forward_context().token_budget is None
    assert not module.get_encoder_forward_context().capturing


def test_qwen3vl_deepstack_splits_only_when_flashcomm_is_enabled(monkeypatch):
    pytest.importorskip("vllm")
    spec = importlib.util.spec_from_file_location("fl_qwen3vl_patch", QWEN_PATCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Embeds:
        tensors = {"deepstack_input_embeds_0": torch.arange(8).view(4, 2)}

    class Context:
        flash_comm_v1_enabled = False

    monkeypatch.setattr(module, "_EXTRA_CTX", Context())
    wrapped = module._deepstack_tp_wrap(lambda: Embeds())
    original = wrapped().tensors["deepstack_input_embeds_0"]
    assert original.shape == (4, 2)

    Context.flash_comm_v1_enabled = True
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    assert torch.equal(wrapped().tensors["deepstack_input_embeds_0"], original.chunk(2)[1])


def test_qwen3vl_attention_fallback_and_fused_mrope_invocations(monkeypatch):
    pytest.importorskip("vllm")
    spec = importlib.util.spec_from_file_location("fl_qwen3vl_patch_invocation", QWEN_PATCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Identity:
        weight = torch.ones(1)
        variance_epsilon = 1e-6
        def __call__(self, value): return value
    class PlainRotary:
        def __call__(self, positions, query, key): return query + 1, key + 1
    class Projection:
        def __call__(self, value): return value, None
    class Attention:
        def __call__(self, query, key, value): return query
    class FakeSelf:
        q_size, kv_size, num_heads, num_kv_heads, head_dim = 2, 1, 2, 1, 1
        q_norm, k_norm, qkv_proj, attn, o_proj = Identity(), Identity(), Projection(), Attention(), Projection()
        rotary_emb = PlainRotary()

    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(module._qwen3_attention_forward(FakeSelf(), torch.tensor([0]), hidden), torch.tensor([[2.0, 3.0]]))

    class FakeMRotary:
        cos_sin_cache = torch.ones(2, 2)
        mrope_section, mrope_interleaved, rotary_dim = [1, 1, 1], True, 1
    captured = {}
    def fused(**kwargs):
        captured.update(kwargs)
        return torch.ones(1, 2), torch.ones(1, 1), torch.ones(1, 1), None
    monkeypatch.setattr(module, "MRotaryEmbedding", FakeMRotary)
    monkeypatch.setattr(module.torch, "ops", SimpleNamespace(vllm=SimpleNamespace(triton_split_qkv_rmsnorm_mrope=fused)))
    fused_self = FakeSelf(); fused_self.rotary_emb = FakeMRotary()
    assert torch.equal(module._qwen3_attention_forward(fused_self, torch.tensor([0]), hidden), torch.ones(1, 2))
    assert captured["mrope_section"] == [1, 1, 1]
