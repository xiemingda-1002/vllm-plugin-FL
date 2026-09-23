"""Behavioral contracts for FL's Ascend OOT MLA wrapper."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")


def test_ascend_mla_wrapper_dispatches_to_registered_graph_boundary(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.ops.mla import (
        AscendMultiHeadLatentAttention,
    )

    wrapper = object.__new__(AscendMultiHeadLatentAttention)
    torch.nn.Module.__init__(wrapper)
    wrapper.hidden_size = 3
    wrapper.tp_size = 1
    wrapper.is_vl_first_layer = False
    wrapper.prefix = "model.layers.0.self_attn"
    wrapper._use_ascend_mla_forward = True

    calls = []

    def mla_forward(hidden_states, gather_q_kv, output, layer_name):
        calls.append((hidden_states, gather_q_kv, output, layer_name))
        output.copy_(hidden_states + 1)

    monkeypatch.setattr(
        torch.ops.vllm,
        "mla_forward",
        mla_forward,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_fl.dispatch.backends.vendor.ascend.ops.mla._EXTRA_CTX",
        SimpleNamespace(flash_comm_v1_enabled=False),
    )

    hidden_states = torch.zeros((2, 3))
    result = wrapper(None, hidden_states)

    assert len(calls) == 1
    assert calls[0][1] is False
    assert calls[0][3] == wrapper.prefix
    torch.testing.assert_close(result, torch.ones((2, 3)))


def test_dense_mla_delegates_to_upstream_wrapper(monkeypatch) -> None:
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
    from vllm_fl.dispatch.backends.vendor.ascend.ops.mla import (
        AscendMultiHeadLatentAttention,
    )

    wrapper = object.__new__(AscendMultiHeadLatentAttention)
    torch.nn.Module.__init__(wrapper)
    wrapper._use_ascend_mla_forward = False
    calls = []

    def upstream_forward(self, positions, hidden_states, llama_4_scaling=None):
        calls.append((self, positions, hidden_states, llama_4_scaling))
        return hidden_states + 2

    monkeypatch.setattr(
        MultiHeadLatentAttentionWrapper,
        "forward",
        upstream_forward,
    )
    hidden_states = torch.zeros((2, 3))
    scale = torch.tensor(0.5)

    result = wrapper(torch.tensor([0, 1]), hidden_states, scale)

    assert len(calls) == 1
    assert calls[0][0] is wrapper
    assert calls[0][3] is scale
    torch.testing.assert_close(result, torch.full((2, 3), 2.0))


def test_dense_mla_initialization_uses_upstream_wrapper(monkeypatch) -> None:
    from vllm.model_executor.layers.mla import (
        MLAModules,
        MultiHeadLatentAttentionWrapper,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.ops.mla import (
        AscendMultiHeadLatentAttention,
    )

    captured = []

    def upstream_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        captured.append((args, kwargs))

    monkeypatch.setattr(MultiHeadLatentAttentionWrapper, "__init__", upstream_init)
    module = torch.nn.Identity()
    mla_modules = MLAModules(
        kv_a_layernorm=module,
        kv_b_proj=module,
        rotary_emb=module,
        o_proj=module,
        fused_qkv_a_proj=None,
        kv_a_proj_with_mqa=module,
        q_a_layernorm=None,
        q_b_proj=None,
        q_proj=module,
        indexer=None,
        is_sparse=False,
        topk_indices_buffer=None,
    )

    wrapper = AscendMultiHeadLatentAttention(
        8, 2, 1.0, 2, 2, 2, None, 4, mla_modules, prefix="dense"
    )

    assert wrapper._use_ascend_mla_forward is False
    assert len(captured) == 1
    assert captured[0][0][8] is mla_modules


def test_ascend_patch_registers_mla_oot_and_privateuse1_op(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend import patch

    registered_ops = []
    registered_layers = []

    monkeypatch.setattr(
        "vllm_fl.dispatch.backends.vendor.ascend.ops.mla.direct_register_custom_op",
        lambda **kwargs: registered_ops.append(kwargs),
    )
    monkeypatch.setattr(
        "vllm.model_executor.custom_op.CustomOp.register_oot",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.custom_op.PluggableLayer.register_oot",
        lambda **kwargs: registered_layers.append(kwargs),
    )
    monkeypatch.setattr(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.layernorm.ensure_ascend_rms_norm_gated_registered",
        lambda: None,
    )
    monkeypatch.setattr(patch, "_op_classes_patched", False)
    monkeypatch.setattr(
        "vllm_fl.dispatch.backends.vendor.ascend.ops.mla._MLA_FORWARD_REGISTERED",
        False,
    )

    patch.patch_op_cls()

    assert registered_ops == [{
        "op_name": "mla_forward",
        "op_func": __import__(
            "vllm_fl.dispatch.backends.vendor.ascend.ops.mla",
            fromlist=["mla_forward"],
        ).mla_forward,
        "mutates_args": ["output"],
        "fake_impl": __import__(
            "vllm_fl.dispatch.backends.vendor.ascend.ops.mla",
            fromlist=["mla_forward_fake"],
        ).mla_forward_fake,
        "dispatch_key": "PrivateUse1",
    }]
    assert any(
        entry["name"] == "MultiHeadLatentAttentionWrapper"
        and entry["_decorated_layer_cls"].__name__
        == "AscendMultiHeadLatentAttention"
        for entry in registered_layers
    )
