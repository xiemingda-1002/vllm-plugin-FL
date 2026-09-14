"""Regression coverage for DeepSeek's Ascend-only forward-context extras."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_fl import ascend_forward_context as afc


def _context(**additional_kwargs):
    return ForwardContext(
        no_compile_layers={"standard-layer": object()},
        attn_metadata={"standard-layer": object()},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        additional_kwargs=additional_kwargs,
    )


def test_extra_context_is_per_forward_and_keeps_standard_context_fields() -> None:
    first = _context(flash_comm_v1_enabled=False, pad_size=0, num_tokens=2)
    second = _context(flash_comm_v1_enabled=True, pad_size=3, num_tokens=5)

    with override_forward_context(first):
        assert afc._EXTRA_CTX.flash_comm_v1_enabled is False
        assert afc._EXTRA_CTX.pad_size == 0
        assert afc._EXTRA_CTX.num_tokens == 2
        # These are upstream ForwardContext fields, not Ascend extras.
        assert afc._EXTRA_CTX.no_compile_layers is first.no_compile_layers
        assert afc._EXTRA_CTX.attn_metadata is first.attn_metadata

    with override_forward_context(second):
        assert afc._EXTRA_CTX.flash_comm_v1_enabled is True
        assert afc._EXTRA_CTX.pad_size == 3
        assert afc._EXTRA_CTX.num_tokens == 5


def test_dsa_consumers_follow_current_forward_extras(monkeypatch) -> None:
    dsa_ops = pytest.importorskip(
        "vllm_fl.dispatch.backends.vendor.ascend.ops.dsa"
    )
    dsa_v1 = pytest.importorskip(
        "vllm_fl.dispatch.backends.vendor.ascend.attention.dsa_v1"
    )
    observed_gather_flags: list[bool] = []

    monkeypatch.setattr(
        torch.ops.vllm,
        "dsa_forward",
        lambda hidden_states, need_gather_q_kv, output, prefix: observed_gather_flags.append(
            need_gather_q_kv
        ),
        raising=False,
    )
    sparse_attention = object.__new__(dsa_ops.AscendDeepseekSparseAttention)
    sparse_attention.prefix = "standard-layer"

    dsa_impl = object.__new__(dsa_v1.AscendDSAImpl)
    dsa_impl.n_local_heads = 1
    dsa_impl.head_dim = 2
    monkeypatch.setattr(dsa_v1, "oproj_tp_enable", lambda: False)

    for enabled, num_tokens in ((False, 2), (True, 5)):
        with override_forward_context(
            _context(
                flash_comm_v1_enabled=enabled,
                pad_size=0,
                num_tokens=num_tokens,
            )
        ):
            sparse_attention.forward(
                torch.empty(num_tokens), torch.empty(num_tokens, 2)
            )
            output = torch.empty(num_tokens, 2)
            assert dsa_impl.forward(
                "standard-layer",
                torch.empty(num_tokens, 2),
                None,
                None,
                output=output,
            ) is output

    assert observed_gather_flags == [False, True]


def test_deepseek_mtp_uses_current_flashcomm_padding(monkeypatch) -> None:
    model_module = pytest.importorskip(
        "vllm_fl.dispatch.backends.vendor.ascend.models.deepseek_v4"
    )
    model = object.__new__(model_module.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.do_not_compile = True
    model.hc_mult = 1
    model.layers = []
    model.start_layer = 0
    model.end_layer = 0
    model._mtp_hidden_buffer = torch.zeros(8, 2)
    model.embed_input_ids = lambda input_ids: input_ids.to(torch.float32).unsqueeze(1).repeat(1, 2)
    model.hc_head = lambda states, *_: states.squeeze(1)
    model.norm = lambda states: states
    model.hc_head_fn = torch.empty(0)
    model.hc_head_scale = torch.empty(0)
    model.hc_head_base = torch.empty(0)
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    input_ids = torch.tensor([10, 20])
    with override_forward_context(
        _context(flash_comm_v1_enabled=False, pad_size=0, num_tokens=2)
    ):
        model(input_ids, torch.empty(2), None)
    assert torch.equal(model._mtp_hidden_buffer[:2], torch.tensor([[10.0, 10.0], [20.0, 20.0]]))

    monkeypatch.setattr(
        model_module,
        "tensor_model_parallel_all_gather",
        lambda states, dim: torch.cat((states, torch.full_like(states[:1], -1)), dim=0),
    )
    with override_forward_context(
        _context(flash_comm_v1_enabled=True, pad_size=1, num_tokens=2)
    ):
        model(input_ids, torch.empty(2), None)
    assert torch.equal(model._mtp_hidden_buffer[:2], torch.tensor([[10.0, 10.0], [20.0, 20.0]]))
