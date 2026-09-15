# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU contracts for the FlashComm1 vocabulary-embedding boundary."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_fl.ascend_forward_context import MoECommType
from vllm_fl.dispatch.backends.vendor.ascend.impl import flashcomm_custom_ops
from vllm_fl.dispatch.backends.vendor.ascend.impl import vocab_parallel_embedding
from vllm_fl.dispatch.backends.vendor.ascend.impl.moe import experts_selector


def _context(**additional_kwargs) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        additional_kwargs=additional_kwargs,
    )


def _embedding(tp_size: int):
    """Make a real _forward_origin receiver without allocating model weights."""
    embedding = vocab_parallel_embedding.AscendVocabParallelEmbedding.__new__(
        vocab_parallel_embedding.AscendVocabParallelEmbedding
    )
    embedding.tp_size = tp_size
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=0,
        org_vocab_end_index=4096,
        num_org_vocab_padding=0,
        added_vocab_start_index=4096,
        added_vocab_end_index=4096,
    )
    embedding.quant_method = SimpleNamespace(
        embedding=lambda _layer, ids: ids.to(torch.float32).unsqueeze(-1)
    )
    return embedding


@pytest.mark.parametrize("rank", range(4))
def test_embedding_reduce_scatter_and_hash_selector_share_tp4_rows(
    monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    """Exercise the actual embedding and selector paths, not a split mock."""
    global_ids = torch.arange(100, 164, dtype=torch.int32)
    global_ids[rank * 16 + 3] = -1
    reduce_scatter_inputs: list[torch.Tensor] = []
    observed_hash: list[tuple[torch.Tensor, int]] = []

    def reduce_scatter(x: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        reduce_scatter_inputs.append(x.clone())
        return x.reshape(4, 16, x.shape[-1])[rank].clone()

    monkeypatch.setattr(
        flashcomm_custom_ops, "tensor_model_parallel_reduce_scatter", reduce_scatter
    )
    monkeypatch.setattr(
        flashcomm_custom_ops, "tensor_model_parallel_all_reduce", lambda x: x + 1000
    )
    monkeypatch.setattr(
        flashcomm_custom_ops,
        "_EXTRA_CTX",
        SimpleNamespace(flash_comm_v1_enabled=True, pad_size=0),
    )
    monkeypatch.setattr(
        vocab_parallel_embedding.torch.ops,
        "vllm",
        SimpleNamespace(
            maybe_pad_and_reduce=flashcomm_custom_ops._maybe_pad_and_reduce_impl
        ),
    )
    monkeypatch.setattr(
        experts_selector,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=rank),
    )

    def hash_op(**kwargs):
        input_ids = kwargs["input_ids"]
        x = kwargs["x"]
        observed_hash.append((input_ids, x.shape[0]))
        assert input_ids.numel() == x.shape[0]
        return torch.ones(x.shape[0], 1), torch.zeros(x.shape[0], 1, dtype=torch.int32), None

    monkeypatch.setattr(
        torch.ops._C_ascend, "moe_gating_top_k_hash", hash_op, raising=False
    )
    comm_method = SimpleNamespace(pad_and_split_input_ids=lambda ids: ids)
    with override_forward_context(
        _context(
            input_ids=global_ids,
            moe_comm_type=MoECommType.MC2,
            moe_comm_method=comm_method,
            flash_comm_v1_enabled=True,
        )
    ):
        hidden_states = _embedding(tp_size=4)._forward_origin(global_ids)
        experts_selector._select_experts_with_fusion_ops(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            top_k=1,
            use_grouped_topk=False,
            renormalize=False,
            e_score_correction_bias=None,
            topk_group=1,
            num_expert_group=1,
            scoring_func="sqrtsoftplus",
            tid2eid=torch.tensor([[0]], dtype=torch.int64),
        )

    assert reduce_scatter_inputs[0].shape == (64, 1)
    assert hidden_states.shape == (16, 1)
    expected_ids = global_ids[rank * 16 : (rank + 1) * 16].to(torch.int64)
    expected_ids[3] = 0
    assert len(observed_hash) == 1
    assert observed_hash[0][1] == 16
    assert torch.equal(observed_hash[0][0], expected_ids)


def test_embedding_keeps_full_rows_when_flashcomm_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    all_reduces: list[torch.Tensor] = []
    monkeypatch.setattr(
        flashcomm_custom_ops, "tensor_model_parallel_all_reduce", lambda x: all_reduces.append(x) or x
    )
    monkeypatch.setattr(
        flashcomm_custom_ops, "_EXTRA_CTX", SimpleNamespace(flash_comm_v1_enabled=False, pad_size=0)
    )
    monkeypatch.setattr(
        vocab_parallel_embedding.torch.ops,
        "vllm",
        SimpleNamespace(maybe_pad_and_reduce=flashcomm_custom_ops._maybe_pad_and_reduce_impl),
    )
    input_ids = torch.arange(64, dtype=torch.int32)
    with override_forward_context(_context()):
        result = _embedding(tp_size=4)._forward_origin(input_ids)

    assert len(all_reduces) == 1
    assert result.shape == (64, 1)


def test_embedding_reduction_has_no_context_allreduce_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    all_reduces: list[torch.Tensor] = []
    monkeypatch.setattr(
        flashcomm_custom_ops, "tensor_model_parallel_all_reduce", lambda x: all_reduces.append(x) or x
    )
    result = flashcomm_custom_ops._maybe_pad_and_reduce_impl(torch.ones(3, 2))

    assert len(all_reduces) == 1
    assert result.shape == (3, 2)
