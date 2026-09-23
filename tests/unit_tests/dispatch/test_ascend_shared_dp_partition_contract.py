# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU contracts for rc1 shared-DP with FlashComm1 ordinary MoE paths."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_fl.ascend_forward_context import MoECommType
from vllm_fl.dispatch.backends.vendor.ascend.impl.moe import experts_selector
from vllm_fl.dispatch.backends.vendor.ascend.impl.moe import prepare_finalize


def _context(dp_metadata=None, **additional_kwargs) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        additional_kwargs=additional_kwargs,
    )


@pytest.mark.parametrize(
    ("prepare_cls", "comm_type"),
    [
        (prepare_finalize.PrepareAndFinalizeWithMC2, MoECommType.MC2),
        (prepare_finalize.PrepareAndFinalizeWithAll2All, MoECommType.ALLTOALL),
    ],
    ids=("mc2", "alltoall"),
)
@pytest.mark.parametrize("enable_shared_expert_dp", [False, True])
@pytest.mark.parametrize("rank", range(4))
def test_flashcomm_ordinary_moe_keeps_one_tp_id_partition(
    monkeypatch: pytest.MonkeyPatch,
    prepare_cls,
    comm_type: MoECommType,
    enable_shared_expert_dp: bool,
    rank: int,
) -> None:
    """Real prepare + selector: FlashComm owns hidden partition, selector owns IDs."""
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(
        prepare_finalize,
        "_EXTRA_CTX",
        SimpleNamespace(
            # 13 real tokens plus a FlashComm-owned three-token tail.
            padded_num_tokens=16,
            mc2_mask=(torch.arange(16) % 2 == 0),
        ),
    )
    prepare = prepare_cls(SimpleNamespace())

    # These are already rank-local after the generic embedding/row-parallel
    # FlashComm reduce-scatter.  prepare must not pad or split them again.
    hidden_states = torch.full((4, 2), float(rank))
    router_logits = torch.full((4, 3), float(rank))
    prepared = prepare.prepare(
        hidden_states,
        router_logits,
        enable_shared_expert_dp=enable_shared_expert_dp,
        replace_allreduce=True,
    )
    assert torch.equal(prepared.hidden_states, hidden_states)
    assert torch.equal(prepared.router_logits, router_logits)
    if comm_type is MoECommType.MC2:
        assert torch.equal(
            prepared.mc2_mask,
            (torch.arange(16) % 2 == 0)[rank * 4 : (rank + 1) * 4],
        )
    else:
        assert prepared.mc2_mask is None

    # `pad_and_split_input_ids` must preserve the pre-selector padded global
    # IDs under replace_allreduce.  The selector performs the sole TP split.
    global_ids = torch.arange(100, 113, dtype=torch.int32)
    global_ids = torch.nn.functional.pad(global_ids, (0, 3), value=-1)
    assert torch.equal(prepare.pad_and_split_input_ids(global_ids), global_ids)

    monkeypatch.setattr(
        experts_selector,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=rank),
    )
    captured: list[tuple[torch.Tensor, int]] = []

    def hash_op(**kwargs):
        ids = kwargs["input_ids"]
        rows = kwargs["x"].shape[0]
        captured.append((ids, rows))
        assert ids.numel() == rows
        return torch.ones(rows, 1), torch.zeros(rows, 1, dtype=torch.int32), None

    monkeypatch.setattr(torch.ops._C_ascend, "moe_gating_top_k_hash", hash_op, raising=False)
    comm_method = SimpleNamespace(pad_and_split_input_ids=prepare.pad_and_split_input_ids)
    # A deliberately unbalanced DP metadata vector belongs to the outer
    # FlashComm gather/reduce owner.  This ordinary local prepare contract must
    # not reinterpret it or add a second padding operation.
    dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([13, 7], dtype=torch.int64))
    with override_forward_context(
        _context(
            input_ids=global_ids,
            moe_comm_type=comm_type,
            moe_comm_method=comm_method,
            flash_comm_v1_enabled=True,
            dp_metadata=dp_metadata,
        )
    ):
        experts_selector._select_experts_with_fusion_ops(
            hidden_states=prepared.hidden_states,
            router_logits=prepared.router_logits,
            top_k=1,
            use_grouped_topk=False,
            renormalize=False,
            e_score_correction_bias=None,
            topk_group=1,
            num_expert_group=1,
            scoring_func="sqrtsoftplus",
            tid2eid=torch.tensor([[0]], dtype=torch.int64),
        )

    expected = global_ids[rank * 4 : (rank + 1) * 4].to(torch.int64)
    expected[expected == -1] = 0
    assert len(captured) == 1
    assert torch.equal(captured[0][0], expected)
    assert captured[0][1] == 4
