# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_fl.ascend_forward_context import MoECommType
from vllm_fl.dispatch.backends.vendor.ascend.impl.moe import experts_selector
from vllm_fl.ops.fused_moe import layer as layer_module


class _FakeRunner:
    _quant_method = object()
    moe_config = SimpleNamespace()


def _context(**additional_kwargs) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        additional_kwargs=additional_kwargs,
    )


def _set_platform(monkeypatch, vendor_name: str, device_type: str) -> None:
    import vllm.platforms as platforms

    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(vendor_name=vendor_name, device_type=device_type),
    )


def test_ascend_factory_consumes_hash_and_preserves_tid2eid_identity(monkeypatch) -> None:
    _set_platform(monkeypatch, "ascend", "npu")
    captured: dict[str, object] = {}
    tid2eid = object()
    custom_runner = object()
    original_runner_args = {"existing": object()}

    def upstream_factory(*args, **kwargs):
        captured.update(kwargs)
        return _FakeRunner()

    monkeypatch.setattr(layer_module, "_OrigFusedMoE", upstream_factory)
    monkeypatch.setattr(layer_module, "replace_router_with_fl", lambda: None)

    layer_module.FusedMoEFL(
        hash=True,
        tid2eid=tid2eid,
        runner_cls=custom_runner,
        runner_args=original_runner_args,
        unrelated="kept",
    )

    assert "hash" not in captured
    assert "tid2eid" not in captured
    assert captured["runner_cls"] is custom_runner
    assert captured["unrelated"] == "kept"
    assert captured["runner_args"] is not original_runner_args
    assert captured["runner_args"] == {
        "existing": original_runner_args["existing"],
        "tid2eid": tid2eid,
    }
    assert original_runner_args == {"existing": original_runner_args["existing"]}


def test_non_ascend_factory_keeps_hash_arguments_unchanged(monkeypatch) -> None:
    _set_platform(monkeypatch, "cuda", "cuda")
    captured: dict[str, object] = {}
    custom_runner = object()

    def upstream_factory(*args, **kwargs):
        captured.update(kwargs)
        return _FakeRunner()

    monkeypatch.setattr(layer_module, "_OrigFusedMoE", upstream_factory)
    monkeypatch.setattr(layer_module, "replace_router_with_fl", lambda: None)

    layer_module.FusedMoEFL(
        hash=True,
        tid2eid="non-ascend-table",
        runner_cls=custom_runner,
        runner_args={"preserved": True},
    )

    assert captured == {
        "hash": True,
        "tid2eid": "non-ascend-table",
        "runner_cls": custom_runner,
        "runner_args": {"preserved": True},
    }


def test_hash_selector_uses_current_extra_context_input_ids(monkeypatch) -> None:
    observed_input_ids: list[torch.Tensor] = []

    def hash_op(**kwargs):
        observed_input_ids.append(kwargs["input_ids"])
        return torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32), None

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "moe_gating_top_k_hash",
        hash_op,
        raising=False,
    )

    comm_method = SimpleNamespace(
        prepare_finalize=SimpleNamespace(
            all_gather_input_id_with_dp_group=lambda input_ids: input_ids
        )
    )
    for input_ids, expected in (
        (torch.tensor([2, 5, -1], dtype=torch.int32), torch.tensor([2, 5, 0])),
        (torch.tensor([7, 1], dtype=torch.int32), torch.tensor([7, 1])),
    ):
        with override_forward_context(
            _context(
                input_ids=input_ids,
                moe_comm_type=MoECommType.ALLGATHER,
                moe_comm_method=comm_method,
                flash_comm_v1_enabled=False,
            )
        ):
            experts_selector._select_experts_with_fusion_ops(
                hidden_states=torch.empty(1, 4),
                router_logits=torch.empty(1, 4),
                top_k=1,
                use_grouped_topk=False,
                renormalize=False,
                e_score_correction_bias=None,
                topk_group=1,
                num_expert_group=1,
                scoring_func="sqrtsoftplus",
                tid2eid=torch.tensor([[3]], dtype=torch.int64),
            )
        assert torch.equal(observed_input_ids[-1], expected)
        assert observed_input_ids[-1].dtype is torch.int64

    assert len(observed_input_ids) == 2


@pytest.mark.parametrize("num_partitions", [1, 2, 4])
def test_rc1_first_dim_split_returns_partitions_and_preserves_views(
    num_partitions: int,
) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.distributed.utils import (
        split_tensor_along_first_dim,
    )

    tensor = torch.arange(num_partitions * 8).reshape(num_partitions * 4, 2)[:, :1]
    assert not tensor.is_contiguous()
    chunks = split_tensor_along_first_dim(tensor, num_partitions)

    assert isinstance(chunks, tuple)
    assert len(chunks) == num_partitions
    assert all(not chunk.is_contiguous() for chunk in chunks)
    contiguous = split_tensor_along_first_dim(
        tensor, num_partitions, contiguous_split_chunks=True
    )
    assert all(chunk.is_contiguous() for chunk in contiguous)


def test_rc1_first_dim_split_rejects_uneven_dimension() -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.distributed.utils import (
        split_tensor_along_first_dim,
    )

    with pytest.raises(AssertionError):
        split_tensor_along_first_dim(torch.arange(5), 2)


@pytest.mark.parametrize("comm_type", [MoECommType.MC2, MoECommType.ALLTOALL])
@pytest.mark.parametrize(
    ("tp_size", "rank"),
    [(tp_size, rank) for tp_size in (1, 2, 4) for rank in range(tp_size)],
)
def test_hash_selector_flashcomm_uses_rank_local_tp_input_ids(
    monkeypatch, comm_type, tp_size, rank
) -> None:
    captured: list[torch.Tensor] = []
    all_ids = torch.arange(100, 100 + tp_size * 4, dtype=torch.int32)
    all_ids[2::4] = -1

    def hash_op(**kwargs):
        captured.append(kwargs["input_ids"])
        return torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32), None

    monkeypatch.setattr(torch.ops._C_ascend, "moe_gating_top_k_hash", hash_op, raising=False)
    monkeypatch.setattr(
        experts_selector,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=tp_size, rank_in_group=rank),
    )
    comm_method = SimpleNamespace(pad_and_split_input_ids=lambda ids: ids)
    with override_forward_context(
        _context(
            input_ids=all_ids,
            moe_comm_type=comm_type,
            moe_comm_method=comm_method,
            flash_comm_v1_enabled=True,
        )
    ):
        experts_selector._select_experts_with_fusion_ops(
            hidden_states=torch.empty(1, 4),
            router_logits=torch.empty(1, 4),
            top_k=1,
            use_grouped_topk=False,
            renormalize=False,
            e_score_correction_bias=None,
            topk_group=1,
            num_expert_group=1,
            scoring_func="sqrtsoftplus",
            tid2eid=torch.tensor([[3]], dtype=torch.int64),
        )

    expected = all_ids[rank * 4 : (rank + 1) * 4].to(torch.int64)
    expected[2] = 0
    assert torch.equal(captured[-1], expected)
    assert captured[-1].is_contiguous()
    assert captured[-1].dtype is torch.int64


def test_hash_selector_flashcomm_keeps_allgather_input_ids_unsplit(monkeypatch) -> None:
    captured: list[torch.Tensor] = []
    all_ids = torch.tensor([10, -1, 12, 13], dtype=torch.int32)

    def hash_op(**kwargs):
        captured.append(kwargs["input_ids"])
        return torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32), None

    monkeypatch.setattr(torch.ops._C_ascend, "moe_gating_top_k_hash", hash_op, raising=False)
    comm_method = SimpleNamespace(
        prepare_finalize=SimpleNamespace(
            all_gather_input_id_with_dp_group=lambda ids: ids
        )
    )
    with override_forward_context(
        _context(
            input_ids=all_ids,
            moe_comm_type=MoECommType.ALLGATHER,
            moe_comm_method=comm_method,
            flash_comm_v1_enabled=True,
        )
    ):
        experts_selector._select_experts_with_fusion_ops(
            hidden_states=torch.empty(1, 4),
            router_logits=torch.empty(1, 4),
            top_k=1,
            use_grouped_topk=False,
            renormalize=False,
            e_score_correction_bias=None,
            topk_group=1,
            num_expert_group=1,
            scoring_func="sqrtsoftplus",
            tid2eid=torch.tensor([[3]], dtype=torch.int64),
        )

    assert torch.equal(captured[-1], torch.tensor([10, 0, 12, 13], dtype=torch.int64))
