# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import torch

from vllm_fl import ascend_forward_context as afc


def _config(
    *,
    is_moe: bool = True,
    enable_ep: bool = True,
    ep_size: int = 4,
    tp_size: int = 2,
    num_experts: int = 128,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            is_moe_model=is_moe,
            enable_expert_parallel=enable_ep,
            world_size_across_dp=ep_size,
            pipeline_parallel_size=1,
            tensor_parallel_size=tp_size,
        ),
        model_config=SimpleNamespace(get_num_experts=lambda: num_experts),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[],
            max_cudagraph_capture_size=0,
        ),
        additional_config={},
    )


def test_a2_qwen_dp2_tp2_ep4_uses_allgather(monkeypatch) -> None:
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)

    assert afc.select_moe_comm_method(2, _config()) is afc.MoECommType.ALLGATHER


def test_a2_mc2_threshold_matches_rc1(monkeypatch) -> None:
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    config = _config(ep_size=16, num_experts=128)

    assert afc.select_moe_comm_method(128, config) is afc.MoECommType.MC2
    assert afc.select_moe_comm_method(129, config) is afc.MoECommType.ALLGATHER


def test_mc2_capacity_matches_rc1_decode_rounding(monkeypatch) -> None:
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
    config = _config(tp_size=8)

    afc.set_mc2_tokens_capacity(
        config,
        max_num_reqs=513,
        uniform_decode_query_len=1,
    )

    assert afc.get_mc2_tokens_capacity() == 520


def test_non_moe_has_no_ascend_communication_method() -> None:
    assert afc.select_moe_comm_method(4, _config(is_moe=False)) is None


def test_context_contains_rc1_allgather_fields(monkeypatch) -> None:
    dummy_method = object()
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    monkeypatch.setattr(afc, "_reserved_mc2_mask", None)
    monkeypatch.setattr(afc, "_get_moe_comm_method", lambda _: dummy_method)
    dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([1, 2]))

    context = afc.build_additional_forward_context(
        attn_metadata=None,
        vllm_config=_config(),
        dp_metadata=dp_metadata,
        num_tokens=1,
    )

    assert context["moe_comm_type"] is afc.MoECommType.ALLGATHER
    assert context["moe_comm_method"] is dummy_method
    assert context["max_tokens_across_dp"] == 2
    assert context["padded_num_tokens"] == 2
    assert context["mmrs_fusion"] is False
    assert context["flash_comm_v1_enabled"] is False
