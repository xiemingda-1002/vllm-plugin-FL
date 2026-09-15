# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import torch
import pytest

from vllm_fl import ascend_forward_context as afc


@pytest.fixture(autouse=True)
def _keep_cpu_contracts_off_the_a3_runtime_path(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.ascend.hardware import AscendDeviceType

    monkeypatch.setattr(afc, "_active_ep_world_size", lambda: 4)
    monkeypatch.setattr(afc, "_get_ascend_device_type", lambda: AscendDeviceType.A2)


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


def test_a3_ordinary_selector_uses_capacity_without_fused_mc2(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.hardware import AscendDeviceType
    from vllm_fl.dispatch.backends.vendor.ascend.impl.moe import compat

    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    monkeypatch.setattr(afc, "_active_ep_world_size", lambda: 16)
    monkeypatch.setattr(afc, "_get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(
        compat, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=0)
    )
    config = _config(ep_size=16, num_experts=256)

    assert afc.select_moe_comm_method(128, config) is afc.MoECommType.MC2
    assert afc.select_moe_comm_method(129, config) is afc.MoECommType.ALLTOALL


def test_a3_fused_mc2_selector_covers_both_capacity_sides(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.hardware import AscendDeviceType
    from vllm_fl.dispatch.backends.vendor.ascend.impl.moe import compat

    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    monkeypatch.setattr(afc, "_active_ep_world_size", lambda: 16)
    monkeypatch.setattr(afc, "_get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(
        compat, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1)
    )
    config = _config(ep_size=16, num_experts=256)

    assert afc.select_moe_comm_method(128, config) is afc.MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(129, config) is afc.MoECommType.FUSED_MC2


def test_a3_fused_mc2_fails_closed_above_ep32(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.hardware import AscendDeviceType
    from vllm_fl.dispatch.backends.vendor.ascend.impl.moe import compat

    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    monkeypatch.setattr(afc, "_active_ep_world_size", lambda: 33)
    monkeypatch.setattr(afc, "_get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(
        compat, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1)
    )
    config = _config(ep_size=33, num_experts=256)

    assert afc.select_moe_comm_method(128, config) is afc.MoECommType.MC2
    assert afc.select_moe_comm_method(129, config) is afc.MoECommType.ALLTOALL


def test_active_ep_one_uses_allgather_despite_config_ep_size(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.hardware import AscendDeviceType

    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    monkeypatch.setattr(afc, "_active_ep_world_size", lambda: 1)
    monkeypatch.setattr(afc, "_get_ascend_device_type", lambda: AscendDeviceType.A3)

    assert afc.select_moe_comm_method(
        128, _config(ep_size=16, num_experts=256)
    ) is afc.MoECommType.ALLGATHER


def test_310p_keeps_allgather_and_a5_is_explicitly_rejected(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.hardware import AscendDeviceType

    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    monkeypatch.setattr(afc, "_active_ep_world_size", lambda: 16)
    config = _config(ep_size=16, num_experts=256)

    monkeypatch.setattr(afc, "_get_ascend_device_type",
                        lambda: AscendDeviceType._310P)
    assert afc.select_moe_comm_method(128, config) is afc.MoECommType.ALLGATHER

    monkeypatch.setattr(afc, "_get_ascend_device_type", lambda: AscendDeviceType.A5)
    with pytest.raises(NotImplementedError, match="A5"):
        afc.select_moe_comm_method(128, config)


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


def test_context_selects_from_synced_dp_max_not_local_tokens(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.hardware import AscendDeviceType

    dummy_method = object()
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    monkeypatch.setattr(afc, "_reserved_mc2_mask", torch.zeros(256, dtype=torch.bool))
    monkeypatch.setattr(afc, "_active_ep_world_size", lambda: 16)
    monkeypatch.setattr(afc, "_get_ascend_device_type", lambda: AscendDeviceType.A3)
    selected = []
    monkeypatch.setattr(afc, "_get_moe_comm_method", lambda comm: selected.append(comm) or dummy_method)

    context = afc.build_additional_forward_context(
        attn_metadata=None,
        vllm_config=_config(ep_size=16, num_experts=256),
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([2, 129])),
        num_tokens=2,
        num_tokens_across_dp=torch.tensor([2, 129]),
    )

    assert context["max_tokens_across_dp"] == 129
    assert context["moe_comm_type"] is afc.MoECommType.ALLTOALL
    assert selected == [afc.MoECommType.ALLTOALL]
    assert context["mc2_mask"].shape == (130,)
    assert context["mc2_mask"].tolist()[:2] == [True, True]
    assert not context["mc2_mask"][2:].any()


def test_normal_forward_actual_count_masks_only_real_rows_and_reuses_buffer(
    monkeypatch,
) -> None:
    """rc1 normal forward: actual 3 in a graph/DP extent of 4 is TTTF."""
    reserved = torch.zeros(16, dtype=torch.bool)
    monkeypatch.setattr(afc, "_reserved_mc2_mask", reserved)
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    monkeypatch.setattr(afc, "_get_moe_comm_method", lambda _: object())
    config = _config(tp_size=4)

    with afc.override_actual_num_tokens(3):
        context = afc.build_additional_forward_context(
            attn_metadata=None,
            vllm_config=config,
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor([4])
            ),
            num_tokens=4,
            num_tokens_across_dp=torch.tensor([4]),
        )

    mask = context["mc2_mask"]
    assert mask.data_ptr() == reserved.data_ptr()
    assert mask.tolist() == [True, True, True, False]
    # PrepareAndFinalizeWithMC2 slices this tensor along TP: rank 3 receives
    # the padded row and must see it inactive.
    assert [part.tolist() for part in torch.tensor_split(mask, 4)] == [
        [True],
        [True],
        [True],
        [False],
    ]

    # Context scope reset preserves rc1 dummy/capture default: padded rows are
    # active and the returned tensor remains a view of the same reserved buffer.
    dummy_context = afc.build_additional_forward_context(
        attn_metadata=None,
        vllm_config=config,
        dp_metadata=None,
        num_tokens=4,
    )
    assert dummy_context["mc2_mask"].data_ptr() == reserved.data_ptr()
    assert dummy_context["mc2_mask"].tolist() == [True, True, True, True]


def test_actual_token_scope_validates_extent_and_no_mask_path(monkeypatch) -> None:
    monkeypatch.setattr(afc, "_reserved_mc2_mask", None)
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", 128)
    monkeypatch.setattr(afc, "_get_moe_comm_method", lambda _: object())

    with afc.override_actual_num_tokens(1):
        context = afc.build_additional_forward_context(
            attn_metadata=None,
            vllm_config=_config(tp_size=4),
            dp_metadata=None,
            num_tokens=4,
        )
    assert context["mc2_mask"] is None

    with afc.override_actual_num_tokens(5), pytest.raises(
        ValueError, match="cannot exceed"
    ):
        afc.build_additional_forward_context(
            attn_metadata=None,
            vllm_config=_config(tp_size=4),
            dp_metadata=None,
            num_tokens=4,
        )
