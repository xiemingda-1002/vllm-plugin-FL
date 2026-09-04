# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    int8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.runner import (
    moe_runner as moe_runner_module,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

from vllm_fl.distributed import ascend_parallel_state
from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization import moe_mc2


def _moe_config(is_sequence_parallel=True):
    parallel_config = SimpleNamespace(
        use_ep=True,
        ep_size=2,
        dp_size=2,
        pp_size=1,
        pcp_size=1,
        enable_eplb=False,
        enable_dbo=False,
    )
    return SimpleNamespace(
        moe_parallel_config=parallel_config,
        is_sequence_parallel=is_sequence_parallel,
        experts_per_token=2,
        activation=MoEActivation.SILU,
    )


def _quant_config(num_local_experts=2, hidden_size=4):
    return int8_w8a8_moe_quant_config(
        w1_scale=torch.ones(num_local_experts, 6),
        w2_scale=torch.ones(num_local_experts, hidden_size),
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=True,
    )


def _communicator():
    return moe_mc2._MC2Communicator(
        process_group=object(),
        group_name="fl-mc2-test",
        rank=0,
        world_size=2,
    )


class _FakeMC2Ops:
    def __init__(self):
        self.dispatch_calls = []
        self.combine_calls = []
        self.combined = None

    def npu_moe_distribute_dispatch_v2(self, **kwargs):
        self.dispatch_calls.append(kwargs)
        expanded = kwargs["x"].repeat_interleave(2, dim=0).to(torch.int8)
        return (
            expanded,
            torch.ones(expanded.shape[0], 1),
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([2, 2], dtype=torch.int64),
            torch.tensor([2, 2], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            None,
        )

    def npu_moe_distribute_combine_v2(self, **kwargs):
        self.combine_calls.append(kwargs)
        assert self.combined is not None
        return self.combined


class _FakeAllToAllDispatcher:
    def __init__(self):
        self.dispatch_calls = []
        self.combine_calls = []
        self.combined = None

    def dispatch(self, hidden_states, topk_weights, topk_ids):
        self.dispatch_calls.append(
            (hidden_states, topk_weights, topk_ids)
        )
        quantized = hidden_states.to(torch.int8)
        scale = torch.ones(hidden_states.shape[0], 1)
        expert_tokens = torch.tensor([1, 1], dtype=torch.int64)
        return quantized, scale, expert_tokens, object()

    def combine(self, hidden_states, state):
        self.combine_calls.append((hidden_states, state))
        assert self.combined is not None
        return self.combined


class _FakeEPGroup:
    def __init__(self):
        self.rank_in_group = 0
        self.world_size = 2
        self.ranks = [0, 1]
        self.device_group = object()
        self.dispatch_calls = []
        self.combine_calls = []

    def dispatch(self, hidden_states, topk_weights, topk_ids, **kwargs):
        self.dispatch_calls.append(
            (hidden_states, topk_weights, topk_ids, kwargs)
        )
        return hidden_states + 1, topk_weights, topk_ids

    def combine(self, hidden_states, **kwargs):
        self.combine_calls.append((hidden_states, kwargs))
        return hidden_states + 2


def test_alltoall_workspace_bound_covers_ep_routing(monkeypatch):
    config = _moe_config(is_sequence_parallel=True)
    config.experts_per_token = 6
    dispatcher = SimpleNamespace(
        ep_group=SimpleNamespace(world_size=16),
    )
    monkeypatch.setattr(
        moe_mc2,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    prepare_finalize = moe_mc2.AscendModelSlimW8A8MC2PrepareFinalize(
        config,
        communicator=None,
        torch_npu_module=None,
        local_num_experts=16,
        alltoall_dispatcher=dispatcher,
    )

    assert prepare_finalize.max_workspace_tokens(8192) == 196608


def test_workspace_bound_without_alltoall_uses_scheduler_budget():
    prepare_finalize = moe_mc2.AscendModelSlimW8A8MC2PrepareFinalize(
        _moe_config(),
        communicator=None,
        torch_npu_module=None,
        local_num_experts=2,
        alltoall_dispatcher=None,
    )

    assert prepare_finalize.max_workspace_tokens(8192) == 8192


def test_mc2_communicator_consumes_early_group_without_late_new_group(
    monkeypatch,
):
    class FakeHCCLBackend:
        def get_hccl_comm_name(self, rank):
            assert rank == 0
            return "early-mc2"

    class FakeProcessGroup:
        def _get_backend(self, device):
            assert device == torch.device("npu")
            return FakeHCCLBackend()

    ep_group = _FakeEPGroup()
    process_group = FakeProcessGroup()
    early_group = SimpleNamespace(
        ranks=list(ep_group.ranks),
        world_size=ep_group.world_size,
        rank_in_group=ep_group.rank_in_group,
        device_group=process_group,
    )
    monkeypatch.setattr(
        ascend_parallel_state,
        "get_ascend_mc2_group",
        lambda: early_group,
    )

    def reject_late_group(*args, **kwargs):
        raise AssertionError("quantization code must not create process groups")

    monkeypatch.setattr(
        torch.distributed,
        "new_group",
        reject_late_group,
    )

    communicator = moe_mc2._create_mc2_communicator(ep_group)

    assert communicator is not None
    assert communicator.process_group is process_group
    assert communicator.group_name == "early-mc2"
    assert communicator.rank == 0
    assert communicator.world_size == 2


@pytest.mark.parametrize(
    "missing_capability",
    (
        "static-ep-config",
        "alltoall-ops",
        "ep-group",
        "expert-layout",
        "weight-layout",
    ),
)
def test_static_capability_failure_preserves_old_quant_method(
    monkeypatch,
    missing_capability,
):
    fake_ops = _FakeMC2Ops()
    fake_ep_group = _FakeEPGroup()
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 4, 6, dtype=torch.int8),
    )

    monkeypatch.setattr(
        moe_mc2,
        "_static_ep_config_supported",
        lambda _: True,
    )
    monkeypatch.setattr(moe_mc2, "get_alltoall_ops", lambda: fake_ops)
    monkeypatch.setattr(
        moe_mc2,
        "AscendW8A8AllToAllDispatcher",
        lambda *args: _FakeAllToAllDispatcher(),
    )
    monkeypatch.setattr(moe_mc2, "_static_config_supported", lambda _: True)
    monkeypatch.setattr(moe_mc2, "_get_mc2_ops", lambda: fake_ops)
    monkeypatch.setattr(moe_mc2, "get_ep_group", lambda: fake_ep_group)
    monkeypatch.setattr(
        moe_mc2,
        "_expert_layout_supported",
        lambda layer, group: True,
    )
    monkeypatch.setattr(
        moe_mc2,
        "_weight_layout_supported",
        lambda layer: True,
    )
    monkeypatch.setattr(
        moe_mc2,
        "_create_mc2_communicator",
        lambda group: _communicator(),
    )

    if missing_capability == "static-ep-config":
        monkeypatch.setattr(
            moe_mc2,
            "_static_ep_config_supported",
            lambda _: False,
        )
    elif missing_capability == "alltoall-ops":
        monkeypatch.setattr(moe_mc2, "get_alltoall_ops", lambda: None)
    elif missing_capability == "ep-group":
        def missing_ep_group():
            raise AssertionError("EP group unavailable")

        monkeypatch.setattr(moe_mc2, "get_ep_group", missing_ep_group)
    elif missing_capability == "expert-layout":
        monkeypatch.setattr(
            moe_mc2,
            "_expert_layout_supported",
            lambda layer, group: False,
        )
    elif missing_capability == "weight-layout":
        monkeypatch.setattr(
            moe_mc2,
            "_weight_layout_supported",
            lambda layer: False,
        )
    kernel = moe_mc2.maybe_make_ordinary_mc2_kernel(
        _moe_config(),
        layer,
        _quant_config(),
    )

    assert kernel is None


def test_factory_builds_upstream_modular_kernel(monkeypatch):
    fake_ops = _FakeMC2Ops()
    fake_ep_group = _FakeEPGroup()
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 4, 6, dtype=torch.int8),
    )
    monkeypatch.setattr(
        moe_mc2,
        "_static_ep_config_supported",
        lambda _: True,
    )
    monkeypatch.setattr(moe_mc2, "get_alltoall_ops", lambda: fake_ops)
    fake_alltoall = _FakeAllToAllDispatcher()
    monkeypatch.setattr(
        moe_mc2,
        "AscendW8A8AllToAllDispatcher",
        lambda *args: fake_alltoall,
    )
    monkeypatch.setattr(moe_mc2, "_static_config_supported", lambda _: True)
    monkeypatch.setattr(moe_mc2, "_get_mc2_ops", lambda: fake_ops)
    monkeypatch.setattr(moe_mc2, "get_ep_group", lambda: fake_ep_group)
    monkeypatch.setattr(
        moe_mc2,
        "_expert_layout_supported",
        lambda layer, group: True,
    )
    monkeypatch.setattr(
        moe_mc2,
        "_weight_layout_supported",
        lambda layer: True,
    )
    monkeypatch.setattr(
        moe_mc2,
        "_create_mc2_communicator",
        lambda group: _communicator(),
    )

    kernel = moe_mc2.maybe_make_ordinary_mc2_kernel(
        _moe_config(),
        layer,
        _quant_config(),
    )

    assert isinstance(kernel, mk.FusedMoEKernel)
    assert isinstance(
        kernel.prepare_finalize,
        moe_mc2.AscendModelSlimW8A8MC2PrepareFinalize,
    )
    assert isinstance(
        kernel.fused_experts,
        moe_mc2.AscendModelSlimW8A8MC2Experts,
    )
    assert kernel.output_is_reduced() is True


def test_factory_keeps_alltoall_when_mc2_is_unavailable(monkeypatch):
    fake_ops = _FakeMC2Ops()
    fake_ep_group = _FakeEPGroup()
    fake_alltoall = _FakeAllToAllDispatcher()
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 4, 6, dtype=torch.int8),
        global_num_experts=4,
    )
    monkeypatch.setattr(
        moe_mc2,
        "_static_ep_config_supported",
        lambda _: True,
    )
    monkeypatch.setattr(moe_mc2, "get_alltoall_ops", lambda: fake_ops)
    monkeypatch.setattr(
        moe_mc2,
        "AscendW8A8AllToAllDispatcher",
        lambda *args: fake_alltoall,
    )
    monkeypatch.setattr(moe_mc2, "get_ep_group", lambda: fake_ep_group)
    monkeypatch.setattr(
        moe_mc2,
        "_expert_layout_supported",
        lambda layer, group: True,
    )
    monkeypatch.setattr(
        moe_mc2,
        "_weight_layout_supported",
        lambda layer: True,
    )
    monkeypatch.setattr(moe_mc2, "_static_config_supported", lambda _: True)
    monkeypatch.setattr(moe_mc2, "_get_mc2_ops", lambda: None)

    kernel = moe_mc2.maybe_make_ordinary_mc2_kernel(
        _moe_config(),
        layer,
        _quant_config(),
    )

    assert isinstance(kernel, mk.FusedMoEKernel)
    assert kernel.prepare_finalize.alltoall_dispatcher is fake_alltoall
    assert kernel.prepare_finalize.communicator is None


def test_full_graph_uses_paired_mc2_v2_contract(monkeypatch):
    fake_ops = _FakeMC2Ops()
    active_mask = torch.tensor([True, False])
    monkeypatch.setattr(moe_mc2, "_is_full_graph_runtime", lambda: True)
    monkeypatch.setattr(
        moe_mc2,
        "_active_mask_from_context",
        lambda num_tokens, is_sequence_parallel: (True, active_mask),
    )
    prepare_finalize = moe_mc2.AscendModelSlimW8A8MC2PrepareFinalize(
        _moe_config(),
        _communicator(),
        fake_ops,
        local_num_experts=2,
    )
    quant_config = _quant_config()
    experts = moe_mc2.AscendModelSlimW8A8MC2Experts(
        _moe_config(),
        quant_config,
        prepare_finalize,
    )
    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    topk_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
    topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    expert_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)

    (
        expanded,
        dynamic_scale,
        expert_metadata,
        dispatched_ids,
        dispatched_weights,
    ) = prepare_finalize.prepare(
        hidden_states,
        topk_weights,
        topk_ids,
        num_experts=4,
        expert_map=expert_map,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
        defer_input_quant=True,
    )

    assert len(fake_ops.dispatch_calls) == 1
    dispatch_kwargs = fake_ops.dispatch_calls[0]
    assert dispatch_kwargs["group_ep"] == "fl-mc2-test"
    assert dispatch_kwargs["group_tp"] == "fl-mc2-test"
    assert dispatch_kwargs["ep_world_size"] == 2
    assert dispatch_kwargs["tp_world_size"] == 1
    assert dispatch_kwargs["global_bs"] == 0
    assert dispatch_kwargs["expert_token_nums_type"] == 1
    assert dispatch_kwargs["quant_mode"] == 2
    assert "expert_scales" not in dispatch_kwargs
    assert dispatch_kwargs["x_active_mask"] is active_mask
    assert dispatch_kwargs["x"] is hidden_states
    assert dispatch_kwargs["x"].shape == hidden_states.shape
    assert dynamic_scale is not None
    assert torch.equal(dispatched_ids, topk_ids)
    assert torch.equal(dispatched_weights, topk_weights)
    assert expert_metadata is not None

    gmm_state = {}

    def fake_run_quantized_gmm(
        hidden,
        input_scale,
        w1,
        w2,
        w1_scale,
        w2_scale,
        expert_tokens,
        activation,
    ):
        del w1, w2, w1_scale, w2_scale, activation
        gmm_state["expert_tokens"] = expert_tokens
        gmm_state["input_scale"] = input_scale
        return hidden.float() + 5

    monkeypatch.setattr(
        experts,
        "_run_quantized_gmm",
        fake_run_quantized_gmm,
    )
    expert_output = torch.empty(expanded.shape, dtype=hidden_states.dtype)
    experts.apply(
        output=expert_output,
        hidden_states=expanded,
        w1=torch.empty(2, 4, 6, dtype=torch.int8),
        w2=torch.empty(2, 3, 4, dtype=torch.int8),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=4,
        expert_map=expert_map,
        a1q_scale=dynamic_scale,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=expert_metadata,
        apply_router_weight_on_input=False,
    )
    assert torch.equal(
        gmm_state["expert_tokens"],
        torch.tensor([2, 2], dtype=torch.int64),
    )
    assert gmm_state["input_scale"] is dynamic_scale
    assert torch.equal(expert_output, expanded.float() + 5)

    fake_ops.combined = torch.full_like(hidden_states, 9)
    output = torch.empty_like(hidden_states)
    prepare_finalize.finalize(
        output,
        expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=TopKWeightAndReduceNoOP(),
    )

    assert torch.equal(output, fake_ops.combined)
    assert len(fake_ops.combine_calls) == 1
    combine_kwargs = fake_ops.combine_calls[0]
    assert combine_kwargs["group_ep"] == dispatch_kwargs["group_ep"]
    assert combine_kwargs["group_tp"] == dispatch_kwargs["group_tp"]
    assert combine_kwargs["assist_info_for_combine"].dtype == torch.int32
    assert combine_kwargs["comm_quant_mode"] == 0
    assert combine_kwargs["x_active_mask"] is active_mask


def test_full_service_prefill_uses_ordinary_alltoall(monkeypatch):
    fake_ops = _FakeMC2Ops()
    fake_alltoall = _FakeAllToAllDispatcher()
    monkeypatch.setattr(moe_mc2, "_is_full_graph_runtime", lambda: False)
    prepare_finalize = moe_mc2.AscendModelSlimW8A8MC2PrepareFinalize(
        _moe_config(),
        _communicator(),
        fake_ops,
        local_num_experts=2,
        alltoall_dispatcher=fake_alltoall,
    )
    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)

    dispatched, input_scale, expert_metadata, dispatched_ids, dispatched_weights = (
        prepare_finalize.prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            num_experts=4,
            expert_map=torch.tensor([0, 1, -1, -1]),
            apply_router_weight_on_input=False,
            quant_config=_quant_config(),
            defer_input_quant=True,
        )
    )

    assert len(fake_ops.dispatch_calls) == 0
    assert len(fake_alltoall.dispatch_calls) == 1
    assert torch.equal(dispatched, hidden_states.to(torch.int8))
    assert input_scale is not None
    assert expert_metadata is not None
    assert torch.equal(dispatched_ids, topk_ids)
    assert torch.equal(dispatched_weights, topk_weights)

    fused_output = dispatched.to(torch.float32) * 2
    fake_alltoall.combined = torch.full_like(hidden_states, 7)
    output = torch.empty_like(hidden_states)
    prepare_finalize.finalize(
        output,
        fused_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=TopKWeightAndReduceNoOP(),
    )

    assert len(fake_alltoall.combine_calls) == 1
    assert torch.equal(output, fake_alltoall.combined)
    assert prepare_finalize._mode == "alltoall"
    assert prepare_finalize.output_is_reduced() is True


def test_non_sequence_parallel_fallback_reduces_once(monkeypatch):
    fake_ep_group = _FakeEPGroup()
    reduce_calls = []
    monkeypatch.setattr(moe_mc2, "_is_full_graph_runtime", lambda: False)
    monkeypatch.setattr(moe_mc2, "get_ep_group", lambda: fake_ep_group)

    def fake_all_reduce(tensor):
        reduce_calls.append(tensor)
        return tensor + 3

    monkeypatch.setattr(
        moe_mc2,
        "tensor_model_parallel_all_reduce",
        fake_all_reduce,
    )
    prepare_finalize = moe_mc2.AscendModelSlimW8A8MC2PrepareFinalize(
        _moe_config(is_sequence_parallel=False),
        _communicator(),
        _FakeMC2Ops(),
        local_num_experts=2,
    )
    hidden_states = torch.ones(2, 4)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    dispatched, _, _, _, _ = prepare_finalize.prepare(
        hidden_states,
        topk_weights,
        topk_ids,
        num_experts=4,
        expert_map=torch.tensor([0, 1, -1, -1]),
        apply_router_weight_on_input=False,
        quant_config=_quant_config(),
        defer_input_quant=True,
    )
    output = torch.empty_like(hidden_states)
    prepare_finalize.finalize(
        output,
        dispatched,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=TopKWeightAndReduceNoOP(),
    )

    assert len(fake_ep_group.combine_calls) == 1
    assert len(reduce_calls) == 1
    assert torch.equal(output, hidden_states + 6)


def test_fallback_reuses_routing_quantization_scale(monkeypatch):
    monkeypatch.setattr(moe_mc2, "get_ep_group", lambda: _FakeEPGroup())
    prepare_finalize = moe_mc2.AscendModelSlimW8A8MC2PrepareFinalize(
        _moe_config(),
        _communicator(),
        _FakeMC2Ops(),
        local_num_experts=2,
    )
    prepare_finalize._mode = "fallback"
    experts = moe_mc2.AscendModelSlimW8A8MC2Experts(
        _moe_config(),
        _quant_config(),
        prepare_finalize,
    )
    hidden_states = torch.ones(2, 4)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    expert_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    sorted_x = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2, 1)
    captured = {}

    from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization import (
        moe as moe_impl,
    )

    monkeypatch.setattr(
        moe_impl,
        "_npu_moe_init_routing",
        lambda *args, **kwargs: (
            sorted_x,
            torch.arange(2, dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int64),
            input_scale,
        ),
    )

    def fake_run_quantized_gmm(
        quantized,
        scale,
        w1,
        w2,
        w1_scale,
        w2_scale,
        expert_tokens,
        activation,
    ):
        del w1, w2, w1_scale, w2_scale, expert_tokens, activation
        captured["quantized"] = quantized
        captured["scale"] = scale
        return quantized.to(torch.float32)

    monkeypatch.setattr(
        experts,
        "_run_quantized_gmm",
        fake_run_quantized_gmm,
    )
    monkeypatch.setattr(
        moe_impl,
        "_npu_moe_token_unpermute",
        lambda routed, expanded_row_idx, weights: routed,
    )

    output = torch.empty_like(hidden_states)
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(2, 4, 6, dtype=torch.int8),
        w2=torch.empty(2, 3, 4, dtype=torch.int8),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=4,
        expert_map=expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert captured["quantized"] is sorted_x
    assert captured["scale"] is input_scale
    assert torch.equal(output, sorted_x.to(output.dtype))


def test_silu_experts_use_fused_gmm_swiglu_quant(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization import (
        moe as moe_impl,
    )

    captured = {}
    quantized_hidden = torch.ones(4, 3, dtype=torch.int8)
    hidden_scale = torch.ones(4, dtype=torch.float32)

    def fake_fused_gmm_swiglu_quant(
        x,
        weight,
        weight_scale,
        per_token_scale,
        expert_tokens,
    ):
        captured["fused"] = (
            x,
            weight,
            weight_scale,
            per_token_scale,
            expert_tokens,
        )
        return quantized_hidden, hidden_scale

    def fake_grouped_quant_matmul(
        x,
        weight,
        weight_scale,
        per_token_scale,
        expert_tokens,
        *,
        output_dtype,
    ):
        captured["gmm2"] = (
            x,
            weight,
            weight_scale,
            per_token_scale,
            expert_tokens,
            output_dtype,
        )
        return torch.full((4, 4), 7, dtype=output_dtype)

    monkeypatch.setattr(
        moe_impl,
        "_npu_grouped_matmul_swiglu_quant",
        fake_fused_gmm_swiglu_quant,
    )
    monkeypatch.setattr(
        moe_impl,
        "_npu_grouped_quant_matmul",
        fake_grouped_quant_matmul,
    )

    x = torch.ones(4, 4, dtype=torch.int8)
    input_scale = torch.ones(4, 1)
    w1 = torch.ones(2, 4, 6, dtype=torch.int8)
    w2 = torch.ones(2, 3, 4, dtype=torch.int8)
    w1_scale = torch.ones(2, 6, dtype=torch.float32)
    w2_scale = torch.ones(2, 4, dtype=torch.bfloat16)
    expert_tokens = torch.tensor([2, 2], dtype=torch.int64)

    output = moe_mc2.AscendModelSlimW8A8MC2Experts._run_quantized_gmm(
        x,
        input_scale,
        w1,
        w2,
        w1_scale,
        w2_scale,
        expert_tokens,
        MoEActivation.SILU,
    )

    fused_call = captured["fused"]
    assert fused_call[0] is x
    assert fused_call[1] is w1
    assert fused_call[2] is w1_scale
    assert fused_call[3] is input_scale
    assert fused_call[4] is expert_tokens
    gmm2_call = captured["gmm2"]
    assert gmm2_call[0] is quantized_hidden
    assert gmm2_call[1] is w2
    assert gmm2_call[2] is w2_scale
    assert gmm2_call[3] is hidden_scale
    assert torch.equal(gmm2_call[4], expert_tokens)
    assert gmm2_call[5] == torch.bfloat16
    assert torch.equal(output, torch.full((4, 4), 7, dtype=torch.bfloat16))


def test_full_graph_input_contract_failure_uses_compatibility_path(monkeypatch):
    fake_ops = _FakeMC2Ops()
    fake_ep_group = _FakeEPGroup()
    monkeypatch.setattr(moe_mc2, "_is_full_graph_runtime", lambda: True)
    monkeypatch.setattr(moe_mc2, "get_ep_group", lambda: fake_ep_group)
    monkeypatch.setattr(
        moe_mc2,
        "_active_mask_from_context",
        lambda num_tokens, is_sequence_parallel: (True, None),
    )
    prepare_finalize = moe_mc2.AscendModelSlimW8A8MC2PrepareFinalize(
        _moe_config(),
        _communicator(),
        fake_ops,
        local_num_experts=2,
    )
    prepare_finalize.prepare(
        torch.ones(2, 4),
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int64),
        num_experts=4,
        expert_map=torch.tensor([0, 1, -1, -1]),
        apply_router_weight_on_input=False,
        quant_config=_quant_config(),
        defer_input_quant=True,
    )

    assert len(fake_ops.dispatch_calls) == 0
    assert len(fake_ep_group.dispatch_calls) == 1
    assert fake_ep_group.dispatch_calls[0][3]["is_sequence_parallel"] is True


def test_full_graph_active_mask_failure_does_not_switch_collective(monkeypatch):
    fake_ep_group = _FakeEPGroup()
    monkeypatch.setattr(moe_mc2, "_is_full_graph_runtime", lambda: True)
    monkeypatch.setattr(moe_mc2, "get_ep_group", lambda: fake_ep_group)
    monkeypatch.setattr(
        moe_mc2,
        "_active_mask_from_context",
        lambda num_tokens, is_sequence_parallel: (False, None),
    )
    prepare_finalize = moe_mc2.AscendModelSlimW8A8MC2PrepareFinalize(
        _moe_config(),
        _communicator(),
        _FakeMC2Ops(),
        local_num_experts=2,
    )

    with pytest.raises(RuntimeError, match="active-mask contract"):
        prepare_finalize.prepare(
            torch.ones(2, 4),
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int32),
            num_experts=4,
            expert_map=torch.tensor([0, 1, -1, -1]),
            apply_router_weight_on_input=False,
            quant_config=_quant_config(),
            defer_input_quant=True,
        )

    assert len(fake_ep_group.dispatch_calls) == 0


def test_reduced_kernel_reduces_shared_and_routed_outputs_once(monkeypatch):
    reduce_calls = []

    def fake_all_reduce(tensor):
        reduce_calls.append(tensor.clone())
        return tensor + 10

    monkeypatch.setattr(
        moe_runner_module,
        "tensor_model_parallel_all_reduce",
        fake_all_reduce,
    )
    runner = MoERunner.__new__(MoERunner)
    runner.moe_config = SimpleNamespace(
        is_sequence_parallel=False,
        tp_size=2,
        ep_size=2,
    )
    runner.quant_method = SimpleNamespace(
        moe_kernel=SimpleNamespace(output_is_reduced=lambda: True)
    )
    shared_output = torch.ones(2, 4)
    routed_output = torch.full((2, 4), 2.0)

    reduced_shared = runner._maybe_reduce_shared_expert_output(shared_output)
    final = runner._maybe_reduce_final_output(
        reduced_shared + routed_output,
        trunc_size=4,
    )

    assert len(reduce_calls) == 1
    assert torch.equal(reduced_shared, shared_output + 10)
    assert torch.equal(final, shared_output + 10 + routed_output)


def test_sequence_parallel_is_static_mc2_capability(monkeypatch):
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=moe_mc2.CUDAGraphMode.FULL_DECODE_ONLY,
        ),
        speculative_config=None,
    )
    monkeypatch.setattr(moe_mc2, "_get_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(moe_mc2, "_is_a3", lambda: True)

    assert moe_mc2._static_config_supported(_moe_config())


def test_sequence_parallel_uses_local_mask_without_splitting_input(monkeypatch):
    full_mask = torch.tensor([True, False, False, True])
    context = SimpleNamespace(mc2_mask=full_mask, additional_kwargs={})
    monkeypatch.setattr(
        moe_mc2,
        "is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(moe_mc2, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        moe_mc2,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        moe_mc2,
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )

    valid, local_mask = moe_mc2._active_mask_from_context(
        num_tokens=2,
        is_sequence_parallel=True,
    )

    assert valid is True
    assert torch.equal(local_mask, full_mask[2:])


def test_operator_schema_requires_all_mc2_v2_arguments():
    compatible = SimpleNamespace(
        arguments=[
            SimpleNamespace(name=name)
            for name in moe_mc2._DISPATCH_V2_KWARGS
        ]
    )
    incompatible = SimpleNamespace(
        arguments=[SimpleNamespace(name="x")]
    )

    assert moe_mc2._schema_accepts_arguments(
        compatible,
        moe_mc2._DISPATCH_V2_KWARGS,
    )
    assert not moe_mc2._schema_accepts_arguments(
        incompatible,
        moe_mc2._DISPATCH_V2_KWARGS,
    )
