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

import sys
from types import SimpleNamespace

import torch

from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization import (
    moe_alltoall,
)


class _FakeNPUOps:
    def npu_dynamic_quant(self, tokens):
        scale = torch.arange(
            1,
            tokens.shape[0] + 1,
            dtype=torch.float32,
        ).unsqueeze(-1)
        return tokens.to(torch.int8), scale

    def npu_moe_token_permute(
        self,
        *,
        tokens,
        indices,
        num_out_tokens=None,
    ):
        flat_indices = indices.reshape(-1)
        order = torch.argsort(flat_indices, stable=True)
        if indices.ndim == 2:
            source_rows = torch.arange(tokens.shape[0]).repeat_interleave(
                indices.shape[1]
            )
        else:
            source_rows = torch.arange(tokens.shape[0])
        if num_out_tokens is not None:
            order = order[:num_out_tokens]
        return tokens[source_rows[order]], order.to(torch.int32)

    def npu_moe_token_unpermute(
        self,
        *,
        permuted_tokens,
        sorted_indices,
        probs=None,
        restore_shape=None,
    ):
        mapping = sorted_indices.to(torch.int64)
        if probs is None:
            output = torch.empty_like(permuted_tokens)
            output[mapping] = permuted_tokens
            return output

        assert restore_shape is not None
        output = permuted_tokens.new_zeros(restore_shape)
        flat_probs = probs.reshape(-1)
        top_k = probs.shape[1]
        for permuted_row, original_slot in enumerate(mapping.tolist()):
            output[original_slot // top_k] += (
                permuted_tokens[permuted_row]
                * flat_probs[original_slot]
            )
        return output


def _ep_group(*, rank=0, world_size=2):
    return SimpleNamespace(
        rank_in_group=rank,
        world_size=world_size,
        device_group=object(),
    )


def test_capability_loader_requires_complete_op_set(monkeypatch):
    fake_ops = _FakeNPUOps()
    monkeypatch.setitem(sys.modules, "torch_npu", fake_ops)
    assert moe_alltoall.get_alltoall_ops() is fake_ops

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_dynamic_quant=lambda tensor: tensor),
    )
    assert moe_alltoall.get_alltoall_ops() is None


def test_capability_requires_contiguous_expert_partition(monkeypatch):
    monkeypatch.setattr(moe_alltoall.dist, "is_available", lambda: True)
    fake_ops = _FakeNPUOps()

    assert moe_alltoall.is_alltoall_supported(
        _ep_group(),
        fake_ops,
        num_experts=4,
        local_num_experts=2,
    )
    assert not moe_alltoall.is_alltoall_supported(
        _ep_group(),
        fake_ops,
        num_experts=5,
        local_num_experts=2,
    )
    assert not moe_alltoall.is_alltoall_supported(
        _ep_group(world_size=1),
        fake_ops,
        num_experts=2,
        local_num_experts=2,
    )


def test_route_plan_uses_histograms_for_unequal_splits(monkeypatch):
    local_counts = torch.tensor([1, 1, 3, 1], dtype=torch.float32)
    global_counts = torch.tensor(
        [
            [1, 1, 3, 1],
            [2, 1, 0, 1],
        ],
        dtype=torch.float32,
    )
    gather_calls = []

    def fake_gather(counts, device_group, world_size):
        gather_calls.append((counts, device_group, world_size))
        return global_counts

    monkeypatch.setattr(
        moe_alltoall,
        "_all_gather_expert_counts",
        fake_gather,
    )

    group = _ep_group()
    plan = moe_alltoall._build_route_plan(
        local_counts,
        group,
        num_experts=4,
        local_num_experts=2,
    )

    assert plan.input_splits == (2, 4)
    assert plan.output_splits == (2, 3)
    assert torch.equal(plan.expert_tokens, torch.tensor([3, 2]))
    assert torch.equal(
        plan.local_expert_indices,
        torch.tensor([0, 1, 0, 0, 1], dtype=torch.int32),
    )
    assert gather_calls == [(local_counts, group.device_group, 2)]


def test_dispatch_exchanges_scale_and_activation_then_combine_inverts(
    monkeypatch,
):
    monkeypatch.setattr(moe_alltoall.dist, "is_available", lambda: True)
    monkeypatch.setattr(
        moe_alltoall,
        "_count_local_tokens_per_expert",
        lambda topk_ids, num_experts: torch.ones(
            num_experts,
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(
        moe_alltoall,
        "_all_gather_expert_counts",
        lambda local_counts, device_group, world_size: local_counts.repeat(
            world_size, 1
        ),
    )
    alltoall_calls = []

    def identity_alltoall(
        input_tensor,
        input_splits,
        output_splits,
        device_group,
    ):
        alltoall_calls.append(
            (
                input_tensor.clone(),
                input_splits,
                output_splits,
                device_group,
            )
        )
        return input_tensor.clone()

    monkeypatch.setattr(
        moe_alltoall,
        "_all_to_all_single",
        identity_alltoall,
    )

    group = _ep_group()
    dispatcher = moe_alltoall.AscendW8A8AllToAllDispatcher(
        group,
        _FakeNPUOps(),
        num_experts=4,
        local_num_experts=2,
    )
    local_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.25, 0.75], [0.60, 0.40]])

    quantized, input_scale, expert_tokens, state = dispatcher.dispatch(
        local_input,
        topk_weights,
        topk_ids,
    )

    assert torch.equal(
        quantized,
        torch.tensor(
            [[1, 2], [1, 2], [3, 4], [3, 4]],
            dtype=torch.int8,
        ),
    )
    assert input_scale.shape == (4, 1)
    assert torch.equal(
        input_scale,
        torch.tensor([[1.0], [3.0], [2.0], [4.0]]),
    )
    assert torch.equal(expert_tokens, torch.tensor([2, 2]))
    assert state.input_splits == (2, 2)
    assert state.output_splits == (2, 2)

    local_output = dispatcher.combine(quantized.to(torch.float32), state)

    assert torch.allclose(local_output, local_input)
    assert len(alltoall_calls) == 3
    assert alltoall_calls[0][0].shape == (4, 1)
    assert alltoall_calls[1][0].shape == (4, 2)
    assert alltoall_calls[2][0].shape == (4, 2)
    assert all(
        input_splits == (2, 2)
        and output_splits == (2, 2)
        and device_group is group.device_group
        for _, input_splits, output_splits, device_group in alltoall_calls
    )
