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
"""Ordinary all-to-all token routing for Ascend Dynamic W8A8 MoE.

The dispatcher operates on the local sequence-parallel token slice supplied
by the model.  It routes quantized activations to contiguous local experts and
combines expert results back into the same local slice.  Tensor-parallel
partitioning and gathering remain owned by the model boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

_REQUIRED_ALLTOALL_OPS = (
    "npu_dynamic_quant",
    "npu_moe_token_permute",
    "npu_moe_token_unpermute",
)


@dataclass(frozen=True)
class _AllToAllRoutePlan:
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    expert_tokens: torch.Tensor
    local_expert_indices: torch.Tensor | None


@dataclass(frozen=True)
class _AllToAllCombineState:
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    topk_weights: torch.Tensor
    reversed_local_input_permutation_mapping: torch.Tensor
    reversed_global_input_permutation_mapping: torch.Tensor | None
    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size


def get_alltoall_ops() -> Any | None:
    """Load the NPU operations required by ordinary all-to-all routing."""
    try:
        import torch_npu
    except (ImportError, OSError):
        return None

    if not all(
        callable(getattr(torch_npu, name, None))
        for name in _REQUIRED_ALLTOALL_OPS
    ):
        return None
    return torch_npu


def is_alltoall_supported(
    ep_group: Any,
    torch_npu_module: Any,
    num_experts: int,
    local_num_experts: int,
) -> bool:
    """Return whether the static ordinary all-to-all contract is available."""
    try:
        world_size = int(ep_group.world_size)
        rank = int(ep_group.rank_in_group)
        device_group = ep_group.device_group
    except (AttributeError, TypeError, ValueError):
        return False

    return bool(
        world_size > 1
        and 0 <= rank < world_size
        and device_group is not None
        and num_experts > 0
        and local_num_experts > 0
        and world_size * local_num_experts == num_experts
        and all(
            callable(getattr(torch_npu_module, name, None))
            for name in _REQUIRED_ALLTOALL_OPS
        )
        and dist.is_available()
    )


def _count_local_tokens_per_expert(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Count routed tokens while keeping the histogram on the input device."""
    return torch.histc(
        topk_ids,
        bins=num_experts,
        min=0,
        max=num_experts,
    )


def _all_gather_expert_counts(
    local_counts: torch.Tensor,
    device_group: Any,
    world_size: int,
) -> torch.Tensor:
    """Gather each rank's complete expert histogram."""
    gathered = local_counts.new_empty(world_size * local_counts.numel())
    dist.all_gather_into_tensor(
        gathered,
        local_counts.contiguous(),
        group=device_group,
    )
    return gathered.view(world_size, local_counts.numel())


def _all_to_all_single(
    input_tensor: torch.Tensor,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    device_group: Any,
) -> torch.Tensor:
    """Exchange a tensor's first dimension with unequal per-rank splits."""
    output = input_tensor.new_empty(
        (sum(output_splits), *input_tensor.shape[1:])
    )
    dist.all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=device_group,
    )
    return output


def _token_permute(
    torch_npu_module: Any,
    tokens: torch.Tensor,
    indices: torch.Tensor,
    *,
    num_out_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs: dict[str, Any] = {
        "tokens": tokens,
        "indices": indices,
    }
    if num_out_tokens is not None:
        kwargs["num_out_tokens"] = num_out_tokens
    return torch_npu_module.npu_moe_token_permute(**kwargs)


def _token_unpermute(
    torch_npu_module: Any,
    tokens: torch.Tensor,
    reversed_mapping: torch.Tensor,
    *,
    topk_weights: torch.Tensor | None = None,
    restore_shape: torch.Size | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "permuted_tokens": tokens,
        "sorted_indices": reversed_mapping,
    }
    if topk_weights is not None:
        kwargs["probs"] = topk_weights
    if restore_shape is not None:
        kwargs["restore_shape"] = restore_shape
    return torch_npu_module.npu_moe_token_unpermute(**kwargs)


def _build_route_plan(
    local_counts: torch.Tensor,
    ep_group: Any,
    num_experts: int,
    local_num_experts: int,
) -> _AllToAllRoutePlan:
    """Build split vectors and the received-token local-expert ordering."""
    world_size = int(ep_group.world_size)
    rank = int(ep_group.rank_in_group)
    local_counts_by_destination = local_counts.reshape(
        world_size, local_num_experts
    )
    input_splits = tuple(
        int(value)
        for value in local_counts_by_destination.sum(dim=1).cpu().tolist()
    )

    global_counts = _all_gather_expert_counts(
        local_counts,
        ep_group.device_group,
        world_size,
    ).reshape(world_size, num_experts)
    first_local_expert = rank * local_num_experts
    received_counts = global_counts[
        :,
        first_local_expert : first_local_expert + local_num_experts,
    ]
    output_splits = tuple(
        int(value) for value in received_counts.sum(dim=1).cpu().tolist()
    )
    expert_tokens = received_counts.sum(dim=0).to(torch.int64)

    local_expert_indices = None
    if local_num_experts > 1:
        expert_ids_per_source = torch.arange(
            local_num_experts,
            dtype=torch.int32,
            device=local_counts.device,
        ).repeat(world_size)
        local_expert_indices = torch.repeat_interleave(
            expert_ids_per_source,
            received_counts.reshape(-1).to(torch.int64),
        )

    return _AllToAllRoutePlan(
        input_splits=input_splits,
        output_splits=output_splits,
        expert_tokens=expert_tokens,
        local_expert_indices=local_expert_indices,
    )


class AscendW8A8AllToAllDispatcher:
    """Dispatch and combine one local token slice across contiguous EP ranks."""

    def __init__(
        self,
        ep_group: Any,
        torch_npu_module: Any,
        num_experts: int,
        local_num_experts: int,
    ) -> None:
        if not is_alltoall_supported(
            ep_group,
            torch_npu_module,
            num_experts,
            local_num_experts,
        ):
            raise ValueError(
                "ordinary all-to-all requires an initialized EP group with "
                "more than one rank, contiguous experts, and Ascend MoE ops"
            )
        self.ep_group = ep_group
        self.device_group = ep_group.device_group
        self.torch_npu = torch_npu_module
        self.num_experts = num_experts
        self.local_num_experts = local_num_experts

    def dispatch(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        _AllToAllCombineState,
    ]:
        """Route and quantize the caller's local sequence-parallel slice."""
        hidden_shape = a1.shape
        if a1.ndim < 2:
            raise ValueError("MoE hidden states must have at least two dimensions")
        flattened_a1 = a1.reshape(-1, a1.shape[-1])
        if (
            topk_weights.ndim != 2
            or topk_ids.ndim != 2
            or topk_weights.shape != topk_ids.shape
            or topk_ids.shape[0] != flattened_a1.shape[0]
        ):
            raise ValueError(
                "top-k weights and IDs must be matching 2D tensors with one "
                "row per local hidden-state token"
            )

        local_counts = _count_local_tokens_per_expert(
            topk_ids,
            self.num_experts,
        )
        permuted_local, reversed_local = _token_permute(
            self.torch_npu,
            flattened_a1,
            topk_ids,
            num_out_tokens=topk_ids.numel(),
        )
        quantized_local, input_scale = self.torch_npu.npu_dynamic_quant(
            permuted_local
        )

        # The count all-gather is the first collective.  From this point onward
        # failures propagate; this dispatcher never makes a rank-local fallback.
        plan = _build_route_plan(
            local_counts,
            self.ep_group,
            self.num_experts,
            self.local_num_experts,
        )
        exchanged_scale = _all_to_all_single(
            input_scale,
            plan.input_splits,
            plan.output_splits,
            self.device_group,
        )
        exchanged_a1 = _all_to_all_single(
            quantized_local,
            plan.input_splits,
            plan.output_splits,
            self.device_group,
        )

        reversed_global = None
        if plan.local_expert_indices is not None:
            exchanged_a1, reversed_global = _token_permute(
                self.torch_npu,
                exchanged_a1,
                plan.local_expert_indices,
            )
            scale_was_vector = exchanged_scale.ndim == 1
            scale_tokens = (
                exchanged_scale.unsqueeze(-1)
                if scale_was_vector
                else exchanged_scale
            )
            scale_tokens, _ = _token_permute(
                self.torch_npu,
                scale_tokens,
                plan.local_expert_indices,
            )
            exchanged_scale = (
                scale_tokens.squeeze(-1) if scale_was_vector else scale_tokens
            )

        state = _AllToAllCombineState(
            input_splits=plan.input_splits,
            output_splits=plan.output_splits,
            topk_weights=topk_weights,
            reversed_local_input_permutation_mapping=reversed_local,
            reversed_global_input_permutation_mapping=reversed_global,
            hidden_shape=hidden_shape,
            hidden_shape_before_permute=flattened_a1.shape,
        )
        return exchanged_a1, exchanged_scale, plan.expert_tokens, state

    def combine(
        self,
        fused_expert_output: torch.Tensor,
        combine_state: _AllToAllCombineState,
    ) -> torch.Tensor:
        """Return expert output to the same local token slice as ``dispatch``."""
        reversed_global = (
            combine_state.reversed_global_input_permutation_mapping
        )
        if reversed_global is not None:
            fused_expert_output = _token_unpermute(
                self.torch_npu,
                fused_expert_output,
                reversed_global,
            )

        returned_local = _all_to_all_single(
            fused_expert_output,
            combine_state.output_splits,
            combine_state.input_splits,
            self.device_group,
        )
        local_output = _token_unpermute(
            self.torch_npu,
            returned_local,
            combine_state.reversed_local_input_permutation_mapping.to(
                torch.int32
            ),
            topk_weights=combine_state.topk_weights,
            restore_shape=combine_state.hidden_shape_before_permute,
        )
        return local_output.view(combine_state.hidden_shape)
