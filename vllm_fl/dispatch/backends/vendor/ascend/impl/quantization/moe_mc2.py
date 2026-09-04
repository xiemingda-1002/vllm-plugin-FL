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
"""Distributed communication for Ascend ModelSlim Dynamic W8A8 MoE.

The modular MoE boundary uses ordinary all-to-all for eager execution and
service prefill. Eligible A3 full-graph decode calls use the paired
``dispatch_v2`` and ``combine_v2`` MC2 operators. Both paths return the local
sequence-parallel token slice expected by the model-level gather, while the
compatibility path completes its own reduction. Consequently the kernel has
one static ``output_is_reduced`` contract for every runtime phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

from .moe_alltoall import (
    AscendW8A8AllToAllDispatcher,
    get_alltoall_ops,
)

logger = init_logger(__name__)

_EXPERT_TOKEN_NUMS_TYPE_COUNT = 1
_DISPATCH_OUTPUT_SIZE = 7

_DISPATCH_V2_KWARGS = frozenset(
    {
        "x",
        "expert_ids",
        "scales",
        "expert_shard_type",
        "shared_expert_rank_num",
        "moe_expert_num",
        "group_ep",
        "ep_world_size",
        "ep_rank_id",
        "group_tp",
        "tp_world_size",
        "tp_rank_id",
        "global_bs",
        "expert_token_nums_type",
        "quant_mode",
        "x_active_mask",
    }
)
_COMBINE_V2_KWARGS = frozenset(
    {
        "expand_x",
        "expert_ids",
        "expert_scales",
        "assist_info_for_combine",
        "ep_send_counts",
        "tp_send_counts",
        "expand_scales",
        "expert_shard_type",
        "shared_expert_rank_num",
        "moe_expert_num",
        "group_ep",
        "ep_world_size",
        "ep_rank_id",
        "group_tp",
        "tp_world_size",
        "tp_rank_id",
        "global_bs",
        "comm_quant_mode",
        "x_active_mask",
    }
)


@dataclass(frozen=True)
class _MC2Communicator:
    process_group: Any
    group_name: str
    rank: int
    world_size: int


@dataclass
class _MC2CombineState:
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    assist_info_for_combine: torch.Tensor
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor
    expand_scales: torch.Tensor | None
    active_mask: torch.Tensor | None
    num_experts: int


def _schema_accepts_arguments(schema: Any, required: frozenset[str]) -> bool:
    """Return whether a registered Torch operator schema has all arguments."""
    arguments = getattr(schema, "arguments", None)
    if arguments is None:
        return False
    names = {getattr(argument, "name", None) for argument in arguments}
    return required.issubset(names)


def _registered_op_schema(name: str) -> Any | None:
    """Read the dispatcher schema behind torch_npu's opaque wrappers."""
    try:
        packet = getattr(torch.ops.npu, name)
        return packet.default._schema
    except (AttributeError, RuntimeError):
        return None


def _get_mc2_ops() -> Any | None:
    try:
        import torch_npu
    except (ImportError, OSError):
        return None

    dispatch = getattr(torch_npu, "npu_moe_distribute_dispatch_v2", None)
    combine = getattr(torch_npu, "npu_moe_distribute_combine_v2", None)
    if not callable(dispatch) or not callable(combine):
        return None
    dispatch_schema = _registered_op_schema("npu_moe_distribute_dispatch_v2")
    combine_schema = _registered_op_schema("npu_moe_distribute_combine_v2")
    if not _schema_accepts_arguments(dispatch_schema, _DISPATCH_V2_KWARGS):
        return None
    if not _schema_accepts_arguments(combine_schema, _COMBINE_V2_KWARGS):
        return None
    return torch_npu


def _is_a3() -> bool:
    try:
        from vllm_fl.cpu_binding import (
            AscendDeviceType,
            get_ascend_device_type,
        )

        return get_ascend_device_type() == AscendDeviceType.A3
    except (ImportError, OSError, RuntimeError):
        return False


def _get_vllm_config() -> Any | None:
    try:
        return get_current_vllm_config()
    except (AssertionError, LookupError, RuntimeError):
        return None


def _static_ep_config_supported(moe_config: Any) -> bool:
    vllm_config = _get_vllm_config()
    if vllm_config is None:
        return False

    parallel_config = getattr(moe_config, "moe_parallel_config", None)
    if parallel_config is None:
        parallel_config = getattr(vllm_config, "parallel_config", None)
    if (
        parallel_config is None
        or not bool(getattr(parallel_config, "use_ep", False))
        or int(getattr(parallel_config, "ep_size", 1)) <= 1
        or int(getattr(parallel_config, "pp_size", 1)) != 1
        or int(getattr(parallel_config, "pcp_size", 1)) != 1
        or bool(getattr(parallel_config, "enable_eplb", False))
        or bool(getattr(parallel_config, "enable_dbo", False))
    ):
        return False
    if getattr(vllm_config, "speculative_config", None) is not None:
        return False
    return True


def _static_config_supported(moe_config: Any) -> bool:
    """Return whether paired ordinary MC2 can serve full-graph decode."""
    if not _static_ep_config_supported(moe_config) or not _is_a3():
        return False
    vllm_config = _get_vllm_config()
    compilation_config = getattr(vllm_config, "compilation_config", None)
    return bool(
        getattr(compilation_config, "cudagraph_mode", None)
        == CUDAGraphMode.FULL_DECODE_ONLY
    )


def _expert_layout_supported(layer: torch.nn.Module, ep_group: Any) -> bool:
    expert_map = getattr(layer, "expert_map", None)
    if expert_map is None or expert_map.ndim != 1:
        return False

    global_num_experts = int(getattr(layer, "global_num_experts", 0))
    local_num_experts = int(getattr(layer, "local_num_experts", 0))
    if (
        global_num_experts <= 0
        or local_num_experts <= 0
        or expert_map.numel() != global_num_experts
        or local_num_experts * int(ep_group.world_size)
        != global_num_experts
    ):
        return False

    # expert_shard_type=0 requires each rank to own one contiguous interval.
    first_expert = int(ep_group.rank_in_group) * local_num_experts
    expected = torch.full_like(expert_map, -1)
    expected[first_expert : first_expert + local_num_experts] = torch.arange(
        local_num_experts,
        dtype=expert_map.dtype,
        device=expert_map.device,
    )
    return bool(torch.equal(expert_map, expected))


def _weight_layout_supported(layer: torch.nn.Module) -> bool:
    tensors = (
        getattr(layer, "w13_weight", None),
        getattr(layer, "w2_weight", None),
        getattr(layer, "w13_weight_scale", None),
        getattr(layer, "w2_weight_scale", None),
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    w13, w2, w13_scale, w2_scale = tensors
    return bool(
        w13.dtype == torch.int8
        and w2.dtype == torch.int8
        and w13.ndim == 3
        and w2.ndim == 3
        and w13_scale.ndim == 2
        and w2_scale.ndim == 2
        and w13.shape[0] == w2.shape[0] == w13_scale.shape[0]
        and w13.shape[0] == w2_scale.shape[0]
        and not bool(getattr(layer, "apply_router_weight_on_input", False))
    )


def _create_mc2_communicator(ep_group: Any) -> _MC2Communicator | None:
    """Resolve the HCCL communicator created during distributed startup."""
    from vllm_fl.distributed.ascend_parallel_state import (
        get_ascend_mc2_group,
    )

    try:
        mc2_group = get_ascend_mc2_group()
        if (
            list(mc2_group.ranks) != list(ep_group.ranks)
            or int(mc2_group.world_size) != int(ep_group.world_size)
            or int(mc2_group.rank_in_group) != int(ep_group.rank_in_group)
        ):
            raise RuntimeError(
                "MC2 and expert-parallel group membership differ"
            )
        process_group = mc2_group.device_group
        group_rank = int(mc2_group.rank_in_group)
        backend = process_group._get_backend(torch.device("npu"))
        group_name = backend.get_hccl_comm_name(group_rank)
    except (AssertionError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning_once(
            "Ascend ordinary MC2 communicator is unavailable; using the "
            "existing ModelSlim MoE path: %s",
            exc,
        )
        return None

    if not isinstance(group_name, str) or not group_name:
        return None
    return _MC2Communicator(
        process_group=process_group,
        group_name=group_name,
        rank=group_rank,
        world_size=int(mc2_group.world_size),
    )


def _active_mask_from_context(
    num_tokens: int,
    is_sequence_parallel: bool,
) -> tuple[bool, torch.Tensor | None]:
    if not is_forward_context_available():
        return True, None
    context = get_forward_context()
    active_mask = getattr(context, "mc2_mask", None)
    if active_mask is None:
        active_mask = getattr(context, "additional_kwargs", {}).get(
            "mc2_mask"
        )
    if active_mask is None:
        return True, None
    if (
        not isinstance(active_mask, torch.Tensor)
        or active_mask.ndim != 1
        or active_mask.dtype != torch.bool
    ):
        return False, None
    if active_mask.numel() == num_tokens:
        return True, active_mask
    if is_sequence_parallel:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if tp_size > 1:
            local_mask = torch.tensor_split(active_mask, tp_size, dim=0)[tp_rank]
            if local_mask.numel() == num_tokens:
                return True, local_mask
    return False, None


def _is_full_graph_runtime() -> bool:
    return bool(
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode
        == CUDAGraphMode.FULL
    )


class AscendModelSlimW8A8MC2PrepareFinalize(
    mk.FusedMoEPrepareAndFinalizeModular
):
    """All-to-all, paired-MC2, and compatibility prepare/finalize path."""

    def __init__(
        self,
        moe_config: Any,
        communicator: _MC2Communicator | None,
        torch_npu_module: Any | None,
        local_num_experts: int,
        alltoall_dispatcher: AscendW8A8AllToAllDispatcher | None = None,
    ) -> None:
        super().__init__()
        self.moe_config = moe_config
        self.communicator = communicator
        self.torch_npu = torch_npu_module
        self.local_num_experts = local_num_experts
        self.alltoall_dispatcher = alltoall_dispatcher
        self._mode: Literal["mc2", "alltoall", "fallback"] = "fallback"
        self._mc2_combine_state: _MC2CombineState | None = None
        self._alltoall_combine_state: Any | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def max_workspace_tokens(self, max_num_tokens: int) -> int:
        """Bound expert tokens received by one rank after all-to-all.

        ``max_num_tokens`` is the scheduler budget of one DP engine. Sequence
        parallelism divides that budget across TP ranks, while EP all-to-all
        can route every top-k assignment in the EP group to experts owned by
        one destination rank. Reserve that correctness bound before graph
        capture locks the shared modular-kernel workspace.
        """
        if max_num_tokens <= 0:
            raise ValueError("max_num_tokens must be positive")
        if self.alltoall_dispatcher is None:
            return max_num_tokens

        ep_size = int(self.alltoall_dispatcher.ep_group.world_size)
        tp_size = get_tensor_model_parallel_world_size()
        if bool(getattr(self.moe_config, "is_sequence_parallel", False)):
            local_tokens = (max_num_tokens + tp_size - 1) // tp_size
        else:
            local_tokens = max_num_tokens
        top_k = int(getattr(self.moe_config, "experts_per_token", 1))
        return local_tokens * ep_size * top_k

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def num_dispatchers(self) -> int:
        return 1

    def output_is_reduced(self) -> bool:
        # MC2 and all-to-all combine complete routed-expert communication.
        # The compatibility finalize performs its own final TP reduction.
        return True

    def _input_contract_supported(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> bool:
        return bool(
            a1.ndim == 2
            and topk_weights.ndim == 2
            and topk_ids.ndim == 2
            and topk_weights.shape == topk_ids.shape
            and topk_ids.shape[0] == a1.shape[0]
            and topk_ids.dtype == torch.int32
            and expert_map is not None
            and expert_map.ndim == 1
            and expert_map.numel() == num_experts
            and not apply_router_weight_on_input
        )

    def _fallback_prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> mk.PrepareResultType:
        self._mode = "fallback"
        self._mc2_combine_state = None
        self._alltoall_combine_state = None
        dispatched = get_ep_group().dispatch(
            a1,
            topk_weights,
            topk_ids,
            is_sequence_parallel=bool(
                getattr(self.moe_config, "is_sequence_parallel", False)
            ),
        )
        if not isinstance(dispatched, (tuple, list)) or len(dispatched) != 3:
            raise RuntimeError(
                "vLLM EP dispatch must return hidden states, top-k weights "
                "and top-k IDs"
            )
        dispatched_a1, dispatched_weights, dispatched_ids = dispatched
        return (
            dispatched_a1,
            None,
            None,
            dispatched_ids,
            dispatched_weights,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        del quant_config
        if not defer_input_quant:
            return self._fallback_prepare(a1, topk_weights, topk_ids)
        if not self._input_contract_supported(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
        ):
            return self._fallback_prepare(a1, topk_weights, topk_ids)

        if not _is_full_graph_runtime():
            if self.alltoall_dispatcher is None:
                return self._fallback_prepare(a1, topk_weights, topk_ids)
            (
                dispatched_a1,
                input_scale,
                expert_token_nums,
                combine_state,
            ) = self.alltoall_dispatcher.dispatch(
                a1,
                topk_weights,
                topk_ids,
            )
            self._mode = "alltoall"
            self._mc2_combine_state = None
            self._alltoall_combine_state = combine_state
            expert_metadata = mk.ExpertTokensMetadata(
                expert_num_tokens=expert_token_nums,
                expert_num_tokens_cpu=None,
            )
            return (
                dispatched_a1,
                input_scale,
                expert_metadata,
                topk_ids,
                topk_weights,
            )

        if self.communicator is None or self.torch_npu is None:
            return self._fallback_prepare(a1, topk_weights, topk_ids)

        mask_is_valid, active_mask = _active_mask_from_context(
            a1.shape[0],
            bool(getattr(self.moe_config, "is_sequence_parallel", False)),
        )
        if not mask_is_valid:
            raise RuntimeError(
                "ordinary MC2 active-mask contract differs across ranks"
            )
        dispatch_output = self.torch_npu.npu_moe_distribute_dispatch_v2(
            x=a1,
            expert_ids=topk_ids,
            scales=None,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=num_experts,
            group_ep=self.communicator.group_name,
            ep_world_size=self.communicator.world_size,
            ep_rank_id=self.communicator.rank,
            group_tp=self.communicator.group_name,
            tp_world_size=1,
            tp_rank_id=0,
            global_bs=0,
            expert_token_nums_type=_EXPERT_TOKEN_NUMS_TYPE_COUNT,
            quant_mode=2,
            x_active_mask=active_mask,
        )
        if (
            not isinstance(dispatch_output, (tuple, list))
            or len(dispatch_output) < _DISPATCH_OUTPUT_SIZE
        ):
            raise RuntimeError("ordinary MC2 dispatch_v2 returned no state")

        (
            expanded,
            dynamic_scale,
            assist_info,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = dispatch_output[:_DISPATCH_OUTPUT_SIZE]
        if not (
            isinstance(expanded, torch.Tensor)
            and expanded.ndim == 2
            and expanded.shape[-1] == a1.shape[-1]
            and isinstance(expert_token_nums, torch.Tensor)
            and expert_token_nums.ndim == 1
            and expert_token_nums.numel() == self.local_num_experts
            and isinstance(assist_info, torch.Tensor)
            and isinstance(ep_recv_counts, torch.Tensor)
            and isinstance(tp_recv_counts, torch.Tensor)
            and (
                dynamic_scale is None
                or isinstance(dynamic_scale, torch.Tensor)
            )
            and (
                expand_scales is None
                or isinstance(expand_scales, torch.Tensor)
            )
        ):
            raise RuntimeError(
                "ordinary MC2 dispatch_v2 returned an incompatible state"
            )

        self._mode = "mc2"
        self._alltoall_combine_state = None
        self._mc2_combine_state = _MC2CombineState(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            assist_info_for_combine=assist_info,
            ep_recv_counts=ep_recv_counts,
            tp_recv_counts=tp_recv_counts,
            expand_scales=expand_scales,
            active_mask=active_mask,
            num_experts=num_experts,
        )
        expert_metadata = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_token_nums,
            expert_num_tokens_cpu=None,
        )
        return expanded, dynamic_scale, expert_metadata, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        del topk_weights, topk_ids, apply_router_weight_on_input
        if self._mode == "mc2":
            state = self._mc2_combine_state
            if state is None:
                raise RuntimeError("ordinary MC2 combine state is missing")
            combined = self.torch_npu.npu_moe_distribute_combine_v2(
                expand_x=fused_expert_output,
                expert_ids=state.topk_ids,
                expert_scales=state.topk_weights.to(torch.float32),
                assist_info_for_combine=state.assist_info_for_combine,
                ep_send_counts=state.ep_recv_counts,
                tp_send_counts=state.tp_recv_counts,
                expand_scales=state.expand_scales,
                expert_shard_type=0,
                shared_expert_rank_num=0,
                moe_expert_num=state.num_experts,
                group_ep=self.communicator.group_name,
                ep_world_size=self.communicator.world_size,
                ep_rank_id=self.communicator.rank,
                group_tp=self.communicator.group_name,
                tp_world_size=1,
                tp_rank_id=0,
                global_bs=0,
                comm_quant_mode=0,
                x_active_mask=state.active_mask,
            )
            if not isinstance(combined, torch.Tensor):
                raise RuntimeError("ordinary MC2 combine_v2 returned no tensor")
            if combined.shape != output.shape:
                raise RuntimeError(
                    "ordinary MC2 combine_v2 returned an incompatible shape: "
                    f"expected {tuple(output.shape)}, got {tuple(combined.shape)}"
                )
            output.copy_(combined)
            return

        if self._mode == "alltoall":
            state = self._alltoall_combine_state
            if self.alltoall_dispatcher is None or state is None:
                raise RuntimeError("ordinary all-to-all combine state is missing")
            combined = self.alltoall_dispatcher.combine(
                fused_expert_output,
                state,
            )
            if combined.shape != output.shape:
                raise RuntimeError(
                    "ordinary all-to-all returned an incompatible shape: "
                    f"expected {tuple(output.shape)}, got {tuple(combined.shape)}"
                )
            output.copy_(combined)
            return

        # Experts already applied route weights and reduced each token's top-k
        # local-expert contributions, so the no-op reducer is intentional.
        if not isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            raise RuntimeError(
                "Ascend ModelSlim fallback experts require a no-op top-k reducer"
            )
        combined = get_ep_group().combine(
            fused_expert_output,
            is_sequence_parallel=bool(
                getattr(self.moe_config, "is_sequence_parallel", False)
            ),
        )
        if not bool(getattr(self.moe_config, "is_sequence_parallel", False)):
            combined = tensor_model_parallel_all_reduce(combined)
        if combined.shape != output.shape:
            raise RuntimeError(
                "vLLM EP combine returned an incompatible shape: "
                f"expected {tuple(output.shape)}, got {tuple(combined.shape)}"
            )
        output.copy_(combined)


class AscendModelSlimW8A8MC2Experts(mk.FusedMoEExpertsModular):
    """ModelSlim W8A8 GMM experts shared by distributed MoE paths."""

    def __init__(
        self,
        moe_config: Any,
        quant_config: FusedMoEQuantConfig,
        prepare_finalize: AscendModelSlimW8A8MC2PrepareFinalize,
    ) -> None:
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.prepare_finalize = prepare_finalize

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        del weight_key, activation_key
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.GELU)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return bool(getattr(moe_parallel_config, "use_ep", False))

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if a1.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
            raise ValueError("Ascend ModelSlim MC2 experts require 2D activations")
        if topk_ids.ndim != 2:
            raise ValueError("Ascend ModelSlim MC2 experts require 2D top-k IDs")
        num_experts = w1.shape[0]
        num_tokens = a1.shape[0]
        hidden_size = a1.shape[-1]
        intermediate_twice = w1.shape[-1]
        return (
            num_experts,
            num_tokens,
            intermediate_twice,
            hidden_size,
            topk_ids.shape[-1],
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, topk, global_num_experts, local_num_experts
        del expert_tokens_meta, activation
        return (M, K), (1,), (M, K)

    @staticmethod
    def _run_quantized_gmm(
        quantized_hidden_states: torch.Tensor,
        input_scale: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        expert_tokens: torch.Tensor,
        activation: MoEActivation,
    ) -> torch.Tensor:
        # Lazy import avoids a module cycle: moe.py owns the established
        # ModelSlim kernels and imports this module only during post-load setup.
        from . import moe as moe_impl

        if activation == MoEActivation.SILU:
            quantized_hidden, hidden_scale = (
                moe_impl._npu_grouped_matmul_swiglu_quant(
                    quantized_hidden_states,
                    w1,
                    w1_scale,
                    input_scale,
                    expert_tokens,
                )
            )
        else:
            gate_up = moe_impl._npu_grouped_quant_matmul(
                quantized_hidden_states,
                w1,
                w1_scale,
                input_scale,
                expert_tokens.to(torch.int64),
                output_dtype=w1_scale.dtype,
            )
            activated = moe_impl._npu_moe_activation(activation, gate_up)
            quantized_hidden, hidden_scale = moe_impl._npu_dynamic_quant(
                activated
            )
        return moe_impl._npu_grouped_quant_matmul(
            quantized_hidden,
            w2,
            w2_scale,
            hidden_scale,
            expert_tokens.to(torch.int64),
            output_dtype=w2_scale.dtype,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del a2_scale, workspace13, workspace2
        if apply_router_weight_on_input:
            raise RuntimeError("ordinary MC2 does not pre-apply router weights")
        if self.w1_scale is None or self.w2_scale is None:
            raise RuntimeError("ModelSlim W8A8 expert scales are missing")

        if self.prepare_finalize._mode in ("mc2", "alltoall"):
            if expert_tokens_meta is None:
                raise RuntimeError(
                    "distributed W8A8 expert token counts are missing"
                )
            if a1q_scale is None or hidden_states.dtype != torch.int8:
                raise RuntimeError(
                    "distributed W8A8 quantized dispatch output is incomplete"
                )
            routed = self._run_quantized_gmm(
                hidden_states,
                a1q_scale,
                w1,
                w2,
                self.w1_scale,
                self.w2_scale,
                expert_tokens_meta.expert_num_tokens,
                activation,
            )
            output.copy_(routed)
            return

        if expert_map is None:
            raise RuntimeError("ModelSlim EP fallback requires an expert map")
        valid = expert_map[topk_ids.long()] != -1
        local_weights = topk_weights.to(hidden_states.dtype) * valid.to(
            hidden_states.dtype
        )
        local_num_experts = w1.shape[0]
        first_expert = int(get_ep_group().rank_in_group) * local_num_experts
        last_expert = first_expert + local_num_experts

        from . import moe as moe_impl

        num_tokens = hidden_states.shape[0]
        sorted_x, expanded_row_idx, expert_tokens, input_scale = (
            moe_impl._npu_moe_init_routing(
                hidden_states,
                topk_ids,
                active_num=num_tokens * topk_ids.shape[-1],
                expert_num=global_num_experts,
                active_expert_range=[first_expert, last_expert],
            )
        )
        # npu_moe_init_routing_custom(quant_mode=1) already returns INT8
        # activations and their per-token scale. Reuse that pair directly;
        # applying dynamic quantization again would pass DT_INT8 to aclnn.
        routed = self._run_quantized_gmm(
            sorted_x,
            input_scale,
            w1,
            w2,
            self.w1_scale,
            self.w2_scale,
            expert_tokens,
            activation,
        )
        output.copy_(
            moe_impl._npu_moe_token_unpermute(
                routed,
                expanded_row_idx,
                local_weights,
            )
        )


def maybe_make_ordinary_mc2_kernel(
    moe_config: Any,
    layer: torch.nn.Module,
    quant_config: FusedMoEQuantConfig,
) -> mk.FusedMoEKernel | None:
    """Build the modular distributed W8A8 kernel when contracts are met."""
    if not _static_ep_config_supported(moe_config):
        return None
    alltoall_ops = get_alltoall_ops()
    if alltoall_ops is None:
        return None
    try:
        ep_group = get_ep_group()
    except (AssertionError, RuntimeError):
        return None
    if not _expert_layout_supported(layer, ep_group):
        return None
    if not _weight_layout_supported(layer):
        return None
    try:
        alltoall_dispatcher = AscendW8A8AllToAllDispatcher(
            ep_group,
            alltoall_ops,
            int(getattr(layer, "global_num_experts", 0)),
            int(layer.w13_weight.shape[0]),
        )
    except ValueError:
        return None
    communicator = None
    torch_npu_module = None
    if _static_config_supported(moe_config):
        torch_npu_module = _get_mc2_ops()
        if torch_npu_module is not None:
            communicator = _create_mc2_communicator(ep_group)
    logger.info_once(
        "Enabled Ascend ModelSlim W8A8 ordinary all-to-all routing; "
        "paired MC2 full-graph decode=%s",
        communicator is not None,
    )

    prepare_finalize = AscendModelSlimW8A8MC2PrepareFinalize(
        moe_config,
        communicator,
        torch_npu_module,
        int(layer.w13_weight.shape[0]),
        alltoall_dispatcher,
    )
    experts = AscendModelSlimW8A8MC2Experts(
        moe_config,
        quant_config,
        prepare_finalize,
    )
    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        shared_experts=None,
        inplace=False,
    )


__all__ = [
    "AscendModelSlimW8A8MC2Experts",
    "AscendModelSlimW8A8MC2PrepareFinalize",
    "maybe_make_ordinary_mc2_kernel",
]
