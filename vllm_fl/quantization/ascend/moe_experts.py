# SPDX-License-Identifier: Apache-2.0
"""Modular experts bridge for dynamic ModelSlim weights."""

import torch

from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)

from vllm_fl.dispatch import CachedOp
from vllm_fl.ops.fused_moe.prepare_finalize import NpuExpertTokensMetadata

_experts = CachedOp("modelslim_w8a8_experts")


class AscendW8A8DynamicExperts(TritonExperts):
    @property
    def expects_unquantized_inputs(self) -> bool:
        # Prepare/AgRs must not quantize inputs through a CUDA-only kernel.
        return True

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        return (0,), (0,), (M, K)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if a1q_scale is not None or a2_scale is not None:
            raise ValueError(
                "ModelSlim dynamic experts require floating prepare inputs"
            )
        if (
            apply_router_weight_on_input
            or self._lora_context is not None
            or self.w1_bias is not None
            or self.w2_bias is not None
        ):
            raise NotImplementedError(
                "ModelSlim experts require output routing and no LoRA/bias"
            )
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("ModelSlim dynamic experts require BF16/FP16 input")
        first_expert = 0
        if expert_map is not None:
            from vllm.distributed import get_ep_group

            group = get_ep_group()
            if (
                expert_map.numel() != w1.shape[0] * group.world_size
                or global_num_experts != expert_map.numel()
            ):
                raise NotImplementedError(
                    "ModelSlim EP requires evenly partitioned experts"
                )
            first_expert = group.rank_in_group * w1.shape[0]
        elif global_num_experts not in (-1, w1.shape[0]):
            raise NotImplementedError("Sharded ModelSlim experts require an expert map")
        output.copy_(
            _experts(
                hidden_states,
                w1,
                w2,
                self.w1_scale,
                self.w2_scale,
                topk_weights,
                topk_ids,
                first_expert=first_expert,
                global_num_experts=global_num_experts,
                activation=activation.value,
                limit=self.gemm1_clamp_limit,
            )
        )


class AscendW8A8DispatchedExperts(mk.FusedMoEExpertsModular):
    """Compute already-routed INT8 rows; communication owns route weighting."""

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        if (
            moe_config.is_lora_enabled
            or self.w1_bias is not None
            or self.w2_bias is not None
            or self.a1_scale is not None
            or self.a2_scale is not None
        ):
            raise NotImplementedError(
                "Dispatched ModelSlim experts require no LoRA/bias/static activation scale"
            )
        if quant_config.gemm1_alpha not in (
            None,
            1.0,
        ) or quant_config.gemm1_beta not in (None, 0.0):
            raise NotImplementedError(
                "Dispatched experts require default SwiGLU alpha/beta"
            )
        self._grouped_experts = CachedOp("modelslim_w8a8_grouped_experts")

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device():
        from vllm.platforms import current_platform

        return current_platform.device_type == "npu"

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        )

    @staticmethod
    def _supports_activation(activation):
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUSTEP)

    @staticmethod
    def _supports_parallel_config(config):
        return (
            config.use_ep
            and not config.is_sequence_parallel
            and config.pcp_size == 1
            and not config.enable_eplb
            and not config.use_batched_activation_format
        )

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        # Routed capacity is independent of the original topk token count.
        if a1.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or topk_ids.ndim != 2:
            raise ValueError("Invalid dispatched expert tensor dimensions")
        return w1.shape[0], a1.shape[0], w1.shape[1], a1.shape[1], topk_ids.shape[1]

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self):
        # P/F combine consumes original topk weights exactly once.
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if apply_router_weight_on_input or a2_scale is not None:
            raise NotImplementedError(
                "Dispatched experts require output routing and dynamic quantization"
            )
        if not isinstance(expert_tokens_meta, NpuExpertTokensMetadata):
            raise ValueError("Dispatched NPU expert metadata is required")
        if a1q_scale is None or hidden_states.dtype != torch.int8:
            raise ValueError("Dispatched experts require INT8 input and token scales")
        if output.dtype != expert_tokens_meta.output_dtype:
            raise ValueError(
                "Dispatched output dtype differs from original hidden dtype"
            )
        if not self._supports_activation(activation):
            raise NotImplementedError("Unsupported dispatched expert activation")
        output.copy_(
            self._grouped_experts(
                hidden_states,
                a1q_scale,
                w1,
                w2,
                self.w1_scale,
                self.w2_scale,
                group_list=expert_tokens_meta.group_list,
                group_list_type=expert_tokens_meta.group_list_type,
                output_dtype=expert_tokens_meta.output_dtype,
                activation=activation.value,
                limit=self.quant_config.gemm1_clamp_limit,
            )
        )
