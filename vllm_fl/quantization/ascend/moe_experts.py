# SPDX-License-Identifier: Apache-2.0
"""Modular experts bridge for dynamic ModelSlim weights."""

import torch

from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

from vllm_fl.dispatch import CachedOp

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
