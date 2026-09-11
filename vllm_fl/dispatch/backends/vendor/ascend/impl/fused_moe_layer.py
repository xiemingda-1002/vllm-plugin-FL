# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Ascend BF16/FP16 fixed-shape unquantized MoE method.

The integration shape follows FL's v0.2 migration: current vLLM keeps the
router, shared experts, runner orchestration, DP/EP dispatch-combine, and late
tensor-parallel reduction.  The Ascend vendor method owns local-expert routing,
grouped expert computation, and token unpermutation.  Native operator
semantics are sourced from vLLM-Ascend 0.24.0rc1; the current-vLLM
allgather/reduce-scatter manager owns DP dispatch and combine.
"""

from __future__ import annotations

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.utils import replace_parameter


_WEIGHTS_PROCESSED_ATTR = "_fl_ascend_gmm_weights_processed"


def _validate_supported_config(moe: FusedMoEConfig) -> None:
    if moe.moe_parallel_config.enable_eplb:
        raise NotImplementedError(
            "FL Ascend fixed-shape MoE does not support EPLB"
        )
    if not moe.use_ep and moe.ep_size != 1:
        raise NotImplementedError(
            "FL Ascend MoE requires ep_size=1 when expert parallelism is disabled"
        )
    if moe.dp_size != 1 and not moe.use_ep:
        raise NotImplementedError(
            "FL Ascend data-parallel MoE requires expert parallelism"
        )
    if moe.pcp_size != 1:
        raise NotImplementedError(
            "FL Ascend fixed-shape MoE does not yet support PCP"
        )
    if moe.has_bias:
        raise NotImplementedError("FL Ascend fixed-shape MoE does not support bias")
    if moe.is_lora_enabled:
        raise NotImplementedError("FL Ascend fixed-shape MoE does not support LoRA")


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Run unquantized experts through Ascend fixed-shape routing and GMM."""

    def __init__(self, moe: FusedMoEConfig):
        _validate_supported_config(moe)
        super().__init__(moe)

    @property
    def is_monolithic(self) -> bool:
        return False

    @property
    def supports_eplb(self) -> bool:
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        del routing_tables
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Weight processing may be invoked more than once (for example by a
        # lifecycle wrapper).  Transposing twice would silently restore the
        # upstream layout, so treat the converted GMM layout as a one-time
        # materialization.  A repeated call is intentionally address-stable.
        if getattr(layer, _WEIGHTS_PROCESSED_ATTR, False):
            return

        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)
        w13 = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        w2 = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()

        # Use current vLLM's replacement helper so the weight_loader attribute
        # survives the first shape-changing replacement.  The one-time guard
        # preserves both Parameter identity and storage address thereafter.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        setattr(layer, _WEIGHTS_PROCESSED_ATTR, True)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        _validate_supported_config(self.moe)
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "FL Ascend fixed-shape MoE requires float16 or bfloat16 input"
            )

        activation = getattr(self.moe.activation, "value", self.moe.activation)
        if activation not in ("silu", "gelu"):
            raise NotImplementedError(
                f"Unsupported FL Ascend MoE activation: {activation}"
            )

        topk_weights = topk_weights.to(x.dtype)
        if layer.apply_router_weight_on_input:
            if topk_weights.shape[-1] != 1:
                raise ValueError(
                    "apply_router_weight_on_input requires top_k == 1 on Ascend"
                )
            x = x * topk_weights

        expert_map = getattr(layer, "expert_map", None)
        if expert_map is None:
            expert_map = getattr(layer, "_expert_map", None)

        if self.moe.use_ep:
            if expert_map is None:
                raise RuntimeError("Ascend expert parallelism requires an expert map")
            valid = expert_map[topk_ids.long()] >= 0
            topk_weights = topk_weights * valid.to(topk_weights.dtype)
            first_expert_idx = self.moe.ep_rank * layer.local_num_experts
            last_expert_idx = first_expert_idx + layer.local_num_experts
            global_num_experts = layer.global_num_experts
        else:
            first_expert_idx = 0
            last_expert_idx = layer.local_num_experts
            global_num_experts = layer.local_num_experts

        num_tokens = x.shape[:-1].numel()
        sorted_x, expanded_row_idx, expert_tokens, _ = (
            torch.ops._C_ascend.npu_moe_init_routing_custom(
                x,
                topk_ids.to(torch.int32),
                active_num=num_tokens * topk_ids.shape[-1],
                expert_num=global_num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[first_expert_idx, last_expert_idx],
                quant_mode=-1,
            )
        )

        gate_up = torch_npu.npu_grouped_matmul(
            x=[sorted_x],
            weight=[layer.w13_weight],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens.to(torch.int64),
        )[0]
        if activation == "silu":
            gate_up = torch_npu.npu_swiglu(gate_up)
        else:
            gate_up = torch_npu.npu_gelu_mul(gate_up)

        routed = torch_npu.npu_grouped_matmul(
            x=[gate_up],
            weight=[layer.w2_weight],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens.to(torch.int64),
        )[0]
        return torch_npu.npu_moe_token_unpermute(
            permuted_tokens=routed,
            sorted_indices=torch.abs(expanded_row_idx),
            probs=None
            if layer.apply_router_weight_on_input
            else topk_weights,
        )


__all__ = ["AscendUnquantizedFusedMoEMethod"]
