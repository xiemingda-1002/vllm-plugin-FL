# Copyright 2026 FlagOS Contributors
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Ascend ModelSlim Dynamic W8A8 MoE method.

Expert selection and distributed output reduction stay in vLLM's
``MoERunner``. This module owns the ModelSlim expert parameter layout and the
fixed-shape Ascend routing, quantized grouped matmul, activation, and token
combine path used by expert computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    int8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from .w8a8 import _npu_dynamic_quant, _npu_format_nz

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEQuantConfig,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEPrepareAndFinalizeModular,
    )


def _npu_moe_activation(
    activation: MoEActivation,
    gate_up: torch.Tensor,
) -> torch.Tensor:
    import torch_npu

    if activation == MoEActivation.SILU:
        return torch_npu.npu_swiglu(gate_up)
    if activation == MoEActivation.GELU:
        return torch_npu.npu_gelu_mul(gate_up)
    raise NotImplementedError(
        "Ascend ModelSlim Dynamic W8A8 MoE supports only silu and gelu "
        f"gated activations, got {activation.value!r}"
    )


def _npu_moe_init_routing(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    active_num: int,
    expert_num: int,
    active_expert_range: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops._C_ascend.npu_moe_init_routing_custom(
        x,
        topk_ids.to(torch.int32),
        active_num=active_num,
        expert_num=expert_num,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=active_expert_range,
        quant_mode=1,
    )


def _npu_grouped_quant_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    expert_tokens: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        scale=[weight_scale],
        per_token_scale=[per_token_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=output_dtype,
    )[0]


def _npu_grouped_matmul_swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    expert_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the first expert GMM, SwiGLU, and output quantization."""
    import torch_npu

    return torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x=x,
        weight=[weight],
        weight_scale=[weight_scale],
        x_scale=per_token_scale.flatten(),
        group_list=expert_tokens.to(torch.int64),
        bias=None,
        dequant_mode=0,
        quant_mode=0,
        quant_dtype=1,
        group_list_type=1,
    )


def _npu_moe_token_unpermute(
    routed: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    topk_weights: torch.Tensor | None,
) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=routed,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )


def _require_zero_moe_weight_offsets(layer: torch.nn.Module) -> None:
    for name in ("w13_weight_offset", "w2_weight_offset"):
        offset = getattr(layer, name)
        if torch.count_nonzero(offset.detach()).item() != 0:
            raise ValueError(
                f"ModelSlim MoE layer {getattr(layer, 'prefix', '<unknown>')!r} "
                f"uses non-zero {name}, but the Ascend Dynamic W8A8 MoE "
                "kernel supports symmetric weights only"
            )


class AscendModelSlimW8A8DynamicMoEMethod(FusedMoEMethodBase):
    """Dynamic per-token activation and per-channel INT8 expert weights."""

    def __init__(self, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        if not moe.is_act_and_mul:
            raise NotImplementedError(
                "Ascend ModelSlim Dynamic W8A8 MoE requires gated activation"
            )
        if moe.has_bias:
            raise NotImplementedError(
                "Ascend ModelSlim Dynamic W8A8 MoE does not support expert bias"
            )
        if moe.is_lora_enabled:
            raise NotImplementedError(
                "Ascend ModelSlim Dynamic W8A8 MoE does not support LoRA"
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_specs = {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
        }
        for name, tensor in weight_specs.items():
            parameter = torch.nn.Parameter(tensor, requires_grad=False)
            set_weight_attrs(parameter, extra_weight_attrs)
            layer.register_parameter(name, parameter)

        channel_specs = {
            "w13_weight_scale": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=params_dtype,
            ),
            "w13_weight_offset": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=params_dtype,
            ),
            "w2_weight_scale": torch.empty(
                num_experts,
                hidden_size,
                1,
                dtype=params_dtype,
            ),
            "w2_weight_offset": torch.empty(
                num_experts,
                hidden_size,
                1,
                dtype=params_dtype,
            ),
        }
        channel_attrs = dict(extra_weight_attrs)
        channel_attrs["quant_method"] = (
            FusedMoeWeightScaleSupported.CHANNEL.value
        )
        for name, tensor in channel_specs.items():
            parameter = torch.nn.Parameter(tensor, requires_grad=False)
            set_weight_attrs(parameter, channel_attrs)
            layer.register_parameter(name, parameter)

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        _require_zero_moe_weight_offsets(layer)
        layer.w13_weight.data = _npu_format_nz(
            layer.w13_weight.data.transpose(1, 2).contiguous()
        )
        layer.w2_weight.data = _npu_format_nz(
            layer.w2_weight.data.transpose(1, 2).contiguous()
        )
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.flatten(1)
        layer.w13_weight_scale_fp32 = layer.w13_weight_scale.data.to(
            torch.float32
        )
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.flatten(1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.flatten(1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.flatten(1)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        from .moe_mc2 import maybe_make_ordinary_mc2_kernel

        self.moe_kernel = maybe_make_ordinary_mc2_kernel(
            self.moe,
            layer,
            self.moe_quant_config,
        )

    def get_fused_moe_quant_config(
        self,
        layer: torch.nn.Module,
    ) -> FusedMoEQuantConfig:
        return int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale_fp32,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=True,
        )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        del routing_tables
        # This method uses vLLM's router and reduction, but owns expert compute.
        return None

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.moe_kernel is not None:
            return self.moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=(
                    layer.apply_router_weight_on_input
                ),
                shared_experts_input=shared_experts_input,
            )

        del shared_experts_input
        if x.ndim != 2:
            raise ValueError(
                "Ascend ModelSlim Dynamic W8A8 MoE expects 2D hidden states, "
                f"got shape {tuple(x.shape)}"
            )
        if topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "ModelSlim MoE topk_weights and topk_ids must have equal shapes"
            )

        topk_weights = topk_weights.to(x.dtype)
        expert_map = layer.expert_map
        configured_num_experts = int(
            getattr(self.moe, "num_experts", layer.w13_weight.shape[0])
        )
        global_num_experts = int(
            getattr(layer, "global_num_experts", configured_num_experts)
        )
        logical_num_experts = int(
            getattr(layer, "logical_num_experts", configured_num_experts)
        )
        global_redundant_expert_num = max(
            0,
            global_num_experts - logical_num_experts,
        )

        apply_router_weight_on_input = bool(
            getattr(layer, "apply_router_weight_on_input", False)
        )
        if apply_router_weight_on_input:
            if topk_weights.shape[-1] != 1:
                raise ValueError(
                    "Ascend apply_router_weight_on_input requires top_k == 1"
                )
            x = x * topk_weights

        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            valid = expert_map[topk_ids.long()] != -1
            topk_weights = topk_weights * valid.to(topk_weights.dtype)
            local_num_experts = int(layer.local_num_experts)
            ep_size = int(
                getattr(
                    layer,
                    "ep_size",
                    max(
                        1,
                        (global_num_experts + local_num_experts - 1)
                        // local_num_experts,
                    ),
                )
            )
            base_experts, remainder = divmod(global_num_experts, ep_size)
            ep_rank = int(layer.ep_rank)
            first_expert_idx = ep_rank * base_experts + min(
                ep_rank,
                remainder,
            )
            last_expert_idx = first_expert_idx + local_num_experts
        else:
            local_num_experts = int(
                getattr(layer, "local_num_experts", layer.w13_weight.shape[0])
            )
            first_expert_idx = 0
            last_expert_idx = local_num_experts
            global_num_experts = local_num_experts

        num_tokens = x.shape[:-1].numel()
        (
            sorted_x,
            expanded_row_idx,
            expert_tokens,
            input_scale,
        ) = _npu_moe_init_routing(
            x,
            topk_ids,
            active_num=num_tokens * topk_ids.shape[-1],
            expert_num=global_num_experts,
            active_expert_range=[first_expert_idx, last_expert_idx],
        )
        expert_tokens = expert_tokens.to(torch.int64)

        gate_up = _npu_grouped_quant_matmul(
            sorted_x,
            layer.w13_weight,
            layer.w13_weight_scale,
            input_scale,
            expert_tokens,
            output_dtype=x.dtype,
        )
        activated = _npu_moe_activation(layer.activation, gate_up)
        quantized_hidden, hidden_scale = _npu_dynamic_quant(activated)
        routed = _npu_grouped_quant_matmul(
            quantized_hidden,
            layer.w2_weight,
            layer.w2_weight_scale,
            hidden_scale,
            expert_tokens,
            output_dtype=x.dtype,
        )

        return _npu_moe_token_unpermute(
            routed,
            expanded_row_idx,
            None if apply_router_weight_on_input else topk_weights,
        )


__all__ = ["AscendModelSlimW8A8DynamicMoEMethod"]
