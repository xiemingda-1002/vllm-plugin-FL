# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) 2023 The vLLM team.
# Copyright (c) 2026 BAAI. All rights reserved.
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

"""Ascend ModelSlim adapters for vLLM parameter loading and layer lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.linear import (
        AscendLinearScheme,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.moe import (
        AscendMoEScheme,
    )


def _flashcomm2_requested() -> bool:
    """Read rc1's FlashComm2 config precedence for the explicit reject gate."""
    from vllm_fl.configs.ascend import (
        get_ascend_additional_config,
    )

    additional_config = get_ascend_additional_config()
    if "enable_flashcomm2_parallel_size" in additional_config:
        parallel_size = additional_config["enable_flashcomm2_parallel_size"]
    else:
        import os

        raw_parallel_size = os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", "0")
        try:
            parallel_size = int(raw_parallel_size)
        except ValueError as exc:
            raise ValueError(
                "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE must be an integer"
            ) from exc
    if isinstance(parallel_size, bool) or not isinstance(parallel_size, int):
        raise ValueError(
            "additional_config.enable_flashcomm2_parallel_size must be an integer"
        )
    return parallel_size > 0


class AscendLinearMethod(LinearMethodBase):
    """rc1 adapter allocating ModelSlim linear parameter contracts."""

    def __init__(self, scheme: AscendLinearScheme) -> None:
        self.quant_method = scheme

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight_dict = self.quant_method.get_weight(
            input_size_per_partition, output_size_per_partition, params_dtype
        )
        packed_dim = weight_dict.pop("_packed_dim", None)
        packed_factor = weight_dict.pop("_packed_factor", None)
        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            if packed_dim is not None and packed_factor is not None:
                set_weight_attrs(
                    param, {"packed_dim": packed_dim, "packed_factor": packed_factor}
                )
            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        layer_type = "row" if isinstance(layer, RowParallelLinear) else "others"
        pertensor_dict = self.quant_method.get_pertensor_param(
            params_dtype, layer_type=layer_type
        )
        for name, value in pertensor_dict.items():
            param = PerTensorScaleParameter(data=value, weight_loader=weight_loader)
            param.ignore_warning = True
            layer.register_parameter(name, param)
            param.weight_loader = weight_loader

        perchannel_dict = self.quant_method.get_perchannel_param(
            output_size_per_partition, params_dtype
        )
        for name, value in perchannel_dict.items():
            param = torch.nn.Parameter(value, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pergroup_dict = self.quant_method.get_pergroup_param(
            input_size_per_partition,
            output_size_per_partition,
            params_dtype,
            layer_type=layer_type,
        )
        scale_packed_dim = pergroup_dict.pop("_packed_dim", None)
        scale_packed_factor = pergroup_dict.pop("_packed_factor", None)
        for name, value in pergroup_dict.items():
            param = torch.nn.Parameter(value, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
            if scale_packed_dim is not None and scale_packed_factor is not None:
                set_weight_attrs(
                    param,
                    {
                        "packed_dim": scale_packed_dim,
                        "packed_factor": scale_packed_factor,
                    },
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.quant_method.process_weights_after_loading(layer)

    def get_computed_params(self) -> set[str]:
        return {"weight_offset", "quant_bias", "deq_scale", "weight_scale"}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(layer, RowParallelLinear):
            if (
                "o_proj" in layer.prefix or "out_proj" in layer.prefix
            ) and _flashcomm2_requested():
                raise NotImplementedError(
                    "FL Ascend ModelSlim linear does not yet migrate "
                    "FlashComm2 tensor-parallel groups"
                )
            tp_rank = get_tensor_model_parallel_rank()
        else:
            tp_rank = 0
        return self.quant_method.apply(layer, x, bias, tp_rank)


class AscendFusedMoEMethod(FusedMoEMethodBase):
    """The rc1 adapter that delegates the layer lifecycle to an MoE scheme."""

    def __init__(
        self, scheme: AscendMoEScheme, moe_config: FusedMoEConfig, tid2eid=None
    ) -> None:
        super().__init__(moe_config)
        self.quant_method = scheme
        self.tid2eid = tid2eid

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # Ascend owns its communication/forward implementation.
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.quant_method.get_weight(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        per_group_param = [
            "weight_scale_second",
            "weight_offset_second",
            "scale_bias",
        ] + (
            ["weight_scale", "weight_offset"]
            if hasattr(self.quant_method, "group_size")
            and self.quant_method.group_size > 0
            else []
        )
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)
            if any(field in param_key for field in per_group_param):
                param.quant_method = FusedMoeWeightScaleSupported.GROUP.value

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.quant_method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_experts=num_experts,
            expert_map=expert_map,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            pertoken_scale=pertoken_scale,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            mc2_mask=mc2_mask,
            tid2eid=self.tid2eid,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.quant_method.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    @property
    def supports_eplb(self):
        return getattr(self.quant_method, "supports_eplb", False)


__all__ = ["AscendLinearMethod", "AscendFusedMoEMethod"]
