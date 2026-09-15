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

"""rc1 ModelSlim W8A8_DYNAMIC fused-MoE scheme for the Ascend FL backend.

This is a vendor-scoped port of ``vllm_ascend`` 0.24.0rc1
``method_adapters.py`` and ``methods/w8a8_dynamic.py``.  It supports the rc1
W8A8 Fused-MC2 scale representation; EPLB remains rejected explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.utils import set_weight_attrs

from vllm_fl.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.compat import get_ascend_config
from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.experts_selector import (
    select_experts,
    zero_experts_compute,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.moe_runtime_args import (
    build_fused_experts_input,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.quant_type import QuantType

ACL_FORMAT_FRACTAL_NZ = 29


def scale_from_float_to_int64(scale: torch.Tensor) -> torch.Tensor:
    """Use rc1's bit-preserving FP32-to-int64 scale representation."""
    return torch.from_numpy(
        np.frombuffer(
            scale.cpu().to(torch.float32).numpy().tobytes(), dtype=np.int32
        ).astype(np.int64)
    ).to(scale.device)


def _torch_npu():
    """Import torch-npu only when a real post-load operation needs it."""
    import torch_npu

    return torch_npu


def get_moe_num_logical_experts(
    layer: torch.nn.Module,
    num_experts: int,
    global_redundant_expert_num: int = 0,
    num_shared_experts: int = 0,
) -> int:
    """Keep rc1's logical-expert override for router-shape validation."""
    moe_config = getattr(layer, "moe_config", None)
    num_logical_experts = getattr(moe_config, "num_logical_experts", None)
    if num_logical_experts is not None:
        return int(num_logical_experts)
    return int(num_experts - global_redundant_expert_num - num_shared_experts)


class AscendMoEScheme:
    """Minimal shared interface for Ascend fused-MoE quantization schemes."""

    quant_type: QuantType = QuantType.NONE


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


class AscendW8A8DynamicFusedMoEMethod(AscendMoEScheme):
    """rc1 W8A8 dynamic activation / channel-scale fused-MoE scheme."""

    quant_type: QuantType = QuantType.W8A8

    def __init__(self) -> None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        # Dynamic EPLB has no FL closure. Fused-MC2 is supported only by this
        # W8A8 scheme and prepares its packed scale representation below.
        if ascend_config.eplb_config.dynamic_eplb:
            raise NotImplementedError(
                "FL Ascend rc1 W8A8_DYNAMIC MoE EPLB is not migrated"
            )
        self.dynamic_eplb = False
        self.in_dtype = vllm_config.model_config.dtype
        self.supports_eplb = False

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes,
                dtype=torch.int8,
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
        }

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "w13_weight_scale": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
            ),
            "w13_weight_offset": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
            ),
            "w2_weight_scale": torch.empty(
                num_experts, hidden_sizes, 1, dtype=params_dtype
            ),
            "w2_weight_offset": torch.empty(
                num_experts, hidden_sizes, 1, dtype=params_dtype
            ),
        }

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
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del is_prefill

        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        n_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
        mix_placement = getattr(layer, "mix_placement", False)
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=n_shared_experts,
        )
        if zero_expert_num == 0 or zero_expert_type is None:
            assert router_logits.shape[1] == num_logical_experts, (
                "[FL/W8A8_DYNAMIC] Number of global experts mismatch "
                f"router_experts={router_logits.shape[1]}, "
                f"expected_experts={num_logical_experts}, "
                f"zero_expert_num={zero_expert_num}, zero_expert_type={zero_expert_type}"
            )

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            mix_placement=mix_placement,
            num_logical_experts=router_logits.shape[1],
            num_shared_experts=n_shared_experts,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )
        assert topk_weights is not None and topk_ids is not None
        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=num_logical_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )
        if enable_force_load_balance:
            random_matrix = torch.rand(
                topk_ids.size(0), num_logical_experts, device=topk_ids.device
            )
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(
                topk_ids.dtype
            )

        fused_scale_flag = (
            _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2
            and get_ascend_config().enable_fused_mc2 == 1
        )
        result = _EXTRA_CTX.moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights.to(self.in_dtype),
                topk_ids=topk_ids,
                w1=[layer.w13_weight],
                w2=[layer.w2_weight],
                quant_type=self.quant_type,
                dynamic_eplb=False,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=[layer.fused_w1_scale]
                if fused_scale_flag else [layer.w13_weight_scale_fp32],
                w2_scale=[layer.fused_w2_scale]
                if fused_scale_flag else [layer.w2_weight_scale],
                w1_scale_bias=[torch.tensor([], dtype=torch.float32)]
                if fused_scale_flag else None,
                w2_scale_bias=[torch.tensor([], dtype=torch.float32)]
                if fused_scale_flag else None,
                swiglu_limit=layer.swiglu_limit,
                swiglu_alpha=getattr(layer, "swiglu_alpha", 1.0),
                swiglu_beta=getattr(layer, "swiglu_beta", 0.0),
            )
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            # The FL MoE runner consumes the complete FusedExpertsResult
            # (including stream events), not just routed_out.
            result.routed_out += zero_expert_result
        return result

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        torch_npu = _torch_npu()
        layer.w13_weight.data = torch_npu.npu_format_cast(
            layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ
        )
        layer.w2_weight.data = torch_npu.npu_format_cast(
            layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ
        )
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
            layer.w13_weight_scale.data.shape[0], -1
        )
        layer.w13_weight_scale_fp32 = layer.w13_weight_scale.data.to(torch.float32)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(
            layer.w13_weight_offset.data.shape[0], -1
        )
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(
            layer.w2_weight_scale.data.shape[0], -1
        )
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(
            layer.w2_weight_offset.data.shape[0], -1
        )
        if get_ascend_config().enable_fused_mc2 == 1:
            layer.fused_w1_scale = scale_from_float_to_int64(
                layer.w13_weight_scale.data
            )
            layer.fused_w2_scale = scale_from_float_to_int64(
                layer.w2_weight_scale.data
            )


def create_moe_scheme(quant_type: str) -> AscendMoEScheme:
    if quant_type.upper() == "W8A8_DYNAMIC":
        return AscendW8A8DynamicFusedMoEMethod()
    raise NotImplementedError(
        f"FL Ascend ModelSlim MoE supports only W8A8_DYNAMIC; got {quant_type!r}"
    )


__all__ = [
    "AscendFusedMoEMethod",
    "AscendMoEScheme",
    "AscendW8A8DynamicFusedMoEMethod",
    "create_moe_scheme",
    "get_moe_num_logical_experts",
    "scale_from_float_to_int64",
]
