# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelSlim checkpoint loading with the upstream modular MoE lifecycle."""

import torch

from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import make_int8_moe_kernel
from vllm.model_executor.utils import set_weight_attrs

from vllm_fl.dispatch import CachedOp

_prepare = CachedOp("modelslim_w8a8_prepare_experts")


class AscendModelSlimMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe_config):
        super().__init__(moe_config)
        parallel = moe_config.moe_parallel_config
        if moe_config.moe_backend not in ("auto", "triton"):
            raise NotImplementedError("Unsupported explicit ModelSlim MoE backend")
        if parallel.is_sequence_parallel and not (
            parallel.use_ep
            and parallel.dp_size > 1
            and parallel.all2all_backend == "allgather_reducescatter"
        ):
            raise NotImplementedError(
                "ModelSlim MoE token splitting requires DP EP AgRs"
            )
        if (
            moe_config.is_lora_enabled
            or moe_config.has_bias
            or parallel.enable_eplb
            or parallel.pcp_size != 1
            or parallel.use_batched_activation_format
        ):
            raise NotImplementedError(
                "ModelSlim experts do not support LoRA/bias/EPLB/PCP/batched routing"
            )
        if parallel.dp_size > 1 and (
            not parallel.use_ep or parallel.all2all_backend != "allgather_reducescatter"
        ):
            raise NotImplementedError("ModelSlim distributed experts require AgRs EP")
        if moe_config.activation.value not in ("silu", "swiglustep"):
            raise NotImplementedError("Unsupported ModelSlim expert activation")
        if moe_config.swiglu_alpha not in (None, 1.0) or moe_config.swiglu_beta not in (
            None,
            0.0,
        ):
            raise NotImplementedError("Unsupported ModelSlim SwiGLU alpha/beta")
        if parallel.use_ep:
            from vllm.config import get_current_vllm_config
            from vllm.platforms import current_platform

            config = get_current_vllm_config()
            if (
                current_platform.dist_backend != "hccl"
                or config.parallel_config.expert_placement_strategy != "linear"
                or config.compilation_config.pass_config.enable_sp
                or moe_config.num_experts % parallel.ep_size
            ):
                raise NotImplementedError(
                    "ModelSlim EP requires HCCL and equal linear experts without compiler SP"
                )

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("ModelSlim dynamic experts require BF16/FP16")
        shapes = {
            "w13": (num_experts, 2 * intermediate_size_per_partition, hidden_size),
            "w2": (num_experts, hidden_size, intermediate_size_per_partition),
        }
        for prefix, shape in shapes.items():
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.int8), requires_grad=False
            )
            set_weight_attrs(param, extra_weight_attrs)
            layer.register_parameter(f"{prefix}_weight", param)
            attrs = {
                **extra_weight_attrs,
                "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value,
            }
            for suffix in ("scale", "offset"):
                factory = torch.zeros if suffix == "offset" else torch.empty
                param = torch.nn.Parameter(
                    factory((*shape[:2], 1), dtype=params_dtype),
                    requires_grad=False,
                )
                set_weight_attrs(param, attrs)
                layer.register_parameter(f"{prefix}_weight_{suffix}", param)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def get_computed_params(self) -> set[str]:
        return {"w13_weight_offset", "w2_weight_offset"}

    def process_weights_after_loading(self, layer):
        for prefix in ("w13", "w2"):
            weight = getattr(layer, f"{prefix}_weight")
            scale = getattr(layer, f"{prefix}_weight_scale")
            offset = getattr(layer, f"{prefix}_weight_offset")
            if weight.dtype != torch.int8 or weight.ndim != 3:
                raise ValueError(
                    "Expected ModelSlim INT8 [experts, output, input] weights"
                )
            if scale.shape != (*weight.shape[:2], 1) or offset.shape != scale.shape:
                raise ValueError("Expected ModelSlim per-channel expert scales/offsets")
            if torch.count_nonzero(offset).item():
                raise NotImplementedError("Nonzero ModelSlim dynamic expert offsets")
        prepared = _prepare(
            layer.w13_weight.data,
            layer.w2_weight.data,
            layer.w13_weight_scale.data,
            layer.w2_weight_scale.data,
        )
        for name, tensor in zip(
            ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"), prepared
        ):
            getattr(layer, name).data = tensor
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_int8_moe_kernel(
            self.moe_quant_config,
            self.moe,
            AscendW8A8DynamicExperts,
            routing_tables=layer._expert_routing_tables(),
        )

    def maybe_make_prepare_finalize(self, routing_tables=None):
        raise ValueError(
            "ModelSlim experts initialize the modular kernel after loading"
        )

    def get_fused_moe_quant_config(self, layer):
        config = int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=True,
        )
        config.gemm1_clamp_limit = self.moe.swiglu_limit
        return config

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if self.moe_kernel is None:
            raise RuntimeError("ModelSlim expert weights are not prepared")
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


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
