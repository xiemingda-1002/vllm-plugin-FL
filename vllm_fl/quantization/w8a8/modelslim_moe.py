# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelSlim checkpoint loading with the upstream modular MoE lifecycle."""

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    modular_kernel as mk,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import make_int8_moe_kernel
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_fl.dispatch import CachedOp
from vllm_fl.ops.fused_moe.prepare_finalize import NpuExpertTokensMetadata

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
        parallel = self.moe.moe_parallel_config
        if (
            parallel.use_ep
            and parallel.ep_size > 1
            and not parallel.is_sequence_parallel
            and 1024 <= self.moe.hidden_dim <= 8192
        ):
            from vllm.config import get_current_vllm_config
            from vllm.distributed import get_ep_group
            from vllm.model_executor.layers.fused_moe.modular_kernel import (
                FusedMoEKernel,
            )

            from vllm_fl.distributed.device_communicators.npu_moe_runtime import (
                is_a3_device,
            )

            if is_a3_device():
                from vllm_fl.ops.fused_moe.prepare_finalize import (
                    NpuMoEPrepareAndFinalize,
                )

                runtime = get_ep_group().device_communicator.get_moe_runtime(
                    get_current_vllm_config(), device=layer.w13_weight.device
                )
                self.moe_kernel = FusedMoEKernel(
                    NpuMoEPrepareAndFinalize(runtime, self.moe.num_experts),
                    AscendW8A8DispatchedExperts(self.moe, self.moe_quant_config),
                )
                return
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
