# Copyright (c) 2026 BAAI. All rights reserved.

"""Current-vLLM linear layers adapted to Ascend FlashComm1."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from vllm.config import get_current_vllm_config
from vllm.distributed import divide
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantizeMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .linear_op import (
    flashcomm1_configured,
    get_parallel_op,
    get_replicated_op,
)
from vllm_fl.dispatch.backends.vendor.ascend.dsa_compat import (
    AscendDeviceType,
    get_ascend_device_type,
    is_310p,
    maybe_trans_nz,
)

_UNQUANTIZED_GEMM_REGISTERED = False


def _should_keep_nd_for_310p_weight(weight: torch.Tensor) -> bool:
    return is_310p() and weight.ndim >= 2 and (
        weight.shape[-1] == 1 or weight.shape[-2] == 1
    )


def _unquantized_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def _unquantized_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del bias
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device
    )


def ensure_ascend_linear_custom_ops_registered() -> None:
    """Register the graph-safe unquantized GEMM boundary once."""
    global _UNQUANTIZED_GEMM_REGISTERED
    if _UNQUANTIZED_GEMM_REGISTERED:
        return
    if hasattr(torch.ops.vllm, "unquantized_gemm"):
        raise RuntimeError(
            "torch.ops.vllm.unquantized_gemm already exists before "
            "FL Ascend registration"
        )
    direct_register_custom_op(
        op_name="unquantized_gemm",
        op_func=_unquantized_gemm,
        fake_impl=_unquantized_gemm_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )
    _UNQUANTIZED_GEMM_REGISTERED = True


class AscendUnquantizedLinearMethod(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Apply the current rc1 Ascend layout used by DeepSeek and Qwen."""
        super().process_weights_after_loading(layer)
        keep_nd_weight = _should_keep_nd_for_310p_weight(layer.weight.data)
        if getattr(layer, "precast_fp32_weight", False):
            weight_fp32 = layer.weight.data.to(torch.float32)
            layer.weight_fp32 = (
                weight_fp32 if keep_nd_weight else maybe_trans_nz(weight_fp32)
            )
        if "conv1d" not in layer.prefix and not keep_nd_weight:
            layer.weight.data = maybe_trans_nz(layer.weight.data)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.unquantized_gemm(x, layer.weight, bias)


class AscendLinearBase(LinearBase):
    """Current-rc1 linear base with a caller-selected communication group."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        if quant_config is None:
            self.quant_method: QuantizeMethodBase | None = (
                AscendUnquantizedLinearMethod()
            )
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        self.return_bias = return_bias
        self.disable_tp = disable_tp


class AscendQKVParallelLinear(QKVParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ) -> None:
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.custom_op, _, tp_size = get_parallel_op(
            disable_tp, prefix, self, "column"
        )
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(
                tp_size, self.total_num_kv_heads
            )
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.v_head_size * tp_size,
        ]
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)


class AscendMergedColumnParallelLinear(MergedColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(
            disable_tp, prefix, self, "column"
        )
        self.output_sizes = output_sizes
        assert all(size % self.tp_size == 0 for size in output_sizes)
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)


class AscendRowParallelLinear(RowParallelLinear):
    unique_prefix_idx = 0

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        out_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(
            disable_tp, prefix, self, "row"
        )
        if flashcomm1_configured():
            compilation_config = get_current_vllm_config().compilation_config
            unique_prefix = prefix
            while unique_prefix in compilation_config.static_forward_context:
                unique_prefix = (
                    f"{prefix}.flashcomm{AscendRowParallelLinear.unique_prefix_idx}"
                )
                AscendRowParallelLinear.unique_prefix_idx += 1
            self.unique_prefix = unique_prefix
            compilation_config.static_forward_context[unique_prefix] = self

        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.out_dtype = out_dtype
        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if not reduce_results and bias and not skip_bias_add:
            raise ValueError(
                "When not reducing results, adding bias can be incorrect"
            )
        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {"output_dim": 0, "weight_loader": self.weight_loader},
            )
        else:
            self.register_parameter("bias", None)
        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(self, input_, **kwargs):
        del kwargs
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)


class AscendColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        output_sizes: list[int] | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(
            disable_tp, prefix, self, "column"
        )
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(size, self.tp_size) for size in self.output_sizes
            ]
        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        self.gather_output = gather_output
        if output_sizes is None:
            output_sizes = [output_size]
        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {"output_dim": 0, "weight_loader": self.weight_loader},
            )
        else:
            self.register_parameter("bias", None)
        if self.custom_op is not None:
            self.custom_op.update_attrs()
        self.prefix = prefix
        if "wo_a" in prefix:
            hf_config = get_current_vllm_config().model_config.hf_text_config
            self.n_local_groups = getattr(hf_config, "o_groups", 0) // self.tp_size
            self.o_lora_rank = getattr(hf_config, "o_lora_rank", 0)

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if "wo_a" in self.prefix and get_ascend_device_type() != AscendDeviceType.A5:
            if self.weight.ndim == 2:
                super().weight_loader(param, loaded_weight)
                self.weight.data = (
                    self.weight.data.view(
                        self.n_local_groups,
                        self.o_lora_rank,
                        -1,
                    )
                    .transpose(2, 1)
                    .contiguous()
                )
            else:
                # Preserve the grouped layout when RL update flows reload a
                # weight after the initial checkpoint transformation.
                shard_size = self.n_local_groups * self.o_lora_rank
                start_idx = self.tp_rank * shard_size
                if loaded_weight.shape[0] != shard_size:
                    loaded_weight = loaded_weight.narrow(
                        0, start_idx, shard_size
                    )
                loaded_weight = (
                    loaded_weight.view(
                        self.n_local_groups,
                        self.o_lora_rank,
                        -1,
                    )
                    .transpose(2, 1)
                    .contiguous()
                )

                if loaded_weight.shape != self.weight.shape:
                    raise ValueError(
                        "Unexpected wo_a weight shape "
                        f"{tuple(loaded_weight.shape)}, expected "
                        f"{tuple(self.weight.shape)}"
                    )
                self.weight.data.copy_(loaded_weight)
        else:
            super().weight_loader(param, loaded_weight)


class AscendReplicatedLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        custom_op, tp_rank, tp_size = get_replicated_op(
            disable_tp, prefix, self
        )
        self.custom_op = custom_op
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.output_partition_sizes = (
            self.output_sizes if hasattr(self, "output_sizes") else [output_size]
        )
        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {"output_dim": 0, "weight_loader": self.weight_loader},
            )
        else:
            self.register_parameter("bias", None)
        if custom_op is not None:
            custom_op.update_attrs()

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)


__all__ = [
    "AscendColumnParallelLinear",
    "AscendLinearBase",
    "AscendMergedColumnParallelLinear",
    "AscendQKVParallelLinear",
    "AscendReplicatedLinear",
    "AscendRowParallelLinear",
    "AscendUnquantizedLinearMethod",
    "ensure_ascend_linear_custom_ops_registered",
    "flashcomm1_configured",
]
