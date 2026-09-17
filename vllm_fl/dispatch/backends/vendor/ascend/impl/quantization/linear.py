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

"""rc1 ModelSlim W8A8 static/dynamic Ascend linear schemes.

Ported from vLLM-Ascend 0.24.0rc1's ``w8a8_static.py``, the linear portion of
``w8a8_dynamic.py``, ``base.py``, and ``method_adapters.py``. Quantized MoE
and FlashComm2 quantized communication remain outside this module; the rc1
DSA-CP large-``wq_b`` split is supported through the Ascend vendor gate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs

from .linear_utils import maybe_trans_nz, register_quantize

COMPRESSED_TENSORS_METHOD = "compressed-tensors"


class AscendLinearScheme(ABC):
    """Base class for the rc1 Ascend linear quantization schemes."""

    @abstractmethod
    def get_weight(
        self, input_size: int, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]: ...

    def get_pertensor_param(
        self, params_dtype: torch.dtype, **kwargs: Any
    ) -> dict[str, Any]:
        return {}

    def get_perchannel_param(
        self, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        return {}

    def get_pergroup_param(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        layer_type: str | None = None,
    ) -> dict[str, Any]:
        return {}

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor: ...

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return


def _torch_npu():
    """Import torch-npu only for a real quantized operation."""
    import torch_npu

    return torch_npu


def _dsa_cp_enabled() -> bool:
    """Use the single rc1-compatible DSA-CP/SP gate for ModelSlim too."""
    from vllm_fl.dispatch.backends.vendor.ascend.dsa_compat import enable_dsa_cp

    return enable_dsa_cp()


def _flashcomm2_requested() -> bool:
    """Read rc1's FlashComm2 config precedence for the explicit reject gate."""
    from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.compat import (
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


class AscendW8A8LinearMethod(AscendLinearScheme):
    """rc1 W8A8 static per-tensor activation, per-channel weight linear."""

    def __init__(self) -> None:
        register_quantize()

    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}

    def get_pertensor_param(
        self, params_dtype: torch.dtype, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "input_scale": torch.empty(1, dtype=params_dtype),
            "input_offset": torch.empty(1, dtype=torch.int8),
        }

    def get_perchannel_param(
        self, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        params_dict: dict[str, Any] = {
            "quant_bias": torch.empty(output_size, dtype=torch.int32),
            "weight_scale": torch.empty(output_size, 1, dtype=params_dtype),
            "weight_offset": torch.empty(output_size, 1, dtype=params_dtype),
        }
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size, dtype=torch.int64)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            quant_comm_config = getattr(layer, "_quant_comm_config", {})
            comm_fn = quant_comm_config.get("communication_fn")
            if comm_fn is not None and (
                "o_proj" in layer.prefix or "out_proj" in layer.prefix
            ):
                raise NotImplementedError(
                    "FL Ascend ModelSlim linear does not yet migrate "
                    "FlashComm2 quantized communication"
                )
            x = torch.ops.vllm.quantize(
                x,
                layer.aclnn_input_scale,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
            )

        quant_bias = layer.quant_bias if tp_rank == 0 else None
        if getattr(layer, "ascend_quant_method", "") == COMPRESSED_TENSORS_METHOD:
            quant_bias = bias

        return _torch_npu().npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=layer.params_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor), requires_grad=False
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor), requires_grad=False
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor), requires_grad=False
        ).to(layer.aclnn_input_scale.dtype)

        layer.weight.data = maybe_trans_nz(
            layer.weight.data.transpose(0, 1).contiguous()
        )
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
        if getattr(layer, "ascend_quant_method", "") == COMPRESSED_TENSORS_METHOD:
            layer.deq_scale = torch.nn.Parameter(
                layer.input_scale.data * layer.weight_scale.data,
                requires_grad=False,
            )


class AscendW8A8DynamicLinearMethod(AscendLinearScheme):
    """rc1 W8A8_DYNAMIC per-token activation, per-channel weight linear."""

    act_quant_type: torch.dtype = torch.int8

    def get_weight(
        self, input_size: int, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}

    def get_perchannel_param(
        self, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        return {
            "weight_scale": torch.empty(output_size, 1, dtype=params_dtype),
            "weight_offset": torch.empty(output_size, 1, dtype=params_dtype),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        quantized_x, pertoken_scale = _torch_npu().npu_dynamic_quant(
            x, dst_type=self.act_quant_type
        )
        need_unsqz = pertoken_scale.dim() == 2
        if need_unsqz:
            quantized_x = quantized_x.squeeze(dim=1)
            pertoken_scale = pertoken_scale.squeeze(dim=1)

        chunk_size = getattr(layer, "_chunk_size", 0)
        if isinstance(chunk_size, int) and chunk_size > 0:
            bias_1 = bias[:chunk_size] if bias is not None else None
            bias_2 = bias[chunk_size:] if bias is not None else None
            output = torch.cat(
                [
                    _torch_npu().npu_quant_matmul(
                        quantized_x,
                        layer.weight_1,
                        layer.weight_1_scale,
                        pertoken_scale=pertoken_scale,
                        bias=bias_1,
                        output_dtype=x.dtype,
                    ),
                    _torch_npu().npu_quant_matmul(
                        quantized_x,
                        layer.weight_2,
                        layer.weight_2_scale,
                        pertoken_scale=pertoken_scale,
                        bias=bias_2,
                        output_dtype=x.dtype,
                    ),
                ],
                dim=-1,
            )
        else:
            output = _torch_npu().npu_quant_matmul(
                quantized_x,
                layer.weight,
                layer.weight_scale,
                pertoken_scale=pertoken_scale,
                bias=bias if self.act_quant_type == torch.int8 else None,
                output_dtype=x.dtype,
            )
        return output.unsqueeze(dim=1) if need_unsqz else output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        dsa_cp_enabled = _dsa_cp_enabled()
        if (
            "wq_b" in getattr(layer, "prefix", "")
            and layer.weight.shape[1] >= 65536
            and dsa_cp_enabled
        ):
            # This is the rc1 DSA-CP workaround for the NPU quant-matmul
            # output-dimension limit.  dsa_cp.py imports this exact scheme
            # class, so its quantized q path observes the same split layout.
            chunk_size = layer.weight.shape[1] // 2
            assert chunk_size < 65536, (
                "Even after chunking, the weight dimension is still larger than 65536."
            )
            layer._chunk_size = chunk_size
            layer.weight_1 = maybe_trans_nz(
                layer.weight.data[:, :chunk_size].contiguous()
            )
            layer.weight_2 = maybe_trans_nz(
                layer.weight.data[:, chunk_size:].contiguous()
            )
            layer.weight_1_scale = (
                layer.weight_scale.data[:chunk_size].flatten().contiguous()
            )
            layer.weight_2_scale = (
                layer.weight_scale.data[chunk_size:].flatten().contiguous()
            )
            layer.weight_1_scale_fp32 = layer.weight_1_scale.to(torch.float32)
            layer.weight_2_scale_fp32 = layer.weight_2_scale.to(torch.float32)
            layer.weight_1_offset = (
                layer.weight_offset.data[:chunk_size].flatten().contiguous()
            )
            layer.weight_2_offset = (
                layer.weight_offset.data[chunk_size:].flatten().contiguous()
            )
            del layer.weight
            del layer.weight_scale
            del layer.weight_offset
        else:
            if self.act_quant_type == torch.int8:
                layer.weight.data = maybe_trans_nz(layer.weight.data)
            layer.weight_scale.data = layer.weight_scale.data.flatten()
            layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
            layer.weight_offset.data = layer.weight_offset.data.flatten()


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


def create_linear_scheme(quant_type: str) -> AscendLinearScheme:
    """Create exactly the two ModelSlim linear schemes in this bounded port."""
    normalized = quant_type.upper()
    if normalized == "W8A8":
        return AscendW8A8LinearMethod()
    if normalized == "W8A8_DYNAMIC":
        return AscendW8A8DynamicLinearMethod()
    raise NotImplementedError(
        "FL Ascend ModelSlim linear supports only W8A8 and W8A8_DYNAMIC; "
        f"got {quant_type!r}"
    )


__all__ = [
    "AscendLinearMethod",
    "AscendLinearScheme",
    "AscendW8A8DynamicLinearMethod",
    "AscendW8A8LinearMethod",
    "create_linear_scheme",
]
