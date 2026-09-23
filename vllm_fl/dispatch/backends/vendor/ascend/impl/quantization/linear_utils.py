# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Current vLLM-Ascend NZ conversion policy for ModelSlim linear weights."""

from __future__ import annotations

import os

import torch

from vllm.config import get_current_vllm_config
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_fl.platforms.ascend.hardware import (
    AscendDeviceType,
    get_ascend_device_type,
)

ACL_FORMAT_FRACTAL_NZ = 29
_QUANTIZE_REGISTERED = False


def _quantize_impl(
    in_tensor: torch.Tensor,
    input_scale: torch.Tensor,
    input_scale_reciprocal: torch.Tensor,
    input_offset: torch.Tensor,
) -> torch.Tensor:
    """rc1 custom-op body; retain div_mode=False and reciprocal scale input."""
    del input_scale
    import torch_npu

    return torch_npu.npu_quantize(
        in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False
    )


def _quantize_impl_fake(
    in_tensor: torch.Tensor,
    input_scale: torch.Tensor,
    input_scale_reciprocal: torch.Tensor,
    input_offset: torch.Tensor,
) -> torch.Tensor:
    """Use the same torch-npu op as rc1's fake implementation."""
    return _quantize_impl(in_tensor, input_scale, input_scale_reciprocal, input_offset)


def register_quantize() -> None:
    """Idempotently install FL's rc1-compatible PrivateUse1 quantize wrapper."""
    global _QUANTIZE_REGISTERED
    if _QUANTIZE_REGISTERED:
        return
    if hasattr(torch.ops.vllm, "quantize"):
        raise RuntimeError(
            "torch.ops.vllm.quantize already exists before FL Ascend registration"
        )
    direct_register_custom_op(
        op_name="quantize",
        op_func=_quantize_impl,
        fake_impl=_quantize_impl_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )
    _QUANTIZE_REGISTERED = True


def _should_trans_nz(weight: torch.Tensor) -> bool:
    """Preserve rc1's dtype, meta tensor, device generation, and config policy."""
    if weight.dtype == torch.float32 or weight.is_meta:
        return False
    if get_ascend_device_type() is AscendDeviceType._310P:
        return True

    # AscendConfig._get_config_value gives additional_config precedence over
    # the legacy environment key; retain that exact rc1 resolution here while
    # keeping this utility independent from parser registration.
    vllm_config = get_current_vllm_config()
    additional_config = vllm_config.additional_config
    if additional_config is None:
        additional_config = {}
    if "weight_nz_mode" in additional_config:
        nz_mode = additional_config["weight_nz_mode"]
    else:
        nz_mode = int(os.getenv("VLLM_ASCEND_ENABLE_NZ", "1"))
    if not nz_mode:
        return False
    if weight.dtype in {torch.bfloat16, torch.float16}:
        return nz_mode == 2
    return True


def maybe_trans_nz(weight: torch.Tensor) -> torch.Tensor:
    """Format a weight as FRACTAL_NZ only when rc1's policy permits it."""
    if not _should_trans_nz(weight):
        return weight
    import torch_npu

    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
