# SPDX-License-Identifier: Apache-2.0
"""Native ModelSlim dynamic INT8 Linear operations."""

import torch


def modelslim_w8a8_prepare_weight(weight, weight_scale, nz_mode):
    import torch_npu

    weight = torch_npu.npu_format_cast(weight, 29) if nz_mode else weight
    # CANN requires FP32 channel scales for FP16 output, but accepts BF16 scales
    # for BF16 output. Prepare once rather than casting inside graph replay.
    if weight_scale.dtype == torch.float16:
        weight_scale = weight_scale.float()
    return weight, weight_scale


def modelslim_w8a8_dynamic_linear(x, weight, weight_scale, bias=None):
    import torch_npu

    quantized, token_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    return torch_npu.npu_quant_matmul(
        quantized,
        weight,
        weight_scale,
        pertoken_scale=token_scale,
        bias=bias,
        output_dtype=x.dtype,
    )
