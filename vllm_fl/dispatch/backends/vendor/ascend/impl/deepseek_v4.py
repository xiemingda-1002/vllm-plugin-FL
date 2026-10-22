# SPDX-License-Identifier: Apache-2.0
"""CANN providers for DeepSeek V4's logical operator contracts."""

import torch


def deepseek_v4_index_quant(x):
    import torch_npu

    if x.ndim < 2 or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Indexer quantization requires floating token/head rows")
    if x.numel() == 0:
        return torch.empty_like(x, dtype=torch.int8), x.new_empty(
            (*x.shape[:-1], 1), dtype=torch.float16
        )
    quantized, scale = torch_npu.npu_dynamic_quant(
        x.reshape(-1, x.shape[-1]).contiguous(), dst_type=torch.int8
    )
    return quantized.reshape(x.shape), scale.reshape(*x.shape[:-1], 1).half()
