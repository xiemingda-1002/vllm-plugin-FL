# SPDX-License-Identifier: Apache-2.0
"""Small eager oracle, including dynamic activation QDQ at both matmuls.

This reference uses float32 accumulation for integer products. It is not a
graph/performance backend and deliberately does not read distributed state.
"""

import math

import torch
import torch.nn.functional as F

from vllm_fl.quantization.w8a8.reference import dynamic_per_token_quant_int8


def modelslim_w8a8_prepare_experts(w1, w2, s1, s2):
    for weight, scale in ((w1, s1), (w2, s2)):
        if weight.ndim != 3 or weight.dtype != torch.int8:
            raise ValueError("Expert weights must be int8 [experts, output, input]")
        if scale.shape != (*weight.shape[:2], 1):
            raise ValueError("Expert scales must be [experts, output, 1]")
    if (
        w1.shape[0] != w2.shape[0]
        or w1.shape[1] != 2 * w2.shape[2]
        or w1.shape[2] != w2.shape[1]
    ):
        raise ValueError("Incompatible gated expert weight shapes")
    return (
        w1.contiguous(),
        w2.contiguous(),
        s1.float().contiguous(),
        s2.float().contiguous(),
    )


def _linear(x, weight, scale):
    quantized, token_scale = dynamic_per_token_quant_int8(x)
    product = quantized.float() @ weight.float().t()
    return (product * token_scale * scale.float().reshape(1, -1)).to(x.dtype)


def modelslim_w8a8_experts(
    x,
    w1,
    w2,
    s1,
    s2,
    topk_weights,
    topk_ids,
    first_expert=0,
    global_num_experts=-1,
    activation="silu",
    limit=None,
):
    if activation not in ("silu", "swiglustep"):
        raise NotImplementedError(f"Unsupported ModelSlim activation: {activation}")
    if activation == "swiglustep" and limit is None:
        limit = 7.0
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError("Activation limit must be finite and positive")
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Expert input must be BF16/FP16 [tokens, hidden]")
    if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
        raise ValueError(
            "Routing IDs and weights must have matching [tokens, topk] shape"
        )
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("Routing shape does not match tokens or has empty topk")
    if global_num_experts == -1:
        global_num_experts = first_expert + w1.shape[0]
    if first_expert < 0 or first_expert + w1.shape[0] > global_num_experts:
        raise ValueError("Local expert range is outside global experts")
    output = torch.zeros_like(x, dtype=torch.float32)
    # Boolean-selected expert batches and Python loops are intentional here:
    # this independent oracle is restricted to bounded eager component tests.
    for expert in range(w1.shape[0]):
        token, route = torch.where(topk_ids == first_expert + expert)
        if token.numel() == 0:
            continue
        gate_up = _linear(x[token], w1[expert], s1[expert])
        gate, up = gate_up.chunk(2, dim=-1)
        if activation == "swiglustep":
            activated = F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
        elif limit is not None:
            gate = gate.clamp(max=limit)
            activated = gate * torch.sigmoid(gate) * up.clamp(-limit, limit)
        else:
            # Fused SwiGLU rounds once on output, not between SiLU and mul.
            activated = (F.silu(gate.float()) * up.float()).to(gate_up.dtype)
        local_output = _linear(activated, w2[expert], s2[expert])
        weighted = local_output.float() * topk_weights[token, route, None].float()
        output.index_add_(0, token, weighted)
    return output.to(x.dtype)


def _validate_grouped_inputs(
    x,
    pertoken_scale,
    w1,
    w2,
    s1,
    s2,
    group_list,
    group_list_type,
    output_dtype,
    activation,
    limit,
):
    # Structural checks only: keep device values on device during capture.
    if x.ndim != 2 or x.dtype != torch.int8:
        raise ValueError("Routed input must be INT8 [capacity, hidden]")
    if pertoken_scale.dtype != torch.float32 or pertoken_scale.shape not in (
        (x.shape[0],),
        (x.shape[0], 1),
    ):
        raise ValueError("Token scales must be FP32 [capacity] or [capacity,1]")
    if output_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Output dtype must be BF16/FP16")
    if group_list_type not in (0, 1):
        raise ValueError("Group list type must be 0 (prefix) or 1 (count)")
    if w1.ndim != 3 or w2.ndim != 3 or w1.dtype != torch.int8 or w2.dtype != torch.int8:
        raise ValueError("Expert weights must be INT8 [experts, output, input]")
    if (
        w1.shape[0] == 0
        or w1.shape[0] != w2.shape[0]
        or w1.shape[1] != 2 * w2.shape[2]
        or w1.shape[2] != w2.shape[1]
        or x.shape[1] != w1.shape[2]
        or x.shape[1] == 0
        or w2.shape[2] == 0
    ):
        raise ValueError("Incompatible gated expert weight shapes")
    if group_list.dtype != torch.int64 or group_list.shape != (w1.shape[0],):
        raise ValueError("Group list must be INT64 [experts]")
    for weight, scale in ((w1, s1), (w2, s2)):
        if scale.shape != (*weight.shape[:2], 1) or scale.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("Prepared channel scales must be BF16/FP32 [E,N,1]")
    if any(t.device != x.device for t in (pertoken_scale, w1, w2, s1, s2, group_list)):
        raise ValueError("Grouped expert tensors must share a device")
    if activation not in ("silu", "swiglustep"):
        raise NotImplementedError(f"Unsupported ModelSlim activation: {activation}")
    if activation == "swiglustep" and limit is None:
        limit = 7.0
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError("Activation limit must be finite and positive")
    return limit


def _quantize_grouped_activation(x):
    """Quantize by the computed scale, preserving half-integer boundaries.

    This scoped oracle uses division rather than the generic vLLM-style
    reciprocal/multiply helper; their finite-precision boundary results
    can differ even when the returned FP32 scales are identical.
    """
    values = x.float()
    scale = values.abs().amax(dim=-1, keepdim=True) / 127
    safe_scale = torch.where(scale != 0, scale, torch.ones_like(scale))
    quantized = (values / safe_scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def modelslim_w8a8_grouped_experts(
    x,
    pertoken_scale,
    w1,
    w2,
    s1,
    s2,
    group_list,
    group_list_type,
    output_dtype,
    activation="silu",
    limit=None,
):
    """Independent eager integer-product oracle; reads counts for loop bounds."""
    limit = _validate_grouped_inputs(
        x,
        pertoken_scale,
        w1,
        w2,
        s1,
        s2,
        group_list,
        group_list_type,
        output_dtype,
        activation,
        limit,
    )
    values = group_list.tolist()
    counts = (
        [value - previous for value, previous in zip(values, [0] + values[:-1])]
        if group_list_type == 0
        else values
    )
    if any(value < 0 for value in counts) or sum(counts) > x.shape[0]:
        raise ValueError("Expert counts must be nonnegative and fit capacity")
    output = torch.zeros(x.shape, dtype=output_dtype, device=x.device)
    start = 0
    for expert, count in enumerate(counts):
        stop = start + count
        if count:
            token_scale = pertoken_scale.reshape(-1)[start:stop, None]
            gate_up = (
                (x[start:stop].float() @ w1[expert].float().t())
                * token_scale
                * s1[expert].float().reshape(1, -1)
            ).to(output_dtype)
            gate, up = gate_up.chunk(2, dim=-1)
            if activation == "swiglustep":
                activated = F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
            elif limit is not None:
                gate = gate.clamp(max=limit)
                activated = gate * torch.sigmoid(gate) * up.clamp(-limit, limit)
            else:
                activated = (F.silu(gate.float()) * up.float()).to(output_dtype)
            quantized, token_scale = _quantize_grouped_activation(activated)
            product = quantized.float() @ w2[expert].float().t()
            output[start:stop] = (
                product * token_scale * s2[expert].float().reshape(1, -1)
            ).to(output_dtype)
        start = stop
    return output
