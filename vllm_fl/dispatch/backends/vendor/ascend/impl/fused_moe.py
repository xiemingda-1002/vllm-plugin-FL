# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/layer.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
import torch.nn.functional as F
import torch_npu

from .device_operator import DeviceOperator


def topk_softmax_ascend(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route softmax top-k through FL's current-rc1 A2/A3 native op.

    vLLM preallocates float32 weights and int32 indices.  The native op emits
    weights in the router-logit dtype, so copy/cast into those caller-owned
    buffers and return the same tensor objects.  Match vLLM's CUDA contract
    for the third caller-owned buffer as well: entry ``[row, k_idx]`` stores
    ``k_idx * num_tokens + row``.
    """
    if gating_output.ndim != 2:
        raise ValueError("gating_output must be a 2D tensor")
    if topk_weights.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError("top-k output buffers must be 2D tensors")
    if topk_weights.shape != topk_indices.shape:
        raise ValueError("top-k weight and index buffers must have the same shape")
    if topk_weights.shape[0] != gating_output.shape[0]:
        raise ValueError("top-k buffers and gating_output must have the same rows")
    if topk_weights.shape[1] < 1 or topk_weights.shape[1] > gating_output.shape[1]:
        raise ValueError("top-k width must be in [1, number of experts]")
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must be preallocated with dtype torch.float32")
    if topk_indices.dtype != torch.int32:
        raise TypeError("topk_indices must be preallocated with dtype torch.int32")
    if token_expert_indices.shape != topk_indices.shape:
        raise ValueError("token_expert_indices must match the top-k output shape")
    if token_expert_indices.dtype != torch.int32:
        raise TypeError(
            "token_expert_indices must be preallocated with dtype torch.int32"
        )

    native_weights, native_indices, _ = DeviceOperator.moe_gating_top_k(
        gating_output,
        k=topk_weights.shape[1],
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=int(renormalize),
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        bias_opt=None,
    )
    topk_weights.copy_(native_weights.to(dtype=topk_weights.dtype))
    topk_indices.copy_(native_indices.to(dtype=topk_indices.dtype))
    num_tokens, top_k = topk_indices.shape
    token_expert_indices.copy_(
        torch.arange(
            num_tokens * top_k,
            device=token_expert_indices.device,
            dtype=token_expert_indices.dtype,
        )
        .reshape(top_k, num_tokens)
        .T
    )
    return topk_weights, topk_indices


def _torch_fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor|None = None,
) -> torch.Tensor:
    """Current-rc1 unquantized grouped-matmul MoE path for A2/A3."""
    num_tokens, hidden_dim = hidden_states.size()
    E, _, N = w1.size()  # preprocessed w1: [E, K_in, N]
    top_k = topk_ids.size(1)

    if global_num_experts == -1:
        global_num_experts = E

    out_hidden_states = hidden_states if inplace else torch.zeros_like(hidden_states)
    if expert_map is not None:
        local_topk_ids = expert_map[topk_ids.long()]
    else:
        local_topk_ids = topk_ids.long()

    flat_experts = local_topk_ids.reshape(-1)
    flat_tokens = torch.arange(
        num_tokens, device=hidden_states.device, dtype=torch.long
    ).repeat_interleave(top_k)
    flat_weights = topk_weights.reshape(-1)
    valid = (flat_experts >= 0) & (flat_experts < E)
    flat_experts = flat_experts[valid]
    flat_tokens = flat_tokens[valid]
    flat_weights = flat_weights[valid]
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    sorted_tokens = flat_tokens[order]
    sorted_weights = flat_weights[order]
    expert_input = hidden_states.index_select(0, sorted_tokens)
    if apply_router_weight_on_input:
        expert_input = expert_input * sorted_weights[:, None].to(expert_input.dtype)

    group_list = torch.bincount(sorted_experts, minlength=E).cumsum(0).to(torch.int64)
    gate_up = torch_npu.npu_grouped_matmul(
        x=[expert_input],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )[0]
    if activation == "silu":
        gate_up = torch_npu.npu_swiglu(gate_up)
    elif activation == "gelu":
        gate, up = gate_up.chunk(2, dim=-1)
        gate_up = F.gelu(gate) * up
    elif activation == "silu_no_mul":
        gate_up = F.silu(gate_up)
    elif activation == "gelu_no_mul":
        gate_up = F.gelu(gate_up)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")
    if not apply_router_weight_on_input:
        gate_up = gate_up * sorted_weights[:, None].to(gate_up.dtype)
    expert_output = torch_npu.npu_grouped_matmul(
        x=[gate_up],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )[0]
    out_hidden_states.index_add_(0, sorted_tokens, expert_output)

    return out_hidden_states


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(1), "Hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(1), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(1)}"
        )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    if any((use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16)):
        raise NotImplementedError(
            "The FL Ascend MoE closure in this migration supports BF16/FP16 "
            "unquantized experts only"
        )
    return _torch_fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )
