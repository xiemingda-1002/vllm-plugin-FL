# Copyright (c) 2026 BAAI. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Ascend-only additions to vLLM's per-forward context.

This module keeps the vLLM-Ascend communication selector and context contract
local to FL.  It deliberately has no ``torch_npu`` import so importing the FL
package on another vendor cannot initialize an Ascend runtime.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import torch
from vllm.forward_context import get_forward_context

from vllm_fl.ascend_flashcomm import (
    flashcomm1_enabled_for_forward,
    is_moe_model,
)


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3


_mc2_tokens_capacity: int | None = None
_reserved_mc2_mask: torch.Tensor | None = None


def _is_moe_model(vllm_config: Any) -> bool:
    """Compatibility alias for existing Ascend forward-context callers."""
    return is_moe_model(vllm_config)


def set_mc2_tokens_capacity(
    vllm_config: Any,
    max_num_reqs: int,
    uniform_decode_query_len: int,
) -> None:
    """Initialize the fixed token threshold used by the rc1 selector."""
    global _mc2_tokens_capacity
    if _mc2_tokens_capacity is not None:
        return

    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if additional_config.get("enable_prefill_mc2", False):
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    elif vllm_config.compilation_config.cudagraph_capture_sizes:
        max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
    else:
        max_num_tokens = max_num_reqs * uniform_decode_query_len
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    tokens_per_tp_rank = min((max_num_tokens + tp_size - 1) // tp_size, 512)
    _mc2_tokens_capacity = tokens_per_tp_rank * tp_size


def get_mc2_tokens_capacity() -> int | None:
    return _mc2_tokens_capacity


def set_mc2_mask(vllm_config: Any, device: torch.device | str) -> None:
    global _reserved_mc2_mask
    if _reserved_mc2_mask is not None:
        return
    if _is_moe_model(vllm_config):
        _reserved_mc2_mask = torch.zeros(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.bool,
            device=device,
        )


def get_mc2_mask() -> torch.Tensor | None:
    return _reserved_mc2_mask


def _ep_world_size(vllm_config: Any) -> int:
    """Avoid accessing a process group before distributed init in unit tests."""
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.world_size_across_dp
        // parallel_config.pipeline_parallel_size
    )


def _select_a2_moe_comm_method(
    num_tokens: int,
    vllm_config: Any,
    mc2_tokens_capacity: int | None,
) -> MoECommType:
    num_experts = vllm_config.model_config.get_num_experts()
    ep_world_size = _ep_world_size(vllm_config)
    num_experts_per_device = num_experts // ep_world_size
    if (
        mc2_tokens_capacity is not None
        and num_experts_per_device <= 24
        and ep_world_size >= 16
        and num_tokens <= mc2_tokens_capacity
    ):
        return MoECommType.MC2
    return MoECommType.ALLGATHER


def select_moe_comm_method(
    num_tokens: int,
    vllm_config: Any,
    is_draft_model: bool = False,
) -> MoECommType | None:
    """Select the current rc1 A2 communication path.

    FL's first supported closure is A2 BF16.  A3/310P/A5 selection must be
    added together with their corresponding communication kernels rather than
    silently selecting an implementation that is not present.
    """
    del is_draft_model
    if not _is_moe_model(vllm_config):
        return None
    if not vllm_config.parallel_config.enable_expert_parallel:
        return MoECommType.ALLGATHER
    if _ep_world_size(vllm_config) == 1:
        return MoECommType.ALLGATHER
    return _select_a2_moe_comm_method(
        num_tokens,
        vllm_config,
        get_mc2_tokens_capacity(),
    )


def _get_moe_comm_method(moe_comm_type: MoECommType) -> Any:
    # Importing the communication implementation pulls in torch_npu and
    # vLLM-Ascend. Keep it behind the Ascend platform hook.
    from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.moe_comm_method import (
        get_moe_comm_method,
    )

    return get_moe_comm_method(moe_comm_type)


def build_additional_forward_context(
    *,
    attn_metadata: Any,
    vllm_config: Any,
    dp_metadata: Any,
    num_tokens: int | None = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: Any = None,
    batch_descriptor: Any = None,
    ubatch_slices: Any = None,
) -> dict[str, Any]:
    """Build the fields consumed by FL's Ascend MoE runner.

    The arguments mirror ``Platform.set_additional_forward_context`` exactly;
    graph metadata remains owned by upstream vLLM.
    """
    del num_tokens_across_dp, cudagraph_runtime_mode, batch_descriptor, ubatch_slices

    if num_tokens is None and attn_metadata:
        num_tokens = next(iter(attn_metadata.values())).num_actual_tokens
    has_num_tokens = num_tokens is not None
    num_tokens = int(num_tokens or 0)

    moe_comm_type = select_moe_comm_method(num_tokens, vllm_config)
    moe_comm_method = None
    if moe_comm_type is not None:
        moe_comm_method = _get_moe_comm_method(moe_comm_type)

    tp_size = vllm_config.parallel_config.tensor_parallel_size
    flash_comm_v1_enabled = flashcomm1_enabled_for_forward(
        vllm_config,
        num_tokens if has_num_tokens else None,
    )
    max_tokens_across_dp = num_tokens
    if dp_metadata is not None:
        max_tokens_across_dp = int(
            dp_metadata.num_tokens_across_dp_cpu.max().item()
        )

    padded_length = None
    pad_size = 0
    if flash_comm_v1_enabled:
        pad_size = (tp_size - (num_tokens % tp_size)) % tp_size
        if dp_metadata is not None:
            padded_length = math.ceil(max_tokens_across_dp / tp_size) * tp_size
            pad_size = padded_length - num_tokens

    padded_num_tokens = math.ceil(max_tokens_across_dp / tp_size) * tp_size
    mc2_mask = None
    reserved_mc2_mask = get_mc2_mask()
    if reserved_mc2_mask is not None:
        mc2_mask = reserved_mc2_mask[:padded_num_tokens]
        mc2_mask[:num_tokens] = True
        mc2_mask[num_tokens:] = False

    return {
        "moe_comm_type": moe_comm_type,
        "moe_comm_method": moe_comm_method,
        "capturing": False,
        "mmrs_fusion": not _is_moe_model(vllm_config),
        "num_tokens": num_tokens,
        "flash_comm_v1_enabled": flash_comm_v1_enabled,
        "flashcomm_v2_enabled": False,
        "pad_size": pad_size,
        "padded_length": padded_length,
        "max_tokens_across_dp": max_tokens_across_dp,
        "max_tokens_across_pcp": 0,
        "mc2_mask": mc2_mask,
        "is_draft_model": False,
        "is_draft_model_prefill": False,
        "in_profile_run": False,
        "padded_num_tokens": padded_num_tokens,
        "sinks": False,
        "eplb_heat_collection_status": False,
    }


class _ExtraForwardContextProxy:
    """Expose Ascend extras without changing upstream ForwardContext."""

    def __getattr__(self, name: str) -> Any:
        context = get_forward_context()
        if name in context.additional_kwargs:
            return context.additional_kwargs[name]
        return getattr(context, name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        get_forward_context().additional_kwargs[name] = value


_EXTRA_CTX = _ExtraForwardContextProxy()


__all__ = [
    "MoECommType",
    "_EXTRA_CTX",
    "build_additional_forward_context",
    "get_mc2_mask",
    "get_mc2_tokens_capacity",
    "select_moe_comm_method",
    "set_mc2_mask",
    "set_mc2_tokens_capacity",
]
