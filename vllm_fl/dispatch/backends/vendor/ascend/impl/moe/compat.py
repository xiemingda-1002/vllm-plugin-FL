"""Narrow FL integration boundary for the rc1 Ascend MoE package."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from vllm_fl.platforms.ascend.hardware import (
    AscendDeviceType as AscendDeviceType,
    get_ascend_device_type as get_ascend_device_type,
)

ACL_FORMAT_FRACTAL_NZ = 29
_SHARED_EXPERTS_CALCULATION_STREAM = None

from vllm_fl.configs.ascend import (
    _nested_config,
    _reject_unsupported,
    _current_vllm_config_or_none,
    balance_scheduling_enabled,
    clear_ascend_config,
    enable_sp,
    enable_sp_by_pass,
    get_ascend_additional_config,
    get_ascend_config,
    init_ascend_config,
)


def require_a2_bf16_allgather(feature: str) -> None:
    raise NotImplementedError(
        f"FL Ascend rc1 MoE currently supports unquantized BF16/FP16 "
        f"ALLGATHER only; {feature} is not migrated"
    )


def enable_custom_op() -> bool:
    return False


def is_hierarchical_communication_enabled() -> bool:
    return False


def should_skip_allreduce_across_dp_group(
    _vllm_config=None,
    is_draft_model: bool = False,
) -> bool:
    """Keep the non-PD synchronized-token rc1 contract on the DP path."""
    return False


def dispose_tensor(_tensor) -> None:
    return None


def maybe_trans_nz(tensor: torch.Tensor) -> torch.Tensor:
    # A2 BF16 grouped matmul accepts the contiguous ND layout produced below.
    return tensor


def npu_stream_switch(target_stream=None, *, enabled: bool = True):
    if not enabled:
        return nullcontext()
    assert target_stream is not None
    return torch.npu.stream(target_stream)


def shared_expert_dp_enabled() -> bool:
    config = get_ascend_config()
    flashcomm1_enabled = enable_sp(
        enable_shared_expert_dp=config.enable_shared_expert_dp
    )
    return bool(
        config.enable_shared_expert_dp or flashcomm1_enabled or enable_sp_by_pass()
    )


def shared_experts_calculation_stream():
    global _SHARED_EXPERTS_CALCULATION_STREAM
    if _SHARED_EXPERTS_CALCULATION_STREAM is None:
        # Keep torch_npu behind the Ascend-only call path so importing FL on a
        # different vendor does not initialize the NPU runtime.
        import torch_npu

        _SHARED_EXPERTS_CALCULATION_STREAM = torch_npu.npu.Stream()
    return _SHARED_EXPERTS_CALCULATION_STREAM


def get_mc2_group():
    """Return the worker-initialized Ascend MC2 group.

    The vendor worker creates this group after upstream model-parallel setup;
    a communication call site must only consume it, never create another
    group with potentially different rank ordering.
    """
    from vllm_fl.distributed.ascend_parallel_state import (
        get_mc2_group as _get_mc2_group,
    )

    return _get_mc2_group()


def split_tensor_along_first_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Compatibility import for rc1's first-dimension partition API."""
    from vllm_fl.distributed.ascend_utils import (
        split_tensor_along_first_dim as _split_tensor_along_first_dim,
    )

    return _split_tensor_along_first_dim(
        tensor,
        num_partitions=num_partitions,
        contiguous_split_chunks=contiguous_split_chunks,
    )


def get_moe_num_logical_experts(
    _layer,
    num_experts: int,
    *,
    global_redundant_expert_num: int = 0,
    num_shared_experts: int = 0,
) -> int:
    """Number of routed experts, excluding redundant and shared experts.

    MiniMax-M3 declares one shared expert alongside its 128 routed experts, so
    subtracting it is required for the routing math (``select_experts`` fills
    the shared slot itself). This is plain arithmetic; it is not an EPLB or
    mixed-placement execution path, so it must not be rejected by the
    A2/ALLGATHER fail-closed gate below.
    """
    if global_redundant_expert_num:
        require_a2_bf16_allgather("EPLB")
    moe_config = getattr(_layer, "moe_config", None)
    num_logical_experts = getattr(moe_config, "num_logical_experts", None)
    if num_logical_experts is not None:
        return int(num_logical_experts)
    return int(num_experts - global_redundant_expert_num - num_shared_experts)


class VllmEplbAdaptor:
    @staticmethod
    def register_layer(_layer) -> None:
        return None


def init_eplb_config(*_args, **_kwargs):
    require_a2_bf16_allgather("EPLB")
