"""Narrow FL integration boundary for the rc1 Ascend MoE package."""

from __future__ import annotations

import logging

import torch
from vllm_fl.platforms.ascend.hardware import (
    AscendDeviceType as AscendDeviceType,
    get_ascend_device_type as get_ascend_device_type,
)

logger = logging.getLogger(__name__)


ACL_FORMAT_FRACTAL_NZ = 29

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
    shared_expert_dp_enabled,
)
from vllm_fl.distributed.ascend_moe_runtime import (
    npu_stream_switch,
    shared_experts_calculation_stream,
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
    """Opt-in BF16 expert NZ weights; convert once during model loading.

    The previous shim ignored weight_nz_mode=2 for ordinary MoE. Keep that
    behavior by default, and constrain the A3 Qwen3-VL topology before
    enabling the native GMM weight layout. No quantization or routing changes.
    """
    extra = get_ascend_additional_config()
    enabled = extra.get("fl_moe_bf16_nz", False)
    if not isinstance(enabled, bool):
        raise TypeError("fl_moe_bf16_nz must be a JSON boolean")
    if not enabled:
        return tensor
    cfg = _current_vllm_config_or_none()
    if cfg is None:
        raise RuntimeError("fl_moe_bf16_nz requires an active worker config")
    parallel = cfg.parallel_config
    model = cfg.model_config
    if not (
        get_ascend_device_type() is AscendDeviceType.A3
        and extra.get("weight_nz_mode") == 2
        and model.dtype == torch.bfloat16
        and model.quantization is None
        and getattr(model.hf_text_config, "model_type", None) == "qwen3_vl_moe_text"
        and parallel.tensor_parallel_size == 2
        and parallel.data_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.enable_expert_parallel
        and not parallel.enable_eplb
        and not parallel.use_ubatching
        and cfg.lora_config is None
        and cfg.speculative_config is None
        and not tensor.is_meta
        and tensor.device.type == "npu"
        and tensor.dtype == torch.bfloat16
        and tuple(tensor.shape) in {(64, 2048, 1536), (64, 768, 2048)}
    ):
        raise ValueError("fl_moe_bf16_nz supports A3 Qwen3-VL BF16 TP2/EP, DP1 only")
    import torch_npu

    result = torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_NZ)
    result_format = torch_npu.get_npu_format(result)
    if result_format != ACL_FORMAT_FRACTAL_NZ:
        raise RuntimeError(
            "BF16 expert NZ conversion did not produce FRACTAL_NZ: "
            f"got format {result_format!r}"
        )
    logger.info(
        "FL BF16 expert NZ enabled: shape=%s, dtype=%s",
        tuple(tensor.shape),
        tensor.dtype,
    )
    return result


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
