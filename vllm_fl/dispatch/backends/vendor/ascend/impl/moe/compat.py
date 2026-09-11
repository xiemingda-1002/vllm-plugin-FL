"""Narrow FL integration boundary for the rc1 Ascend MoE package."""

from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
from types import SimpleNamespace

import torch

from vllm_fl.ascend_flashcomm import (
    enable_flashcomm1,
)


ACL_FORMAT_FRACTAL_NZ = 29
_SHARED_EXPERTS_CALCULATION_STREAM = None


class AscendDeviceType(Enum):
    A2 = "A2"
    A3 = "A3"
    A5 = "A5"
    _310P = "310P"


def _additional_config() -> dict:
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config().additional_config or {}
    except (AssertionError, AttributeError):
        return {}


def get_ascend_config():
    """Return the rc1 fields consumed by MoE, with conservative defaults."""
    extra = _additional_config()
    return SimpleNamespace(
        enable_fused_mc2=int(extra.get("enable_fused_mc2", 0)),
        enable_shared_expert_dp=bool(extra.get("enable_shared_expert_dp", False)),
        multistream_overlap_shared_expert=bool(
            extra.get("multistream_overlap_shared_expert", False)
        ),
        mix_placement=bool(extra.get("mix_placement", False)),
        enable_mc2_hierarchy_comm=bool(
            extra.get("enable_mc2_hierarchy_comm", False)
        ),
        mega_moe_max_tokens=int(extra.get("mega_moe_max_tokens", 131072)),
        ascend_compilation_config=SimpleNamespace(enable_static_kernel=False),
        ascend_fusion_config=SimpleNamespace(fusion_ops_gmmswigluquant=False),
        eplb_config=SimpleNamespace(
            dynamic_eplb=bool(extra.get("dynamic_eplb", False)),
            expert_map_path=extra.get("expert_map_path"),
            num_redundant_experts=int(extra.get("num_redundant_experts", 0)),
            eplb_policy_type=int(extra.get("eplb_policy_type", 0)),
            expert_heat_collection_interval=int(
                extra.get("expert_heat_collection_interval", 1)
            ),
        ),
    )


def require_a2_bf16_allgather(feature: str) -> None:
    raise NotImplementedError(
        f"FL Ascend rc1 MoE currently supports A2 BF16/FP16 ALLGATHER only; "
        f"{feature} is not migrated"
    )


def enable_sp(
    vllm_config=None,
    enable_shared_expert_dp: bool = False,
) -> bool:
    """Compatibility name for the current rc1 FlashComm1 runtime gate."""
    return enable_flashcomm1(
        vllm_config,
        enable_shared_expert_dp=enable_shared_expert_dp,
    )


def enable_sp_by_pass() -> bool:
    """Keep compiler-pass SP distinct from the FlashComm1 runtime gate."""
    try:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
    except AssertionError:
        return False

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or getattr(model_config, "enforce_eager", False):
        return False
    compilation_config = getattr(vllm_config, "compilation_config", None)
    pass_config = getattr(compilation_config, "pass_config", None)
    return bool(getattr(pass_config, "enable_sp", False))


def enable_custom_op() -> bool:
    return False


def get_ascend_device_type():
    return AscendDeviceType.A2


def is_hierarchical_communication_enabled() -> bool:
    return False


def should_skip_allreduce_across_dp_group() -> bool:
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
        config.enable_shared_expert_dp
        or flashcomm1_enabled
        or enable_sp_by_pass()
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
    require_a2_bf16_allgather("MC2 communication")


def split_tensor_along_first_dim(tensor: torch.Tensor, group) -> torch.Tensor:
    world_size = group.world_size
    rank = group.rank_in_group
    if tensor.shape[0] % world_size != 0:
        raise ValueError("first dimension must be divisible by the group size")
    return tensor.chunk(world_size, dim=0)[rank]


def get_moe_num_logical_experts(
    _layer,
    num_experts: int,
    *,
    global_redundant_expert_num: int = 0,
    num_shared_experts: int = 0,
) -> int:
    if global_redundant_expert_num or num_shared_experts:
        require_a2_bf16_allgather("EPLB or mixed shared-expert placement")
    return num_experts


class VllmEplbAdaptor:
    @staticmethod
    def register_layer(_layer) -> None:
        return None


def init_eplb_config(*_args, **_kwargs):
    require_a2_bf16_allgather("EPLB")
