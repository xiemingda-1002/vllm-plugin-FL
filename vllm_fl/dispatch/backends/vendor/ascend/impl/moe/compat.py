"""Narrow FL integration boundary for the rc1 Ascend MoE package."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from vllm_fl.ascend_flashcomm import (
    enable_flashcomm1,
)
from vllm_fl.dispatch.backends.vendor.ascend.hardware import (
    AscendDeviceType as AscendDeviceType,
    get_ascend_device_type as get_ascend_device_type,
)

ACL_FORMAT_FRACTAL_NZ = 29
_SHARED_EXPERTS_CALCULATION_STREAM = None


def _additional_config() -> Mapping[str, object]:
    """Return vLLM's additional config, only defaulting without a context.

    rc1 nests the MoE-related settings below ``additional_config``.  Do not
    turn a malformed in-context configuration into an empty mapping: doing so
    makes an explicitly requested but unsupported execution mode look like the
    supported AllGather default.
    """
    try:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
    except AssertionError:
        return {}
    extra = vllm_config.additional_config
    if extra is None:
        return {}
    if not isinstance(extra, Mapping):
        raise TypeError(
            "vllm additional_config must be a mapping for FL Ascend MoE; "
            f"got {type(extra).__name__}"
        )
    return extra


def _nested_config(extra: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = extra.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"additional_config.{name} must be a mapping for FL Ascend MoE; "
            f"got {type(value).__name__}"
        )
    return value


def _reject_unsupported(enabled: bool, feature: str) -> None:
    if enabled:
        require_a2_bf16_allgather(feature)


def get_ascend_config():
    """Return the rc1 MoE config view and fail closed for unmigrated opt-ins."""
    extra = _additional_config()
    compilation = _nested_config(extra, "ascend_compilation_config")
    fusion = _nested_config(extra, "ascend_fusion_config")
    eplb = _nested_config(extra, "eplb_config")
    finegrained_tp = _nested_config(extra, "finegrained_tp_config")
    finegrained_defaults = {
        "lmhead_tensor_parallel_size": 0,
        "oproj_tensor_parallel_size": 0,
        "embedding_tensor_parallel_size": 0,
        "mlp_tensor_parallel_size": 0,
        "olora_tensor_parallel_size": 0,
    }
    unknown_finegrained = set(finegrained_tp) - set(finegrained_defaults)
    if unknown_finegrained:
        raise ValueError(
            "Config has no attribute "
            f"'{sorted(unknown_finegrained)[0]}' in finegrained_tp_config"
        )
    requested_finegrained = {
        key: finegrained_tp.get(key, default)
        for key, default in finegrained_defaults.items()
    }
    for key, value in requested_finegrained.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"finegrained_tp_config.{key} must be a non-negative integer"
            )
        if value > 0:
            raise NotImplementedError(
                f"FL Ascend fine-grained TP ({key}) is not migrated"
            )

    # rc1's EplbConfig defaults.  Preserve dormant tuning values, but reject
    # the settings that actually select the unmigrated EPLB execution chain.
    eplb_defaults = {
        "dynamic_eplb": False,
        "expert_map_path": None,
        "expert_heat_collection_interval": 600,
        "algorithm_execution_interval": 50,
        "expert_map_record_path": None,
        "num_redundant_experts": 0,
        "eplb_policy_type": 2,
        "eplb_heat_collection_stage": "all",
    }
    unknown_eplb = set(eplb) - set(eplb_defaults)
    if unknown_eplb:
        raise ValueError(f"Config has no attribute '{sorted(unknown_eplb)[0]}'")
    # rc1 validates dormant tuning too; accepting it does not imply EPLB is
    # active, but malformed values should not survive until a later request.
    for key in (
        "expert_heat_collection_interval",
        "algorithm_execution_interval",
        "num_redundant_experts",
    ):
        value = eplb.get(key, eplb_defaults[key])
        if not isinstance(value, int):
            raise TypeError(f"eplb_config.{key} must be an integer")
        if value < 0:
            raise ValueError(f"eplb_config.{key} must be non-negative")
    if eplb.get("eplb_policy_type", 2) not in (0, 1, 2, 3):
        raise ValueError("eplb_config.eplb_policy_type must be 0, 1, 2, or 3")
    if eplb.get("eplb_heat_collection_stage", "all") not in (
        "all",
        "prefill",
        "decode",
    ):
        raise ValueError(
            "eplb_config.eplb_heat_collection_stage must be all, prefill, or decode"
        )
    # In rc1, an expert-map recording path enables dynamic EPLB even if the
    # caller omitted dynamic_eplb.  Its presence (including an empty path),
    # rather than truthiness, is the activation semantic.
    eplb_requested = (
        bool(eplb.get("dynamic_eplb", False))
        or eplb.get("expert_map_path") is not None
        or eplb.get("expert_map_record_path") is not None
        or bool(eplb.get("num_redundant_experts", 0))
    )
    legacy_eplb_keys = set(eplb_defaults) & set(extra)
    if legacy_eplb_keys:
        key = sorted(legacy_eplb_keys)[0]
        raise ValueError(
            f"additional_config.{key} is not an rc1 MoE key; use "
            f"additional_config.eplb_config.{key}"
        )

    # Match rc1's additional-config-over-environment precedence.  A process
    # environment request must fail closed too; an explicit config value of 0
    # deliberately overrides an inherited environment value of 1.
    if "enable_fused_mc2" in extra:
        enable_fused_mc2 = extra["enable_fused_mc2"]
    else:
        env_enable_fused_mc2 = os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2")
        try:
            enable_fused_mc2 = (
                int(env_enable_fused_mc2) if env_enable_fused_mc2 is not None else 0
            )
        except ValueError as exc:
            raise ValueError(
                "VLLM_ASCEND_ENABLE_FUSED_MC2 must be 0 or 1, "
                f"got {env_enable_fused_mc2!r}"
            ) from exc
    if enable_fused_mc2 not in (0, 1):
        raise ValueError(
            "additional_config.enable_fused_mc2 must be 0 or 1, "
            f"got {enable_fused_mc2!r}"
        )
    _reject_unsupported(enable_fused_mc2 == 1, "fused MC2 communication")
    _reject_unsupported(
        bool(extra.get("enable_shared_expert_dp", False)), "shared-expert DP"
    )
    _reject_unsupported(
        bool(extra.get("mix_placement", False)), "mixed shared-expert placement"
    )
    _reject_unsupported(
        bool(extra.get("enable_mc2_hierarchy_comm", False)),
        "MC2 hierarchy communication",
    )
    _reject_unsupported(eplb_requested, "EPLB")
    _reject_unsupported(
        bool(compilation.get("enable_static_kernel", False)), "static kernel generation"
    )
    _reject_unsupported(
        bool(fusion.get("fusion_ops_gmmswigluquant", False)),
        "gmmswigluquant fusion",
    )

    mega_moe_max_tokens = extra.get("mega_moe_max_tokens", 131072)
    if not isinstance(mega_moe_max_tokens, int) or mega_moe_max_tokens <= 0:
        raise ValueError(
            "additional_config.mega_moe_max_tokens must be a positive integer, "
            f"got {mega_moe_max_tokens!r}"
        )

    enable_mlapo = extra.get(
        "enable_mlapo",
        bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
    )
    weight_nz_mode = extra.get(
        "weight_nz_mode",
        int(os.getenv("VLLM_ASCEND_ENABLE_NZ", "1")),
    )
    if weight_nz_mode not in (0, 1, 2):
        raise ValueError(
            "additional_config.weight_nz_mode must be 0, 1, or 2, "
            f"got {weight_nz_mode!r}"
        )

    return SimpleNamespace(
        enable_fused_mc2=enable_fused_mc2,
        # Keep the rc1-shaped fields for consumers. Unsupported true requests
        # above never reach them; the supported default remains false.
        enable_shared_expert_dp=False,
        multistream_overlap_shared_expert=bool(
            extra.get("multistream_overlap_shared_expert", False)
        ),
        mix_placement=False,
        enable_mc2_hierarchy_comm=False,
        mega_moe_max_tokens=mega_moe_max_tokens,
        multistream_dsv4_dsa_overlap=bool(
            extra.get("multistream_dsv4_dsa_overlap", True)
        ),
        pa_shape_list=extra.get("pa_shape_list", []),
        enable_mlapo=bool(enable_mlapo),
        weight_nz_mode=weight_nz_mode,
        recompute_scheduler_enable=bool(
            extra.get("recompute_scheduler_enable", False)
        ),
        finegrained_tp_config=SimpleNamespace(
            **requested_finegrained,
        ),
        ascend_compilation_config=SimpleNamespace(
            enable_npugraph_ex=compilation.get("enable_npugraph_ex", True),
            enable_static_kernel=False,
            fuse_norm_quant=compilation.get("fuse_norm_quant", True),
            fuse_qknorm_rope=compilation.get("fuse_qknorm_rope", True),
            fuse_allreduce_rms=compilation.get("fuse_allreduce_rms", False),
            fuse_muls_add=compilation.get("fuse_muls_add", True),
        ),
        # FL's current AllGather MoE never enables this quantized fusion.
        ascend_fusion_config=SimpleNamespace(fusion_ops_gmmswigluquant=False),
        eplb_config=SimpleNamespace(
            dynamic_eplb=False,
            expert_map_path=eplb.get("expert_map_path"),
            expert_heat_collection_interval=eplb.get(
                "expert_heat_collection_interval",
                eplb_defaults["expert_heat_collection_interval"],
            ),
            algorithm_execution_interval=eplb.get(
                "algorithm_execution_interval",
                eplb_defaults["algorithm_execution_interval"],
            ),
            expert_map_record_path=eplb.get("expert_map_record_path"),
            num_redundant_experts=eplb.get("num_redundant_experts", 0),
            eplb_policy_type=eplb.get(
                "eplb_policy_type", eplb_defaults["eplb_policy_type"]
            ),
            eplb_heat_collection_stage=eplb.get(
                "eplb_heat_collection_stage",
                eplb_defaults["eplb_heat_collection_stage"],
            ),
        ),
    )


def require_a2_bf16_allgather(feature: str) -> None:
    raise NotImplementedError(
        f"FL Ascend rc1 MoE currently supports unquantized BF16/FP16 "
        f"ALLGATHER only; "
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
    from vllm_fl.dispatch.backends.vendor.ascend.distributed.parallel_state import (
        get_mc2_group as _get_mc2_group,
    )

    return _get_mc2_group()


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
