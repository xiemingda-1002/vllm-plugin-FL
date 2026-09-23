"""Ascend runtime configuration ownership and compatibility gates.

This module intentionally has no dependency on the MoE implementation package.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import SimpleNamespace

from vllm.logger import init_logger

from vllm_fl.ascend_flashcomm import (
    enable_flashcomm1,
    shared_expert_dp_enabled_for_config,
)

logger = init_logger(__name__)

_WORKER_VLLM_CONFIG = None


def init_ascend_config(vllm_config) -> None:
    """Retain the worker owner, as rc1 does during worker initialization.

    Store the VllmConfig rather than a computed compatibility view: explicit
    current-config contexts still take precedence and refreshed settings are
    not frozen into a stale SimpleNamespace.
    """
    global _WORKER_VLLM_CONFIG
    _WORKER_VLLM_CONFIG = vllm_config


def clear_ascend_config() -> None:
    global _WORKER_VLLM_CONFIG
    _WORKER_VLLM_CONFIG = None


def _current_vllm_config_or_none():
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()
    except AssertionError:
        return _WORKER_VLLM_CONFIG


def get_ascend_additional_config() -> Mapping[str, object]:
    """Read active/worker config, defaulting only without either owner.

    rc1 nests the MoE-related settings below ``additional_config``.  Do not
    turn a malformed in-context configuration into an empty mapping: doing so
    makes an explicitly requested but unsupported execution mode look like the
    supported AllGather default.
    """
    vllm_config = _current_vllm_config_or_none()
    if vllm_config is None:
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


def balance_scheduling_enabled(vllm_config=None) -> bool:
    """Resolve rc1 balance scheduling with config-over-environment precedence.

    ``VLLM_ASCEND_BALANCE_SCHEDULING`` is retained for compatibility with rc1,
    but an explicit ``additional_config`` value always wins.  The opt-in is
    deliberately false by default so importing FL never changes scheduling.
    """
    if vllm_config is None:
        vllm_config = _current_vllm_config_or_none()
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional_config, Mapping):
        raise TypeError("vllm additional_config must be a mapping")
    if "enable_balance_scheduling" in additional_config:
        return bool(additional_config["enable_balance_scheduling"])
    value = os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0")
    try:
        return bool(int(value))
    except ValueError as exc:
        raise ValueError(
            "VLLM_ASCEND_BALANCE_SCHEDULING must be an integer, "
            f"got {value!r}"
        ) from exc


# Keep the internal name for the existing MoE compatibility callers.  Other
# Ascend implementation closures (for example ModelSlim linear) use the
# public name above so they share the same current-context/worker-owner
# lifetime semantics rather than reaching into vLLM's context directly.
_additional_config = get_ascend_additional_config


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
        raise NotImplementedError(
            "FL Ascend rc1 MoE currently supports unquantized BF16/FP16 "
            "ALLGATHER only; "
            f"{feature} is not migrated"
        )


def get_ascend_config():
    """Return the rc1 MoE config view and fail closed for unmigrated opt-ins."""
    vllm_config = _current_vllm_config_or_none()
    extra = _additional_config()
    # rc1 ignores this shared option on PD producers, rejects it outside
    # disaggregated PD, and activates a special scheduler on PD consumers.
    # FL has not migrated that consumer-side scheduler/KV chain.
    if bool(extra.get("recompute_scheduler_enable", False)):
        kv_config = getattr(vllm_config, "kv_transfer_config", None)
        kv_role = getattr(kv_config, "kv_role", None)
        if kv_role == "kv_consumer":
            raise NotImplementedError(
                "FL Ascend PD decode recompute scheduling is not migrated"
            )
        if kv_role != "kv_producer":
            raise ValueError(
                "recompute_scheduler_enable can only be enabled on "
                "PD-disaggregated D nodes (kv_role='kv_consumer', "
                f"but got kv_role={kv_role!r})"
            )
        logger.warning_once(
            "recompute_scheduler_enable is ignored on PD-disaggregated P nodes; "
            "configure it only on D nodes"
        )
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

    enable_shared_expert_dp = shared_expert_dp_enabled_for_config(vllm_config)
    if enable_shared_expert_dp:
        assert enable_sp(
            vllm_config=vllm_config,
            enable_shared_expert_dp=True,
        )

    multistream_overlap_shared_expert = bool(
        extra.get("multistream_overlap_shared_expert", False)
    )
    if enable_fused_mc2 == 1 and multistream_overlap_shared_expert:
        multistream_overlap_shared_expert = False
        logger.warning_once(
            "enable_fused_mc2 and multistream_overlap_shared_expert cannot "
            "be enabled together; disabling shared-expert overlap."
        )

    # rc1 gates sparse C8 on non-compressed SFA models. The currently
    # migrated W8A8 path stores BF16/FP16 KV; reject C8 before selecting
    # kernels or allocating an incompatible cache.
    enable_sparse_c8 = False
    if bool(extra.get("enable_sparse_c8", False)):
        from vllm_fl.dispatch.backends.vendor.ascend.dsa_compat import (
            model_uses_sfa_sparse,
        )

        enable_sparse_c8 = model_uses_sfa_sparse(
            getattr(vllm_config, "model_config", None)
        )
    if enable_sparse_c8:
        raise NotImplementedError("FL Ascend SFA sparse-C8 cache is not migrated")

    def is_sparse_c8_layer(layer_name):
        # Matches rc1's first branch when sparse C8 is disabled.
        return enable_sparse_c8

    return SimpleNamespace(
        enable_balance_scheduling=balance_scheduling_enabled(vllm_config),
        enable_sparse_c8=enable_sparse_c8,
        c8_enable_reshape_optim=False,
        is_sparse_c8_layer=is_sparse_c8_layer,
        enable_fused_mc2=enable_fused_mc2,
        enable_shared_expert_dp=enable_shared_expert_dp,
        multistream_overlap_shared_expert=multistream_overlap_shared_expert,
        mix_placement=False,
        enable_mc2_hierarchy_comm=False,
        mega_moe_max_tokens=mega_moe_max_tokens,
        multistream_dsv4_dsa_overlap=bool(
            extra.get("multistream_dsv4_dsa_overlap", True)
        ),
        pa_shape_list=extra.get("pa_shape_list", []),
        enable_mlapo=bool(enable_mlapo),
        weight_nz_mode=weight_nz_mode,
        recompute_scheduler_enable=False,
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


def enable_sp(
    vllm_config=None,
    enable_shared_expert_dp: bool = False,
) -> bool:
    """Compatibility name for the current rc1 FlashComm1 runtime gate."""
    return enable_flashcomm1(
        _current_vllm_config_or_none() if vllm_config is None else vllm_config,
        enable_shared_expert_dp=enable_shared_expert_dp,
    )


def enable_sp_by_pass() -> bool:
    """Keep compiler-pass SP distinct from the FlashComm1 runtime gate."""
    vllm_config = _current_vllm_config_or_none()
    if vllm_config is None:
        return False

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or getattr(model_config, "enforce_eager", False):
        return False
    compilation_config = getattr(vllm_config, "compilation_config", None)
    pass_config = getattr(compilation_config, "pass_config", None)
    return bool(getattr(pass_config, "enable_sp", False))
