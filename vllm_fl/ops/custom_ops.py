# Copyright (c) 2025 BAAI. All rights reserved.

import logging
from typing import List, Optional

from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from .layernorm import *  # noqa F403 F401
from .activation import *  # noqa F403 F401
from .rotary_embedding import *  # noqa F403 F401
from .fused_moe import *  # noqa F403 F401

logger = logging.getLogger(__name__)

# Mapping from OOT operator name (op_name, internal/whitelist) to (class, registration_name).
# registration_name is passed to CustomOp.register_oot and must match what vLLM uses
# when looking up the OOT op (typically the base class name).
# item example as follows:
# op_name: (class, registration_name of vllm's CustomOp.register_oot)
# note: cannot control inner gems op of UnquantizedFusedMoEMethodFL via env variable.
OOT_OPS = {
    "silu_and_mul": (SiluAndMulFL, "SiluAndMul"),  # noqa F405
    "gelu_and_mul": (GeluAndMulFL, "GeluAndMul"),  # noqa F405
    "rms_norm": (RMSNormFL, "RMSNorm"),  # noqa F405
    "rotary_embedding": (RotaryEmbeddingFL, "RotaryEmbedding"),  # noqa F405
    # NOTE: fused_moe is NOT registered via PluggableLayer/CustomOp.register_oot.
    # In vllm >= 0.24.0, FusedMoE is a factory function (not a class), so the
    # PluggableLayer OOT path is incompatible.  Instead, FusedMoEFL is injected
    # via monkey-patch in register_oot_ops() below.
    # "fused_moe": (FusedMoEFL, "FusedMoE"),
    # unquantized_fused_moe_method is also handled via FusedMoEFL factory —
    # no separate registration needed.
    # "unquantized_fused_moe_method": (UnquantizedFusedMoEMethodFL, "UnquantizedFusedMoEMethod"),
}

# These public vLLM registration names are owned by the Ascend vendor lifecycle.
# The corresponding generic FL operators must not overwrite them after
# ``apply_ascend_patches`` has installed the current-vLLM-Ascend implementation.
_ASCEND_VENDOR_OWNED_REGISTRATIONS = frozenset({"RMSNorm"})


def _register_oot_once(op_cls: type, registration_name: str) -> None:
    """Register an OOT class idempotently without hiding ownership conflicts."""
    from vllm.model_executor.custom_op import op_registry_oot

    existing_cls = op_registry_oot.get(registration_name)
    if existing_cls is op_cls:
        logger.debug(
            "OOT op '%s' is already registered by %s; keeping the existing owner",
            registration_name,
            op_cls,
        )
        return
    if existing_cls is not None:
        raise RuntimeError(
            f"OOT op '{registration_name}' is already registered by "
            f"{existing_cls!r}; refusing to replace it with {op_cls!r}"
        )

    if issubclass(op_cls, PluggableLayer):
        PluggableLayer.register_oot(
            _decorated_layer_cls=op_cls, name=registration_name
        )
    else:
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=registration_name)


def _patch_unquantized_moe_oracle() -> None:
    """
    Monkey-patch the upstream select_unquantized_moe_backend so it does not
    short-circuit to (OOT, None) on our platform.  Instead it falls through
    to the normal CUDA/ROCm backend priority selection — the same logic that
    select_unquantized_moe_backend_oot uses.

    This is needed when FusedMoEFL is NOT registered (PREFER_ENABLED=0 or
    fused_moe blacklisted): without the patch, the in-tree UnquantizedFusedMoEMethod
    would get (OOT, None), skip _setup_kernel, and crash at inference time.
    """
    import vllm.model_executor.layers.fused_moe.oracle.unquantized as _oracle_mod
    from vllm_fl.ops.fused_moe.fused_moe_utils import select_unquantized_moe_backend_oot
    _oracle_mod.select_unquantized_moe_backend = select_unquantized_moe_backend_oot
    # Also patch the import in unquantized_fused_moe_method module
    import vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method as _method_mod
    _method_mod.select_unquantized_moe_backend = select_unquantized_moe_backend_oot
    logger.info("Patched select_unquantized_moe_backend to bypass OOT short-circuit")


def register_oot_ops(whitelist: Optional[List[str]] = None) -> None:
    """
    Register OOT (out-of-tree) custom operators.

    Args:
        whitelist: If provided, only register operators in this list.
                   If None, check VLLM_FL_OOT_WHITELIST env var.
                   If neither is set, register all operators.

    Operators in VLLM_FL_OOT_BLACKLIST or platform config oot_blacklist
    will be excluded from registration.

    When fused_moe is not registered (PREFER_ENABLED=0 or blacklisted),
    the upstream select_unquantized_moe_backend oracle is monkey-patched
    so it picks native CUDA backends instead of returning (OOT, None).
    """
    from vllm.platforms import current_platform
    from vllm_fl.utils import get_oot_blacklist, get_oot_whitelist, is_oot_enabled, use_flaggems_op

    # Vendor lifecycle patches are required independently of the generic OOT
    # allowlist. In particular USE_FLAGGEMS=0 and an empty OOT whitelist must
    # still install the FL-owned Qwen GDN/attention/MoE providers before model
    # construction.
    is_ascend = (
        current_platform.vendor_name == "ascend"
        and current_platform.device_type == "npu"
    )
    if is_ascend:
        # Patch the factory before importing any Qwen model module.  qwen3_next
        # binds ``FusedMoE`` in its module globals at import time, so doing this
        # after apply_ascend_patches() would leave Qwen3.5/3.6 on the upstream
        # factory even though the package-level symbol had been replaced.
        _patch_fused_moe_factory()
        from vllm_fl.dispatch.backends.vendor.ascend.patch import apply_ascend_patches

        apply_ascend_patches()
    elif current_platform.device_type == "ptpu":
        from vllm_fl.dispatch.backends.vendor.sunrise.patch import apply_sunrise_patches

        apply_sunrise_patches()

    # Check if OOT registration is enabled
    if not is_oot_enabled():
        # Patch the upstream oracle so in-tree FusedMoE works on this platform.
        _patch_unquantized_moe_oracle()
        return

    # Get blacklist (from env var or platform config)
    blacklist = get_oot_blacklist() or []

    # Determine which operators to register
    env_whitelist = get_oot_whitelist()
    if env_whitelist is not None:
        ops_to_register = env_whitelist
    elif whitelist is not None:
        ops_to_register = whitelist
    else:
        ops_to_register = list(OOT_OPS.keys())

    # Apply blacklist
    ops_to_register = [op for op in ops_to_register if op not in blacklist]

    # If fused_moe is excluded (blacklisted or not in whitelist), patch the
    # upstream oracle so the in-tree FusedMoE doesn't crash on OOT platforms.
    if "fused_moe" not in ops_to_register:
        _patch_unquantized_moe_oracle()

    for op_name in ops_to_register:
        if op_name not in OOT_OPS:
            logger.warning(f"OOT op '{op_name}' not found in OOT_OPS, skipping.")
            continue

        # unquantized_fused_moe_method only registers when use_flaggems_op is True
        if op_name == "unquantized_fused_moe_method" and not use_flaggems_op(op_name):
            logger.debug(f"Skipping '{op_name}': use_flaggems_op returned False")
            continue

        op_cls, registration_name = OOT_OPS[op_name]
        if (
            is_ascend
            and registration_name in _ASCEND_VENDOR_OWNED_REGISTRATIONS
        ):
            logger.debug(
                "Skipping generic OOT op '%s': Ascend vendor lifecycle owns '%s'",
                op_name,
                registration_name,
            )
            continue
        logger.info(f"Registering oot op: {op_name} as '{registration_name}'")
        _register_oot_once(op_cls, registration_name)
    # --- FusedMoE monkey-patch (vllm >= 0.24.0) ---
    # FusedMoE is a factory function in vllm 0.24.0+, not a PluggableLayer
    # subclass, so it cannot be registered via CustomOp/PluggableLayer.register_oot.
    # Instead we replace the factory function in the two places vllm imports it
    # from, so all model code transparently gets FusedMoEFL.
    if not is_ascend and "fused_moe" not in (blacklist or []):
        _patch_fused_moe_factory()


def _patch_fused_moe_factory() -> None:
    """Replace the FusedMoE factory function with FusedMoEFL in all relevant
    vllm modules so that model code picks up the FL version automatically."""
    import inspect
    import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
    import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer

    # Patch at the module level so `from vllm...fused_moe import FusedMoE` picks it up.
    _fused_moe_layer.FusedMoE = FusedMoEFL  # noqa F405
    _fused_moe_pkg.FusedMoE = FusedMoEFL   # noqa F405
    # Be robust when a caller imported qwen3_next before plugin registration.
    # This is intentionally explicit: the Qwen module is the current-0.24
    # consumer in this migration, and scanning/mutating arbitrary modules would
    # make the patch ordering harder to reason about.
    import sys

    qwen_module = sys.modules.get("vllm.model_executor.models.qwen3_next")
    if qwen_module is not None:
        qwen_module.FusedMoE = FusedMoEFL  # noqa F405
    logger.info("Monkey-patched FusedMoE factory -> FusedMoEFL")
