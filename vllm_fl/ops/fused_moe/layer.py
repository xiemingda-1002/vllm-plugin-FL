# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from vllm/model_executor/layers/fused_moe/layer.py (v0.24.0)

import vllm.model_executor.layers.fused_moe as _fused_moe_pkg

# Save the original FusedMoE factory BEFORE any monkey-patching occurs.
# custom_ops.py patches _fused_moe_pkg.FusedMoE = FusedMoEFL at runtime,
# so calling _fused_moe_pkg.FusedMoE() inside FusedMoEFL would recurse
# infinitely.  Capturing it here breaks the cycle.
_OrigFusedMoE = _fused_moe_pkg.FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.logger import init_logger

from .fused_moe_utils import select_unquantized_moe_backend_oot
from vllm_fl.ops.fused_moe.router import replace_router_with_fl


logger = init_logger(__name__)


class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    """OOT replacement for UnquantizedFusedMoEMethod that routes computation
    through flaggems operators."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend_oot(
            moe_config=self.moe
        )

    @property
    def is_monolithic(self) -> bool:
        if self.moe_kernel is None:
            if self.experts_cls is None:
                return True
            return self.experts_cls.is_monolithic()
        return self.moe_kernel.is_monolithic


def FusedMoEFL(*args, **kwargs) -> MoERunner:
    """
    OOT factory replacement for FusedMoE (vllm >= 0.24.0).

    In vllm 0.24.0, FusedMoE changed from a class to a factory function that
    returns a MoERunner instance.  FusedMoEFL mirrors this pattern: it
    delegates to the standard FusedMoE() factory, replaces the router, and
    substitutes the FL experts only for unquantized MoE.

    Registration: op_registry_oot maps FusedMoE -> FusedMoEFL so that all
    MoE layers in a model use the FL router transparently.
    """
    from vllm.platforms import current_platform

    is_ascend = (
        current_platform.vendor_name == "ascend"
        and current_platform.device_type == "npu"
    )

    # vLLM-Ascend 0.24rc1 owns MoE prepare/dispatch/compute/combine/finalize in
    # its runner.  Inject that runner while the upstream factory is building
    # RoutedExperts; replacing only the quant method afterwards leaves the
    # upstream AgRs lifecycle active and is not semantically equivalent.
    if is_ascend and kwargs.get("runner_cls") is None:
        from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.fused_moe import (
            AscendMoERunner,
        )

        kwargs["runner_cls"] = AscendMoERunner

    # Use the original factory captured before monkey-patching to avoid
    # recursion.  Explicit caller-owned runner_cls remains authoritative.
    runner: MoERunner = _OrigFusedMoE(*args, **kwargs)

    # 2. Replace only an upstream unquantized method with the vendor-specific
    # implementation. kwargs are passed through unchanged above, so a caller's
    # explicit runner_cls/runner_args remain authoritative.
    # Quantized methods own their weight/activation scaling metadata and must
    # remain attached to the runner.
    if is_ascend:
        # AscendMoERunner installs its rc1 quant method during construction.
        # An explicit custom runner owns its own method and is not rewritten.
        pass
    elif isinstance(runner._quant_method, UnquantizedFusedMoEMethod):
        fl_quant_method = UnquantizedFusedMoEMethodFL(runner.moe_config)
        runner._replace_quant_method(fl_quant_method)
    else:
        logger.info_once(
            "Preserving upstream quantized MoE method %s in FusedMoEFL.",
            type(runner._quant_method).__name__,
        )

    # 3. Replace router _compute_routing with FL version via monkey-patch.
    #    replace_router_with_fl() patches the class method so the router
    #    instance built by FusedMoE() above uses FL dispatch without needing
    #    to re-construct the router (which would require re-passing all init
    #    args and risks signature mismatch across vllm versions).
    replace_router_with_fl()

    return runner


__all__ = ["FusedMoEFL", "UnquantizedFusedMoEMethodFL"]
