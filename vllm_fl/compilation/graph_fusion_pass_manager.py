# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend graph-fusion pass orchestration owned by the FL plugin."""

from __future__ import annotations

from typing import Any

from torch import fx
from vllm.compilation.passes.inductor_pass import InductorPass, get_pass_context
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger

from vllm_fl.compilation.config import ASCEND_COMPILATION_DEFAULTS

logger = init_logger(__name__)


class GraphFusionPassManager(InductorPass):
    """Register and order the current Ascend fusion-pass closure.

    npugraph_ex consumes the replacements registered while these passes are
    constructed. The manager remains a valid InductorPass for vLLM's current
    custom-pass API, but FL's compiler does not apply it a second time.
    """

    def __init__(self) -> None:
        self.passes: list[InductorPass] = []
        self._options: dict[str, bool] = dict(ASCEND_COMPILATION_DEFAULTS)
        self._configured = False

    def __call__(self, graph: fx.Graph) -> None:
        compile_range = get_pass_context().compile_range
        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
            else:
                logger.debug("Skipping %s for compile range %s", pass_, compile_range)

    def add(self, pass_: InductorPass) -> None:
        if not isinstance(pass_, InductorPass):
            raise TypeError(
                "FL graph fusion manager accepts only vLLM InductorPass instances"
            )
        raise NotImplementedError(
            "FL Ascend npugraph_ex compilation does not yet support external "
            "InductorPass instances; accepting one would silently skip it"
        )

    def configure(self, config: VllmConfig) -> None:
        raw_options = (config.additional_config or {}).get(
            "ascend_compilation_config", {}
        )
        options = {
            name: raw_options.get(name, default)
            for name, default in ASCEND_COMPILATION_DEFAULTS.items()
        }
        for name, value in options.items():
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        if options["fuse_allreduce_rms"]:
            raise NotImplementedError(
                "FL Ascend graph compilation does not yet support "
                "fuse_allreduce_rms=True"
            )
        if config.compilation_config.pass_config.enable_sp:
            raise NotImplementedError(
                "FL Ascend graph compilation does not yet support "
                "sequence-parallel fusion"
            )

        defaults: list[InductorPass] = []
        with set_current_vllm_config(config, check_compile=False):
            if options["fuse_norm_quant"]:
                from .passes.norm_quant_fusion_pass import (
                    AddRMSNormQuantFusionPass,
                )

                defaults.append(AddRMSNormQuantFusionPass(config))
            if options["fuse_qknorm_rope"]:
                from .passes.qknorm_rope_fusion_pass import QKNormRopeFusionPass

                defaults.append(QKNormRopeFusionPass(config))
            if options["fuse_muls_add"]:
                from .passes.muls_add_pass import MulsAddFusionPass

                defaults.append(MulsAddFusionPass(config))

        self._options = options
        self.passes = defaults
        self._configured = True

    def uuid(self) -> str:
        state: dict[str, Any] = {
            "options": self._options,
            "passes": [pass_.uuid() for pass_ in self.passes],
        }
        try:
            state["compile_range"] = str(get_pass_context().compile_range)
        except AssertionError:
            state["compile_range"] = None
        return InductorPass.hash_dict(state)
