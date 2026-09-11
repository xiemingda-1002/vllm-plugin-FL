# Copyright (c) 2026 BAAI. All rights reserved.

"""Norm/quant fusion boundary for the first FL Ascend graph closure."""

from __future__ import annotations

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import init_logger

logger = init_logger(__name__)


class AddRMSNormQuantFusionPass(VllmInductorPass):
    """Register norm/quant fusions only after quant support is migrated.

    Dense BF16/FP16 models have no quantization subgraph to fuse, so this is an
    explicit zero-pattern pass. Quantized models fail closed instead of silently
    running with a partially migrated fusion set.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes = PatternMatcherPass(
            pass_name="rmsnorm_quant_fusion_pass"
        )
        self.matched_count = 0
        if getattr(vllm_config, "quant_config", None) is not None:
            raise NotImplementedError(
                "FL Ascend graph norm/quant fusion for quantized models is not "
                "included in the current closure"
            )
        logger.debug("Dense model: norm/quant fusion registered zero patterns")

    def __call__(self, graph: torch.fx.Graph) -> None:
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s norm/quant patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
