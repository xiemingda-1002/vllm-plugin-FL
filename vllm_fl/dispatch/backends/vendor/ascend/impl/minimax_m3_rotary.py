# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-local partial rotary adapter through the selected operator bundle."""

from __future__ import annotations

import logging

import torch

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

logger = logging.getLogger(__name__)


class AscendApplyRotaryEmb(ApplyRotaryEmb):
    """Partial-rotation-correct ``ApplyRotaryEmb`` backed by ``npu_rotary_mul``."""

    def forward_oot(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)

        from vllm_fl.dispatch.backends.vendor.ascend.attention_selector import (
            get_minimax_sparse_kernels,
        )

        output = get_minimax_sparse_kernels().rotary(x, cos, sin, self.is_neox_style)
        return self._post_process(output, origin_shape, origin_dtype)


def install_apply_rotary_emb_override() -> bool:
    """Bind the rotary adapter only in MiniMax's vision construction module."""
    from vllm.models.minimax_m3.common import vision_tower

    if not hasattr(vision_tower, "ApplyRotaryEmb"):
        raise RuntimeError("MiniMax-M3 vision tower requires ApplyRotaryEmb")
    vision_tower.ApplyRotaryEmb = AscendApplyRotaryEmb
    return True


__all__ = ["AscendApplyRotaryEmb", "install_apply_rotary_emb_override"]
