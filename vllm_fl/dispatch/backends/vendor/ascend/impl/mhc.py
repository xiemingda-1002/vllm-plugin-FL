# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend implementations of DeepSeek manifold-constrained hyper-connections."""

from __future__ import annotations

import torch


def mhc_pre_ascend(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the native pre-mixing path and preserve the FL dispatch contract."""
    del hc_pre_eps, hc_post_mult_value, n_splits
    layer_input, post_mix, res_mix = torch.ops._C_ascend.npu_hc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        hc_mult=residual.shape[-2],
        hc_sinkhorn_iters=sinkhorn_repeat,
        norm_eps=rms_eps,
        hc_eps=hc_sinkhorn_eps,
    )
    return post_mix, res_mix, layer_input


def mhc_post_ascend(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Run native post-mixing, adding the batch dimension expected by CANN."""
    return torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(0),
        residual.unsqueeze(0),
        post.unsqueeze(0),
        comb.unsqueeze(0),
    ).squeeze(0)


__all__ = ["mhc_post_ascend", "mhc_pre_ascend"]
