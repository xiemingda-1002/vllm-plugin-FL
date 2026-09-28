# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend activation operator implementations.
"""

from __future__ import annotations

import torch


def silu_and_mul_ascend(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using Ascend NPU.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    import torch_npu

    return torch_npu.npu_swiglu(x)


def gelu_and_mul_ascend(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using a fused
    torch_npu operator.

    Used by MiniMax-M3's vision tower and multimodal projector, whose config
    selects ``gelu``. Without this implementation those layers fall back to the
    PyTorch reference (``USE_FLAGGEMS=0``), which is both slower and outside
    this repository's own operator set.

    Args:
        obj: The calling obj (``GeluAndMul``), which carries ``approximate``.
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    # ``npu_gelu_mul`` splits x into two halves exactly like GeluAndMul, but the
    # torch_npu op only accepts "none"/"tanh"; other variants keep the eager
    # PyTorch path so behaviour is unchanged for models we do not target.
    if approximate not in ("none", "tanh"):
        import torch.nn.functional as F

        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=approximate) * x[..., d:]
    return torch.ops.npu.npu_gelu_mul(x, approximate=approximate)
