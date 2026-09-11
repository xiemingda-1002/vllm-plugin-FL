"""Shape helpers shared by the current Qwen GDN causal-convolution path."""

from __future__ import annotations

import torch


def extract_last_width(x: torch.Tensor, start_loc: torch.Tensor, width: int):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)
    return x[:, indices].permute(1, 0, 2)
