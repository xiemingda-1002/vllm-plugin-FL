"""Ascend activation helpers required by the rc1 MoE MLP interface."""

import torch
import torch.nn.functional as F


class AscendSwigluOAIAndMul:
    @staticmethod
    def swiglu_oai_forward(
        x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0
    ) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return gate * torch.sigmoid(alpha * gate) * (up + 1.0)


class AscendSwigluStepAndMul:
    @staticmethod
    def swiglustep_forward(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate).clamp(max=limit) * up.clamp(
            min=-limit, max=limit
        )
