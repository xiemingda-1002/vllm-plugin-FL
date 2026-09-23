# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend-only layer-normalization implementations."""

from __future__ import annotations

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm,
    RMSNorm,
    RMSNormGated,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .fla.layernorm_gated import layer_norm_fwd_npu

_REGISTERED = False


def _ascend_rms_norm_gated_impl(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
) -> torch.Tensor:
    """Run the existing Ascend kernel behind an opaque dispatcher boundary."""
    original_shape = x.shape
    if z.shape != original_shape:
        raise ValueError(
            f"gate shape {tuple(z.shape)} must match input shape "
            f"{tuple(original_shape)}"
        )
    x = x.reshape(-1, x.shape[-1]).contiguous()
    z = z.reshape(-1, z.shape[-1]).contiguous()
    y, _, _ = layer_norm_fwd_npu(
        x,
        weight.contiguous(),
        None,
        eps,
        z=z,
        group_size=None if group_size < 0 else group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )
    return y.reshape(original_shape)


def _ascend_rms_norm_gated_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
) -> torch.Tensor:
    del z, weight, eps, group_size, norm_before_gate
    return torch.empty_like(x)


def ensure_ascend_rms_norm_gated_registered() -> None:
    """Register the functional op once during the Ascend patch lifecycle."""
    global _REGISTERED
    if _REGISTERED:
        return
    if hasattr(torch.ops.vllm, "ascend_rms_norm_gated"):
        raise RuntimeError(
            "torch.ops.vllm.ascend_rms_norm_gated already exists before "
            "FL Ascend registration"
        )
    direct_register_custom_op(
        op_name="ascend_rms_norm_gated",
        op_func=_ascend_rms_norm_gated_impl,
        fake_impl=_ascend_rms_norm_gated_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )
    _REGISTERED = True


class AscendGemmaRMSNorm(GemmaRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        weight = 1.0 + self.weight
        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, weight, self.variance_epsilon
            )
            return x, residual
        x, _ = torch_npu.npu_rms_norm(x, weight, self.variance_epsilon)
        return x


class AscendRMSNorm(RMSNorm):
    """BF16 RMSNorm with FlashComm1 residual-shard alignment."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        vllm_config = get_current_vllm_config()
        self.bias = None
        self.bias_loaded = False

        # Quantization with anti_method m4 can supply a non-zero norm bias.
        # Keep its parameter and loader on the norm module so the normal vLLM
        # weight-loading lifecycle finds ``q_a_layernorm.bias``.
        if vllm_config.quant_config is not None and any(
            "norm.bias" in name
            for name in vllm_config.quant_config.quant_description
        ):
            self.bias = torch.nn.Parameter(
                torch.zeros(hidden_size), requires_grad=False
            )
            self.bias.weight_loader = self._bias_weight_loader

    def _bias_weight_loader(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Scalar checkpoint values have no compatible shape to compare;
            # preserve the rc1 broadcast behavior.
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) into "
                f"parameter ({param.size()})"
            )
            param.data.copy_(loaded_weight)
        self.bias_loaded = True

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if not self.has_weight:
            return super().forward_native(x, residual)
        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon
            )
            # FL does not ship rc1's fused add-RMSNorm-bias extension. The
            # torch_npu path is the rc1 fallback and preserves residual
            # chunking while applying an allocated (possibly zero) bias.
            if self.bias is not None:
                x.add_(self.bias)
            return x, residual
        x, _ = torch_npu.npu_rms_norm(
            x, self.weight, self.variance_epsilon
        )
        if self.bias_loaded:
            x.add_(self.bias)
        return x


class AscendRMSNormGated(RMSNormGated):
    def forward_oot(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        if z is None or self.activation not in {"silu", "swish"}:
            return super().forward_native(x, z)
        return torch.ops.vllm.ascend_rms_norm_gated(
            x,
            z,
            self.weight,
            self.eps,
            -1 if self.group_size is None else self.group_size,
            self.norm_before_gate,
        )


__all__ = [
    "AscendGemmaRMSNorm",
    "AscendRMSNorm",
    "AscendRMSNormGated",
    "ensure_ascend_rms_norm_gated_registered",
]
