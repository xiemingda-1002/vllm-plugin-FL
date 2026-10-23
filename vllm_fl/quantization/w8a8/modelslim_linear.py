# SPDX-License-Identifier: Apache-2.0
"""Bounded ModelSlim W8A8_DYNAMIC linear parameter contract."""

import torch

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm_fl.dispatch import CachedOp

_prepare_weight = CachedOp("modelslim_w8a8_prepare_weight")
_linear = CachedOp("modelslim_w8a8_dynamic_linear")


class AscendW8A8DynamicLinearMethod(LinearMethodBase):
    """Load [N,K] int8 weights and channel scales; execute prepared [K,N].

    The caller supplies the resolved weight_nz_mode when config integration is
    added. Modes 1 and 2 both NZ-cast int8; mode 0 retains ND. This component
    supports A2/A3 without FlashComm2 quantized communication or DSA-CP splitting.
    """

    def __init__(self, weight_nz_mode: int = 1):
        if weight_nz_mode not in (0, 1, 2):
            raise ValueError("weight_nz_mode must be 0, 1 or 2")
        self.weight_nz_mode = weight_nz_mode

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError("ModelSlim dynamic Linear requires BF16/FP16")
        n = sum(output_partition_sizes)
        weight = torch.nn.Parameter(
            torch.empty(n, input_size_per_partition, dtype=torch.int8),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)
        for name in ("weight_scale", "weight_offset"):
            param = torch.nn.Parameter(
                (torch.zeros if name == "weight_offset" else torch.empty)(
                    n, 1, dtype=params_dtype
                ),
                requires_grad=False,
            )
            set_weight_attrs(param, {"output_dim": 0})
            set_weight_attrs(param, extra_weight_attrs)
            layer.register_parameter(name, param)

    def get_computed_params(self) -> set[str]:
        # Symmetric offsets can be computed rather than required weights.
        return {"weight_offset"}

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype != torch.int8 or layer.weight.ndim != 2:
            raise ValueError("Expected ModelSlim [N,K] int8 weight")
        if layer.weight_scale.shape != (layer.weight.shape[0], 1):
            raise ValueError("Expected one ModelSlim scale per output channel")
        if layer.weight_offset.shape != layer.weight_scale.shape:
            raise ValueError("Expected one ModelSlim offset per output channel")
        # Dynamic INT8 arithmetic loads offsets but does not apply them.
        # Restrict this bounded component to the symmetric contract explicitly.
        if torch.count_nonzero(layer.weight_offset).item():
            raise NotImplementedError("Nonzero ModelSlim dynamic weight offsets")
        if layer.weight.shape[0] >= 65536:
            raise NotImplementedError(
                "Large-output DSA-CP Linear splitting is not enabled"
            )
        weight, weight_scale = _prepare_weight(
            layer.weight.data.transpose(0, 1).contiguous(),
            layer.weight_scale.data.flatten(),
            self.weight_nz_mode,
        )
        layer.weight.data = weight
        layer.weight_scale.data = weight_scale
        layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply(self, layer, x, bias=None):
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(
                "ModelSlim dynamic Linear requires BF16/FP16 input"
            )
        if x.ndim < 2 or x.shape[-1] != layer.weight.shape[0]:
            raise ValueError("Expected [..., K] activations matching prepared [K,N]")
        if bias is not None and bias.shape != (layer.weight.shape[1],):
            raise ValueError("Expected bias [N]")
        shape = (*x.shape[:-1], layer.weight.shape[1])
        return _linear(
            x.reshape(-1, x.shape[-1]).contiguous(),
            layer.weight,
            layer.weight_scale,
            bias,
        ).reshape(shape)
