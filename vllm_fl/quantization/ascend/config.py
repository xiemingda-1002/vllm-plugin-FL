# SPDX-License-Identifier: Apache-2.0
"""Model-independent ModelSlim metadata and dynamic INT8 method selection."""

import json
from pathlib import Path

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

MODELSLIM_CONFIG_FILENAME = "quant_model_description.json"
_EXPERT_SHARDS = (
    "experts.0.gate_proj",
    "experts.0.up_proj",
    "experts.0.down_proj",
)


class AscendModelSlimConfig(QuantizationConfig):
    """Load dynamic/FLOAT metadata without implicitly enabling a model.

    Checkpoint naming belongs to the model's hf_to_vllm_mapper; absent a mapper,
    names are preserved. Quantized attention/cache and static INT8 need their
    own methods. Registration is an explicit action by platform integration.
    """

    def __init__(self, quant_config=None, *, weight_nz_mode=1):
        super().__init__()
        if weight_nz_mode not in (0, 1, 2):
            raise ValueError("weight_nz_mode must be 0, 1 or 2")
        self.weight_nz_mode = weight_nz_mode
        self.packed_modules_mapping = {
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        self.hf_to_vllm_mapper = None
        self.quant_description = {}
        self._set_description(quant_config or {})

    def _set_description(self, description):
        if not isinstance(description, dict):
            raise ValueError("ModelSlim description must be a JSON object")
        for key, value in description.items():
            if (
                key.endswith((".weight", ".weight_scale", ".weight_offset"))
                or key == "model_quant_type"
            ) and value not in ("FLOAT", "W8A8_DYNAMIC"):
                raise NotImplementedError(
                    f"Unsupported ModelSlim type {value!r} at {key}"
                )
            if (
                key.endswith("quant_type")
                and key != "model_quant_type"
                and value != "FLOAT"
            ):
                raise NotImplementedError(
                    f"Quantized attention/cache metadata at {key}"
                )
            if key == "kv_cache_type" and value != "FLOAT":
                raise NotImplementedError("Quantized ModelSlim KV cache is unsupported")
            if key == "group_size" and value != 0:
                raise NotImplementedError(
                    "Grouped ModelSlim quantization is unsupported"
                )
        self.quant_description = dict(description)
        if self.hf_to_vllm_mapper is not None:
            self.quant_description = self.hf_to_vllm_mapper.apply_dict(
                self.quant_description
            )

    @classmethod
    def get_name(cls):
        return "ascend"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        raise NotImplementedError("Ascend does not have a CUDA compute capability")

    @staticmethod
    def get_config_filenames():
        # The descriptor is loaded explicitly by maybe_update_config.
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        if self.quant_description:
            return
        path = Path(model_name) / MODELSLIM_CONFIG_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Local ModelSlim descriptor missing: {path}")
        self._set_description(json.loads(path.read_text()))

    def apply_vllm_mapper(self, mapper):
        if self.hf_to_vllm_mapper is mapper:
            return
        if self.hf_to_vllm_mapper is not None:
            raise ValueError("ModelSlim mapper already applied; create a fresh config")
        self.quant_description = mapper.apply_dict(self.quant_description)
        self.hf_to_vllm_mapper = mapper

    def get_layer_quant_type(self, prefix, *, is_moe=False):
        # Prefixes come from runtime modules. Only checkpoint keys are mapped;
        # reapplying a model mapper here can remap an already canonical prefix.
        name = prefix.rsplit(".", 1)[-1]
        base = prefix[: -len(name)]
        # configure_quant_config replaces the model's Linear packing map by
        # reference. Expert-zero aliases describe the ModelSlim checkpoint
        # format, so they must survive that independent framework operation.
        shards = (
            _EXPERT_SHARDS
            if is_moe
            else self.packed_modules_mapping.get(name, [name])
        )
        keys = [base + shard + ".weight" for shard in shards]
        missing = [key for key in keys if key not in self.quant_description]
        if missing:
            raise ValueError(f"Missing ModelSlim weight metadata: {missing}")
        types = {self.quant_description[key] for key in keys}
        if len(types) != 1:
            raise ValueError(f"Mixed ModelSlim quantization for packed layer {prefix}")
        return types.pop()

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        is_moe = isinstance(layer, (RoutedExperts, MoERunner))
        if not isinstance(layer, LinearBase) and not is_moe:
            return None
        quant_type = self.get_layer_quant_type(prefix, is_moe=is_moe)
        if quant_type == "FLOAT":
            # The existing FL MoE factory retains its unquantized method path.
            return None if is_moe else UnquantizedLinearMethod()
        if quant_type != "W8A8_DYNAMIC":
            raise NotImplementedError(f"Unsupported ModelSlim type {quant_type!r}")
        if is_moe:
            from .moe import AscendModelSlimMoEMethod

            return AscendModelSlimMoEMethod(layer.moe_config)
        from .linear import AscendW8A8DynamicLinearMethod

        return AscendW8A8DynamicLinearMethod(self.weight_nz_mode)
