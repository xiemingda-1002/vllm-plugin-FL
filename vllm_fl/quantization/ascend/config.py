# SPDX-License-Identifier: Apache-2.0
"""Explicit ModelSlim configuration for dynamic INT8 DeepSeek weights."""

import json
from pathlib import Path

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import WeightsMapper

MODELSLIM_CONFIG_FILENAME = "quant_model_description.json"


def _deepseek_mapper():
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed_tokens.",
            "head.": "lm_head.",
        },
        orig_to_new_substr={
            ".attn.": ".self_attn.",
            ".w1.": ".gate_proj.",
            ".w2.": ".down_proj.",
            ".w3.": ".up_proj.",
            ".ffn.": ".mlp.",
            ".ffn_norm.": ".post_attention_layernorm.",
            ".attn_norm.": ".input_layernorm.",
        },
    )


class AscendModelSlimConfig(QuantizationConfig):
    """Load dynamic/FLOAT metadata without implicitly enabling a model.

    Both checkpoint and runtime prefixes use the same DeepSeek name mapper.
    Other model catalogues, quantized attention/cache and static INT8 are outside
    this component. Registration is an explicit action by platform integration.
    """

    def __init__(self, quant_config=None, *, weight_nz_mode=1):
        super().__init__()
        if weight_nz_mode not in (0, 1, 2):
            raise ValueError("weight_nz_mode must be 0, 1 or 2")
        self.weight_nz_mode = weight_nz_mode
        self.packed_modules_mapping = {
            "gate_up_proj": ["gate_proj", "up_proj"],
            "experts": [
                "experts.0.gate_proj",
                "experts.0.up_proj",
                "experts.0.down_proj",
            ],
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
        self.quant_description = _deepseek_mapper().apply_dict(description)
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
        if hf_config is not None and hf_config.model_type != "deepseek_v4":
            raise NotImplementedError("This ModelSlim component supports deepseek_v4")
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

    def get_layer_quant_type(self, prefix):
        # A suffix allows the mapper to rename final projection components too.
        prefix = _deepseek_mapper()._map_name(prefix + ".weight")[:-7]
        name = prefix.rsplit(".", 1)[-1]
        base = prefix[: -len(name)]
        shards = self.packed_modules_mapping.get(name, [name])
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
        quant_type = self.get_layer_quant_type(prefix)
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
