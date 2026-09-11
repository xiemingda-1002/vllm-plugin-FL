# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations


def get_rope_dim(vllm_config) -> int:
    """Return the rotary width using current vLLM model metadata."""
    model_config = vllm_config.model_config
    if model_config.use_mla:
        return int(model_config.hf_text_config.qk_rope_head_dim)
    rope_dim = int(model_config.get_head_size())
    if hasattr(model_config.hf_text_config, "partial_rotary_factor"):
        rope_dim = int(
            rope_dim * model_config.hf_text_config.partial_rotary_factor
        )
    elif hasattr(model_config.hf_text_config, "rotary_dim"):
        rope_dim = int(model_config.hf_text_config.rotary_dim)
    return rope_dim
