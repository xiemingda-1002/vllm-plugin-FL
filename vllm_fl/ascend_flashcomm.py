# Copyright (c) 2026 BAAI. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Torch-NPU-free configuration helpers for Ascend FlashComm1.

Common FL runner and worker modules are imported for every vendor.  Keep the
pure configuration and shape gate outside the Ascend implementation package,
whose package initializer loads Ascend operator implementations.
"""

from __future__ import annotations

from typing import Any


_ENABLE_FLASHCOMM1: bool | None = None
_IS_VL_MODEL: bool | None = None


def is_vl_model(vllm_config: Any | None = None) -> bool | None:
    """Return the current-rc1 process-cached VL-model classification.

    Model-runner initialization passes its explicit config to seed this cache.
    Linear operators later run outside a current-config context and consume the
    same value through the no-argument form.
    """
    global _IS_VL_MODEL

    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    model_config = (
        getattr(vllm_config, "model_config", None)
        if vllm_config is not None
        else None
    )
    if _IS_VL_MODEL is None and model_config is not None:
        if model_config.hf_config is not model_config.hf_text_config:
            _IS_VL_MODEL = True
        else:
            hf_config = model_config.hf_config.to_dict()
            _IS_VL_MODEL = (
                "thinker_config" in hf_config
                or "vision_config" in hf_config
            )
    return _IS_VL_MODEL


def is_moe_model(vllm_config: Any) -> bool:
    """Read current-vLLM's normalized MoE metadata when it is available."""
    parallel_config = getattr(vllm_config, "parallel_config", None)
    configured = getattr(parallel_config, "is_moe_model", None)
    if configured is not None:
        return bool(configured)

    model_config = getattr(vllm_config, "model_config", None)
    get_num_experts = getattr(model_config, "get_num_experts", None)
    return bool(get_num_experts and get_num_experts() > 0)


def enable_flashcomm1(
    vllm_config: Any | None = None,
    *,
    enable_shared_expert_dp: bool = False,
) -> bool:
    """Return the cached current-rc1 FlashComm1 configuration gate."""
    global _ENABLE_FLASHCOMM1

    if vllm_config is None:
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        except AssertionError:
            vllm_config = None

    additional_config = (
        getattr(vllm_config, "additional_config", None)
        if vllm_config is not None
        else None
    )
    refresh = bool(additional_config and additional_config.get("refresh", False))
    if _ENABLE_FLASHCOMM1 is None or refresh:
        _ENABLE_FLASHCOMM1 = bool(
            additional_config
            and additional_config.get("enable_flashcomm1", False)
        )
        if not _ENABLE_FLASHCOMM1 and enable_shared_expert_dp:
            _ENABLE_FLASHCOMM1 = True

    return bool(_ENABLE_FLASHCOMM1)


def flashcomm1_enabled_for_forward(
    vllm_config: Any,
    num_tokens: int | None,
    *,
    is_draft_model: bool = False,
) -> bool:
    """Apply rc1's model/token gate for one forward invocation."""
    if not enable_flashcomm1(vllm_config) or num_tokens is None:
        return False
    if is_moe_model(vllm_config):
        return True
    if is_draft_model:
        return False
    return num_tokens > 1000


def flashcomm1_attention_output_tokens(
    *,
    layer_idx: int,
    num_tokens: int,
    tp_size: int,
    enabled: bool,
) -> int:
    """Return the current-rc1 Qwen attention output length for FlashComm1."""
    if layer_idx == 0 and enabled:
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
        return (num_tokens + tp_size - 1) // tp_size
    return num_tokens


def validate_and_update_flashcomm1_config(vllm_config: Any) -> None:
    """Validate FlashComm1 topology and keep graph shapes TP divisible."""
    if not enable_flashcomm1(vllm_config):
        return

    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    assert tp_size > 1, "Flash Comm v1 is only supported when tp_size > 1."
    assert not is_moe_model(vllm_config) or parallel_config.enable_expert_parallel, (
        "Flash Comm v1 requires enable_expert_parallel=True for MoE models."
    )

    compilation_config = vllm_config.compilation_config
    model_config = getattr(vllm_config, "model_config", None)
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    graph_enabled = bool(
        cudagraph_mode is not None
        and getattr(cudagraph_mode, "name", str(cudagraph_mode)) != "NONE"
    )
    capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    if (
        not graph_enabled
        or model_config is None
        or getattr(model_config, "enforce_eager", False)
        or not capture_sizes
    ):
        return

    update_sizes = getattr(vllm_config, "update_sizes_for_sequence_parallelism", None)
    if update_sizes is not None:
        flashcomm1_sizes = update_sizes(capture_sizes)
    else:
        flashcomm1_sizes = [size for size in capture_sizes if size % tp_size == 0]
    assert flashcomm1_sizes, (
        f"cudagraph_capture_sizes {capture_sizes} does not contain values that "
        f"are multiples of tp_size {tp_size}"
    )
    if len(flashcomm1_sizes) != len(capture_sizes):
        compilation_config.max_cudagraph_capture_size = flashcomm1_sizes[-1]
        compilation_config.cudagraph_capture_sizes = flashcomm1_sizes


__all__ = [
    "enable_flashcomm1",
    "flashcomm1_attention_output_tokens",
    "flashcomm1_enabled_for_forward",
    "is_vl_model",
    "is_moe_model",
    "validate_and_update_flashcomm1_config",
]
