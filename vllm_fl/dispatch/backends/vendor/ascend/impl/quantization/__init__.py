# Copyright (c) 2026 BAAI. All rights reserved.
"""Ascend-only ModelSlim registration; imports do not initialize an NPU."""

ASCEND_QUANTIZATION_METHOD = "ascend"
MODELSLIM_CONFIG_FILENAME = "quant_model_description.json"
_registered = False


def register_modelslim(parser=None):
    """Mirror rc1 registration for both CLI and programmatic engine creation."""
    global _registered
    if not _registered:
        from vllm.model_executor.layers.quantization import register_quantization_config

        from .config import AscendModelSlimConfig

        register_quantization_config(ASCEND_QUANTIZATION_METHOD)(AscendModelSlimConfig)
        _registered = True
    if parser is not None:
        action = parser._option_string_actions.get("--quantization")
        if (
            action is not None
            and action.choices is not None
            and ASCEND_QUANTIZATION_METHOD not in action.choices
        ):
            action.choices = [*action.choices, ASCEND_QUANTIZATION_METHOD]


def maybe_auto_detect_quantization(vllm_config):
    # Config recreation requires the registry also for callers constructing
    # VllmConfig directly rather than going through EngineArgs.
    register_modelslim()
    from .utils import maybe_auto_detect_quantization as detect

    return detect(vllm_config)
