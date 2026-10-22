# SPDX-License-Identifier: Apache-2.0
"""ModelSlim loading components; registration is intentionally explicit."""

_modelslim_registered = False


def register_modelslim_config():
    """Register only when an Ascend integration explicitly requests ModelSlim."""
    global _modelslim_registered
    if _modelslim_registered:
        return
    from vllm.model_executor.layers.quantization import register_quantization_config

    from .config import AscendModelSlimConfig

    register_quantization_config(AscendModelSlimConfig.get_name())(
        AscendModelSlimConfig
    )
    _modelslim_registered = True
