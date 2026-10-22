# SPDX-License-Identifier: Apache-2.0
"""ModelSlim loading components; registration is intentionally explicit."""


def register_modelslim_config():
    """Register only when an Ascend integration explicitly requests ModelSlim."""
    from vllm.model_executor.layers.quantization import register_quantization_config

    from .config import AscendModelSlimConfig

    register_quantization_config(AscendModelSlimConfig.get_name())(
        AscendModelSlimConfig
    )
