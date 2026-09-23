# Copyright (c) 2026 BAAI. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Compatibility facade for the torch-NPU-free FlashComm1 helpers."""

from vllm_fl.ascend_flashcomm import (
    enable_flashcomm1,
    flashcomm1_enabled_for_forward,
    is_moe_model,
    validate_and_update_flashcomm1_config,
)

__all__ = [
    "enable_flashcomm1",
    "flashcomm1_enabled_for_forward",
    "is_moe_model",
    "validate_and_update_flashcomm1_config",
]
