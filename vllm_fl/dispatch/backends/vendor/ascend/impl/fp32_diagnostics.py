# Copyright (c) 2026 BAAI. All rights reserved.

"""Compatibility import for the explicit validation worker extension.

New validation tools should use vllm_fl.validation.fp32_diagnostics directly.
Normal attention execution does not import this module.
"""

from vllm_fl.validation.fp32_diagnostics import FP32Diagnostics

__all__ = ["FP32Diagnostics"]
