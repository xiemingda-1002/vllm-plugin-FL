# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the AscendC to Triton operator migrations.

Source: the AscendC operators under csrc/ascend/ on the vllm-fl branch
feature/v024-ascend-platform-runtime.

Each operator lives in its own module in this package and keeps the function name of
the original operator. All implementations use only standard triton /
triton.language and do not depend on vllm / triton_ascend / cann.extension.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["npu_device", "ceil_div", "next_pow2", "IS_NPU"]

IS_NPU = hasattr(torch, "npu") and torch.npu.is_available()


def npu_device(idx: int = 0) -> str:
    """Return the device string of the currently available device."""
    return f"npu:{idx}" if IS_NPU else "cpu"


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())


def grid_1d(n: int, block: int) -> tuple[int]:
    return (ceil_div(n, block),)
