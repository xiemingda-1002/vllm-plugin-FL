# Copyright (c) 2026 BAAI. All rights reserved.

"""Compatibility exports for the canonical Ascend Triton utilities.

Keep a single set of device-property globals.  Ascend kernels live below the
``impl.triton`` package and share these canonical utility exports.
"""

from .triton.triton_utils import (
    get_aicore_num,
    get_vectorcore_num,
    init_device_properties_triton,
)

__all__ = [
    "get_aicore_num",
    "get_vectorcore_num",
    "init_device_properties_triton",
]
