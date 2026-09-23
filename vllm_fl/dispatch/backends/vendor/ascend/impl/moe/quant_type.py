"""Quantization identifiers used by the migrated Ascend MoE contracts.

Only ``NONE`` is executable in the first FL migration slice.  Keeping the
complete enum makes unsupported rc1 payloads fail at the execution boundary
instead of being misclassified as unquantized.
"""

from enum import Enum


class QuantType(Enum):
    NONE = 0
    W8A8 = 1
    W4A8 = 2
    W8A8MXFP = 3
    W4A16 = 4
    W4A4MXFP = 5
    W4A8MXFP = 6
    W8A8FP = 7
    W4A16MXFP = 8
