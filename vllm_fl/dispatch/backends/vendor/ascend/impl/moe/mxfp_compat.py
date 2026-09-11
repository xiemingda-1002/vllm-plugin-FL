"""Import-safe MXFP symbols for rc1 contract preservation."""

import torch

try:
    import torch_npu
except ImportError:  # pragma: no cover - exercised only on non-NPU test hosts
    torch_npu = None


FLOAT8_E8M0FNU_DTYPE = getattr(
    torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None)
)


def ensure_mxfp8_moe_available(_feature: str) -> None:
    raise NotImplementedError("FL Ascend rc1 MoE MXFP execution is not migrated")
