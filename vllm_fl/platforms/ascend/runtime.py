# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend runtime prerequisites, applied only by an initialized NPU worker."""


def configure_native_runtime() -> None:
    """Allow real NZ allocations/casts, as in Ascend rc1 model_runner_v1.

    Without this setting torch-npu may silently keep weights in base format
    even when the quantization method explicitly requests FRACTAL_NZ.
    Keep the import and mutation out of other vendors and model inspection.
    """
    import torch_npu

    torch_npu.npu.config.allow_internal_format = True
