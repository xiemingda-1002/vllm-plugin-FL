"""Ascend shared-expert execution resources owned by the FL runtime."""

from __future__ import annotations

from contextlib import nullcontext

import torch

_SHARED_EXPERTS_CALCULATION_STREAM = None


def npu_stream_switch(target_stream=None, *, enabled: bool = True):
    if not enabled:
        return nullcontext()
    assert target_stream is not None
    return torch.npu.stream(target_stream)


def shared_experts_calculation_stream():
    global _SHARED_EXPERTS_CALCULATION_STREAM
    if _SHARED_EXPERTS_CALCULATION_STREAM is None:
        # Keep torch_npu behind the Ascend-only call path so importing FL on a
        # different vendor does not initialize the NPU runtime.
        import torch_npu

        _SHARED_EXPERTS_CALCULATION_STREAM = torch_npu.npu.Stream()
    return _SHARED_EXPERTS_CALCULATION_STREAM
