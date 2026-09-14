#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Compute-stream synchronization fence for Ascend MemCache transfers."""

from __future__ import annotations

import threading

import torch

_lock = threading.RLock()
_attention_compute_start_gate: AttentionComputeStartGate | None = None


class AttentionComputeStartGate:
    """Open when the compute stream reaches an attention operation."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._event: torch.npu.Event | None = None

    def record(self, stream: torch.npu.Stream | None = None) -> None:
        stream = stream or torch.npu.current_stream()
        event = torch.npu.Event()
        event.record(stream)
        with self._condition:
            if self._event is None:
                self._event = event
                self._condition.notify_all()

    def wait(self, timeout: float = 10.0) -> bool:
        with self._condition:
            while self._event is None:
                if not self._condition.wait(timeout=timeout):
                    return False
            event = self._event

        event.synchronize()
        return True


def reset_attention_compute_start_gate() -> AttentionComputeStartGate:
    """Create and publish a new gate for layerwise MemCache work."""
    global _attention_compute_start_gate
    gate = AttentionComputeStartGate()
    with _lock:
        _attention_compute_start_gate = gate
    return gate


def get_attention_compute_start_gate() -> AttentionComputeStartGate:
    with _lock:
        gate = _attention_compute_start_gate
    if gate is None:
        gate = reset_attention_compute_start_gate()
    return gate


def record_attention_compute_start() -> None:
    """Record the compute-stream boundary immediately before attention."""
    with _lock:
        gate = _attention_compute_start_gate
    if gate is not None:
        gate.record()
