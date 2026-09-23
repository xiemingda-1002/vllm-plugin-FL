# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Tensor lifetime helpers from vLLM-Ascend v0.24.0rc1 utils.py."""

from typing import Any

import torch
import torch_npu
from vllm.sequence import IntermediateTensors


def weak_ref_tensor(tensor: Any) -> Any:
    """Share tensor storage without retaining the owning graph allocation."""
    if isinstance(tensor, torch.Tensor):
        return torch_npu._C._weak_ref_tensor(tensor)
    return tensor


def weak_ref_tensors(tensors):
    """Preserve rc1 NPU weak-reference semantics for graph-owned buffers."""
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    if isinstance(tensors, IntermediateTensors):
        return IntermediateTensors(
            {key: weak_ref_tensor(value) for key, value in tensors.tensors.items()}
        )
    raise ValueError("Invalid type for tensors")
