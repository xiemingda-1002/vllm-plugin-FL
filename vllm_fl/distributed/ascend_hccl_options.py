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
# Derived from vllm-ascend's utils.py HCCL process-group configuration.
"""rc1-compatible HCCL process-group option construction for Ascend."""

from __future__ import annotations

import math

import torch_npu

_DEFAULT_BUFFER_SIZE = 200
_MIN_DP_BUFFER_SIZE = 50
_DYNAMIC_EPLB_BUFFER_SIZE = 100


def calculate_dp_buffer_size() -> int:
    from vllm.config import get_current_vllm_config

    dp_size = get_current_vllm_config().parallel_config.data_parallel_size
    return max(math.ceil((dp_size + 1) * 4 / (1024 * 1024)), _MIN_DP_BUFFER_SIZE)


def get_hccl_config_for_pg_options(group_name: str) -> dict | None:
    if group_name and "mc2" in group_name:
        return None
    configs = {
        "dp": {"hccl_buffer_size": calculate_dp_buffer_size()},
        "dynamic_eplb": {"hccl_buffer_size": _DYNAMIC_EPLB_BUFFER_SIZE},
    }
    return configs.get(group_name, {"hccl_buffer_size": _DEFAULT_BUFFER_SIZE})


def create_hccl_pg_options(group_name: str):
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    hccl_config = get_hccl_config_for_pg_options(group_name) or {}
    hccl_config["group_name"] = group_name
    options.hccl_config = hccl_config
    return options
