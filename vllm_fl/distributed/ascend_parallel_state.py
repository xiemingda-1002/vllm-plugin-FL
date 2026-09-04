# Copyright 2026 FlagOS Contributors
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
"""Ascend-specific model-parallel communication groups.

MC2 operators require a private HCCL communicator with the same membership
as expert parallelism.  The group is initialized with the other distributed
groups so every rank creates collectives in a deterministic order; model and
quantization code only consume the resulting coordinator.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_ep_group,
    get_world_group,
    init_model_parallel_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_MC2: GroupCoordinator | None = None
_MC2_TOPOLOGY: tuple[tuple[int, ...], ...] | None = None


def _mc2_required(vllm_config: Any) -> bool:
    """Return whether this configuration can consume the Ascend MC2 group."""
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return bool(
        model_config is not None
        and parallel_config is not None
        and getattr(model_config, "quantization", None) == "ascend"
        and bool(getattr(model_config, "is_moe", False))
        and bool(getattr(parallel_config, "enable_expert_parallel", False))
    )


def _build_mc2_group_ranks(
    world_size: int,
    data_parallel_size: int,
    pipeline_parallel_size: int,
    prefill_context_parallel_size: int,
    tensor_parallel_size: int,
) -> list[list[int]]:
    """Build independent groups with the same rank layout as vLLM EP."""
    model_parallel_size = (
        data_parallel_size
        * pipeline_parallel_size
        * prefill_context_parallel_size
        * tensor_parallel_size
    )
    if model_parallel_size <= 0 or world_size % model_parallel_size != 0:
        raise ValueError(
            "World size must be divisible by DP * PP * PCP * TP when "
            "initializing the Ascend MC2 group."
        )
    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_parallel_size,
        prefill_context_parallel_size,
        tensor_parallel_size,
    )
    ranks = (
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            data_parallel_size
            * prefill_context_parallel_size
            * tensor_parallel_size,
        )
        .unbind(0)
    )
    return [rank_group.tolist() for rank_group in ranks]


def init_ascend_mc2_group(vllm_config: Any) -> GroupCoordinator | None:
    """Initialize the private EP-shaped group used by Ascend MC2 operators."""
    if not _mc2_required(vllm_config):
        return None
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "The distributed environment must be initialized before the "
            "Ascend MC2 group."
        )

    parallel_config = vllm_config.parallel_config
    world_group = get_world_group()
    group_ranks = _build_mc2_group_ranks(
        int(world_group.world_size),
        int(parallel_config.data_parallel_size),
        int(parallel_config.pipeline_parallel_size),
        int(parallel_config.prefill_context_parallel_size),
        int(parallel_config.tensor_parallel_size),
    )
    topology = tuple(tuple(ranks) for ranks in group_ranks)

    global _MC2, _MC2_TOPOLOGY
    if _MC2 is not None:
        if _MC2_TOPOLOGY != topology:
            raise RuntimeError(
                "Ascend MC2 group is already initialized with a different "
                "parallel topology; destroy it before reconfiguration."
            )
        return _MC2

    rank = int(world_group.rank)
    expected_ranks = next(
        (ranks for ranks in group_ranks if rank in ranks),
        None,
    )
    if expected_ranks is None:
        raise RuntimeError(f"Rank {rank} is absent from the Ascend MC2 topology.")

    ep_group = get_ep_group()
    if (
        list(ep_group.ranks) != expected_ranks
        or int(ep_group.world_size) != len(expected_ranks)
        or int(ep_group.rank_in_group) != expected_ranks.index(rank)
    ):
        raise RuntimeError(
            "Ascend MC2 group membership must match the initialized expert "
            "parallel group."
        )

    backend = torch.distributed.get_backend(world_group.device_group)
    _MC2 = init_model_parallel_group(
        group_ranks,
        int(world_group.local_rank),
        backend,
        group_name="ascend_mc2",
        use_device_communicator=False,
    )
    _MC2_TOPOLOGY = topology
    logger.info_once(
        "Initialized Ascend MC2 communication group with %d ranks per group",
        _MC2.world_size,
    )
    return _MC2


def is_ascend_mc2_group_initialized() -> bool:
    """Return whether the private Ascend MC2 group exists."""
    return _MC2 is not None


def get_ascend_mc2_group() -> GroupCoordinator:
    """Return the initialized private Ascend MC2 group."""
    assert _MC2 is not None, "Ascend MC2 group is not initialized"
    return _MC2


def destroy_ascend_mc2_group() -> None:
    """Destroy the private Ascend MC2 group before distributed teardown."""
    global _MC2, _MC2_TOPOLOGY
    if _MC2 is not None:
        _MC2.destroy()
    _MC2 = None
    _MC2_TOPOLOGY = None


__all__ = [
    "destroy_ascend_mc2_group",
    "get_ascend_mc2_group",
    "init_ascend_mc2_group",
    "is_ascend_mc2_group_initialized",
]
