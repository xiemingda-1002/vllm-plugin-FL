"""rc1 fine-grained Ascend tensor-parallel process groups.

These groups are distinct from vLLM's ordinary TP group.  DeepSeek-V4 uses
the output-projection group when fine-grained OProj TP is enabled.
"""

from __future__ import annotations

import torch

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_world_group,
    init_model_parallel_group,
)

from vllm_fl.dispatch.backends.vendor.ascend.dsa_compat import get_ascend_config

_MC2: GroupCoordinator | None = None
_MLP_TP: GroupCoordinator | None = None
_OTP: GroupCoordinator | None = None
_LMTP: GroupCoordinator | None = None
_EMBED_TP: GroupCoordinator | None = None


def init_ascend_model_parallel(parallel_config: ParallelConfig) -> None:
    """Initialize the rc1 Ascend-only groups from the active vLLM topology."""
    global _MC2, _MLP_TP, _OTP, _LMTP, _EMBED_TP
    if _MC2 is not None:
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    world_group = get_world_group()
    backend = torch.distributed.get_backend(world_group.device_group)
    global_tp_size = parallel_config.tensor_parallel_size
    global_dp_size = parallel_config.data_parallel_size
    global_pp_size = parallel_config.pipeline_parallel_size
    global_pcp_size = parallel_config.prefill_context_parallel_size
    all_ranks = torch.arange(world_size).reshape(
        -1,
        global_dp_size,
        global_pp_size,
        global_pcp_size,
        global_tp_size,
    )

    mc2_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(-1, global_dp_size * global_pcp_size * global_tp_size)
        .unbind(0)
    )
    _MC2 = init_model_parallel_group(
        [ranks.tolist() for ranks in mc2_ranks],
        world_group.local_rank,
        backend,
        group_name="mc2",
    )

    finegrained = get_ascend_config().finegrained_tp_config
    group_cache: dict[int, GroupCoordinator] = {}

    def create_or_get_group(group_size: int, group_name: str) -> GroupCoordinator | None:
        if group_size <= 0:
            return None
        if group_size > global_dp_size or global_dp_size % group_size:
            raise ValueError(
                f"{group_name} size {group_size} must divide data-parallel size "
                f"{global_dp_size}."
            )
        if group_size not in group_cache:
            rank_grid = torch.arange(world_size).reshape(
                global_pp_size, global_dp_size, global_tp_size
            )
            group_ranks: list[list[int]] = []
            for pp_idx in range(global_pp_size):
                stage_ranks = rank_grid[pp_idx]
                for chunk in range(global_dp_size // group_size):
                    for tp_idx in range(global_tp_size):
                        group_ranks.append(
                            stage_ranks[
                                chunk * group_size : (chunk + 1) * group_size,
                                tp_idx,
                            ].tolist()
                        )
            group_cache[group_size] = init_model_parallel_group(
                group_ranks,
                world_group.local_rank,
                backend,
                group_name=group_name,
            )
        return group_cache[group_size]

    _OTP = create_or_get_group(
        finegrained.oproj_tensor_parallel_size, "otp"
    )
    _LMTP = create_or_get_group(
        finegrained.lmhead_tensor_parallel_size, "lmheadtp"
    )
    _EMBED_TP = create_or_get_group(
        finegrained.embedding_tensor_parallel_size, "emtp"
    )
    _MLP_TP = create_or_get_group(
        finegrained.mlp_tensor_parallel_size, "mlptp"
    )


def model_parallel_initialized() -> bool:
    return _MC2 is not None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, "mc2 group is not initialized"
    return _MC2


def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, "mlp group is not initialized"
    return _MLP_TP


def get_otp_group() -> GroupCoordinator:
    assert _OTP is not None, "output tensor parallel group is not initialized"
    return _OTP


def get_lmhead_tp_group() -> GroupCoordinator:
    assert _LMTP is not None, "lm head tensor parallel group is not initialized"
    return _LMTP


def get_embed_tp_group() -> GroupCoordinator:
    assert _EMBED_TP is not None, "emtp group is not initialized"
    return _EMBED_TP


def destroy_ascend_model_parallel() -> None:
    global _MC2, _MLP_TP, _OTP, _LMTP, _EMBED_TP
    for group in (_MC2, _MLP_TP, _OTP, _LMTP, _EMBED_TP):
        if group is not None:
            group.destroy()
    _MC2 = _MLP_TP = _OTP = _LMTP = _EMBED_TP = None
