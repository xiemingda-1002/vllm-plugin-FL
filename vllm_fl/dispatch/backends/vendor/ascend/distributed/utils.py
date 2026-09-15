"""Small Ascend distributed helpers shared by DSA-CP."""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator


def all_gather_async(
    input: torch.Tensor,
    group: GroupCoordinator,
    output: torch.Tensor | None = None,
    async_op: bool = True,
):
    """Gather a TP-sharded tensor without introducing a new process group."""
    if group.world_size == 1:
        return input, None
    if output is None:
        output = torch.empty(
            (input.shape[0] * group.world_size,) + input.shape[1:],
            dtype=input.dtype,
            device=input.device,
        )
    return output, dist.all_gather_into_tensor(
        output, input, group=group.device_group, async_op=async_op
    )


def split_tensor_along_first_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Split the first dimension into equal rc1-style TP partitions."""
    from vllm.distributed.utils import divide

    first_dim_size = divide(tensor.size()[0], num_partitions)
    tensor_list = torch.split(tensor, first_dim_size, dim=0)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list
