"""HCCL-backed device communicator for the FL Ascend platform."""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger


logger = init_logger(__name__)


class _NpuAll2AllManager:
    @property
    def support_fault_tolerance(self) -> bool:
        return False

    def query_fault(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    def query_active_mask(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
    ) -> None:
        super().__init__(cpu_group, device, device_group, unique_name)
        self.device = torch.npu.current_device()
        self.ca_comm = None
        self.all2all_manager = _NpuAll2AllManager()
        if self.use_all2all:
            if self.all2all_backend not in (
                "naive",
                "allgather_reducescatter",
            ):
                logger.warning(
                    "`%s` all2all manager is not supported on NPU. "
                    "Falling back to `allgather_reducescatter` manager.",
                    self.all2all_backend,
                )
            from vllm.distributed.device_communicators.all2all import (
                AgRsAll2AllManager,
            )

            self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
            logger.info("Using allgather_reducescatter all2all manager on NPU.")

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        scatter_dim %= input_.dim()
        gather_dim %= input_.dim()
        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [
                tensor.contiguous()
                for tensor in torch.split(input_, scatter_sizes, scatter_dim)
            ]
            base = input_list[self.rank].size()
            output_list = []
            for size in gather_sizes:
                shape = list(base)
                shape[gather_dim] = size
                output_list.append(
                    torch.empty(shape, dtype=input_.dtype, device=input_.device)
                )
        else:
            input_list = [
                tensor.contiguous()
                for tensor in torch.tensor_split(input_, self.world_size, scatter_dim)
            ]
            output_list = [torch.empty_like(tensor) for tensor in input_list]
        dist.all_to_all(output_list, input_list, group=self.device_group)
        return torch.cat(output_list, dim=gather_dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")

        if sizes is not None and all(size == sizes[0] for size in sizes):
            sizes = None

        def _all_gather_single(
            tensor: torch.Tensor,
            sizes: list[int] | None,
        ) -> torch.Tensor:
            if sizes is None:
                return self.all_gather(tensor, dim=0)

            assert len(sizes) == self.world_size
            assert tensor.shape[0] == sizes[self.rank_in_group], (
                f"{tensor.shape[0]} != {sizes[self.rank_in_group]}"
            )
            gathered = [
                torch.empty(
                    (size,) + tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                for size in sizes
            ]
            dist.all_gather(gathered, tensor, group=self.device_group)
            return torch.cat(gathered, dim=0)

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)
        return [_all_gather_single(tensor, sizes) for tensor in input_]

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == self.world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % self.world_size == 0
            chunk_size = input_tensor.shape[0] // self.world_size

        output = torch.empty(
            (chunk_size,) + input_tensor.shape[1:],
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
            input_splits = list(input_tensor.split(sizes, dim=0))
            dist.reduce_scatter(output, input_splits, group=self.device_group)
        else:
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
        return output.movedim(0, dim).contiguous()

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        if not self.use_all2all:
            return super().dispatch_router_logits(
                hidden_states,
                router_logits,
                is_sequence_parallel,
                extra_tensors,
            )
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        if not self.use_all2all:
            return super().dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
            )
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if not self.use_all2all:
            return super().combine(hidden_states, is_sequence_parallel)
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )
