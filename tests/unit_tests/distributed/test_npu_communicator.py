# Copyright (c) 2026 BAAI. All rights reserved.

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.device_communicators import all2all as all2all_module
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


_MODULE_PATH = (
    Path(__file__).parents[3]
    / "vllm_fl"
    / "distributed"
    / "device_communicators"
    / "npu_communicator.py"
)
_SPEC = importlib.util.spec_from_file_location("fl_test_npu_communicator", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
npu_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(npu_module)


def _patch_communicator_init(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_all2all: bool,
    all2all_backend: str | None,
) -> None:
    def fake_base_init(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name="",
    ) -> None:
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name
        self.use_all2all = use_all2all
        self.all2all_backend = all2all_backend
        self.rank = 0
        self.world_size = 1

    monkeypatch.setattr(DeviceCommunicatorBase, "__init__", fake_base_init)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_device=lambda: 3),
        raising=False,
    )


@pytest.mark.parametrize("backend", ["naive", "allgather_reducescatter"])
def test_npu_communicator_uses_agrs_manager_for_supported_ep_backends(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    _patch_communicator_init(
        monkeypatch,
        use_all2all=True,
        all2all_backend=backend,
    )
    cpu_group = object()
    manager = object()
    manager_cls = MagicMock(return_value=manager)
    monkeypatch.setattr(all2all_module, "AgRsAll2AllManager", manager_cls)

    communicator = npu_module.NPUCommunicator(cpu_group)

    manager_cls.assert_called_once_with(cpu_group)
    assert communicator.all2all_manager is manager
    assert communicator.device == 3
    assert communicator.ca_comm is None


def test_npu_communicator_falls_back_to_agrs_for_unsupported_ep_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_communicator_init(
        monkeypatch,
        use_all2all=True,
        all2all_backend="deepep_high_throughput",
    )
    cpu_group = object()
    manager = object()
    manager_cls = MagicMock(return_value=manager)
    warning = MagicMock()
    monkeypatch.setattr(all2all_module, "AgRsAll2AllManager", manager_cls)
    monkeypatch.setattr(npu_module.logger, "warning", warning)

    communicator = npu_module.NPUCommunicator(cpu_group)

    manager_cls.assert_called_once_with(cpu_group)
    assert communicator.all2all_manager is manager
    warning.assert_called_once_with(
        "`%s` all2all manager is not supported on NPU. "
        "Falling back to `allgather_reducescatter` manager.",
        "deepep_high_throughput",
    )


def test_npu_communicator_preserves_non_ep_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_communicator_init(
        monkeypatch,
        use_all2all=False,
        all2all_backend=None,
    )
    manager_cls = MagicMock()
    monkeypatch.setattr(all2all_module, "AgRsAll2AllManager", manager_cls)

    communicator = npu_module.NPUCommunicator(object())

    manager_cls.assert_not_called()
    assert isinstance(communicator.all2all_manager, npu_module._NpuAll2AllManager)


def test_npu_communicator_delegates_ep_dispatch_and_combine() -> None:
    communicator = object.__new__(npu_module.NPUCommunicator)
    communicator.use_all2all = True
    communicator.all2all_manager = MagicMock()

    hidden_states = object()
    router_logits = object()
    topk_weights = object()
    topk_ids = object()
    extra_tensors = [object()]
    router_result = object()
    dispatch_result = object()
    combine_result = object()
    communicator.all2all_manager.dispatch_router_logits.return_value = router_result
    communicator.all2all_manager.dispatch.return_value = dispatch_result
    communicator.all2all_manager.combine.return_value = combine_result

    assert (
        communicator.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel=True,
            extra_tensors=extra_tensors,
        )
        is router_result
    )
    assert (
        communicator.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel=True,
            extra_tensors=extra_tensors,
        )
        is dispatch_result
    )
    assert (
        communicator.combine(hidden_states, is_sequence_parallel=True) is combine_result
    )

    communicator.all2all_manager.dispatch_router_logits.assert_called_once_with(
        hidden_states,
        router_logits,
        True,
        extra_tensors,
    )
    communicator.all2all_manager.dispatch.assert_called_once_with(
        hidden_states,
        topk_weights,
        topk_ids,
        True,
        extra_tensors=extra_tensors,
    )
    communicator.all2all_manager.combine.assert_called_once_with(hidden_states, True)


def test_npu_communicator_preserves_non_ep_dispatch_noops() -> None:
    communicator = object.__new__(npu_module.NPUCommunicator)
    communicator.use_all2all = False
    communicator.all2all_manager = npu_module._NpuAll2AllManager()
    hidden_states = object()
    router_logits = object()
    topk_weights = object()
    topk_ids = object()
    extra_tensors = [object()]

    assert communicator.dispatch_router_logits(
        hidden_states,
        router_logits,
        extra_tensors=extra_tensors,
    ) == (hidden_states, router_logits, extra_tensors)
    assert communicator.dispatch(
        hidden_states,
        topk_weights,
        topk_ids,
        extra_tensors=extra_tensors,
    ) == (hidden_states, topk_weights, topk_ids, extra_tensors)
    assert communicator.combine(hidden_states) is hidden_states


def _collective_communicator(
    *,
    world_size: int = 2,
    rank_in_group: int = 1,
) -> npu_module.NPUCommunicator:
    communicator = object.__new__(npu_module.NPUCommunicator)
    communicator.world_size = world_size
    communicator.rank_in_group = rank_in_group
    communicator.device_group = object()
    return communicator


def test_npu_all_gatherv_equal_sizes_uses_all_gather_fastpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = _collective_communicator()
    inputs = [torch.arange(4).reshape(2, 2), torch.arange(2).reshape(2, 1)]
    expected = [torch.arange(8).reshape(4, 2), torch.arange(4).reshape(4, 1)]
    all_gather = MagicMock(side_effect=expected)
    monkeypatch.setattr(communicator, "all_gather", all_gather)

    outputs = communicator.all_gatherv(inputs, dim=0, sizes=[2, 2])

    assert outputs is not None
    assert len(outputs) == 2
    assert torch.equal(outputs[0], expected[0])
    assert torch.equal(outputs[1], expected[1])
    assert all_gather.call_count == 2
    assert all_gather.call_args_list[0].args[0] is inputs[0]
    assert all_gather.call_args_list[0].kwargs == {"dim": 0}
    assert all_gather.call_args_list[1].args[0] is inputs[1]
    assert all_gather.call_args_list[1].kwargs == {"dim": 0}


def test_npu_all_gatherv_uneven_sizes_uses_list_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = _collective_communicator(rank_in_group=1)
    input_ = torch.full((2, 2), 2.0)
    all_gather = MagicMock()

    def gather(outputs, tensor, group) -> None:
        assert group is communicator.device_group
        outputs[0].fill_(1.0)
        outputs[1].copy_(tensor)

    all_gather.side_effect = gather
    monkeypatch.setattr(npu_module.dist, "all_gather", all_gather)

    output = communicator.all_gatherv(input_, dim=0, sizes=[1, 2])

    assert torch.equal(
        output,
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]),
    )
    all_gather.assert_called_once()


def test_npu_reduce_scatterv_equal_sizes_uses_tensor_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = _collective_communicator(rank_in_group=1)
    input_ = torch.arange(8).reshape(4, 2)
    reduce_scatter_tensor = MagicMock()
    reduce_scatter = MagicMock()

    def reduce(output, tensor, group) -> None:
        assert group is communicator.device_group
        output.copy_(tensor[2:])

    reduce_scatter_tensor.side_effect = reduce
    monkeypatch.setattr(
        npu_module.dist,
        "reduce_scatter_tensor",
        reduce_scatter_tensor,
    )
    monkeypatch.setattr(npu_module.dist, "reduce_scatter", reduce_scatter)

    output = communicator.reduce_scatterv(input_, dim=0, sizes=[2, 2])

    assert torch.equal(output, input_[2:])
    reduce_scatter_tensor.assert_called_once()
    reduce_scatter.assert_not_called()


def test_npu_reduce_scatterv_uneven_sizes_uses_list_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = _collective_communicator(rank_in_group=1)
    input_ = torch.arange(6).reshape(3, 2)
    reduce_scatter = MagicMock()
    reduce_scatter_tensor = MagicMock()

    def reduce(output, input_splits, group) -> None:
        assert group is communicator.device_group
        assert [tensor.shape[0] for tensor in input_splits] == [1, 2]
        output.copy_(input_splits[1])

    reduce_scatter.side_effect = reduce
    monkeypatch.setattr(npu_module.dist, "reduce_scatter", reduce_scatter)
    monkeypatch.setattr(
        npu_module.dist,
        "reduce_scatter_tensor",
        reduce_scatter_tensor,
    )

    output = communicator.reduce_scatterv(input_, dim=0, sizes=[1, 2])

    assert torch.equal(output, input_[1:])
    reduce_scatter.assert_called_once()
    reduce_scatter_tensor.assert_not_called()
