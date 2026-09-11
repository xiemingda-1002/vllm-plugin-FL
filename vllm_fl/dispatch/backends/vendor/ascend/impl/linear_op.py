# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend BF16 linear operations required by FlashComm1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.models.utils import extract_layer_index

from vllm_fl.ascend_flashcomm import enable_flashcomm1, is_vl_model
from vllm_fl.ascend_forward_context import _EXTRA_CTX

from .device_operator import DeviceOperator
from .moe.compat import shared_expert_dp_enabled


def flashcomm1_configured() -> bool:
    """Read the public current-vLLM additional-config switch."""
    return enable_flashcomm1()


def _sequence_column_needs_all_gather(
    prefix: str, is_vl: bool | None
) -> bool:
    """Match rc1's first-layer VL attention FlashComm1 exception."""
    return not (
        extract_layer_index(prefix) == 0 and is_vl and "attn" in prefix
    )


class CustomLinearOp:
    def __init__(self, layer):
        self.layer = layer
        self.bias = None
        self.skip_bias_add = False
        self.return_bias = True
        self.quant_method = None
        self.prefix = ""

    @property
    def comm_group(self):
        return get_tp_group()

    @property
    def tp_rank(self) -> int:
        return self.comm_group.rank_in_group

    @property
    def tp_size(self) -> int:
        return self.comm_group.world_size

    def update_attrs(self) -> None:
        self.bias = getattr(self.layer, "bias", None)
        self.skip_bias_add = self.layer.skip_bias_add
        self.return_bias = self.layer.return_bias
        self.quant_method = self.layer.quant_method
        self.prefix = self.layer.prefix

    def apply_impl(self, input_: torch.Tensor):
        raise NotImplementedError

    def apply(self, input_: torch.Tensor):
        output, output_bias = self.apply_impl(input_)
        if not self.return_bias:
            return output
        return output, output_bias


class CustomColumnParallelOp(CustomLinearOp):
    def update_attrs(self) -> None:
        super().update_attrs()
        self.gather_output = self.layer.gather_output


class CustomRowParallelOp(CustomLinearOp):
    def update_attrs(self) -> None:
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.reduce_results = self.layer.reduce_results
        self.input_size_per_partition = self.layer.input_size_per_partition

    def get_input_parallel(self, input_: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            return input_
        shards = split_tensor_along_last_dim(input_, self.tp_size)
        return shards[self.tp_rank].contiguous()


class CustomReplicatedOp(CustomLinearOp):
    def apply_impl(self, input_: torch.Tensor):
        bias = self.bias if not self.skip_bias_add else None
        output = self.quant_method.apply(self.layer, input_, bias)
        return output, self.bias if self.skip_bias_add else None


class SequenceColumnParallelOp(CustomColumnParallelOp):
    def apply_impl(self, input_: torch.Tensor):
        bias = self.bias if not self.skip_bias_add else None
        need_all_gather = _sequence_column_needs_all_gather(
            self.layer.prefix, is_vl_model()
        )
        input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            input_, label=need_all_gather
        )
        output = self.quant_method.apply(self.layer, input_, bias)
        if self.gather_output:
            output = self.comm_group.all_gather(output)
        return output, self.bias if self.skip_bias_add else None


class SequenceRowParallelOp(CustomRowParallelOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.unique_prefix: str | None = None

    def update_attrs(self) -> None:
        super().update_attrs()
        self.unique_prefix = self.layer.unique_prefix

    def apply_impl(self, input_: torch.Tensor):
        input_parallel = self.get_input_parallel(input_)
        bias = None if self.tp_rank > 0 or self.skip_bias_add else self.bias
        if self.tp_size == 1 or not self.reduce_results:
            output = self.quant_method.apply(self.layer, input_parallel, bias=bias)
        else:
            output = torch.ops.vllm.matmul_and_reduce(
                input_parallel, self.unique_prefix
            )
        return output, self.bias if self.skip_bias_add else None

    def matmul_and_reduce(
        self,
        input_parallel: torch.Tensor,
        bias: Parameter | None,
    ) -> torch.Tensor:
        try:
            enabled = _EXTRA_CTX.flash_comm_v1_enabled
            mmrs_fusion = _EXTRA_CTX.mmrs_fusion
        except AssertionError:
            enabled = False
            mmrs_fusion = False

        if not enabled:
            output = self.quant_method.apply(
                self.layer, input_parallel, bias=bias
            )
            return tensor_model_parallel_all_reduce(output)

        x = input_parallel
        if _EXTRA_CTX.pad_size > 0:
            x = F.pad(x, (0, 0, 0, _EXTRA_CTX.pad_size))

        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        if mmrs_fusion:
            if not isinstance(self.layer.quant_method, UnquantizedLinearMethod):
                raise NotImplementedError(
                    "FL FlashComm1 MMRS currently supports BF16 "
                    "unquantized linear layers only"
                )
            backend = get_tp_group().device_group._get_backend(
                torch.device("npu")
            )
            hcomm_name = backend.get_hccl_comm_name(self.layer.tp_rank)
            output = DeviceOperator.npu_mm_reduce_scatter_base(
                x,
                self.layer.weight.t(),
                hcomm_name,
                self.layer.tp_size,
                reduce_op="sum",
                bias=None,
                comm_turn=0,
            )
            if bias is not None:
                output.add_(bias)
            return output

        output = self.quant_method.apply(self.layer, x, bias=bias)
        return tensor_model_parallel_reduce_scatter(output, 0)


_COLUMN_PREFIXES = (
    "gate_up_proj",
    "in_proj",
    "qkv_proj",
    "conv1d",
    "query_key_value",
    "indexer_proj",
    "g_proj",
)
_ROW_PREFIXES = (
    "o_proj",
    "out_proj",
    "down_proj",
    "attention.dense",
    "wo_b",
)


def _is_shared_expert(prefix: str) -> bool:
    return any(name in prefix for name in ("shared_expert", "share_expert"))


def get_parallel_op(disable_tp, prefix, layer, direction):
    if disable_tp or (_is_shared_expert(prefix) and shared_expert_dp_enabled()):
        return None, 0, 1

    custom_op = None
    if flashcomm1_configured() and not _is_shared_expert(prefix):
        if direction == "column" and any(name in prefix for name in _COLUMN_PREFIXES):
            custom_op = SequenceColumnParallelOp(layer)
        elif direction == "row" and any(name in prefix for name in _ROW_PREFIXES):
            custom_op = SequenceRowParallelOp(layer)

    if custom_op is not None:
        return custom_op, custom_op.tp_rank, custom_op.tp_size
    tp_group = get_tp_group()
    return None, tp_group.rank_in_group, tp_group.world_size


def get_replicated_op(disable_tp, prefix, layer):
    del prefix
    if disable_tp:
        return None, None, None
    custom_op = CustomReplicatedOp(layer)
    return custom_op, custom_op.tp_rank, custom_op.tp_size


__all__ = [
    "CustomReplicatedOp",
    "SequenceColumnParallelOp",
    "SequenceRowParallelOp",
    "flashcomm1_configured",
    "get_parallel_op",
    "get_replicated_op",
]
