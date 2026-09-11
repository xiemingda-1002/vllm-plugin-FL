# Copyright (c) 2026 BAAI. All rights reserved.

"""Graph-safe FlashComm1 dispatcher boundaries for the Ascend vendor."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_fl.ascend_forward_context import _EXTRA_CTX

_REGISTERED = False


def _maybe_chunk_residual_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Return the residual shard matching a FlashComm1 hidden-state shard."""
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) == residual.size(0):
        return residual
    if _EXTRA_CTX.pad_size > 0:
        residual = F.pad(residual, (0, 0, 0, _EXTRA_CTX.pad_size))
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    return torch.chunk(residual, tp_size, dim=0)[tp_rank]


def _maybe_chunk_residual_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    del residual
    return torch.empty_like(x)


def _maybe_all_gather_and_maybe_unpad_impl(
    x: torch.Tensor,
    label: bool,
    is_ep_comm: bool = False,
) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    if not (_EXTRA_CTX.flash_comm_v1_enabled and label):
        return x

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        x = tensor_model_parallel_all_gather(x, 0)
        if _EXTRA_CTX.pad_size > 0:
            x = x[: -_EXTRA_CTX.pad_size]
        return x

    x = get_ep_group().all_gather(x, 0)
    token_counts = dp_metadata.num_tokens_across_dp_cpu
    total_tokens = int(token_counts.sum().item())
    result = torch.empty(
        (total_tokens, *x.shape[1:]), dtype=x.dtype, device=x.device
    )
    dp_size = get_dp_group().world_size
    x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
    offset = 0
    for idx in range(dp_size):
        count = int(token_counts[idx].item())
        result[offset : offset + count].copy_(x[idx, :count])
        offset += count
    return result


def _maybe_all_gather_and_maybe_unpad_fake(
    x: torch.Tensor,
    label: bool,
    is_ep_comm: bool = False,
) -> torch.Tensor:
    del is_ep_comm
    if _EXTRA_CTX.flash_comm_v1_enabled and label:
        return torch.empty(
            (x.shape[0] * get_tensor_model_parallel_world_size(), *x.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )
    return x


def _maybe_pad_and_reduce_impl(
    x: torch.Tensor,
    is_ep_comm: bool = False,
) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    if not _EXTRA_CTX.flash_comm_v1_enabled:
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        if _EXTRA_CTX.pad_size > 0:
            x = F.pad(x, (0, 0, 0, _EXTRA_CTX.pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)

    dp_size = get_dp_group().world_size
    token_counts = dp_metadata.num_tokens_across_dp_cpu
    padded_x = torch.empty(
        (dp_size, _EXTRA_CTX.padded_length, *x.shape[1:]),
        dtype=x.dtype,
        device=x.device,
    )
    offset = 0
    for idx in range(dp_size):
        count = int(token_counts[idx].item())
        padded_x[idx, :count].copy_(x[offset : offset + count])
        offset += count
    return get_ep_group().reduce_scatter(
        padded_x.view(-1, *x.shape[1:]), 0
    )


def _maybe_pad_and_reduce_fake(
    x: torch.Tensor,
    is_ep_comm: bool = False,
) -> torch.Tensor:
    del is_ep_comm
    if _EXTRA_CTX.flash_comm_v1_enabled:
        return torch.empty(
            (x.shape[0] // get_tensor_model_parallel_world_size(), *x.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )
    return x


def _matmul_and_reduce_impl(
    input_parallel: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    if layer.custom_op is None:
        raise RuntimeError(f"FlashComm1 linear op is missing for {layer_name}")
    bias = None if layer.tp_rank > 0 or layer.skip_bias_add else layer.bias
    return layer.custom_op.matmul_and_reduce(input_parallel, bias)


def _matmul_and_reduce_fake(
    input_parallel: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    num_tokens = input_parallel.size(0)
    if _EXTRA_CTX.flash_comm_v1_enabled:
        num_tokens //= layer.tp_size
    return torch.empty(
        (num_tokens, layer.output_size_per_partition),
        dtype=input_parallel.dtype,
        device=input_parallel.device,
    )


def ensure_ascend_flashcomm_custom_ops_registered() -> None:
    """Register the rc1 FlashComm1 Python operators exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return

    registrations = (
        (
            "maybe_chunk_residual",
            _maybe_chunk_residual_impl,
            _maybe_chunk_residual_fake,
        ),
        (
            "maybe_all_gather_and_maybe_unpad",
            _maybe_all_gather_and_maybe_unpad_impl,
            _maybe_all_gather_and_maybe_unpad_fake,
        ),
        (
            "maybe_pad_and_reduce",
            _maybe_pad_and_reduce_impl,
            _maybe_pad_and_reduce_fake,
        ),
        ("matmul_and_reduce", _matmul_and_reduce_impl, _matmul_and_reduce_fake),
    )
    collisions = [
        name for name, _, _ in registrations if hasattr(torch.ops.vllm, name)
    ]
    if collisions:
        joined = ", ".join(f"torch.ops.vllm.{name}" for name in collisions)
        raise RuntimeError(
            f"{joined} already exist before FL Ascend registration"
        )
    for name, implementation, fake in registrations:
        direct_register_custom_op(
            op_name=name,
            op_func=implementation,
            fake_impl=fake,
            mutates_args=[],
            dispatch_key="PrivateUse1",
        )
    _REGISTERED = True


__all__ = ["ensure_ascend_flashcomm_custom_ops_registered"]
