# Copyright (c) 2026 BAAI. All rights reserved.

"""Graph-safe custom ops owned by the FL Ascend MoE lifecycle."""

from __future__ import annotations

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_fl.ascend_forward_context import _EXTRA_CTX, MoECommType

_REGISTERED = False


def _maybe_all_reduce_tensor_model_parallel_impl(
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Match vLLM-Ascend rc1's MoE final-reduction contract."""
    moe_comm_type = _EXTRA_CTX.moe_comm_type
    if (
        moe_comm_type
        in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
        or _EXTRA_CTX.flash_comm_v1_enabled
    ):
        return final_hidden_states
    return tensor_model_parallel_all_reduce(final_hidden_states)


def _maybe_all_reduce_tensor_model_parallel_fake(
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    return final_hidden_states


def ensure_ascend_moe_custom_ops_registered() -> None:
    """Register the rc1 reduction op once during Ascend patching."""
    global _REGISTERED
    if _REGISTERED:
        return
    if hasattr(torch.ops.vllm, "maybe_all_reduce_tensor_model_parallel"):
        raise RuntimeError(
            "torch.ops.vllm.maybe_all_reduce_tensor_model_parallel already "
            "exists before FL Ascend registration"
        )
    direct_register_custom_op(
        op_name="maybe_all_reduce_tensor_model_parallel",
        op_func=_maybe_all_reduce_tensor_model_parallel_impl,
        fake_impl=_maybe_all_reduce_tensor_model_parallel_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )
    _REGISTERED = True


__all__ = ["ensure_ascend_moe_custom_ops_registered"]
