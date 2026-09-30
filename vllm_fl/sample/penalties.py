# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# apply_all_penalties for AscendSampler - uses Triton-Ascend kernels.

import torch
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad

from vllm_fl.dispatch.backends.vendor.ascend.ops.triton.penalty import (
    apply_penalties_triton,
)


def _convert_to_tensors(
    output_token_ids: list[list[int]], vocab_size: int, device: torch.device
) -> torch.Tensor:
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
    reduce_sample: bool = False,
) -> torch.Tensor:
    """Apply penalties to logits via Triton-Ascend."""
    _, vocab_size = logits.shape
    tp_rank = 0
    padding_token_id = vocab_size
    if reduce_sample:
        from vllm.distributed.parallel_state import get_tp_group

        tp_rank = get_tp_group().rank_in_group
        # A local ``vocab_size`` is a valid global token on higher shards.
        # Use this shard's exclusive global end as the padding sentinel.
        padding_token_id = (tp_rank + 1) * vocab_size
    output_tokens_t = _convert_to_tensors(output_token_ids, padding_token_id, logits.device)
    output_tokens_t.masked_fill_(output_tokens_t == -1, padding_token_id)
    return apply_penalties_triton(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        tp_rank,
    )
