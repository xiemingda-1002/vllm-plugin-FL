# SPDX-License-Identifier: Apache-2.0
"""Fused exact greedy verification for the FL Ascend DSv4 MTP adapter.

Uses existing device cumulative draft counts, without a new counts upload.
The vocabulary argmax and logits processors remain owned by the sampler.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _greedy_verify_kernel(
    draft_ptr,
    cu_draft_ptr,
    target_ptr,
    bonus_ptr,
    output_ptr,
    MAX_SPEC_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per request; all its output slots, including padding, are
    # written here. Different requests may have different draft counts.
    row = tl.program_id(0)
    start = tl.load(cu_draft_ptr + row - 1, mask=row > 0, other=0)
    end = tl.load(cu_draft_ptr + row)
    count = end - start
    col = tl.arange(0, BLOCK)
    has_draft = col < count
    draft = tl.load(draft_ptr + start + col, mask=has_draft, other=0)
    target = tl.load(target_ptr + start + col, mask=has_draft, other=0)
    mismatch = has_draft & (draft != target)
    first_rejection = tl.minimum(
        tl.min(tl.where(mismatch, col, MAX_SPEC_LEN), axis=0), count
    )
    bonus = tl.load(bonus_ptr + row)
    # Accept the matching prefix, then emit exactly one replacement or
    # bonus token. A zero-draft row emits its bonus in column zero.
    replacement = tl.where(has_draft, target, bonus)
    tokens = tl.where(
        col < first_rejection, draft,
        tl.where(col == first_rejection, replacement, -1),
    )
    tl.store(output_ptr + row * (MAX_SPEC_LEN + 1) + col,
             tokens, mask=col < MAX_SPEC_LEN + 1)


def fused_greedy_rejection_tokens(metadata, target_argmax, bonus_ids):
    """Return int32 [requests, max_spec_len + 1], with -1 after the result."""
    batch = len(metadata.num_draft_tokens)
    width = metadata.max_spec_len + 1
    output = torch.empty((batch, width), dtype=torch.int32, device=bonus_ids.device)
    if batch:
        _greedy_verify_kernel[(batch,)](
            metadata.draft_token_ids, metadata.cu_num_draft_tokens,
            target_argmax, bonus_ids, output,
            MAX_SPEC_LEN=metadata.max_spec_len,
            BLOCK=triton.next_power_of_2(width),
        )
    return output
