# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Triton-Ascend implementation of get_token_bin_counts_and_mask.
# Migrated from model_executor/layers/utils.get_token_bin_counts_and_mask.

import torch
from vllm.triton_utils import tl, triton

from vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils import (
    get_vectorcore_num,
)


@triton.jit(do_not_specialize=["batch_size", "seq_len"])
def token_bin_counts_and_mask_kernel(
    tokens_ptr,
    tokens_batch_stride,
    tokens_seq_stride,
    batch_size,
    seq_len,
    vocab_size,
    bin_counts_ptr,
    tp_rank,
    counts_batch_stride,
    counts_vocab_stride,
    total_blocks,
    SEQ_BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    vocab_start_idx = tp_rank * vocab_size
    num_seq_blocks = tl.cdiv(seq_len, SEQ_BLOCK)
    for linear_block in tl.range(pid, total_blocks, num_programs):
        batch_idx = linear_block // num_seq_blocks
        seq_block_id = linear_block - batch_idx * num_seq_blocks
        positions = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
        position_mask = positions < seq_len
        tokens = tl.load(
            tokens_ptr + batch_idx * tokens_batch_stride
            + positions * tokens_seq_stride,
            mask=position_mask,
            other=vocab_size + vocab_start_idx,
        )
        local_tokens = tokens - vocab_start_idx
        in_vocab = position_mask & (tokens >= vocab_start_idx) & (
            local_tokens < vocab_size
        )
        safe_tokens = tl.where(in_vocab, local_tokens, 0)
        tl.atomic_add(
            bin_counts_ptr + batch_idx * counts_batch_stride
            + safe_tokens * counts_vocab_stride,
            1,
            mask=in_vocab,
        )


def get_token_bin_counts_and_mask_triton(
    tokens: torch.Tensor, vocab_size: int, num_seqs: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int32 counts and a presence mask for full-vocabulary logits.

    FL's current sampler does not expose rc1 reduce-sample/sharded-vocabulary
    mode, therefore ``tp_rank`` remains zero (the established full-vocab
    contract) while the kernel retains the generic parameter.
    """
    n_rows, n_cols = tokens.shape
    if num_seqs is not None and num_seqs > 0:
        assert n_rows == num_seqs, (
            "tokens rows must match num_seqs: "
            f"tokens.shape[0]={n_rows}, num_seqs={num_seqs}"
        )
    n_rows = num_seqs if num_seqs is not None else n_rows
    bin_counts = torch.zeros(
        (n_rows, vocab_size), dtype=torch.int32, device=tokens.device
    )
    if n_rows == 0 or n_cols == 0:
        return bin_counts, bin_counts > 0
    if not tokens.is_contiguous():
        tokens = tokens.contiguous()
    seq_block = 256
    total_blocks = n_rows * triton.cdiv(n_cols, seq_block)
    token_bin_counts_and_mask_kernel[(min(get_vectorcore_num(), total_blocks),)](
        tokens,
        tokens.stride(0),
        tokens.stride(1),
        n_rows,
        n_cols,
        vocab_size,
        bin_counts,
        0,
        bin_counts.stride(0),
        bin_counts.stride(1),
        total_blocks,
        SEQ_BLOCK=seq_block,
        multibuffer=False,
    )
    return bin_counts, bin_counts > 0
