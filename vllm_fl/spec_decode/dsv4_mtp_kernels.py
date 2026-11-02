# SPDX-License-Identifier: Apache-2.0
"""FL-owned input preparation kernels for greedy, sequential DSv4 MTP."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _next_tokens_kernel(sampled, discard, backup, next_ids, counts,
                        VOCAB: tl.constexpr, WIDTH: tl.constexpr,
                        ROW_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token = tl.load(sampled + row * ROW_STRIDE + col, col < WIDTH, other=-1)
    valid = (col < WIDTH) & (token >= 0) & (token < VOCAB)
    last = tl.max(tl.where(valid, col, -1), 0)
    count = tl.sum(valid.to(tl.int32), 0)
    chosen = tl.load(sampled + row * ROW_STRIDE + last, last >= 0, other=0)
    old = tl.load(backup + row)
    dropped = tl.load(discard + row)
    tl.store(next_ids + row, tl.where((last >= 0) & ~dropped, chosen, old))
    tl.store(counts + row, tl.where(dropped, 0, count))


@triton.jit
def _padded_indices_kernel(cu_draft, valid_counts, query_start, indices,
                           rejected, N, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    end = tl.load(cu_draft + row, row < N, other=0)
    start = tl.load(cu_draft + row - 1, (row > 0) & (row < N), other=0)
    count = end - start
    valid = tl.load(valid_counts + row, row < N, other=0)
    reject = tl.where(count > 0, count + 1 - valid, 0)
    qend = tl.load(query_start + row + 1, row < N, other=0)
    tl.store(indices + row, qend - 1 - reject, row < N)
    tl.store(rejected + row, reject, row < N)


@triton.jit
def _advance_kernel(pos, table, lengths, active, out_pos, out_slots,
                    out_lengths, out_active, N,
                    COLS, TABLE_STRIDE,
                    BLOCK_SIZE: tl.constexpr, MAX_LEN: tl.constexpr,
                    HAS_ACTIVE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    next_pos = tl.load(pos + row, row < N, other=0).to(tl.int64) + 1
    block_index = next_pos // BLOCK_SIZE
    valid = (next_pos >= 0) & (next_pos < MAX_LEN) & (block_index < COLS)
    if HAS_ACTIVE:
        valid = valid & tl.load(active + row, row < N, other=0)
    safe = tl.minimum(tl.maximum(next_pos, 0), MAX_LEN - 1)
    column = tl.minimum(safe // BLOCK_SIZE, COLS - 1)
    block = tl.load(table + row * TABLE_STRIDE + column, row < N, other=-1)
    valid = valid & (block >= 0)
    seq = tl.load(lengths + row, row < N, other=0)
    slot = block.to(tl.int64) * BLOCK_SIZE + safe % BLOCK_SIZE
    tl.store(out_pos + row, tl.where(valid, next_pos, 0), row < N)
    tl.store(out_slots + row, tl.where(valid, slot, -1), row < N)
    tl.store(out_lengths + row, tl.where(valid, tl.minimum(seq + 1, MAX_LEN), 1), row < N)
    tl.store(out_active + row, valid, row < N)


def next_tokens(sampled, discard, backup, vocab):
    n, width = sampled.shape
    ids = torch.empty(n, dtype=torch.int32, device=sampled.device)
    counts = torch.empty_like(ids)
    if n:
        _next_tokens_kernel[(n,)](
            sampled, discard, backup, ids, counts, vocab, width,
            sampled.stride(0), triton.next_power_of_2(max(width, 1)))
    return ids, counts


def padded_indices(cu_draft, valid_counts, query_start):
    n = valid_counts.numel()
    indices = torch.empty(n, dtype=torch.int32, device=valid_counts.device)
    rejected = torch.empty_like(indices)
    if n:
        _padded_indices_kernel[(triton.cdiv(n, 128),)](
            cu_draft, valid_counts, query_start, indices, rejected, n, 128)
    return indices, rejected


def advance_positions(pos, table, lengths, block_size, max_len, active, out):
    n = pos.numel()
    output = tuple(t[:n] for t in out)
    if table.shape[1] == 0:
        output[0].zero_()
        output[1].fill_(-1)
        output[2].fill_(1)
        output[3].zero_()
    elif n:
        _advance_kernel[(triton.cdiv(n, 128),)](
            pos, table, lengths, active if active is not None else pos,
            *output, n, table.shape[1], table.stride(0), block_size, max_len,
            active is not None, 128)
    return output
