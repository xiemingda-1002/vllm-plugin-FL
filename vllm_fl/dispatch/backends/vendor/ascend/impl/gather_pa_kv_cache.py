# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation replacing the torch_npu built-in of the same name.
# SPDX-License-Identifier: Apache-2.0
"""gather_pa_kv_cache -- torch_npu built-in replaced by a Triton implementation.

Source
------
There is **no AscendC source for this operator under csrc/ascend**: a grep for
``gather_pa`` / ``GatherPa`` / ``gather_kv`` over the whole csrc tree returns
nothing. ``torch_npu.npu_gather_pa_kv_cache`` is a built-in of torch_npu itself
(the paged-cache sibling of ``npu_scatter_pa_kv_cache``), so the contract below
is derived from the one call site, from the two upstream vllm-ascend call sites
and from the upstream end-to-end test that uses this operator as its own golden
generator. Every claim carries the file and line it comes from.

Call site (1 in this tree)
--------------------------
impl/../device/device_op.py L1476-1485 (A5DeviceAdaptor.kv_cache_load)::

    torch_npu.npu_gather_pa_kv_cache(
        cache_kv_c,                    # key_cache   [num_blocks, block_size, *row]
        cache_k_pe,                    # value_cache [num_blocks, block_size, *row]
        block_table,                   # [num_seqs, max_blocks] int32/int64
        context_seq_len_npu.contiguous(),   # [num_seqs] int32
        seq_offset=seq_offset,         # [num_seqs] int32  (5th positional in
                                       #  BaseDeviceAdaptor.kv_cache_load L305)
        key=key,                       # out [sum(seq_len), *row]
        value=value,                   # out [sum(seq_len), *row]
    )

Its callers (upstream vllm-ascend, the shape oracle for this port):
  * _repos/vllm-ascend/vllm_ascend/attention/mla_v1.py L1332-1355: chunked MLA
    prefill. ``kv_c_normed = torch.empty(toks, num_heads, latent_kv_dim)`` with
    ``toks = prefill_metadata.chunked_context.seq_tot[i]`` (L1333) and
    ``seq_tot = chunk_seq_lens.sum(dim=1)`` -- i.e. the buffer holds **exactly
    sum(seq_len) rows, packed**, and ``starts[i]`` (L1352) is the chunk's start
    position *inside the sequence* (``chunk_starts = arange(num_chunks) *
    max_context_chunk``, mla_v1.py L620-625), not an output offset.
  * _repos/vllm-ascend/vllm_ascend/attention/context_parallel/attention_cp.py
    L486-505: same shape pattern, comment "slot offsets of current chunk".
  * _repos/vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/
    test_transpose_kv_cache_by_block.py L33-51: one sequence, ``seq_len =
    num_blocks * block_size``, ``seq_offset = 0``, output
    ``[block_len, num_kv_head, head_dim]`` -- the flattened block order
    ``out[j] = cache[block_table[0, j // block_size], j % block_size]``.
  * _repos/vllm-ascend/tests/ut/attention/test_mla_nope_chunked_gather.py L93:
    ``key.shape == (seq_len, 1, latent_kv_dim)`` -- again packed by seq_len.

Semantics
---------
::

    cursor = 0
    for s in range(num_seqs):
        L   = max(seq_len[s], 0)
        off = seq_offset[s] if seq_offset is not None else 0     # source position
        for blk in range(ceil(L / block_size)):
            n = min(block_size, L - blk * block_size)
            for i in range(n):
                pos = off + blk * block_size + i
                bid = block_table[s, pos // block_size]
                key[cursor + blk * block_size + i]   = key_cache[bid, pos % block_size]
                value[cursor + blk * block_size + i] = value_cache[bid, pos % block_size]
        cursor += L

i.e. per sequence, the context tokens [off, off+L) are copied out of the paged
cache into a **packed** contiguous run (``cursor`` is the exclusive prefix sum of
``seq_len``); ``seq_offset`` only moves the *source* window and never affects the
output layout. Rows are full rows: the row width is ``prod(cache.shape[2:])`` and
may differ between key and value (the MLA call site hands a latent cache to
``key`` and a rope cache to ``value``, mla_v1.py L1295-1345).

Out-of-range protection: ``block_table`` entries are clamped into
[0, num_blocks) and the block index into [0, block_table.shape[1]); the Ascend
backend raises on wild address computation even for masked lanes
(impl/scatter_nd_update_v2.py L172-176), so every address is clamped instead of
relying on the mask alone.

No host sync: ``cursor`` comes from a 1-program device scan of ``seq_len``, so
the operator is capture safe (it never reads a device value on the host).

Dtypes: the copy is dtype agnostic (int8/uint8/fp16/bf16/fp32 and any other plain
element type); key and value may use different dtypes, matching the MLA quantized
cache.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b



__all__ = ["gather_pa_kv_cache"]

#: Elements per 1-D copy tile. UB cost is 4 bytes per lane, independent of the
#: sequence length (C2 in _agent_logs/TRITON_OPERATOR_PITFALLS.md).
_BK = 1024
#: Cache blocks one program may walk (program_id(1) stride). The host doubles it
#: until num_seqs * ceil(blocks / BLK_B) fits under MAX_PROGRAMS.
_BLK_B_MAX = 64
#: AI Core block_dim cap measured in _agent_logs/TRITON_OPERATOR_PITFALLS.md C1.
MAX_PROGRAMS = 32_768


@triton.jit
def _seq_cursor_kernel(SEQLEN, CURSOR, NSEQ, BLK: tl.constexpr):
    """CURSOR = exclusive prefix sum of the clamped seq_len (1 program, no sync).

    One program walks the sequences in chunks; `run` carries the total of the
    previous chunks and the in-chunk `cumsum` gives each lane its offset, so the
    result is the packed output cursor the gather kernel needs.
    """
    run = 0
    for c0 in range(0, tl.cdiv(NSEQ, BLK)):
        o = c0 * BLK + tl.arange(0, BLK)
        m = o < NSEQ
        v = tl.where(m, tl.load(SEQLEN + o, mask=m, other=0), 0).to(tl.int32)
        v = tl.maximum(v, 0)                    # padded rows carry -1 or 0
        excl = tl.cumsum(v, axis=0) - v
        tl.store(CURSOR + o, run + excl, mask=m)
        run = run + tl.sum(v, axis=0)


@triton.jit
def _gather_pa_kv_cache_kernel(
    KC, VC, BT, SEQLEN, SEQOFF, CURSOR, KEY, VAL,
    BS, NB, NC, WK, WV,
    BLK_B: tl.constexpr,
    BK: tl.constexpr,
    HAS_OFF: tl.constexpr,
):
    """Per (sequence, group of BLK_B cache blocks): copy the block rows out.

    One 1-D tile of BK elements at a time; the row index inside the tile is
    derived with `i // WK`, so a tile may straddle rows and a chunk may straddle
    cache blocks (which happens whenever ``seq_offset`` is not block aligned).
    Every address is clamped (see the module docstring) and every access is
    masked.
    """
    s = tl.program_id(0)
    b0 = tl.program_id(1) * BLK_B
    L = tl.maximum(tl.load(SEQLEN + s).to(tl.int32), 0)
    if HAS_OFF:
        off = tl.maximum(tl.load(SEQOFF + s).to(tl.int32), 0)
    else:
        off = 0
    cur = tl.load(CURSOR + s).to(tl.int32)
    for bb in range(BLK_B):
        base = (b0 + bb) * BS
        if base < L:
            n = tl.minimum(L - base, BS)
            pos0 = off + base
            ob = (cur + base).to(tl.int64)
            nk = n * WK
            for c0 in range(0, tl.cdiv(nk, BK)):
                i = c0 * BK + tl.arange(0, BK)
                m = i < nk
                row = i // WK
                col = i - row * WK
                pos = pos0 + row
                bi = tl.minimum(tl.maximum(pos // BS, 0), NB - 1)
                bid = tl.load(BT + s * NB + bi, mask=m, other=0).to(tl.int64)
                bid = tl.minimum(tl.maximum(bid, 0), NC - 1)
                src = (bid * BS + (pos - bi * BS)) * WK + col
                v = tl.load(KC + src, mask=m, other=0)
                tl.store(KEY + ob * WK + i, v, mask=m)
            nv = n * WV
            for c0 in range(0, tl.cdiv(nv, BK)):
                i = c0 * BK + tl.arange(0, BK)
                m = i < nv
                row = i // WV
                col = i - row * WV
                pos = pos0 + row
                bi = tl.minimum(tl.maximum(pos // BS, 0), NB - 1)
                bid = tl.load(BT + s * NB + bi, mask=m, other=0).to(tl.int64)
                bid = tl.minimum(tl.maximum(bid, 0), NC - 1)
                src = (bid * BS + (pos - bi * BS)) * WV + col
                v = tl.load(VC + src, mask=m, other=0)
                tl.store(VAL + ob * WV + i, v, mask=m)


def gather_pa_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: torch.Tensor,
    seq_offset: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent of ``torch_npu.npu_gather_pa_kv_cache`` (paged cache -> packed KV).

    Args:
        key_cache: [num_blocks, block_size, *row_k], contiguous.
        value_cache: [num_blocks, block_size, *row_v]; the block count and
            block_size must match ``key_cache`` (same block table), the row shape
            and dtype may differ.
        block_table: [num_seqs, max_blocks] int32/int64, contiguous. Entries are
            cache block ids; every sequence uses its own row.
        seq_len: [num_seqs] int32/int64, number of context tokens to gather per
            sequence.
        seq_offset: optional [num_seqs] int32/int64, first context position to
            gather per sequence (default 0). This is a *source* position: the
            output is always packed, ``cursor`` = exclusive prefix sum of
            ``seq_len`` (see the module docstring).
        key: output buffer [sum(seq_len), *row_k], contiguous, same dtype as
            ``key_cache``. Required: the row count is only known on the device,
            so it cannot be allocated here without a sync.
        value: output buffer [sum(seq_len), *row_v], same rules as ``key``.

    Returns:
        (key, value) -- the buffers that were passed in, updated in place. The
        torch_npu operator returns them too, so both call styles work.
    """
    if key is None or value is None:
        raise NotImplementedError(
            "key/value must be preallocated: the output row count is sum(seq_len), "
            "which is only known on the device; allocating it here would need a "
            "host sync inside the forward (and breaks ACL graph capture). The call "
            "site (device_op.py L1477) always passes key= and value=."
        )
    if key_cache.dim() < 2 or value_cache.dim() < 2:
        raise ValueError(
            f"key_cache/value_cache must be [num_blocks, block_size, *row], got "
            f"{tuple(key_cache.shape)} / {tuple(value_cache.shape)}"
        )
    if tuple(key_cache.shape[:2]) != tuple(value_cache.shape[:2]):
        raise ValueError(
            "key_cache and value_cache must share (num_blocks, block_size), got "
            f"{tuple(key_cache.shape[:2])} vs {tuple(value_cache.shape[:2])}"
        )
    if block_table.dim() != 2:
        raise ValueError(f"block_table must be 2-D [num_seqs, max_blocks], got {tuple(block_table.shape)}")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"block_table must be int32/int64, got {block_table.dtype}")
    if seq_len.dim() != 1 or int(seq_len.numel()) != int(block_table.shape[0]):
        raise ValueError(
            f"seq_len must be [num_seqs={int(block_table.shape[0])}], got "
            f"{tuple(seq_len.shape)}"
        )
    if seq_len.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"seq_len must be int32/int64, got {seq_len.dtype}")
    if seq_offset is not None:
        if seq_offset.dim() != 1 or int(seq_offset.numel()) != int(seq_len.numel()):
            raise ValueError(
                f"seq_offset must be [num_seqs={int(seq_len.numel())}], got "
                f"{tuple(seq_offset.shape)}"
            )
        if seq_offset.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"seq_offset must be int32/int64, got {seq_offset.dtype}")
    for nm, t in (("key_cache", key_cache), ("value_cache", value_cache),
                  ("block_table", block_table), ("key", key), ("value", value)):
        if not t.is_contiguous():
            raise ValueError(
                f"{nm} must be contiguous: the kernel addresses it linearly "
                f"(shape={tuple(t.shape)}, stride={tuple(t.stride())})"
            )
    if tuple(key.shape[1:]) != tuple(key_cache.shape[2:]):
        raise ValueError(
            f"key shape {tuple(key.shape)} does not match key_cache rows "
            f"{tuple(key_cache.shape[2:])}"
        )
    if tuple(value.shape[1:]) != tuple(value_cache.shape[2:]):
        raise ValueError(
            f"value shape {tuple(value.shape)} does not match value_cache rows "
            f"{tuple(value_cache.shape[2:])}"
        )
    if key.dtype != key_cache.dtype:
        raise ValueError(f"key dtype {key.dtype} != key_cache dtype {key_cache.dtype}")
    if value.dtype != value_cache.dtype:
        raise ValueError(f"value dtype {value.dtype} != value_cache dtype {value_cache.dtype}")

    nseq = int(block_table.shape[0])
    nb = int(block_table.shape[1])
    if nseq == 0 or nb == 0:
        return key, value
    # Tiny device tensors stay out of the hot path; .contiguous() on an already
    # contiguous [num_seqs] tensor is free.
    sl = seq_len.contiguous()
    so = seq_offset.contiguous() if seq_offset is not None else None
    cursor = torch.zeros(nseq, dtype=torch.int32, device=key.device)
    _seq_cursor_kernel[(1,)](sl, cursor, nseq, BLK=64)

    blk_b = 1
    while nseq * _ceil_div(nb, blk_b) > MAX_PROGRAMS and blk_b < _BLK_B_MAX:
        blk_b *= 2
    if nseq * _ceil_div(nb, blk_b) > MAX_PROGRAMS:
        # A larger grid is measured to crash with `aivec` (C1 in
        # _agent_logs/TRITON_OPERATOR_PITFALLS.md), so refuse instead of launching.
        raise NotImplementedError(
            f"grid {nseq} x {_ceil_div(nb, blk_b)} exceeds the {MAX_PROGRAMS}-program cap "
            f"(num_seqs={nseq}, block_table width={nb}); split the sequences across calls"
        )
    wk = 1
    for d in key_cache.shape[2:]:
        wk *= int(d)
    wv = 1
    for d in value_cache.shape[2:]:
        wv *= int(d)
    _gather_pa_kv_cache_kernel[(nseq, _ceil_div(nb, blk_b))](
        key_cache, value_cache, block_table, sl,
        so if so is not None else sl,       # placeholder, not read when HAS_OFF=False
        cursor, key, value,
        int(key_cache.shape[1]), nb, int(key_cache.shape[0]), wk, wv,
        BLK_B=blk_b, BK=_BK, HAS_OFF=seq_offset is not None,
    )
    return key, value
