# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""store_kv_block -- AscendC to Triton

Source: csrc/attention/store_kv_block/
Interface (op_host/store_kv_block_def.cpp L27-57, **positional arguments**):
    Input : keyIn            int8/fp16/bf16  [numTokens, numHeads, headSize...] (2..7 dims)
            keyCacheIn       int8/fp16/bf16  [numCache, ...] (numCache = dim0*dim1)
            groupLen         int32  [numGroups]
            groupKeyIdx      int32  [numGroups]
            groupKeyCacheIdx int32  [numGroups]
    Attr  : blockSize
    Output: keyCacheIn (written back in place)

Call site (vllm_ascend/attention/indexer.py L261-268):
    torch.ops._C_ascend.store_kv_block(
        k_li,                    # keyIn
        indexer_k_cache,         # keyCacheIn
        group_len,               # groupLen
        group_key_idx,           # groupKeyIdx
        group_key_cache_idx,     # groupKeyCacheIdx
        block_size,              # blockSize
    )

Formula source (op_kernel/store_kv_block.h):
    L71-74   tokenSize = elements per row (numHeads * headSize...), tokenByteSize = tokenSize*byte
    L112-126 Process main loop:
        for i in corePerNum:
            idx = coreId + i*blockNum
            if groupLen[idx] <= 0 || groupKeyIdx[idx] < 0 || groupKeyCacheIdx[idx] < 0: continue
            blockLen = groupLen[idx] * tokenByteSize
            DataCopyPad(tokenLocal, keyInputGt[groupKeyIdx[idx]*tokenSize], blockLen)  # copy out
            DataCopyPad(keyCacheInputGt[groupKeyCacheIdx[idx]*tokenSize], tokenLocal, blockLen)  # in
    => Semantics: a **contiguous row-block copy** per group
        keyCache[dst : dst+len] = keyIn[src : src+len]
        where len = groupLen[g], src = groupKeyIdx[g], dst = groupKeyCacheIdx[g]
        len<=0 or src<0 or dst<0 -> the whole group is skipped (this is exactly the reshape
        optimization used by indexer.py, which replaces a per-token scatter with "contiguous
        runs per group": within a run both src and dst are contiguous, so a single block copy
        is enough)

Implementation notes:
    The block copy is flattened to **1-D** (src*TOKEN + [0, len*TOKEN) -> dst*TOKEN + the same
    range), with one program per group, to avoid a scalar being broadcast through [:, None] into
    a 2-D pointer base (a known issue on the Ascend backend).
    Pure copy operator: the output is **bit-identical** to the input (int8/fp16/bf16 alike).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div

__all__ = ["store_kv_block"]


@triton.jit
def _store_kv_block_kernel(
    KEYIN, KEYCACHE, GRLEN, GRKEY, GRCACHE,
    TOKEN,                       # elements per row
    BLK: tl.constexpr,
):
    g = tl.program_id(0)         # one program per group
    ln = tl.load(GRLEN + g)
    src = tl.load(GRKEY + g)
    dst = tl.load(GRCACHE + g)
    # Invalid group (a `continue` in AscendC): zero the length and the bases, then skip it
    # entirely through the mask.
    ok = (ln > 0) & (src >= 0) & (dst >= 0)
    ln = tl.where(ok, ln, 0)
    src = tl.where(ok, src, 0)
    dst = tl.where(ok, dst, 0)
    total = ln * TOKEN
    s_base = src * TOKEN         # scalar base (added to a 1-D arange only, no [:, None] broadcast)
    d_base = dst * TOKEN
    for c in range(0, tl.cdiv(total, BLK)):
        i = c * BLK + tl.arange(0, BLK)
        m = i < total
        v = tl.load(KEYIN + s_base + i, mask=m, other=0)
        tl.store(KEYCACHE + d_base + i, v, mask=m)


def store_kv_block(
    key_in: torch.Tensor,
    key_cache_in: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Equivalent to the AscendC StoreKVBlock (writes into key_cache_in in place and returns it).

    Args:
        key_in:   [numTokens, numHeads, headSize...] (contiguous elements, one row per token)
        key_cache_in: [numCache, ...] with the same row width as key_in; written in place
        group_len: [numGroups] int32, number of rows to copy per group
        group_key_idx: [numGroups] int32, source start row
        group_key_cache_idx: [numGroups] int32, destination start row
        block_size: attribute (used by tiling for the UB capacity check)

    Returns:
        key_cache_in (updated)
    """
    for nm, t in (("group_len", group_len), ("group_key_idx", group_key_idx),
                  ("group_key_cache_idx", group_key_cache_idx)):
        if t.dtype != torch.int32:
            raise ValueError(f"{nm} must be int32, got {t.dtype}")
    if not (group_len.numel() == group_key_idx.numel() == group_key_cache_idx.numel()):
        raise ValueError("group_len / group_key_idx / group_key_cache_idx must have equal length")
    if key_in.dtype not in (torch.int8, torch.float16, torch.bfloat16):
        raise ValueError(f"keyIn only supports int8/fp16/bf16, got {key_in.dtype}")
    if key_cache_in.dtype != key_in.dtype:
        raise ValueError(f"keyCacheIn dtype {key_cache_in.dtype} != keyIn {key_in.dtype}")
    # The row width (elements per token) is taken from **keyIn**: keyIn is split by dim0 into
    # rows of tokenSize elements (in the AscendC kernel tokenSize = product(shape[1:]) and the
    # cache is just a 1-D buffer laid out with that row width: the kernel locates it through
    # keyCacheInputGt[groupKeyCacheIdx*tokenSize] and ignores its own shape).
    if key_in.dim() < 2 or key_cache_in.dim() < 2:
        raise ValueError("keyIn / keyCacheIn must have at least 2 dims")
    token = key_in.numel() // int(key_in.shape[0])
    if token * int(key_in.shape[0]) != key_in.numel():
        raise ValueError("keyIn numel is not a multiple of dim0")
    if key_cache_in.numel() % token != 0:
        # `token` is the row width in elements (see above).
        raise ValueError(f"keyCacheIn numel {key_cache_in.numel()} is not a multiple of {token}")
    ng = group_len.numel()
    if ng == 0:
        return key_cache_in
    blk = 1024
    grid = (ng,)
    _store_kv_block_kernel[grid](
        key_in.reshape(-1), key_cache_in.reshape(-1),
        group_len.contiguous(), group_key_idx.contiguous(), group_key_cache_idx.contiguous(),
        token, BLK=blk,
    )
    return key_cache_in


def _token_size(t: torch.Tensor) -> int:
    """Elements per row (one token) = product of shape[1:]."""
    n = 1
    for s in t.shape[1:]:
        n *= int(s)
    return n
