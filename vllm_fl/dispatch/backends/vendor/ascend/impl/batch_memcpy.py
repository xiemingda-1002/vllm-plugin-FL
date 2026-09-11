# Adapted from vLLM-Ascend v0.24.0rc1.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend-safe Triton kernel for batched pointer-based copies."""

from vllm.triton_utils import tl, triton


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    # Ascend's Triton compiler requires the pointer reinterpretation outside
    # the loop. Keeping the typed pointers loop-invariant also avoids repeated
    # pointer conversions in the generated kernel.
    src_ptr = src_ptr.to(tl.pointer_type(tl.uint8))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.uint8))

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size
        curr_src_ptr = src_ptr + i + offsets
        curr_dst_ptr = dst_ptr + i + offsets

        # This is streaming state data, so bypass L1 on both sides.
        data = tl.load(curr_src_ptr, mask=mask, cache_modifier=".cg")
        tl.store(curr_dst_ptr, data, mask=mask, cache_modifier=".cg")


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    """Copy byte ranges described by equal-length pointer/size tensors."""
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    # vLLM-Ascend uses the larger tile to avoid the faulty upstream launch
    # pattern and improve this streaming copy on Ascend.
    block_size = 8192
    batch_memcpy_kernel[(batch,)](
        src_ptrs,
        dst_ptrs,
        sizes,
        BLOCK_SIZE=block_size,
    )
