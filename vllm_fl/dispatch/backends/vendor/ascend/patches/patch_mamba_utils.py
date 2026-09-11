# Copyright (c) 2026 BAAI. All rights reserved.

"""Install the Ascend-only vLLM Mamba batch-copy corrections."""

import itertools
import logging


logger = logging.getLogger(__name__)
_PATCH_MARKER = "_vllm_fl_ascend_mamba_batch_copy_patched"


def patch_mamba_batch_copy() -> bool:
    """Patch vLLM's Mamba copy kernel and unsupported uint64 buffers.

    Imports stay local so non-Ascend plugin initialization never imports the
    Triton implementation or mutates vLLM's Mamba worker module. The marker is
    stored on that upstream module, which keeps this idempotent even if this
    FL module is reloaded.

    Returns ``True`` only when this call installs the patch.
    """
    from vllm.platforms import current_platform

    if not (
        current_platform.vendor_name == "ascend"
        and current_platform.device_type == "npu"
    ):
        return False

    import torch
    from vllm.utils.math_utils import cdiv
    from vllm.v1.worker import mamba_utils

    if getattr(mamba_utils, _PATCH_MARKER, False):
        return False

    from ..impl.batch_memcpy import batch_memcpy, batch_memcpy_kernel

    original_create = mamba_utils.MambaCopyBuffers.create

    @classmethod
    def create_with_int64_pointers(
        cls,
        max_num_reqs,
        kv_cache_config,
        copy_funcs,
        make_buffer,
    ):
        del cls

        def make_ascend_buffer(n, dtype):
            if dtype == torch.uint64:
                dtype = torch.int64
            return make_buffer(n, dtype=dtype)

        return original_create(
            max_num_reqs,
            kv_cache_config,
            copy_funcs,
            make_ascend_buffer,
        )

    mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
    mamba_utils.batch_memcpy = batch_memcpy
    mamba_utils.MambaCopyBuffers.create = create_with_int64_pointers

    def preprocess_mamba_deferred(
        scheduler_output,
        kv_cache_config,
        cache_config,
        mamba_state_idx,
        input_batch,
        requests,
        forward_context,
        mamba_state_copy_funcs,
        copy_bufs,
    ):
        """Collect Mamba state-copy metadata without executing the copy.

        Current vLLM-Ascend v1 defers the actual device copy until the live
        forward context is installed and KV-transfer loading has completed.
        ``cache_config`` remains in the signature for exact vLLM compatibility.
        """
        del cache_config
        mamba_group_ids = copy_bufs.mamba_group_ids
        mamba_spec = copy_bufs.mamba_spec
        num_speculative_blocks = mamba_spec.num_speculative_blocks
        block_size = mamba_spec.block_size
        finished_req_ids = scheduler_output.finished_req_ids
        preempted_req_ids = scheduler_output.preempted_req_ids or set()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        for req_id in itertools.chain(
            finished_req_ids,
            preempted_req_ids,
            resumed_req_ids,
        ):
            mamba_state_idx.pop(req_id, None)

        copy_bufs.offset = 0
        for i, req_id in enumerate(input_batch.req_ids):
            req_state = requests[req_id]
            prev_state_idx = mamba_state_idx.get(req_id)
            if prev_state_idx is None:
                prev_state_idx = (
                    req_state.num_computed_tokens - 1
                ) // block_size

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id
            ]
            num_blocks = (
                cdiv(
                    req_state.num_computed_tokens + num_scheduled_tokens,
                    block_size,
                )
                + num_speculative_blocks
            )
            curr_state_idx = num_blocks - 1 - num_speculative_blocks
            mamba_state_idx[req_id] = curr_state_idx
            if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
                mamba_utils.collect_mamba_copy_meta(
                    copy_bufs,
                    kv_cache_config,
                    mamba_state_copy_funcs,
                    mamba_group_ids,
                    prev_state_idx,
                    curr_state_idx,
                    input_batch.num_accepted_tokens_cpu[i] - 1,
                    req_state,
                    forward_context,
                )
                input_batch.num_accepted_tokens_cpu[i] = 1

    mamba_utils.preprocess_mamba = preprocess_mamba_deferred
    setattr(mamba_utils, _PATCH_MARKER, True)
    logger.info(
        "Patched Mamba batch memcpy, pointer buffers, and deferred copy for "
        "Ascend"
    )
    return True
