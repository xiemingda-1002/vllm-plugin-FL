# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in torch-npu slot mapping using the runner's uploaded request indices."""

import os

import torch

from vllm_fl.models.ocr_runtime import _ocr_can_audit, _write_audit, ocr_runtime_enabled

_SLOT_AUDITED = False


def _supported_device(device):
    return device.type in ("npu", "privateuseone")


def compute_ocr_slot_mapping(block_table_manager, num_reqs, req_indices, positions):
    """Return False without writes when any cache group is unsupported.

    Request indices and positions must already satisfy the runner's indexing
    contract. No device values are read back or validated on the host.
    """
    if (
        os.environ.get("VLLM_FL_OCR_TORCH_SLOT_MAPPING", "1") != "1"
        or not ocr_runtime_enabled()
    ):
        return False
    if not isinstance(req_indices, torch.Tensor) or not isinstance(
        positions, torch.Tensor
    ):
        return False
    if (
        req_indices.ndim != 1
        or positions.ndim != 1
        or req_indices.shape != positions.shape
        or req_indices.dtype not in (torch.int32, torch.int64)
        or positions.dtype != torch.int64
        or req_indices.device != positions.device
        or not _supported_device(positions.device)
        or not isinstance(num_reqs, int)
        or num_reqs < 0
    ):
        return False
    tables = getattr(block_table_manager, "block_tables", None)
    if not isinstance(tables, (list, tuple)) or not tables:
        return False
    num_tokens = positions.numel()
    contracts = []
    for group in tables:
        mode = getattr(group, "slot_mapping_mode", None)
        # Older core BlockTable versions have only ordinary token slot mapping.
        if mode is not None and getattr(mode, "name", None) != "TOKEN_TO_KV_SLOT":
            return False
        block_size = getattr(group, "block_size", None)
        capacity = getattr(group, "max_num_batched_tokens", None)
        table = getattr(getattr(group, "block_table", None), "gpu", None)
        slots = getattr(getattr(group, "slot_mapping", None), "gpu", None)
        if (
            getattr(group, "pcp_world_size", None) != 1
            or getattr(group, "dcp_world_size", None) != 1
            or getattr(group, "blocks_per_kv_block", None) != 1
            or getattr(group, "use_hybrid_blocks", None) is not False
            or not isinstance(block_size, int)
            or block_size <= 0
            or getattr(group, "kv_cache_block_size", block_size) != block_size
            or not isinstance(capacity, int)
            or capacity < num_tokens
            or not isinstance(table, torch.Tensor)
            or not isinstance(slots, torch.Tensor)
        ):
            return False
        if (
            table.ndim != 2
            or table.shape[0] < num_reqs
            or table.stride(1) != 1
            or table.dtype not in (torch.int32, torch.int64)
            or table.device != positions.device
            or slots.ndim != 1
            or slots.numel() < capacity
            or slots.dtype != torch.int64
            or slots.device != positions.device
            or not slots.is_contiguous()
        ):
            return False
        contracts.append((table, slots, block_size, capacity))
    # Preflight all groups before touching any graph-visible output storage.
    for table, slots, block_size, capacity in contracts:
        if num_tokens:
            block_indices = positions // block_size
            physical_blocks = table[req_indices, block_indices].to(torch.int64)
            slots[:num_tokens].copy_(
                physical_blocks * block_size + positions % block_size
            )
        slots[num_tokens:capacity].fill_(-1)
    global _SLOT_AUDITED
    if not _SLOT_AUDITED and _ocr_can_audit():
        _write_audit(
            "slot-mapping",
            {
                "implementation": "vllm_fl.models.ocr_slots.compute_ocr_slot_mapping",
                "event": "torch_ops_enqueued",
                "num_tokens": num_tokens,
                "num_groups": len(contracts),
                "block_sizes": [item[2] for item in contracts],
                "device": str(positions.device),
                "core_triton_slot_kernel_called": False,
            },
        )
        _SLOT_AUDITED = True
    return True
