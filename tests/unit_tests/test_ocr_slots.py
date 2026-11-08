# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU numerical contracts; production dispatch remains NPU-only."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def slots_module(monkeypatch):
    runtime = ModuleType("vllm_fl.models.ocr_runtime")
    runtime.ocr_runtime_enabled = lambda: True
    runtime._ocr_can_audit = lambda: False
    runtime._write_audit = lambda *args: None
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    path = Path(__file__).resolve().parents[2] / "vllm_fl/models/ocr_slots.py"
    spec = importlib.util.spec_from_file_location("ocr_slots_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("VLLM_FL_OCR_TORCH_SLOT_MAPPING", raising=False)
    monkeypatch.setattr(module, "_supported_device", lambda device: True)
    return module


def group(noncontiguous=False):
    table = torch.tensor([[8, 3, 12], [19, 2, 5]], dtype=torch.int32)
    if noncontiguous:
        storage = torch.empty((4, 3), dtype=table.dtype)
        storage[::2].copy_(table)
        table = storage[::2]
    return SimpleNamespace(
        block_size=4,
        blocks_per_kv_block=1,
        use_hybrid_blocks=False,
        pcp_world_size=1,
        dcp_world_size=1,
        max_num_batched_tokens=10,
        block_table=SimpleNamespace(gpu=table),
        slot_mapping=SimpleNamespace(gpu=torch.full((12,), 777, dtype=torch.int64)),
    )


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_values_padding_and_storage(slots_module, noncontiguous):
    groups = [group(noncontiguous), group()]
    positions = torch.tensor([0, 3, 4, 7, 8, 11], dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    pointers = [g.slot_mapping.gpu.data_ptr() for g in groups]
    assert slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=groups), 2, requests, positions
    )
    for g, pointer in zip(groups, pointers):
        assert g.slot_mapping.gpu.tolist() == [
            32,
            35,
            12,
            11,
            20,
            23,
            -1,
            -1,
            -1,
            -1,
            777,
            777,
        ]
        assert g.slot_mapping.gpu.data_ptr() == pointer


@pytest.mark.parametrize(
    "field,value",
    [
        ("pcp_world_size", 2),
        ("dcp_world_size", 2),
        ("blocks_per_kv_block", 2),
        ("use_hybrid_blocks", True),
        ("block_size", 0),
        ("kv_cache_block_size", 8),
        ("max_num_batched_tokens", 0),
        ("slot_mapping_mode", SimpleNamespace(name="NONE")),
    ],
)
def test_group_fallback_is_atomic(slots_module, field, value):
    groups = [group(), group()]
    setattr(groups[1], field, value)
    assert not slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=groups), 2, torch.tensor([0]), torch.tensor([0])
    )
    assert all(torch.all(g.slot_mapping.gpu == 777) for g in groups)


def test_empty_tokens(slots_module):
    g = group()
    empty = torch.empty(0, dtype=torch.int64)
    assert slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 0, empty, empty
    )
    assert g.slot_mapping.gpu.tolist() == [-1] * 10 + [777, 777]


def test_opt_in_and_device_gates(slots_module, monkeypatch):
    g = group()
    manager = SimpleNamespace(block_tables=[g])
    args = (manager, 1, torch.tensor([0]), torch.tensor([0]))
    monkeypatch.setenv("VLLM_FL_OCR_TORCH_SLOT_MAPPING", "0")
    assert not slots_module.compute_ocr_slot_mapping(*args)
    monkeypatch.setenv("VLLM_FL_OCR_TORCH_SLOT_MAPPING", "1")
    monkeypatch.setattr(slots_module, "ocr_runtime_enabled", lambda: False)
    assert not slots_module.compute_ocr_slot_mapping(*args)
    monkeypatch.setattr(slots_module, "ocr_runtime_enabled", lambda: True)
    monkeypatch.setattr(slots_module, "_supported_device", lambda device: False)
    assert not slots_module.compute_ocr_slot_mapping(*args)
    assert torch.all(g.slot_mapping.gpu == 777)


def test_invalid_tensor_contract(slots_module):
    g = group()
    g.slot_mapping.gpu = g.slot_mapping.gpu.to(torch.int32)
    assert not slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 2, torch.tensor([0]), torch.tensor([0])
    )
    assert torch.all(g.slot_mapping.gpu == 777)


def test_column_strided_table_falls_back(slots_module):
    g = group()
    g.block_table.gpu = g.block_table.gpu.T.contiguous().T
    assert not slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 1, torch.tensor([0]), torch.tensor([0])
    )
    assert torch.all(g.slot_mapping.gpu == 777)


def test_audit_only_after_success_and_never_during_capture(slots_module, monkeypatch):
    events = []
    monkeypatch.setattr(slots_module, "_write_audit", lambda *args: events.append(args))
    monkeypatch.setattr(slots_module, "_ocr_can_audit", lambda: False)
    g = group()
    args = (SimpleNamespace(block_tables=[g]), 1, torch.tensor([0]), torch.tensor([0]))
    assert slots_module.compute_ocr_slot_mapping(*args)
    assert not events and not slots_module._SLOT_AUDITED
    monkeypatch.setattr(slots_module, "_ocr_can_audit", lambda: True)
    assert slots_module.compute_ocr_slot_mapping(*args)
    assert slots_module.compute_ocr_slot_mapping(*args)
    assert len(events) == 1 and events[0][0] == "slot-mapping"
    assert events[0][1]["event"] == "torch_ops_enqueued"
