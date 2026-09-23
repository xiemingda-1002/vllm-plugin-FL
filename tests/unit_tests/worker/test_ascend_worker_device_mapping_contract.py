# Copyright (c) 2026 BAAI. All rights reserved.

"""Static contracts for current-vLLM DP device assignment on Ascend."""

import ast
from pathlib import Path


WORKER_PATH = Path(__file__).parents[3] / "vllm_fl" / "worker" / "worker.py"
SOURCE = WORKER_PATH.read_text()
TREE = ast.parse(SOURCE)


def _init_device_source() -> str:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == "WorkerFL":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "init_device":
                    source = ast.get_source_segment(SOURCE, item)
                    assert source is not None
                    return source
    raise AssertionError("WorkerFL.init_device was not found")


def test_preassigned_dp_device_shard_skips_second_rank_offset() -> None:
    source = _init_device_source()

    offset = "self.local_rank += dp_local_rank * tp_pp_world_size"
    npu_guard = 'current_platform.device_type != "npu"'
    assigned_guard = "self.parallel_config.assigned_physical_gpu_ids is None"
    assert npu_guard in source
    assert assigned_guard in source
    assert source.index(npu_guard) < source.index(assigned_guard) < source.index(offset)


def test_unassigned_dp_devices_keep_rank_offset_and_bounds_check() -> None:
    source = _init_device_source()

    assert "dp_local_rank * tp_pp_world_size" in source
    assert "DP adjusted local rank" in source


def test_non_ascend_assigned_devices_keep_existing_dp_offset() -> None:
    source = _init_device_source()

    # The OR makes every non-NPU platform retain the pre-existing DP offset,
    # including CUDA-like vendors with a user-provided full device list.
    assert (
        'current_platform.device_type != "npu"\n'
        "                or self.parallel_config.assigned_physical_gpu_ids is None"
        in source
    )


def test_assigned_devices_are_published_and_indexed_by_local_rank() -> None:
    source = _init_device_source()

    assert "set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)" in source
    assert "self.local_rank < len(assigned_physical_gpu_ids)" in source
    assert "logical_device_id_to_visible_device_id(\n            self.local_rank" in source
