# Copyright (c) 2026 BAAI. All rights reserved.

"""Execute memory-lifecycle branches without importing the NPU runtime.

These tests check dispatch and logging, not device allocation or graph replay.
"""

import ast
import sys
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _method(filename, name):
    tree = ast.parse((ROOT / "vllm_fl/worker" / filename).read_text())
    return next(
        item
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )


def _execute(node, namespace):
    module = ast.Module(body=[node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), "<memory-branch>", "exec"), namespace
    )


class Comm(Enum):
    MC2 = auto()
    FUSED_MC2 = auto()
    ALLTOALL = auto()
    ALLGATHER = auto()


@pytest.mark.parametrize(
    "device,vendor",
    [("npu", "ascend"), ("cuda", "nvidia"), ("cpu", None), ("npu", "other")],
)
@pytest.mark.parametrize("capacity", [None, 64, 128, 256])
@pytest.mark.parametrize("comm", [None, *Comm])
def test_capacity_warmup_dispatch(monkeypatch, device, vendor, capacity, comm):
    context = ModuleType("vllm_fl.ascend_forward_context")
    context.MoECommType = Comm
    context.get_mc2_tokens_capacity = Mock(return_value=capacity)
    context.select_moe_comm_method = Mock(return_value=comm)
    monkeypatch.setitem(sys.modules, context.__name__, context)
    runner = SimpleNamespace(
        max_num_tokens=128, vllm_config=object(), _dummy_run=Mock()
    )
    branch = _method("model_runner.py", "profile_run").body[0]
    assert isinstance(branch, ast.If)
    _execute(
        branch,
        {
            "self": runner,
            "current_platform": SimpleNamespace(device_type=device, vendor_name=vendor),
        },
    )

    expected = (
        device == "npu"
        and vendor == "ascend"
        and capacity is not None
        and capacity < 128
        and comm in {Comm.MC2, Comm.FUSED_MC2}
    )
    if expected:
        runner._dummy_run.assert_called_once_with(capacity, is_profile=True)
    else:
        runner._dummy_run.assert_not_called()
    if vendor != "ascend":
        context.get_mc2_tokens_capacity.assert_not_called()
        context.select_moe_comm_method.assert_not_called()


def test_dummy_profile_flag_reaches_main_model():
    source = ast.unparse(_method("model_runner.py", "_dummy_run"))
    flag = "get_forward_context().additional_kwargs['in_profile_run'] = is_profile"
    assert source.index(flag) < source.index("outputs = self.model(")


@pytest.mark.parametrize(
    "device,vendor",
    [("npu", "ascend"), ("cuda", "nvidia"), ("cpu", None), ("npu", "other")],
)
def test_memory_summary_logging_is_vendor_scoped(device, vendor):
    method = _method("worker.py", "compile_or_warm_up_model")
    branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and "msg.replace" in ast.unparse(node)
        and "current_platform.device_type == 'npu'" in ast.unparse(node.test)
        and "vendor_name" in ast.unparse(node.test)
    )
    memory_api = SimpleNamespace(
        memory_reserved=Mock(return_value=2 << 30),
        memory_allocated=Mock(return_value=1 << 30),
    )
    logger = Mock()
    memory_logger = Mock()
    init_logger = Mock(return_value=memory_logger)
    namespace = {
        "self": SimpleNamespace(device="npu:0"),
        "current_platform": SimpleNamespace(
            device_type=device, vendor_name=vendor, torch_device_fn=memory_api
        ),
        "logger": logger,
        "init_logger": init_logger,
        "GiB": lambda value: round(value / (1 << 30), 2),
        "msg": "Measured CUDAGraph memory.",
    }
    _execute(branch, namespace)
    if device == "npu" and vendor == "ascend":
        init_logger.assert_called_once_with("vllm.vllm_fl.ascend.memory")
        memory_logger.info.assert_called_once()
        message = memory_logger.info.call_args.args[0]
        assert "NPU graph memory" in message
        assert "reserved memory 2.0 GiB" in message
        assert "allocated memory 1.0 GiB" in message
        memory_api.memory_reserved.assert_called_once_with("npu:0")
        memory_api.memory_allocated.assert_called_once_with("npu:0")
        logger.debug.assert_not_called()
    else:
        logger.debug.assert_called_once_with("Measured CUDAGraph memory.")
        logger.info.assert_not_called()
        init_logger.assert_not_called()
        memory_api.memory_reserved.assert_not_called()
        memory_api.memory_allocated.assert_not_called()
