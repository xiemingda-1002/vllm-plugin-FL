from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


class _Options:
    def __init__(self):
        self.hccl_config = {}
        self.backend = "hccl"
        self.global_ranks_in_group = ()
        self.group_id = ""
        self.group_name = ""
        self.is_high_priority_stream = False


def _load_options(monkeypatch, dp_size: int):
    torch_npu = types.SimpleNamespace(
        _C=types.SimpleNamespace(
            _distributed_c10d=types.SimpleNamespace(
                ProcessGroupHCCL=types.SimpleNamespace(Options=_Options)
            )
        )
    )
    config = types.ModuleType("vllm.config")
    config.get_current_vllm_config = lambda: types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(data_parallel_size=dp_size)
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    spec = importlib.util.spec_from_file_location("hccl_options_test", _ROOT / "vllm_fl/distributed/ascend_hccl_options.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_registry():
    spec = importlib.util.spec_from_file_location("hccl_registry_test", _ROOT / "vllm_fl/distributed/ascend_hccl_registry.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rc1_hccl_buffer_options(monkeypatch):
    options = _load_options(monkeypatch, 1)
    assert options.create_hccl_pg_options("tp").hccl_config == {
        "hccl_buffer_size": 200, "group_name": "tp"
    }
    assert options.create_hccl_pg_options("dp").hccl_config == {
        "hccl_buffer_size": 50, "group_name": "dp"
    }
    assert options.create_hccl_pg_options("dynamic_eplb").hccl_config == {
        "hccl_buffer_size": 100, "group_name": "dynamic_eplb"
    }
    assert options.create_hccl_pg_options("mc2").hccl_config == {"group_name": "mc2"}


def test_dp_buffer_scales_and_group_name_participates_in_key(monkeypatch):
    options = _load_options(monkeypatch, 30_000_000)
    assert options.create_hccl_pg_options("dp").hccl_config["hccl_buffer_size"] > 50
    registry = _load_registry()
    world = options.create_hccl_pg_options("world")
    tp = options.create_hccl_pg_options("tp")
    assert registry.make_hccl_pg_key([0, 1], "hccl", world, "shared") != registry.make_hccl_pg_key([0, 1], "hccl", tp, "shared")
    assert registry.make_hccl_pg_key([0, 1], "hccl", world, "shared") == registry.make_hccl_pg_key([0, 1], "hccl", world, "shared")


def test_unknown_nondefault_option_fails_closed(monkeypatch):
    options = _load_options(monkeypatch, 1)
    registry = _load_registry()
    value = options.create_hccl_pg_options("tp")
    value.unrecognized_non_default = "unsafe"
    assert registry.make_hccl_pg_key([0, 1], "hccl", value, "shared") is None
