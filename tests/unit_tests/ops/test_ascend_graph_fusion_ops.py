# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def test_npu_rotary_forward_emits_canonical_functional_op(monkeypatch) -> None:
    import vllm_fl.ops.rotary_embedding as rotary_module

    calls = []
    monkeypatch.setattr(rotary_module.current_platform, "device_type", "npu")
    monkeypatch.setattr(
        torch.ops.vllm,
        "npu_rotary_embedding",
        lambda *args: calls.append(args) or (args[1], args[2]),
        raising=False,
    )

    layer = object.__new__(rotary_module.RotaryEmbeddingFL)
    torch.nn.Module.__init__(layer)
    layer.head_size = 128
    layer.rotary_dim = 128
    layer.is_neox_style = True
    layer.cos_sin_cache = torch.empty(16, 128)
    positions = torch.arange(3)
    query = torch.empty(3, 1024)
    key = torch.empty(3, 256)
    result = layer.forward_oot(positions, query, key)

    assert result[0] is query and result[1] is key
    assert len(calls) == 1
    assert calls[0][0] is positions
    assert calls[0][1] is query and calls[0][2] is key
    assert calls[0][3] is layer.cos_sin_cache
    assert calls[0][4:] == (128, 128, True)


def test_graph_fusion_op_registration_in_fresh_process() -> None:
    source_root = Path(__file__).resolve().parents[3]
    script = r'''
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

assert not hasattr(torch.ops.vllm, "npu_rotary_embedding")
assert not hasattr(torch.ops.vllm, "qkv_rmsnorm_rope")
from vllm_fl.dispatch.backends.vendor.ascend.impl.graph_fusion_ops import (
    ensure_graph_fusion_ops_registered,
)
ensure_graph_fusion_ops_registered()
ensure_graph_fusion_ops_registered()
assert hasattr(torch.ops.vllm, "npu_rotary_embedding")
assert hasattr(torch.ops.vllm, "qkv_rmsnorm_rope")
with FakeTensorMode():
    positions = torch.arange(3)
    query = torch.empty(3, 1024)
    key = torch.empty(3, 256)
    cache = torch.empty(16, 128)
    q_rot, k_rot = torch.ops.vllm.npu_rotary_embedding(
        positions, query, key, cache, 128, 128, True
    )
    assert q_rot.shape == query.shape and k_rot.shape == key.shape
    q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
        torch.empty(3, 1536), cache, positions,
        torch.empty(128), torch.empty(128), 1024, 256, 128, 1e-6
    )
    assert q.shape == (3, 1024)
    assert k.shape == v.shape == (3, 256)
print("FRESH_PROCESS_REGISTRATION_PASS")
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root)
    env["VLLM_PLUGINS"] = "fl"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FRESH_PROCESS_REGISTRATION_PASS" in completed.stdout


def test_first_graph_closure_has_no_vllm_ascend_dependency() -> None:
    import vllm_fl.compilation.graph_fusion_pass_manager as manager_module
    import vllm_fl.compilation.passes.base_pattern as base_module
    import vllm_fl.compilation.passes.muls_add_pass as muls_module
    import vllm_fl.compilation.passes.norm_quant_fusion_pass as norm_module
    import vllm_fl.compilation.passes.qknorm_rope_fusion_pass as qk_module
    import vllm_fl.dispatch.backends.vendor.ascend.impl.canonical_rotary as rotary
    import vllm_fl.dispatch.backends.vendor.ascend.impl.graph_fusion_ops as ops

    modules = (
        manager_module,
        base_module,
        muls_module,
        norm_module,
        qk_module,
        rotary,
        ops,
    )
    assert all("vllm_ascend" not in inspect.getsource(module) for module in modules)
    assert not any(
        name == "vllm_ascend" or name.startswith("vllm_ascend.")
        for name in sys.modules
    )
