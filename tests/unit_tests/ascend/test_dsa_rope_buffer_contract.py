"""CPU coverage of the production DSA RoPE graph-buffer contract."""

import ast
from pathlib import Path
from typing import Any

import pytest
import torch


def _load_rope_contract():
    # Execute unchanged production definitions without loading NPU registration.
    source = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/dispatch/backends/vendor/ascend/ops/rope_dsv4.py"
    )
    names = {"RopeGlobalState", "RopeDataProxy", "get_cos_and_sin_dsa"}
    tree = ast.parse(source.read_text())
    definitions = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = {"torch": torch, "Any": Any}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    state = namespace["RopeGlobalState"]()
    namespace["_ROPE_STATE"] = state
    return state, namespace["get_cos_and_sin_dsa"]


@pytest.mark.parametrize("group", ["default", "local"])
def test_decode_rope_updates_capture_storage_in_place(group):
    state, get_rope = _load_rope_contract()
    full_cos = torch.arange(64, dtype=torch.float32).reshape(16, 1, 1, 4)
    full_sin = -full_cos
    buffers = (torch.zeros(8, 1, 1, 4), torch.zeros(8, 1, 1, 4))
    state.full_rope_cache["config"] = (full_cos, full_sin)
    state.registry_summary["config"] = {group}
    state.layer_info["layer"] = ("config", [group])
    state.runtime_buffer["config"] = {group: buffers}

    captured = get_rope({group: torch.tensor([1, 2])}, use_cache=True)
    original = captured[0]["layer"].clone()
    updated = get_rope({group: torch.tensor([7, 8])}, use_cache=True)
    for index, table in enumerate((full_cos, full_sin)):
        assert captured[index]["layer"].data_ptr() == buffers[index].data_ptr()
        assert updated[index]["layer"].data_ptr() == captured[index]["layer"].data_ptr()
        torch.testing.assert_close(captured[index]["layer"], table[[7, 8]])
    assert not torch.equal(original, captured[0]["layer"])

    # Prefill may allocate independent tensors; it must not replace or mutate
    # the storage retained by a captured decode graph.
    prefill = get_rope({group: torch.tensor([3, 4])}, use_cache=False)
    assert prefill[0]["layer"].data_ptr() != buffers[0].data_ptr()
    torch.testing.assert_close(prefill[0]["layer"], full_cos[[3, 4]])
    torch.testing.assert_close(captured[0]["layer"], full_cos[[7, 8]])
