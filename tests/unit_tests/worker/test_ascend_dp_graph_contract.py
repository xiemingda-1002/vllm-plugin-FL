# Copyright (c) 2026 BAAI. All rights reserved.

"""Static contracts for Ascend DP graph metadata and dummy execution.

These tests deliberately parse ``model_runner.py`` instead of importing it so
they remain runnable in the Python-only lint environment. Real DP graph
capture/replay still requires the dedicated multi-NPU validation harness.
"""

import ast
from pathlib import Path


MODEL_RUNNER_PATH = (
    Path(__file__).parents[3] / "vllm_fl" / "worker" / "model_runner.py"
)
SOURCE = MODEL_RUNNER_PATH.read_text()
TREE = ast.parse(SOURCE)


def _method_node(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerFL":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"ModelRunnerFL.{name} was not found")


def _method_source(name: str) -> str:
    node = _method_node(name)
    source = ast.get_source_segment(SOURCE, node)
    assert source is not None
    return source


def _calls_in(node: ast.AST) -> set[str]:
    calls = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            calls.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            calls.add(child.func.attr)
    return calls


def test_ascend_dp_sync_uses_active_group_and_synced_mode_for_padding() -> None:
    source = _method_source("_sync_metadata_across_dp")

    assert "get_dp_group()" in source
    assert "world_size" in source
    assert "rank_in_group" in source
    assert "DP topology mismatch" in source
    assert "torch.distributed.all_reduce" in source
    assert "synced_cudagraph_mode != CUDAGraphMode.NONE" in source
    assert "or force_dp_padding" in source
    assert "dp_allreduce_on_npu" in source


def test_dp1_fast_path_and_non_ascend_coordination_are_preserved() -> None:
    sync_source = _method_source("_sync_metadata_across_dp")
    dp1_guard = sync_source.index("if dp_size == 1:")
    active_group = sync_source.index("get_dp_group()")
    assert dp1_guard < active_group

    determine = _method_node("_determine_batch_execution_and_padding")
    npu_branch = next(
        node
        for node in ast.walk(determine)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "current_platform.device_type == 'npu'"
    )
    assert "_sync_metadata_across_dp" in _calls_in(ast.Module(body=npu_branch.body))
    assert "coordinate_batch_across_dp" not in _calls_in(
        ast.Module(body=npu_branch.body)
    )
    assert "coordinate_batch_across_dp" in _calls_in(
        ast.Module(body=npu_branch.orelse)
    )


def test_dummy_keeps_collective_vector_immutable_and_avoids_npu_tail_op() -> None:
    source = _method_source("_dummy_run")

    assert "num_tokens_across_dp[:]" not in source
    assert "forward_num_tokens_across_dp = num_tokens_across_dp" in source
    assert "torch.full_like(" in source
    assert "num_tokens_across_dp=forward_num_tokens_across_dp" in source
    assert 'if current_platform.device_type == "npu":' in source
    assert "return hidden_states, hidden_states" in source

    # Non-Ascend keeps upstream terminal-state extraction; Ascend moves it out
    # of the DP dummy lifecycle to avoid an extra device op after graph replay.
    assert "logit_indices_device" in source


def test_ascend_dummy_sampler_selects_terminal_states_after_dummy_return() -> None:
    source = _method_source("_dummy_sampler_run")

    assert 'if current_platform.device_type == "npu":' in source
    assert "min_tokens_per_req" in source
    assert "logit_indices = np.cumsum(num_scheduled_tokens) - 1" in source
    assert "hidden_states = hidden_states[logit_indices]" in source


def test_ascend_dummy_sampler_returns_logits_without_sampling() -> None:
    method = _method_node("_dummy_sampler_run")
    npu_branch = next(
        (index, node)
        for index, node in enumerate(method.body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "current_platform.device_type == 'npu'"
    )
    _, branch = npu_branch
    branch_source = ast.unparse(ast.Module(body=branch.body, type_ignores=[]))

    assert branch_source.endswith("return self.model.compute_logits(hidden_states)")
    assert "self.sampler" not in branch_source


def test_non_ascend_dummy_sampler_keeps_generic_warmup() -> None:
    method = _method_node("_dummy_sampler_run")
    npu_index = next(
        index
        for index, node in enumerate(method.body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "current_platform.device_type == 'npu'"
    )
    generic_source = ast.unparse(
        ast.Module(body=method.body[npu_index + 1 :], type_ignores=[])
    )

    assert "hidden_states = torch.rand_like(hidden_states)" in generic_source
    assert "dummy_metadata = SamplingMetadata" in generic_source
    assert "sampler_output = self.sampler" in generic_source
