from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODEL_RUNNER_PATH = ROOT / "vllm_fl/worker/model_runner.py"


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    return ast.unparse(node.func)


def test_ascend_mamba_copy_runs_once_inside_context_before_forward() -> None:
    tree = ast.parse(MODEL_RUNNER_PATH.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerFL"
    )
    execute = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
    )
    copy_calls = [
        node
        for node in ast.walk(execute)
        if _call_name(node) == "mamba_utils.do_mamba_copy_block"
    ]
    assert len(copy_calls) == 1

    forward_context = next(
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.With)
        and any(
            _call_name(item.context_expr) == "set_forward_context"
            for item in node.items
        )
        and any(
            _call_name(candidate) == "self._model_forward"
            for candidate in ast.walk(node)
        )
    )
    gated_copy = forward_context.body[-2]
    model_forward = forward_context.body[-1]
    assert isinstance(gated_copy, ast.If)
    condition = ast.unparse(gated_copy.test)
    assert "mamba_cache_mode == 'align'" in condition
    assert "vendor_name', None) == 'ascend'" in condition
    assert "device_type == 'npu'" in condition
    assert len(gated_copy.body) == 1
    assert any(call is copy_calls[0] for call in ast.walk(gated_copy))
    assert isinstance(model_forward, ast.Assign)
    assert any(
        _call_name(candidate) == "self._model_forward"
        for candidate in ast.walk(model_forward)
    )
    assert copy_calls[0].lineno < model_forward.lineno
