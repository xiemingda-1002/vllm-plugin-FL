"""CPU behavior contracts for GLM-5.2 SFA runner metadata.

These execute the production expressions/method body without constructing an
NPU runner, so the test covers the runner boundary rather than text presence.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_fl/worker/model_runner.py"


def _tree() -> ast.Module:
    return ast.parse(RUNNER.read_text(encoding="utf-8"))


def _method(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _eval(expr: ast.expr, **locals_: object) -> object:
    expression = ast.Expression(expr)
    ast.fix_missing_locations(expression)
    return eval(compile(expression, str(RUNNER), "eval"), {}, locals_)


def _call_keyword(method_name: str, call_attr: str, keyword: str) -> ast.expr:
    method = _method(method_name)
    call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == call_attr)
            or (isinstance(node.func, ast.Name) and node.func.id == call_attr)
        )
    )
    value = next(item.value for item in call.keywords if item.arg == keyword)
    assert value is not None
    return value


def _state_bridge():
    method = _method("_set_dsa_common_attention_state")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"CommonAttentionMetadata": object, "Any": object}
    exec(compile(module, "glm52_sfa_state_bridge", "exec"), namespace)
    return namespace["_set_dsa_common_attention_state"]


def test_sfa_uses_extended_metadata_without_becoming_compressed_dsa() -> None:
    sfa = SimpleNamespace(use_compress=False, use_ascend_sfa=True)
    generic = SimpleNamespace(use_compress=False, use_ascend_sfa=False)

    # Execute the production bridge: a GLM SFA real phase and a graph dummy
    # phase both reach the Ascend extension, while a generic runner does not.
    bridge = _state_bridge()
    metadata = SimpleNamespace(attn_state=None)
    bridge(sfa, metadata, "real-sfa-state")
    assert metadata.attn_state == "real-sfa-state"
    bridge(sfa, metadata, "decode-dummy-state")
    assert metadata.attn_state == "decode-dummy-state"
    untouched = SimpleNamespace(attn_state="upstream")
    bridge(generic, untouched, "must-not-apply")
    assert untouched.attn_state == "upstream"

    # Execute the normal and dummy call-site expressions. SFA retains padded
    # input capacity for RoPE/slot metadata but reports real forward tokens.
    normal_padded = _call_keyword("execute_model", "_build_attention_metadata", "num_tokens_padded")
    dummy_padded = _call_keyword("_dummy_run", "_build_attention_metadata", "num_tokens_padded")
    values = {"pad_attn": False, "num_tokens_padded": 8}
    assert _eval(normal_padded, self=sfa, **values) == 8
    assert _eval(dummy_padded, self=sfa, **values) == 8
    assert _eval(normal_padded, self=generic, **values) is None
    assert _eval(dummy_padded, self=generic, **values) is None

    actual_tokens = _call_keyword(
        "_build_attention_metadata", "common_metadata_cls", "num_actual_tokens"
    )
    assert _eval(actual_tokens, self=sfa, num_tokens=5, num_tokens_padded=8) == 5
    assert _eval(actual_tokens, self=generic, num_tokens=5, num_tokens_padded=8) == 8
