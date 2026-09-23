"""Source-level regression for GLM-5.2's Ascend SFA cache contract.

The host unit environment intentionally does not import torch_npu.  Inspect the
production method's AST so this remains a CPU-only regression while pinning the
real relocated implementation class used by the runner.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_fl/worker/model_runner.py"


def _method() -> ast.FunctionDef:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_kv_cache_spec"
    )


def _isinstance_of(node: ast.expr, class_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == class_name
    )


def test_relocated_ascend_sfa_impl_selects_mla_tuple_cache_only_on_ascend() -> None:
    method = _method()
    imports = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm_fl.attention.ascend.sfa_v1"
    ]
    assert any(any(alias.name == "AscendSFAImpl" for alias in node.names) for node in imports)

    sfa_checks = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and _isinstance_of(node, "AscendSFAImpl")
        and isinstance(node.args[0], ast.Attribute)
        and isinstance(node.args[0].value, ast.Name)
        and node.args[0].value.id == "attn_module"
        and node.args[0].attr == "impl"
    ]
    assert len(sfa_checks) == 1

    gates = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and any(_isinstance_of(value, "MLAAttention") for value in node.test.values)
        and any(
            isinstance(value, ast.Compare)
            and "vendor_name" in ast.unparse(value)
            and ast.unparse(value).endswith("== 'ascend'")
            for value in node.test.values
        )
    ]
    assert len(gates) == 1
    gate = gates[0]
    assert imports[0] in gate.body
    sfa_branches = [
        node for node in gate.body
        if isinstance(node, ast.If) and node.test is sfa_checks[0]
    ]
    assert len(sfa_branches) == 1
    assert not sfa_branches[0].orelse

    # The outer vendor gate keeps dense MLA and every non-Ascend backend on
    # their generic spec path; no import-path suffix is used for dispatch.
    source = ast.unparse(method)
    assert any(
        isinstance(node, ast.Compare)
        and any(
            isinstance(value, ast.Constant) and value.value == "ascend"
            for value in node.comparators
        )
        and "vendor_name" in ast.unparse(node)
        for node in ast.walk(method)
    )
    assert ".__module__.endswith(" not in source

    spec_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AscendMLAAttentionSpec"
    ]
    assert len(spec_calls) == 1
    assert spec_calls[0] in list(ast.walk(sfa_branches[0]))
    keywords = {item.arg: ast.unparse(item.value) for item in spec_calls[0].keywords}
    # AscendSFAImpl owns the main MLA cache as one K/V tuple per block.  Its
    # physical width must stay kv_lora_rank + qk_rope_head_dim.
    assert keywords["num_kv_heads"] == "1"
    assert keywords["head_size"] == (
        "self.model_config.hf_text_config.kv_lora_rank + "
        "self.model_config.hf_text_config.qk_rope_head_dim"
    )
