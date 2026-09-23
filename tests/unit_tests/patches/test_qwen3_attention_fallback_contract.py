# Copyright (c) 2026 BAAI. All rights reserved.

"""Structural guard for the rc1 Qwen3 ordinary-attention fallback."""

from __future__ import annotations

import ast
from pathlib import Path


PATCH = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3vl.py"
)


def _is_explicit_head_view(node: ast.Call, tensor_name: str) -> bool:
    """Match ``x.view(*x.shape[:-1], x.shape[-1] // head_dim, head_dim)``."""
    if not (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == tensor_name
        and node.func.attr == "view"
        and len(node.args) == 3
        and isinstance(node.args[0], ast.Starred)
    ):
        return False
    head_count, head_dim = node.args[1:]
    last_dimension = (
        isinstance(head_count, ast.BinOp)
        and isinstance(head_count.left, ast.Subscript)
        and isinstance(head_count.left.slice, ast.UnaryOp)
        and isinstance(head_count.left.slice.op, ast.USub)
        and isinstance(head_count.left.slice.operand, ast.Constant)
        and head_count.left.slice.operand.value == 1
    )
    return (
        last_dimension
        and isinstance(head_count, ast.BinOp)
        and isinstance(head_count.op, ast.FloorDiv)
        and isinstance(head_count.left, ast.Subscript)
        and isinstance(head_count.left.value, ast.Attribute)
        and isinstance(head_count.left.value.value, ast.Name)
        and head_count.left.value.value.id == tensor_name
        and head_count.left.value.attr == "shape"
        and isinstance(head_count.right, ast.Attribute)
        and isinstance(head_count.right.value, ast.Name)
        and head_count.right.value.id == "self"
        and head_count.right.attr == "head_dim"
        and isinstance(head_dim, ast.Attribute)
        and isinstance(head_dim.value, ast.Name)
        and head_dim.value.id == "self"
        and head_dim.attr == "head_dim"
    )


def test_qwen3_ordinary_fallback_retains_rc1_explicit_head_views() -> None:
    tree = ast.parse(PATCH.read_text(encoding="utf-8"))
    fallback = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_qwen3_attention_forward"
    )
    calls = [node for node in ast.walk(fallback) if isinstance(node, ast.Call)]

    assert any(_is_explicit_head_view(node, "q") for node in calls)
    assert any(_is_explicit_head_view(node, "k") for node in calls)
