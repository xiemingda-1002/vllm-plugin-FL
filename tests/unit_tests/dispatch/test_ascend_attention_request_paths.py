# Copyright (c) 2026 BAAI. All rights reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = (
    Path(__file__).parents[3]
    / "vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py"
)


def _method_node(name: str) -> ast.FunctionDef:
    module = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
    impl = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AscendAttentionBackendImpl"
    )
    return next(
        node
        for node in impl.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class _AttentionState:
    PrefillNoCache = object()
    PrefillCacheHit = object()
    DecodeOnly = object()
    ChunkedPrefill = object()


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.view_args = None

    def view(self, *shape):
        self.view_args = shape
        return self


def _load_get_fia_params():
    method = _method_node("_get_fia_params")
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted)
    namespace = {"AscendAttentionState": _AttentionState}
    exec(compile(extracted, str(_SOURCE), "exec"), namespace)
    return namespace["_get_fia_params"]


@pytest.mark.parametrize(
    "attn_state",
    [_AttentionState.DecodeOnly, _AttentionState.ChunkedPrefill],
)
def test_fia_params_derive_block_layout_from_kv_cache(attn_state) -> None:
    key_cache = _FakeTensor((3, 16, 2, 64))
    value_cache = _FakeTensor((3, 16, 2, 64))
    impl = SimpleNamespace(key_cache=key_cache, value_cache=value_cache)
    block_tables = object()
    metadata = SimpleNamespace(
        attn_state=attn_state,
        block_tables=block_tables,
        seq_lens_list=[17],
    )

    key, value, block_size, returned_tables, seq_lens = (
        _load_get_fia_params()(impl, object(), object(), metadata)
    )

    assert key is key_cache
    assert value is value_cache
    assert key_cache.view_args == (3, 16, -1)
    assert value_cache.view_args == (3, 16, -1)
    assert block_size == 16
    assert returned_tables is block_tables
    assert seq_lens == [17]


def test_forward_uses_metadata_state_for_cache_contiguity() -> None:
    method = _method_node("forward")
    cache_gate = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "contiguous"
            for child in ast.walk(node)
        )
    )

    assert ast.dump(cache_gate.test, include_attributes=False) == ast.dump(
        ast.Compare(
            left=ast.Attribute(
                value=ast.Name(id="attn_metadata", ctx=ast.Load()),
                attr="attn_state",
                ctx=ast.Load(),
            ),
            ops=[ast.NotEq()],
            comparators=[
                ast.Attribute(
                    value=ast.Name(id="AscendAttentionState", ctx=ast.Load()),
                    attr="DecodeOnly",
                    ctx=ast.Load(),
                )
            ],
        ),
        include_attributes=False,
    )


def test_forward_signature_matches_current_attention_contract() -> None:
    method = _method_node("forward")
    assert [argument.arg for argument in method.args.args] == [
        "self",
        "layer",
        "query",
        "key",
        "value",
        "kv_cache",
        "attn_metadata",
        "output",
        "output_scale",
        "output_block_scale",
    ]
    assert len(method.args.defaults) == 3
    assert all(
        isinstance(default, ast.Constant) and default.value is None
        for default in method.args.defaults
    )
