# SPDX-License-Identifier: Apache-2.0
"""Device-independent guards for the stateful M3 provider selection contract."""

import ast
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[3] / "vllm_fl"


@pytest.fixture
def selector(monkeypatch):
    path = ROOT / "dispatch/backends/vendor/ascend/attention_selector.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_minimax_sparse_kernels"
    )
    dispatch = ModuleType("vllm_fl.dispatch")
    calls = []

    class Provider:
        cache_contract = "minimax_m3_paged_v1"
        replay_contract = "persistent_metadata_v1"
        insert_main_cache = insert_index_cache = run_attention = attend = staticmethod(
            lambda *args: None
        )
        qknorm_rope_cache = rotary = gemma_weight = activation = staticmethod(
            lambda *args: None
        )
        index_score = index_topk = index_decode = staticmethod(lambda *args: None)

    dispatch.call_op = lambda name: calls.append(name) or Provider
    monkeypatch.setitem(sys.modules, "vllm_fl.dispatch", dispatch)
    namespace = {"cache": cache}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[node.name], dispatch, Provider, calls


def test_provider_is_pinned_before_stateful_execution(selector):
    select, dispatch, provider, calls = selector
    assert select() is provider
    dispatch.call_op = lambda name: (_ for _ in ()).throw(AssertionError("reselected"))
    assert select() is provider
    assert calls == ["minimax_m3_sparse_kernels"]


@pytest.mark.parametrize("bad", [None, object(), type("Bad", (), {})])
def test_invalid_provider_is_not_cached(selector, bad):
    select, dispatch, provider, _ = selector
    dispatch.call_op = lambda name: bad
    with pytest.raises((TypeError, ValueError)):
        select()
    dispatch.call_op = lambda name: provider
    assert select() is provider


@pytest.mark.parametrize(
    "attribute",
    [
        "cache_contract",
        "replay_contract",
        "insert_main_cache",
        "insert_index_cache",
        "run_attention",
        "attend",
        "qknorm_rope_cache",
        "rotary",
        "gemma_weight",
        "activation",
        "index_score",
        "index_topk",
        "index_decode",
    ],
)
def test_partial_contract_is_rejected(selector, attribute):
    select, _, provider, _ = selector
    setattr(provider, attribute, None)
    with pytest.raises((TypeError, ValueError)):
        select()


def test_stateful_body_has_no_dispatch_fallback_and_explicit_layout():
    path = ROOT / "models/minimax_m3_ascend.py"
    tree = ast.parse(path.read_text())
    body = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sparse_forward_body"
    )
    assert not any(isinstance(node, ast.Try) for node in ast.walk(body))
    source = ast.unparse(body)
    assert "call_op" not in source and "CachedOp" not in source
    assert (
        source.index("kernels.insert_main_cache(")
        < source.index("kernels.insert_index_cache(")
        < source.index("kernels.run_attention(")
    )
    assert "plane_major=layer._fl_sparse_plane_major" in source
    attend = (
        ROOT / "dispatch/backends/vendor/ascend/impl/attention/sparse_attn_m3.py"
    ).read_text()
    assert "kernels = layer._fl_sparse_kernels" in attend
    assert attend.count("plane_major=plane_major") == 2
