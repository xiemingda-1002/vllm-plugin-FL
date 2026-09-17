"""Contracts for the Ascend-owned MLA prefill construction boundary."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")


def test_ascend_mla_prefill_patch_replaces_captured_selector(monkeypatch) -> None:
    import vllm.model_executor.layers.attention.mla_attention as mla_attention
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_mla_prefill_backend import (
        AscendMLAPrefillBackend,
        apply_ascend_mla_prefill_backend_patch,
    )

    sentinel = object()
    monkeypatch.setattr(
        mla_attention,
        "get_mla_prefill_backend",
        lambda _config: sentinel,
    )

    apply_ascend_mla_prefill_backend_patch()

    assert mla_attention.get_mla_prefill_backend(SimpleNamespace()) is AscendMLAPrefillBackend


def test_ascend_mla_prefill_backend_is_constructible_but_never_executes() -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_mla_prefill_backend import (
        AscendMLAPrefillBackend,
    )

    backend = AscendMLAPrefillBackend(
        num_heads=8,
        scale=0.125,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        vllm_config=SimpleNamespace(),
    )
    tensor = torch.empty(0)

    assert backend.get_name() == "ASCEND"
    assert backend.is_available()
    with pytest.raises(NotImplementedError, match="handled by Ascend"):
        backend.run_prefill_new_tokens(tensor, tensor, tensor, False)
    with pytest.raises(NotImplementedError, match="handled by Ascend"):
        backend.run_prefill_context_chunk(0, tensor, tensor, tensor)


def test_ascend_startup_installs_prefill_patch_before_glm_constructor() -> None:
    import ast
    from pathlib import Path

    source = Path(
        "vllm_fl/dispatch/backends/vendor/ascend/patch.py"
    ).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_ascend_patches"
    )
    calls = [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert calls.index("apply_ascend_mla_prefill_backend_patch") < calls.index(
        "apply_glm52_shared_indexer_patch"
    )
