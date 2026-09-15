"""CPU contracts for the rc1 DSA attention-state lifecycle."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_fl.dispatch.backends.vendor.ascend.attention.utils import (
    classify_dsa_attention_state,
    get_dsa_dummy_attention_state,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
    AscendAttentionState,
)


_ROOT = Path(__file__).resolve().parents[3]


def _cp_has_prefill():
    """Execute the production CP consumer predicate without NPU construction."""
    source = (
        _ROOT
        / "vllm_fl"
        / "dispatch"
        / "backends"
        / "vendor"
        / "ascend"
        / "attention"
        / "context_parallel"
        / "dsa_cp.py"
    ).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_has_prefill"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"AscendAttentionState": AscendAttentionState}
    exec(compile(module, "dsa_cp_has_prefill", "exec"), namespace)
    return namespace["_has_prefill"]


def _runner_state_bridge():
    """Execute the production runner bridge with a CPU metadata stub."""
    source = (_ROOT / "vllm_fl" / "worker" / "model_runner.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_set_dsa_common_attention_state"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"CommonAttentionMetadata": object, "Any": object}
    exec(compile(module, "dsa_runner_state_bridge", "exec"), namespace)
    return namespace["_set_dsa_common_attention_state"]


def test_dsa_real_batch_phase_contract_matches_rc1() -> None:
    # New prompt, cache-hit prefill, full decode, and mixed/chunked batches.
    assert classify_dsa_attention_state([0], [1], [1], False, None) is (
        AscendAttentionState.PrefillNoCache
    )
    assert classify_dsa_attention_state([4], [3], [3], False, None) is (
        AscendAttentionState.PrefillCacheHit
    )
    decode_state = classify_dsa_attention_state([4, 9], [1, 1], [1, 1], True, None)
    assert decode_state is AscendAttentionState.DecodeOnly
    assert _cp_has_prefill()(decode_state) is False
    assert classify_dsa_attention_state([4, 0], [1, 4], [1, 4], True, None) is (
        AscendAttentionState.ChunkedPrefill
    )


def test_runner_bridge_uses_real_state_and_explicit_dummy_override() -> None:
    bridge = _runner_state_bridge()
    metadata = SimpleNamespace(attn_state=None)
    runner = SimpleNamespace(
        use_compress=True,
        _dsa_attn_state=AscendAttentionState.PrefillCacheHit,
    )

    # Real metadata uses the freshly classified state stored by _prepare_inputs.
    bridge(runner, metadata, None)
    assert metadata.attn_state is AscendAttentionState.PrefillCacheHit

    # A dummy's DecodeOnly override wins over a stale real state.
    bridge(runner, metadata, get_dsa_dummy_attention_state())
    assert metadata.attn_state is AscendAttentionState.DecodeOnly
    assert _cp_has_prefill()(metadata.attn_state) is False

    # This is the pre-fix failure mode: None would select CP's prefill path.
    assert _cp_has_prefill()(None) is True

    non_dsa_metadata = SimpleNamespace(attn_state="unchanged")
    bridge(SimpleNamespace(use_compress=False), non_dsa_metadata, None)
    assert non_dsa_metadata.attn_state == "unchanged"


def test_dsa_spec_and_dummy_states_use_common_metadata_semantics() -> None:
    # Retain rc1's classifier behavior for future callers without enabling
    # speculative DSA in FL: MTP remains SpecDecoding while a non-MTP draft
    # stores ChunkedPrefill before metadata construction.
    assert classify_dsa_attention_state([5], [1], [1], False, "mtp") is (
        AscendAttentionState.SpecDecoding
    )
    assert classify_dsa_attention_state([5], [2], [1], False, "other") is (
        AscendAttentionState.ChunkedPrefill
    )

    # Capture and idle dummy metadata are explicitly DecodeOnly, independent
    # of the previous real batch and independent of graph-capture mode.
    dummy_state = get_dsa_dummy_attention_state()
    assert dummy_state is AscendAttentionState.DecodeOnly
    assert _cp_has_prefill()(dummy_state) is False
    with pytest.raises(NotImplementedError, match="compressed DSA"):
        get_dsa_dummy_attention_state(create_mixed_batch=True)
