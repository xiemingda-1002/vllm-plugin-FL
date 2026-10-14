"""Runner-to-FIA regression for cache-aware ordinary Ascend attention."""
from __future__ import annotations

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.v1.worker.ubatch_utils import UBatchSlice, split_attn_metadata

from vllm_fl.attention.ascend.attention import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)
from vllm_fl.attention.ascend.utils import (
    AscendCommonAttentionMetadata,
    classify_ascend_attention_state,
    get_ascend_dummy_attention_state,
)

ROOT = Path(__file__).resolve().parents[3]


def _runner_method(name):
    # Exercise the production method without constructing the full NPU runner.
    tree = ast.parse((ROOT / "vllm_fl/worker/model_runner.py").read_text())
    method = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    annotations = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(body=[annotations, method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "CommonAttentionMetadata": object, "Any": Any}
    exec(compile(module, "ascend_runner_phase_bridge", "exec"), namespace)
    return namespace[name]


def _runner(computed, scheduled, chunked=True, method=None):
    runner = SimpleNamespace(
        use_ascend_attention=True,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked),
        speculative_config=SimpleNamespace(method=method) if method else None,
        _ascend_num_valid_tokens=np.array(scheduled),
        _ascend_attn_state=classify_ascend_attention_state(
            computed, scheduled, scheduled, chunked, method
        ),
    )
    bridge = _runner_method("_set_ascend_common_attention_state")
    runner._set_ascend_common_attention_state = lambda cm, phase: bridge(
        runner, cm, phase
    )
    return runner


def _common(computed, scheduled):
    ends = torch.tensor(np.cumsum([0] + scheduled), dtype=torch.int32)
    seq_lens = torch.tensor(np.array(computed) + scheduled, dtype=torch.int32)
    return AscendCommonAttentionMetadata(
        query_start_loc=ends,
        query_start_loc_cpu=ends,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens,
        _num_computed_tokens_cpu=torch.tensor(computed, dtype=torch.int32),
        seq_lens_cpu_upper_bound=seq_lens,
        num_reqs=len(computed),
        num_actual_tokens=sum(scheduled),
        max_query_len=max(scheduled),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=torch.ones((len(computed), 512), dtype=torch.int32),
        slot_mapping=torch.arange(sum(scheduled)),
    )


@pytest.fixture
def builder(monkeypatch):
    # Building on CPU should not require a host accelerator or allocate masks.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    result = object.__new__(AscendAttentionMetadataBuilder)
    result.device = torch.device("cpu")
    result.model_config = SimpleNamespace(runner_type="generate", use_mla=False)
    result.decode_threshold = 1
    monkeypatch.setattr(result, "_make_attention_mask", lambda state: "splitfuse-mask")
    return result


def _fia():
    impl = object.__new__(AscendAttentionBackendImpl)
    impl.key_cache = torch.empty((512, 128, 1, 1))
    impl.value_cache = torch.empty_like(impl.key_cache)
    return impl


def test_four_16k_chunks_select_all_accumulated_kv(builder):
    impl = _fia()
    raw_key, raw_value = torch.empty((16384, 1, 1)), torch.empty((16384, 1, 1))
    for chunk in range(4):
        computed = [chunk * 16384]
        runner = _runner(computed, [16384])
        common = _common(computed, [16384])
        runner._set_ascend_common_attention_state(common, None)
        metadata = builder.build(0, common)
        key, value, _, blocks, lengths = impl._get_fia_params(
            raw_key, raw_value, metadata
        )
        assert lengths == [(chunk + 1) * 16384]
        if chunk == 0:
            assert metadata.attn_state is AscendAttentionState.PrefillNoCache
            assert key is raw_key and value is raw_value and blocks is None
        else:
            assert metadata.attn_state is AscendAttentionState.ChunkedPrefill
            assert key.data_ptr() == impl.key_cache.data_ptr()
            assert value.data_ptr() == impl.value_cache.data_ptr()
            assert blocks is metadata.block_tables


@pytest.mark.parametrize("computed,scheduled,chunked,phase", [
    ([0], [1], True, AscendAttentionState.PrefillNoCache),
    ([32], [1], True, AscendAttentionState.DecodeOnly),
    ([32], [7], False, AscendAttentionState.PrefillCacheHit),
    ([32, 0], [1, 7], True, AscendAttentionState.ChunkedPrefill),
])
def test_ordinary_builder_consumes_runner_phase(
    builder, computed, scheduled, chunked, phase
):
    runner = _runner(computed, scheduled, chunked)
    common = _common(computed, scheduled)
    runner._set_ascend_common_attention_state(common, None)
    metadata = builder.build(0, common)
    assert metadata.attn_state is phase
    assert metadata.attn_mask == "splitfuse-mask"
    params = _fia()._get_fia_params(
        torch.empty((sum(scheduled), 1, 1)),
        torch.empty((sum(scheduled), 1, 1)),
        metadata,
    )
    expected = (
        np.cumsum(scheduled).tolist()
        if phase is AscendAttentionState.PrefillNoCache
        else (np.array(computed) + scheduled).tolist()
    )
    assert params[-1] == expected


def test_dummy_after_cached_prefill_overrides_without_mutating_real_phase(builder):
    runner = _runner([32], [7], False)
    common = _common([32], [1])
    dummy = get_ascend_dummy_attention_state()
    runner._set_ascend_common_attention_state(common, dummy)
    assert builder.build(0, common).attn_state is AscendAttentionState.DecodeOnly
    assert (
        builder.build_for_cudagraph_capture(common).attn_state
        is AscendAttentionState.DecodeOnly
    )
    assert runner._ascend_attn_state is AscendAttentionState.PrefillCacheHit
    assert (
        get_ascend_dummy_attention_state(speculative_method="mtp", use_mla=True)
        is AscendAttentionState.SpecDecoding
    )
    assert (
        get_ascend_dummy_attention_state(speculative_method="mtp")
        is AscendAttentionState.ChunkedPrefill
    )


def test_non_ascend_phase_bridge_is_noop():
    common = SimpleNamespace(attn_state="unchanged")
    _runner_method("_set_ascend_common_attention_state")(
        SimpleNamespace(use_ascend_attention=False), common, None
    )
    assert common.attn_state == "unchanged"


def test_unpadded_and_group_copy_preserve_phase(builder):
    common = _common([32, 0], [4, 1])
    runner = _runner([32, 0], [4, 1])
    runner._set_ascend_common_attention_state(common, None)
    unpadded = common.unpadded(4, 1)
    assert unpadded.attn_state is common.attn_state
    assert builder.build(0, copy(unpadded)).seq_lens_list == [36]


def test_token_split_of_initial_prompt_uses_history_in_second_ubatch(builder):
    runner = _runner([0], [16])
    parent = _common([0], [16])
    runner._set_ascend_common_attention_state(parent, None)
    slices = [
        UBatchSlice(slice(0, 1), slice(0, 8)),
        UBatchSlice(slice(0, 1), slice(8, 16)),
    ]
    subbatches = split_attn_metadata(slices, parent)
    phases, kv_lengths = [], []
    for ubatch_slice, common in zip(slices, subbatches):
        _runner_method("_set_ascend_ubatch_attention_state")(
            runner, common, parent, ubatch_slice, 1, None
        )
        metadata = builder.build(0, common)
        phases.append(metadata.attn_state)
        kv_lengths.append(_fia()._get_fia_params(
            torch.empty((8, 1, 1)), torch.empty((8, 1, 1)), metadata
        )[-1])
    assert phases == [
        AscendAttentionState.PrefillNoCache, AscendAttentionState.ChunkedPrefill
    ]
    assert kv_lengths == [[8], [16]]


def test_ubatch_valid_draft_prefix_and_dummy_override():
    runner = _runner([5], [4], method="other")
    runner._ascend_num_valid_tokens = np.array([1])
    parent = _common([5], [4])
    runner._set_ascend_common_attention_state(parent, None)
    ubatch_slice = UBatchSlice(slice(0, 1), slice(0, 2))
    common = split_attn_metadata([ubatch_slice], parent)[0]
    _runner_method("_set_ascend_ubatch_attention_state")(
        runner, common, parent, ubatch_slice, 1, None
    )
    assert common.attn_state is AscendAttentionState.ChunkedPrefill
    _runner_method("_set_ascend_ubatch_attention_state")(
        runner, common, parent, ubatch_slice, 1, AscendAttentionState.DecodeOnly
    )
    assert common.attn_state is AscendAttentionState.DecodeOnly
