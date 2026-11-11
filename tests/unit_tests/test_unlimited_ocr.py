# Copyright (c) 2026 BAAI. All rights reserved.
"""Unlimited OCR regression tests."""

import ast
import importlib.util
import json
import logging
import math
import os
import sys
import unittest
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


# Grouped R-SWA masks and attention routing


ATTENTION_DIR = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl"
SPEC = importlib.util.spec_from_file_location(
    "rswa_grouped_standard_mask", ATTENTION_DIR / "attention_mask.py"
)
mask_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mask_module)
ENV = "VLLM_FL_RSWA_MASK_MAX_MIB"


def load_helpers():
    """Compile only the pure helpers; backend imports must never run on CPU."""
    source = ast.parse((ATTENTION_DIR / "attention.py").read_text(encoding="utf-8"))
    names = {
        "AscendAttentionState",
        "AscendMetadata",
        "RswaAttentionGroup",
        "_build_grouped_rswa_masks",
        "_rswa_mask_request_lengths",
        "_run_grouped_rswa_fia",
    }
    nodes = [
        node
        for node in source.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    builder = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AscendAttentionMetadataBuilder"
    )
    nodes.append(
        next(
            node
            for node in builder.body
            if isinstance(node, ast.FunctionDef) and node.name == "build"
        )
    )
    prelude = ast.parse(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\nfrom enum import Enum\nfrom typing import *\nimport torch\n"
        "_SWA_INT_MAX = 2147483647\n"
    ).body
    module = ModuleType("rswa_grouped_helpers_under_test")
    # dataclass resolves annotations through sys.modules on recent Python.
    sys.modules[module.__name__] = module
    module.__dict__.update(os=os, logger=logging.getLogger(__name__))
    exec(
        compile(
            ast.Module(body=prelude + nodes, type_ignores=[]),
            str(ATTENTION_DIR / "attention.py"),
            "exec",
        ),
        module.__dict__,
    )
    return module


helpers = load_helpers()


class TestGroupedRswaMasks(unittest.TestCase):
    def setUp(self):
        self.builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))
        self.environment = mock.patch.dict(
            os.environ, {"VLLM_FL_RSWA_MASK_MAX_MIB": "512"}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def build(
        self,
        prefix,
        seq,
        query,
        decodes,
        decode_tokens=None,
        kv=32,
        window=3,
        builder=None,
    ):
        return helpers._build_grouped_rswa_masks(
            self.builder if builder is None else builder,
            torch.tensor(prefix, dtype=torch.int64),
            torch.tensor(seq, dtype=torch.int64),
            torch.tensor(query, dtype=torch.int64),
            decodes,
            decodes if decode_tokens is None else decode_tokens,
            kv,
            window,
        )

    def test_masks_match_full_mask_and_scalar_positions(self):
        cases = [
            ([0, 2, 12, 0], [13, 8, 5, 9], [1, 1, 3, 5], 2, 3),
            ([0, 3, 2], [7, 9, 13], [1, 4, 5], 1, 1),
            ([0, 3, 2], [7, 9, 13], [1, 4, 5], 1, 32),
        ]
        for prefix, seq, query, decodes, window in cases:
            with self.subTest(query=query, window=window):
                groups = self.build(prefix, seq, query, decodes, window=window)
                self.assertEqual(len(groups), 2)
                whole = self.builder.get_rswa_attn_mask(
                    torch.tensor(prefix),
                    torch.tensor(seq),
                    torch.tensor(query),
                    32,
                    max(query),
                    window,
                )
                token_start = 0
                for group in groups:
                    start, end = group.request_start, group.request_end
                    local_q = query[start:end]
                    self.assertEqual(group.token_start, token_start)
                    token_start += sum(local_q)
                    self.assertEqual(group.token_end, token_start)
                    self.assertEqual(
                        group.actual_seq_lengths_q,
                        [sum(local_q[: i + 1]) for i in range(len(local_q))],
                    )
                    self.assertEqual(group.attn_mask.dtype, torch.int8)
                    self.assertTrue(group.attn_mask.is_contiguous())
                    self.assertEqual(
                        tuple(group.attn_mask.shape), (end - start, 1, max(local_q), 32)
                    )
                    for local, request in enumerate(range(start, end)):
                        qlen = query[request]
                        self.assertTrue(
                            torch.equal(
                                group.attn_mask[local, 0, :qlen],
                                whole[request, 0, :qlen],
                            )
                        )
                        for offset in range(max(local_q)):
                            position = seq[request] - qlen + offset
                            expected = [
                                int(
                                    not (
                                        offset < qlen
                                        and key < seq[request]
                                        and key <= position
                                        and (
                                            key < prefix[request]
                                            or key > position - window
                                        )
                                    )
                                )
                                for key in range(32)
                            ]
                            self.assertEqual(
                                group.attn_mask[local, 0, offset].tolist(), expected
                            )
                self.assertEqual(
                    (groups[0].request_start, groups[0].request_end), (0, decodes)
                )
                self.assertEqual(
                    (groups[1].request_start, groups[1].request_end),
                    (decodes, len(query)),
                )

    def test_invalid_or_nonmixed_metadata_falls_back_before_mask_allocation(self):
        cases = [
            ([], [], [], 0, 0),
            ([0, 0], [2, 2], [1, 1], 2, 2),
            ([0, 0], [2, 2], [2, 2], 0, 0),
            ([0, 0, 0], [2, 2, 2], [1, 0, 2], 1, 1),
            ([0, 0, 0], [2, 2, 2], [1, 2, 1], 1, 1),
            ([0, 0], [2, 2], [2, 1], 1, 1),
            ([0, 0], [2, 2], [1, 2], 1, 2),
            ([0], [2, 2], [1, 2], 1, 1),
            ([0, 0], [2], [1, 2], 1, 1),
            ([0, 0], [2, 2], [1, 2], -1, 0),
            ([0, 0], [2, 2], [1, 2], 3, 1),
        ]
        for prefix, seq, query, decodes, tokens in cases:
            with self.subTest(query=query, decodes=decodes, tokens=tokens):
                builder = mock.Mock()
                self.assertIsNone(
                    self.build(prefix, seq, query, decodes, tokens, builder=builder)
                )
                builder.get_rswa_attn_mask.assert_not_called()

    def test_63_decode_one_prefill_allocates_only_group_masks(self):
        # Record realistic production shapes without allocating GiB of CPU RAM.
        shapes = []
        builder = mock.Mock()

        def record_shape(**kwargs):
            shapes.append(
                (
                    len(kwargs["query_lens"]),
                    1,
                    kwargs["max_query_len"],
                    kwargs["max_kv_len"],
                )
            )
            return torch.empty(0, dtype=torch.int8)

        builder.get_rswa_attn_mask.side_effect = record_shape
        groups = self.build(
            [0] * 64, [1024] * 64, [1] * 63 + [961], 63, kv=32768, builder=builder
        )
        self.assertEqual(builder.get_rswa_attn_mask.call_count, 2)
        self.assertEqual(shapes, [(63, 1, 1, 32768), (1, 1, 961, 32768)])
        allocated = sum(batch * qmax * kv for batch, _, qmax, kv in shapes)
        self.assertEqual(allocated, (63 + 961) * 32768)
        self.assertLess(allocated * 60, 64 * 961 * 32768)
        self.assertEqual(groups[0].actual_seq_lengths_q, list(range(1, 64)))
        self.assertEqual(groups[1].actual_seq_lengths_q, [961])
        self.assertEqual((groups[1].token_start, groups[1].token_end), (63, 1024))


class TestGroupedRswaFia(unittest.TestCase):
    def test_fia_slices_requests_and_tokens_and_scatter_output(self):
        builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))
        groups = helpers._build_grouped_rswa_masks(
            builder,
            torch.tensor([0, 2, 1, 0]),
            torch.tensor([8, 9, 10, 12]),
            torch.tensor([1, 1, 3, 5]),
            2,
            2,
            16,
            3,
        )
        query = torch.arange(10 * 2 * 4, dtype=torch.float32).reshape(10, 2, 4)
        key, value = torch.randn(4, 8, 1, 4), torch.randn(4, 8, 1, 4)
        # Force a noncontiguous source so the FIA contiguity guarantee is tested.
        block_table = torch.arange(24, dtype=torch.int32).reshape(6, 4).t()
        kv_lengths = [8, 9, 10, 12]
        output = torch.full((12, 2, 4), -999.0)
        calls = []

        def fake_fia(**kwargs):
            calls.append(kwargs)
            return kwargs["query"] * 2 + 7, None

        result = helpers._run_grouped_rswa_fia(
            fake_fia,
            groups,
            query,
            key,
            value,
            block_table,
            kv_lengths,
            8,
            2,
            1,
            4,
            0.5,
            output,
        )
        self.assertIs(result, output)
        self.assertTrue(torch.equal(output[:10], query * 2 + 7))
        self.assertTrue(torch.all(output[10:] == -999))
        self.assertEqual(len(calls), 2)
        for group, call in zip(groups, calls):
            self.assertIs(call["key"], key)
            self.assertIs(call["value"], value)
            self.assertTrue(
                torch.equal(call["query"], query[group.token_start : group.token_end])
            )
            self.assertTrue(
                torch.equal(
                    call["block_table"],
                    block_table[group.request_start : group.request_end],
                )
            )
            self.assertTrue(call["block_table"].is_contiguous())
            self.assertEqual(call["actual_seq_lengths"], group.actual_seq_lengths_q)
            self.assertEqual(
                call["actual_seq_lengths_kv"],
                kv_lengths[group.request_start : group.request_end],
            )
            self.assertIs(call["atten_mask"], group.attn_mask)
            for name, expected in {
                "input_layout": "TND",
                "block_size": 8,
                "num_heads": 2,
                "num_key_value_heads": 1,
                "scale": 0.5,
                "sparse_mode": 0,
                "pre_tokens": 2147483647,
                "next_tokens": 2147483647,
            }.items():
                self.assertEqual(call[name], expected)


class TestRswaLargeMask(unittest.TestCase):
    def setUp(self):
        self.builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))

    def build(self, prefix, seq, query, kv, qmax, window, limit=None):
        # Isolate from a developer's configured opt-in value.
        with mock.patch.dict(os.environ):
            os.environ.pop(ENV, None)
            if limit is not None:
                os.environ[ENV] = str(limit)
            return self.builder.get_rswa_attn_mask(
                torch.tensor(prefix, dtype=torch.int64),
                torch.tensor(seq, dtype=torch.int64),
                torch.tensor(query, dtype=torch.int64),
                kv,
                qmax,
                window,
            )

    def reference(self, prefix, seq, query, kv, qmax, window):
        result = torch.ones((len(prefix), 1, qmax, kv), dtype=torch.int8)
        for request in range(len(prefix)):
            for offset in range(query[request]):
                position = seq[request] - query[request] + offset
                for key in range(seq[request]):
                    if key <= position and (
                        key < prefix[request] or key > position - window
                    ):
                        result[request, 0, offset, key] = 0
        return result

    def test_opt_in_matches_default_and_scalar_reference(self):
        cases = [
            # Early prefill: full prompt prefix exceeds processed KV length.
            ([12], [5], [3], 17, 5, 3),
            # Decode window, padded queries, and an entirely padded request.
            ([2, 0, 5], [13, 8, 5], [1, 3, 0], 17, 5, 3),
            # Window edges and a window wider than the available sequence.
            ([0, 3], [7, 9], [4, 5], 17, 5, 1),
            ([0, 3], [7, 9], [4, 5], 17, 5, 32),
        ]
        for args in cases:
            with self.subTest(args=args):
                expected = self.reference(*args)
                default = self.build(*args)
                with mock.patch.object(mask_module, "_RSWA_MASK_CHUNK_ELEMENTS", 64):
                    opted_in = self.build(*args, limit=1024)
                self.assertTrue(torch.equal(default, expected))
                self.assertTrue(torch.equal(opted_in, expected))
                self.assertEqual(opted_in.dtype, torch.int8)
                self.assertTrue(opted_in.is_contiguous())

    def test_no_requests_preserves_four_dimensional_shape(self):
        for limit in (None, 1024, 8192):
            with self.subTest(limit=limit):
                result = self.build([], [], [], 17, 5, 3, limit=limit)
                self.assertEqual(tuple(result.shape), (0, 1, 5, 17))
                self.assertEqual(result.dtype, torch.int8)

    def test_default_limit_rejects_before_tensor_assertions_or_allocation(self):
        with (
            mock.patch.object(torch, "_assert", side_effect=AssertionError("too late")),
            mock.patch.object(torch, "arange", side_effect=AssertionError("allocated")),
            mock.patch.object(torch, "empty", side_effect=AssertionError("allocated")),
            self.assertRaisesRegex(ValueError, "configured limit=512 MiB"),
        ):
            self.build([0], [1], [1], 32768, 16385, 1)

    def test_ocr_default_capacity_and_explicit_override(self):
        self.builder.ocr_mode = True
        with (
            mock.patch.object(
                torch, "empty", side_effect=RuntimeError("allocation reached")
            ),
            self.assertRaisesRegex(RuntimeError, "allocation reached"),
        ):
            self.build([1] * 64, [1] * 64, [1] * 64, 32768, 1024, 128)
        with self.assertRaisesRegex(ValueError, "configured limit=1 MiB"):
            self.build([0], [1], [1], 1024, 1025, 1, limit=1)

    def test_configured_limit_is_enforced(self):
        # Just above 1 MiB, without constructing a large output tensor.
        with (
            mock.patch.object(torch, "_assert", side_effect=AssertionError("too late")),
            self.assertRaisesRegex(ValueError, "configured limit=1 MiB"),
        ):
            self.build([0], [1], [1], 1024, 1025, 1, limit=1)

    def test_64_request_capacity_reaches_allocation_without_allocating_gib(self):
        with (
            mock.patch.object(
                torch, "empty", side_effect=RuntimeError("allocation reached")
            ) as allocate,
            self.assertRaisesRegex(RuntimeError, "allocation reached"),
        ):
            self.build([1] * 64, [1] * 64, [1] * 64, 32768, 1024, 128, limit=2048)
        self.assertEqual(allocate.call_args.args[0], (64, 1, 1024, 32768))

    def test_invalid_environment_limit(self):
        for value in ("", "no", "1.5", "0", "-1", "8193", " 512", "+512", "虏"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "integer from 1 to 8192"),
            ):
                self.build([0], [1], [1], 1, 1, 1, limit=value)


state = helpers.AscendAttentionState

cases = [
    ("OCR default enabled", None, state.ChunkedPrefill, None, -1, None, 10, True),
    (
        "ordinary default disabled",
        None,
        state.ChunkedPrefill,
        None,
        -1,
        None,
        10,
        False,
    ),
    ("mixed enabled", "1", state.ChunkedPrefill, None, -1, None, 10, True),
    ("explicit disabled", "0", state.ChunkedPrefill, None, -1, None, 10, False),
    ("pure decode", "1", state.DecodeOnly, None, -1, None, 4, False),
    ("unpaged prefill", "1", state.PrefillNoCache, None, -1, None, 10, False),
    ("speculative fallback", "1", state.ChunkedPrefill, object(), -1, None, 10, False),
    ("graph padding fallback", "1", state.ChunkedPrefill, None, 10, None, 10, False),
    ("request padding fallback", "1", state.ChunkedPrefill, None, -1, 3, 10, False),
    ("token padding fallback", "1", state.ChunkedPrefill, None, -1, None, 9, False),
]


@pytest.mark.parametrize(
    "name,enabled,phase,speculative,graph_pad,actual_reqs,total,grouped",
    cases,
    ids=[case[0] for case in cases],
)
def test_metadata_grouped_routing(
    name, enabled, phase, speculative, graph_pad, actual_reqs, total, grouped
):
    lengths = [1, 1, 1, 1] if phase == state.DecodeOnly else [1, 1, 3, 5]
    starts = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()))
    common = SimpleNamespace(
        num_reqs=4,
        num_actual_tokens=total,
        query_start_loc_cpu=starts,
        block_table_tensor=torch.arange(16).view(4, 4),
        _seq_lens_cpu=torch.tensor([13, 8, 5, 9]),
        slot_mapping=torch.arange(total),
        attn_state=phase,
        rswa_prefix_lens=torch.tensor([0, 2, 12, 0]),
        rswa_num_actual_reqs=actual_reqs,
        graph_pad_size=graph_pad,
        causal=True,
        max_query_len=max(lengths),
    )
    calls = []
    real_builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))

    def build_mask(**kwargs):
        calls.append((kwargs["query_lens"].numel(), kwargs["max_query_len"]))
        return real_builder.get_rswa_attn_mask(**kwargs)

    fake_builder = SimpleNamespace(get_rswa_attn_mask=build_mask)
    decodes = 4 if phase == state.DecodeOnly else 2
    self = SimpleNamespace(
        device=torch.device("cpu"),
        model_config=SimpleNamespace(
            rswa_window=3,
            max_model_len=32,
            runner_type="generate",
            hf_config=SimpleNamespace(
                model_type="unlimited-ocr" if name.startswith("OCR") else "ordinary"
            ),
        ),
        vllm_config=SimpleNamespace(quant_config=None, speculative_config=speculative),
        _make_attention_mask=lambda phase: None,
        _get_mask_builder=lambda: fake_builder,
        _split_decodes_and_prefills=lambda metadata: (
            decodes,
            4 - decodes,
            decodes,
            total - decodes,
        ),
    )
    environment = dict(os.environ)
    environment["VLLM_FL_OCR_TORCH_NPU"] = "1"
    environment.pop("VLLM_FL_RSWA_GROUPED_MASK", None)
    if enabled is not None:
        environment["VLLM_FL_RSWA_GROUPED_MASK"] = enabled
    environment["VLLM_FL_RSWA_MASK_MAX_MIB"] = "2048"
    with (
        mock.patch.dict(
            os.environ,
            environment,
            clear=True,
        ),
        mock.patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
    ):
        result = helpers.build(self, 0, common)
    assert (result.rswa_groups is not None) == grouped, name
    if grouped:
        assert result.attn_mask is None and calls == [(2, 1), (2, 5)], (name, calls)
    else:
        assert len(calls) == 1 and result.attn_mask is not None, (name, calls)


def test_full_padding_uses_explicit_real_request_boundary():
    prefix, query = torch.tensor([2, 0, 99]), torch.tensor([1, 1, 1])
    real_prefix, real_query = helpers._rswa_mask_request_lengths(prefix, query, 2)
    assert real_prefix.tolist() == [2, 0, 0]
    assert real_query.tolist() == [1, 1, 0]
    assert prefix.tolist() == [2, 0, 99]

    builder = mask_module.AttentionMaskBuilder(torch.device("cpu"))
    attention_mask = builder.get_rswa_attn_mask(
        real_prefix, torch.tensor([8, 5, 0]), real_query, 32, 1, 4
    )
    assert torch.all(attention_mask[2] == 1)


# Grouped MoE router


def load_router():
    path = ROOT / "vllm_fl/ops/fused_moe/router.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_fl_grouped_topk"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[function.name]


def original(logits, topk, renormalize, groups, selected, scoring, scaling):
    scores = (
        torch.softmax(logits, dim=-1) if scoring == "softmax" else torch.sigmoid(logits)
    )
    group_size = scores.size(-1) // groups
    group_scores = scores.view(-1, groups, group_size).amax(dim=-1)
    _, selected_groups = torch.topk(group_scores, selected, dim=-1, sorted=False)
    mask = torch.zeros_like(scores)
    for i in range(selected):
        start = selected_groups[:, i] * group_size
        for j in range(group_size):
            mask.scatter_(1, (start + j).unsqueeze(1), 1.0)
    weights, ids = torch.topk(scores * mask, topk, dim=-1, sorted=False)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    if scaling != 1.0:
        weights = weights * scaling
    return weights.float(), ids.int()


@pytest.mark.parametrize("groups,selected", [(1, 1), (2, 2), (8, 8), (4, 2)])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("scaling", [1.0, 2.5])
@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("case", ["random", "near_tie", "tie"])
def test_matches_original(groups, selected, renormalize, scaling, scoring, case):
    logits = torch.randn(64, 64, generator=torch.Generator().manual_seed(17))
    if case == "near_tie":
        logits *= 1e-6
    elif case == "tie":
        logits.zero_()
    kwargs = dict(
        topk=6,
        renormalize=renormalize,
        num_expert_group=groups,
        topk_group=selected,
        scoring_func=scoring,
        routed_scaling_factor=scaling,
    )
    actual = load_router()(torch.empty(64, 1), logits, **kwargs)
    expected = original(logits, 6, renormalize, groups, selected, scoring, scaling)
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.int32
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)


def test_all_groups_does_not_scatter(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("all-group routing must not scatter")

    monkeypatch.setattr(torch.Tensor, "scatter_", forbidden)
    logits = torch.zeros(64, 64)
    load_router()(torch.empty(64, 1), logits, 6, True, 1, 1)


# Batched n-gram masking


class Matrix:
    device = "cpu"

    def __init__(self, rows=4, vocab=16):
        self.ndim = 2
        self.shape = (rows, vocab)
        self.values = [[float(i) for i in range(vocab)] for _ in range(rows)]
        self.assignments = 0

    def __getitem__(self, index):
        return Row(self.values[index])

    def __setitem__(self, indices, value):
        self.assignments += 1
        for row, token in zip(*indices):
            self.values[row][token] = value


class Row:
    def __init__(self, values):
        self.values = values

    def __setitem__(self, tokens, value):
        for token in tokens:
            self.values[token] = value


def core_definitions():
    configured = os.getenv("VLLM_FL_TEST_CORE_OCR")
    if configured:
        source = Path(configured)
    else:
        spec = importlib.util.find_spec("vllm")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if not locations:
            pytest.skip("Set VLLM_FL_TEST_CORE_OCR to the target core deepseek_ocr.py")
        source = Path(locations[0]) / "model_executor/models/deepseek_ocr.py"
    if not source.is_file():
        pytest.skip("Set VLLM_FL_TEST_CORE_OCR to the target core deepseek_ocr.py")
    adapter = source.parents[2] / "v1/sample/logits_processor/__init__.py"
    definitions = []
    for file, names in (
        (adapter, {"AdapterLogitsProcessor"}),
        (source, {"NoRepeatNGramLogitsProcessor", "NGramPerReqLogitsProcessor"}),
    ):
        definitions.extend(
            n
            for n in ast.parse(file.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name in names
        )
    tree = ast.Module(body=definitions, type_ignores=[])
    namespace = dict(
        LogitsProcessor=object,
        abstractmethod=lambda f: f,
        partial=partial,
        torch=SimpleNamespace(Tensor=object),
    )
    # Future annotations leave the model's unrelated dependencies unimported.
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(tree),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace


@pytest.fixture
def batch(monkeypatch):
    core = core_definitions()
    calls = []

    def tensor(data, **kwargs):
        calls.append((data, kwargs))
        return data

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(Tensor=object, tensor=tensor, int64="int64"),
    )
    module_name = "vllm.model_executor.models.deepseek_ocr"
    core_module = ModuleType(module_name)
    core_module.__dict__.update(core)
    monkeypatch.setitem(sys.modules, module_name, core_module)
    path = ROOT / "vllm_fl/models/ocr_ngram.py"
    spec = importlib.util.spec_from_file_location("ocr_ngram_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("VLLM_FL_OCR_BATCH_NGRAM", raising=False)
    processor = module.NGramPerReqLogitsProcessorFL(None, None, False)
    return processor, core, calls


def check(batch, states):
    processor, core, calls = batch
    processor.req_info = states
    expected, actual = Matrix(), Matrix()
    reference = core["NGramPerReqLogitsProcessor"](None, None, False)
    reference.req_info = states
    reference.apply(expected)
    assert processor.apply(actual) is actual
    assert actual.values == expected.values
    assert len(calls) == actual.assignments <= 1
    if calls:
        assert calls[0][1] == {"dtype": "int64", "device": "cpu"}
        assert len(calls[0][0]) == 2
    return actual


@pytest.mark.parametrize(
    "n,window,ids,white",
    [
        (2, 35, [1, 2, 1, 3, 1], set()),
        (2, 128, [1, 2, 1, 3, 1], {2}),
        (2, 3, [1, 2, 1, 3, 1], set()),
        (2, 4, [1, 2, 1, 3, 1], set()),
        (1, 35, [1, 1, 1], set()),
        (4, 2, [1, 1, 1, 1, 1], set()),
        (3, 35, [1], set()),
        (2, 35, [1, 2, 3], set()),
        (2, 35, [1, 2, 1], {2}),
        (2, 35, [], set()),
    ],
)
def test_exact_core_scan(batch, n, window, ids, white):
    core = batch[1]
    check(
        batch, {0: partial(core["NoRepeatNGramLogitsProcessor"](n, window, white), ids)}
    )


def test_batch_submission_and_live_moved_swapped_dropped_refs(batch):
    processor, core, calls = batch
    make = lambda ids: partial(core["NoRepeatNGramLogitsProcessor"](2, 35), ids)
    first, second = [1, 2, 1], [3, 4, 3]
    a, b = make(first), make(second)
    check(batch, {0: a, 1: b})
    assert len(calls) == 1
    calls.clear()
    first.extend([5, 1])
    check(batch, {0: b, 3: a})  # moved/swapped rows, same live references
    calls.clear()
    check(batch, {3: a})  # dropped request
    assert (
        processor.update_state.__func__
        is core["NGramPerReqLogitsProcessor"].update_state
    )
    assert (
        processor._new_state.__func__ is core["NGramPerReqLogitsProcessor"]._new_state
    )


def test_no_ngram_no_device_tensor(batch):
    assert check(batch, {}).assignments == 0


@pytest.mark.parametrize("unsupported", ["function", "keywords", "subclass"])
def test_whole_batch_fallback(batch, unsupported):
    processor, core, calls = batch
    cls = core["NoRepeatNGramLogitsProcessor"]
    valid = partial(cls(2, 35), [1, 2, 1])
    if unsupported == "function":
        state = partial(lambda logits: logits)
    elif unsupported == "keywords":
        state = partial(cls(2, 35), output_ids=[3, 4, 3])
        processor.req_info = {0: valid, 1: state}
        with pytest.raises(TypeError, match="multiple values"):
            processor.apply(Matrix())
        assert calls == []
        return
    else:

        class Custom(cls):
            pass

        state = partial(Custom(2, 35), [3, 4, 3])
    check(batch, {0: valid, 1: state})
    assert calls == []


def test_explicitly_disabled(batch, monkeypatch):
    monkeypatch.setenv("VLLM_FL_OCR_BATCH_NGRAM", "0")
    core = batch[1]
    check(batch, {0: partial(core["NoRepeatNGramLogitsProcessor"](2, 35), [1, 2, 1])})
    assert batch[2] == []


def test_device_api_error_propagates(batch, monkeypatch):
    processor, core, _ = batch
    processor.req_info = {
        0: partial(core["NoRepeatNGramLogitsProcessor"](2, 35), [1, 2, 1])
    }

    def fail(*args, **kwargs):
        raise RuntimeError("tensor API failure")

    monkeypatch.setattr(sys.modules["torch"], "tensor", fail)
    with pytest.raises(RuntimeError, match="tensor API failure"):
        processor.apply(Matrix())


def test_real_torch_noncontiguous(monkeypatch):
    torch = pytest.importorskip("torch")
    core = core_definitions()
    name = "vllm.model_executor.models.deepseek_ocr"
    module = ModuleType(name)
    module.__dict__.update(core)
    monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "vllm_fl/models/ocr_ngram.py"
    spec = importlib.util.spec_from_file_location("ocr_ngram_real_test", path)
    target = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target)
    monkeypatch.setenv("VLLM_FL_OCR_BATCH_NGRAM", "1")
    processor = target.NGramPerReqLogitsProcessorFL(None, None, False)
    processor.req_info = {
        0: partial(core["NoRepeatNGramLogitsProcessor"](2, 35), [1, 2, 1])
    }
    logits = torch.arange(64, dtype=torch.float32).reshape(16, 4).t()
    assert not logits.is_contiguous()
    expected = logits.clone()
    reference = core["NGramPerReqLogitsProcessor"](None, None, False)
    reference.req_info = processor.req_info
    reference.apply(expected)
    assert processor.apply(logits) is logits
    assert torch.equal(logits, expected)


# OCR runtime contracts


@pytest.fixture
def runtime():
    # Load independently for each test to isolate one-time audit state.
    path = ROOT / "vllm_fl/models/ocr_runtime.py"
    spec = importlib.util.spec_from_file_location("ocr_runtime_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Rotary embeddings


def rope_layer(head_size, rotary_dim, neox, device="cpu", dtype=torch.float32):
    phase = torch.arange(19)[:, None] * torch.linspace(0.01, 0.8, rotary_dim // 2)
    return SimpleNamespace(
        head_size=head_size,
        rotary_dim=rotary_dim,
        is_neox_style=neox,
        cos_sin_cache=torch.cat((phase.cos(), phase.sin()), -1).to(device, dtype),
    )


def rope_reference(op, positions, x):
    if x is None:
        return None
    shaped = x.reshape(positions.numel(), -1, op.head_size)
    out = shaped.clone()
    coeff = op.cos_sin_cache[positions.flatten().long()].float()
    c, s = coeff.chunk(2, -1)
    if op.is_neox_style:
        a = shaped[..., : op.rotary_dim // 2].float()
        b = shaped[..., op.rotary_dim // 2 : op.rotary_dim].float()
        out[..., : op.rotary_dim // 2] = a * c[:, None] - b * s[:, None]
        out[..., op.rotary_dim // 2 : op.rotary_dim] = b * c[:, None] + a * s[:, None]
    else:
        a = shaped[..., : op.rotary_dim : 2].float()
        b = shaped[..., 1 : op.rotary_dim : 2].float()
        out[..., : op.rotary_dim : 2] = a * c[:, None] - b * s[:, None]
        out[..., 1 : op.rotary_dim : 2] = b * c[:, None] + a * s[:, None]
    return out.reshape(x.shape)


@pytest.mark.parametrize(
    "neox,rotary_dim,key_heads",
    [
        (True, 128, 2),
        (True, 64, 1),
        (False, 128, 2),
        (False, 64, None),
    ],
)
@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_cpu_reference(runtime, monkeypatch, neox, rotary_dim, key_heads, enabled):
    if enabled is None:
        monkeypatch.delenv("VLLM_FL_OCR_FUSED_ROPE", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_OCR_FUSED_ROPE", enabled)

    def forbidden(*args):
        raise AssertionError("CPU must never dispatch to torch_npu")

    monkeypatch.setattr(runtime, "_ocr_fused_rotary_mul", forbidden)
    op = rope_layer(128, rotary_dim, neox)
    positions = torch.tensor([[7, 1], [7, 3]])
    query = torch.randn(4, 4 * 128)
    key = None if key_heads is None else torch.randn(4, key_heads, 128)
    q_before, k_before = query.clone(), None if key is None else key.clone()
    q, k = runtime._ocr_rotary_forward(op, positions, query, key)
    torch.testing.assert_close(q, rope_reference(op, positions, query))
    torch.testing.assert_close(query, q_before)
    if key is None:
        assert k is None
    else:
        torch.testing.assert_close(k, rope_reference(op, positions, key))
        torch.testing.assert_close(key, k_before)


@pytest.mark.parametrize("neox", [True, False])
def test_public_api_adapter(runtime, monkeypatch, neox):
    monkeypatch.setattr(runtime, "_ocr_can_audit", lambda: True)
    calls = []

    def api(x, c, s, rotary_mode="half", rotate=None):
        calls.append((x, c, s, rotary_mode, rotate))
        assert x.is_contiguous() and c.is_contiguous() and s.is_contiguous()
        return x.clone()

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=api))
    x = torch.randn(3, 2, 128)[..., :64]
    c, s = torch.ones(3, 1, 1, 64), torch.zeros(3, 1, 1, 64)
    result = runtime._ocr_fused_rotary_mul(x, c, s, neox)
    torch.testing.assert_close(result, x)
    assert calls[0][0].shape == (3, 1, 2, 64)
    assert calls[0][3] == ("half" if neox else "interleave")


def test_public_api_error_is_visible(runtime, monkeypatch):
    def unsupported(*args, **kwargs):
        raise RuntimeError("unsupported contract")

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=unsupported)
    )
    with pytest.raises(RuntimeError, match="unsupported contract"):
        runtime._ocr_fused_rotary_mul(
            torch.ones(1, 1, 128),
            torch.ones(1, 1, 1, 128),
            torch.zeros(1, 1, 1, 128),
            True,
        )


def test_fused_audit_follows_success_once(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "_ocr_can_audit", lambda: True)
    evidence = []
    monkeypatch.setattr(runtime, "_OCR_FUSED_ROPE_AUDITED", False)
    monkeypatch.setattr(
        runtime, "_write_audit", lambda stage, data: evidence.append((stage, data))
    )

    def api(x, *args, **kwargs):
        return x.clone()

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=api))
    x = torch.ones(1, 1, 128)
    c, s = torch.ones(1, 1, 1, 128), torch.zeros(1, 1, 1, 128)
    runtime._ocr_fused_rotary_mul(x, c, s, True)
    runtime._ocr_fused_rotary_mul(x, c, s, True)
    assert len(evidence) == 1
    assert evidence[0][0] == "rope-fused"
    assert evidence[0][1]["event"] == "api_returned"


@pytest.mark.parametrize(
    "neox,rotary_dim,key_heads",
    [
        (True, 128, 2),
        (True, 64, 1),
        (False, 128, 2),
        (False, 64, None),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_fused_matches_reference(
    runtime, monkeypatch, neox, rotary_dim, key_heads, dtype
):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU unavailable")
    monkeypatch.setenv("VLLM_FL_OCR_FUSED_ROPE", "1")
    op = rope_layer(128, rotary_dim, neox, "npu", dtype)
    positions = torch.tensor([[7, 1], [7, 3]], device="npu")
    # Strided input exercises adapter contiguity and original shape preservation.
    query = torch.randn(4, 4, 256, device="npu", dtype=dtype)[..., ::2]
    key = (
        None
        if key_heads is None
        else torch.randn(4, key_heads * 128, device="npu", dtype=dtype)
    )
    before = query.clone()
    q, k = runtime._ocr_rotary_forward(op, positions, query, key)
    tolerance = 0.032 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(
        q, rope_reference(op, positions, query), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(query, before)
    if key is None:
        assert k is None
    else:
        torch.testing.assert_close(
            k, rope_reference(op, positions, key), atol=tolerance, rtol=tolerance
        )
    if rotary_dim < op.head_size:
        torch.testing.assert_close(
            q[..., rotary_dim:], query[..., rotary_dim:], atol=0, rtol=0
        )


@pytest.mark.parametrize("compiling,capturing", [(True, False), (False, True)])
def test_audit_skipped_during_compile_or_capture(
    runtime, monkeypatch, compiling, capturing
):
    monkeypatch.setattr(runtime, "_OCR_FUSED_ROPE_AUDITED", False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    capture_checks = []

    def is_capturing():
        capture_checks.append(True)
        return capturing

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=is_capturing),
        raising=False,
    )

    def forbidden_audit(*args):
        raise AssertionError("audit must not run during compile or capture")

    monkeypatch.setattr(runtime, "_write_audit", forbidden_audit)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_rotary_mul=lambda x, *args, **kwargs: x.clone()),
    )
    x = torch.ones(1, 1, 128)
    c, s = torch.ones(1, 1, 1, 128), torch.zeros(1, 1, 1, 128)
    result = runtime._ocr_fused_rotary_mul(x, c, s, True)
    torch.testing.assert_close(result, x)
    assert runtime._OCR_FUSED_ROPE_AUDITED is False
    assert len(capture_checks) == (0 if compiling else 1)


@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("rotary_dim", [4, 8])
@pytest.mark.parametrize("key_heads", [None, 1, 3])
def test_rope_complex_reference(runtime, neox, rotary_dim, key_heads):
    generator = torch.Generator().manual_seed(321)
    positions = torch.tensor([7, 0, 3])
    head_size = 8
    angles = (
        torch.arange(12)[:, None]
        * torch.tensor([0.31, 0.17, 0.07, 0.02][: rotary_dim // 2])[None, :]
    )
    layer = SimpleNamespace(
        head_size=head_size,
        rotary_dim=rotary_dim,
        is_neox_style=neox,
        cos_sin_cache=torch.cat([angles.cos(), angles.sin()], -1),
    )
    query = torch.randn(3, 3 * head_size, generator=generator)
    key = (
        None
        if key_heads is None
        else torch.randn(3, key_heads * head_size, generator=generator)
    )

    def reference(tensor):
        if tensor is None:
            return None
        shaped = tensor.reshape(3, -1, head_size)
        rotary = shaped[..., :rotary_dim]
        if neox:
            real = rotary[..., : rotary_dim // 2]
            imag = rotary[..., rotary_dim // 2 :]
            complex_input = torch.complex(real, imag)
        else:
            complex_input = torch.view_as_complex(
                rotary.reshape(3, -1, rotary_dim // 2, 2).contiguous()
            )
        result = (
            complex_input
            * torch.polar(torch.ones_like(angles[positions]), angles[positions])[
                :, None, :
            ]
        )
        rotated = (
            torch.cat([result.real, result.imag], -1)
            if neox
            else torch.view_as_real(result).flatten(-2)
        )
        return torch.cat([rotated, shaped[..., rotary_dim:]], -1).reshape_as(tensor)

    original = query.clone()
    actual_query, actual_key = runtime._ocr_rotary_forward(layer, positions, query, key)
    torch.testing.assert_close(actual_query, reference(query))
    if key is None:
        assert actual_key is None
    else:
        torch.testing.assert_close(actual_key, reference(key))
    assert torch.equal(query, original)


# SAM attention


class TensorContract:
    def __init__(self, shape=(2, 12, 196, 64), dtype="bf16", device="npu"):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.device = SimpleNamespace(type=device)
        self.contiguous_calls = 0

    def contiguous(self):
        self.contiguous_calls += 1
        return self


@pytest.fixture
def contracts(monkeypatch):
    q, k, v = (TensorContract() for _ in range(3))
    k.device = v.device = q.device
    bias = TensorContract((2, 12, 196, 196))
    bias.device = q.device
    calls = []
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                scaled_dot_product_attention=lambda *a, **kw: "sdpa"
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_fused_infer_attention_score=lambda *a, **kw: (
                calls.append((a, kw)) or "fia",
                None,
            )
        ),
    )
    monkeypatch.delenv("VLLM_FL_OCR_FUSED_SAM", raising=False)
    return q, k, v, bias, calls


@pytest.mark.parametrize("with_bias", [True, False])
def test_public_api_contract(runtime, contracts, monkeypatch, with_bias):
    q, k, v, bias, calls = contracts
    monkeypatch.setattr(runtime, "_ocr_can_audit", lambda: False)
    assert runtime._ocr_sam_attention(q, k, v, bias if with_bias else None) == "fia"
    args, kwargs = calls[0]
    assert args == (q, k, v)
    assert kwargs == dict(
        pse_shift=bias if with_bias else None,
        num_heads=12,
        num_key_value_heads=12,
        input_layout="BNSD",
        scale=64**-0.5,
    )
    assert all(t.contiguous_calls == 1 for t in (q, k, v))
    assert bias.contiguous_calls == int(with_bias)
    assert not runtime._OCR_FUSED_SAM_AUDITED


@pytest.mark.parametrize("reason", ["disabled", "cpu", "fp32", "head", "heads"])
def test_unsupported_contract_falls_back(runtime, contracts, monkeypatch, reason):
    q, k, v, bias, calls = contracts
    if reason == "disabled":
        monkeypatch.setenv("VLLM_FL_OCR_FUSED_SAM", "0")
    elif reason == "cpu":
        q.device.type = "cpu"
    elif reason == "fp32":
        q.dtype = "fp32"
    else:
        q.shape = (2, 12, 196, 32) if reason == "head" else (2, 8, 196, 64)
    assert runtime._ocr_sam_attention(q, k, v, bias) == "sdpa"
    assert calls == []


def test_supported_api_error_propagates(runtime, contracts, monkeypatch):
    q, k, v, bias, _ = contracts

    def fail(*args, **kwargs):
        raise RuntimeError("device API failed")

    monkeypatch.setattr(
        sys.modules["torch_npu"], "npu_fused_infer_attention_score", fail
    )
    with pytest.raises(RuntimeError, match="device API failed"):
        runtime._ocr_sam_attention(q, k, v, bias)


def test_success_audited_once(runtime, contracts, monkeypatch):
    q, k, v, bias, _ = contracts
    events = []
    monkeypatch.setattr(runtime, "_ocr_can_audit", lambda: True)
    monkeypatch.setattr(runtime, "_write_audit", lambda *args: events.append(args))
    runtime._ocr_sam_attention(q, k, v, bias)
    runtime._ocr_sam_attention(q, k, v, bias)
    assert len(events) == 1
    assert events[0][0] == "sam-fused"
    assert events[0][1]["event"] == "api_returned"


@pytest.mark.parametrize("use_rel_pos", [False, True])
def test_cpu_forward_matches_sdpa_geometry(runtime, monkeypatch, use_rel_pos):
    torch.manual_seed(17)
    layer = SimpleNamespace(
        num_heads=3,
        use_rel_pos=use_rel_pos,
        qkv=torch.nn.Linear(24, 72),
        proj=torch.nn.Linear(24, 24),
        rel_pos_h=None,
        rel_pos_w=None,
    )
    # Non-square geometry catches flattening/broadcast mistakes.
    x = torch.randn(2, 2, 3, 24)
    rh = torch.randn(6, 6, 2, 1)
    rw = torch.randn(6, 6, 1, 3)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.deepencoder",
        SimpleNamespace(add_decomposed_rel_pos=lambda *a: (rh, rw)),
    )
    q, k, v = layer.qkv(x).reshape(2, 6, 3, 3, 8).permute(2, 0, 3, 1, 4)
    bias = (rh + rw).reshape(2, 3, 6, 6) if use_rel_pos else None
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    expected = layer.proj(expected.transpose(1, 2).reshape(2, 2, 3, 24))
    torch.testing.assert_close(runtime._ocr_sam_forward(layer, x), expected)


# Slot mapping


@pytest.fixture
def slots_module(runtime, monkeypatch):
    monkeypatch.delenv("VLLM_FL_OCR_TORCH_SLOT_MAPPING", raising=False)
    monkeypatch.setattr(runtime, "ocr_runtime_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_ocr_can_audit", lambda: False)
    monkeypatch.setattr(runtime, "_supported_device", lambda device: True)
    return runtime


def slot_group(noncontiguous=False):
    table = torch.tensor([[8, 3, 12], [19, 2, 5]], dtype=torch.int32)
    if noncontiguous:
        storage = torch.empty((4, 3), dtype=table.dtype)
        storage[::2].copy_(table)
        table = storage[::2]
    return SimpleNamespace(
        block_size=4,
        blocks_per_kv_block=1,
        use_hybrid_blocks=False,
        pcp_world_size=1,
        dcp_world_size=1,
        max_num_batched_tokens=10,
        block_table=SimpleNamespace(gpu=table),
        slot_mapping=SimpleNamespace(gpu=torch.full((12,), 777, dtype=torch.int64)),
    )


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_values_padding_and_storage(slots_module, noncontiguous):
    groups = [slot_group(noncontiguous), slot_group()]
    positions = torch.tensor([0, 3, 4, 7, 8, 11], dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    pointers = [g.slot_mapping.gpu.data_ptr() for g in groups]
    assert slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=groups), 2, requests, positions
    )
    for g, pointer in zip(groups, pointers):
        assert g.slot_mapping.gpu.tolist() == [
            32,
            35,
            12,
            11,
            20,
            23,
            -1,
            -1,
            -1,
            -1,
            777,
            777,
        ]
        assert g.slot_mapping.gpu.data_ptr() == pointer


@pytest.mark.parametrize(
    "field,value",
    [
        ("pcp_world_size", 2),
        ("dcp_world_size", 2),
        ("blocks_per_kv_block", 2),
        ("use_hybrid_blocks", True),
        ("block_size", 0),
        ("kv_cache_block_size", 8),
        ("max_num_batched_tokens", 0),
        ("slot_mapping_mode", SimpleNamespace(name="NONE")),
    ],
)
def test_group_fallback_is_atomic(slots_module, field, value):
    groups = [slot_group(), slot_group()]
    setattr(groups[1], field, value)
    assert not slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=groups), 2, torch.tensor([0]), torch.tensor([0])
    )
    assert all(torch.all(g.slot_mapping.gpu == 777) for g in groups)


def test_empty_tokens(slots_module):
    g = slot_group()
    empty = torch.empty(0, dtype=torch.int64)
    assert slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 0, empty, empty
    )
    assert g.slot_mapping.gpu.tolist() == [-1] * 10 + [777, 777]


def test_opt_in_and_device_gates(slots_module, monkeypatch):
    g = slot_group()
    manager = SimpleNamespace(block_tables=[g])
    args = (manager, 1, torch.tensor([0]), torch.tensor([0]))
    monkeypatch.setenv("VLLM_FL_OCR_TORCH_SLOT_MAPPING", "0")
    assert not slots_module.compute_ocr_slot_mapping(*args)
    monkeypatch.setenv("VLLM_FL_OCR_TORCH_SLOT_MAPPING", "1")
    monkeypatch.setattr(slots_module, "ocr_runtime_enabled", lambda: False)
    assert not slots_module.compute_ocr_slot_mapping(*args)
    monkeypatch.setattr(slots_module, "ocr_runtime_enabled", lambda: True)
    monkeypatch.setattr(slots_module, "_supported_device", lambda device: False)
    assert not slots_module.compute_ocr_slot_mapping(*args)
    assert torch.all(g.slot_mapping.gpu == 777)


def test_invalid_tensor_contract(slots_module):
    g = slot_group()
    g.slot_mapping.gpu = g.slot_mapping.gpu.to(torch.int32)
    assert not slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 2, torch.tensor([0]), torch.tensor([0])
    )
    assert torch.all(g.slot_mapping.gpu == 777)


def test_column_strided_table_falls_back(slots_module):
    g = slot_group()
    g.block_table.gpu = g.block_table.gpu.T.contiguous().T
    assert not slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 1, torch.tensor([0]), torch.tensor([0])
    )
    assert torch.all(g.slot_mapping.gpu == 777)


def test_audit_only_after_success_and_never_during_capture(slots_module, monkeypatch):
    events = []
    monkeypatch.setattr(slots_module, "_write_audit", lambda *args: events.append(args))
    monkeypatch.setattr(slots_module, "_ocr_can_audit", lambda: False)
    g = slot_group()
    args = (SimpleNamespace(block_tables=[g]), 1, torch.tensor([0]), torch.tensor([0]))
    assert slots_module.compute_ocr_slot_mapping(*args)
    assert not events and not slots_module._SLOT_AUDITED
    monkeypatch.setattr(slots_module, "_ocr_can_audit", lambda: True)
    assert slots_module.compute_ocr_slot_mapping(*args)
    assert slots_module.compute_ocr_slot_mapping(*args)
    assert len(events) == 1 and events[0][0] == "slot-mapping"
    assert events[0][1]["event"] == "torch_ops_enqueued"


# Sampling


@pytest.mark.parametrize(
    "k,p,expected",
    [
        (None, None, [0.6, 0.3, 0.1]),
        (2, None, [2 / 3, 1 / 3, 0.0]),
        (None, 0.8, [2 / 3, 1 / 3, 0.0]),
        (2, 0.5, [1.0, 0.0, 0.0]),
    ],
)
def test_sampling_probability_contract(runtime, monkeypatch, k, p, expected):
    captured = []
    generator = torch.Generator().manual_seed(13)

    def multinomial(probs, count, generator=None):
        captured.append((probs.clone(), generator))
        return probs.argmax().reshape(1)

    monkeypatch.setattr(torch, "multinomial", multinomial)
    samples, logprobs = runtime.OCRTopKTopPSampler()(
        torch.tensor([[0.6, 0.3, 0.1]]).log(),
        {0: generator},
        None if k is None else torch.tensor([k]),
        None if p is None else torch.tensor([p]),
    )
    torch.testing.assert_close(captured[0][0], torch.tensor(expected))
    assert captured[0][1] is generator
    assert samples.tolist() == [0]
    assert logprobs is None


def test_penalties_prompt_output_and_padding(runtime, monkeypatch):
    # Stub only the framework base to exercise the FL penalties implementation.
    stub = ModuleType("vllm.v1.sample.sampler")
    stub.Sampler = torch.nn.Module
    with monkeypatch.context() as context:
        context.setitem(sys.modules, stub.__name__, stub)
        sampler_cls = runtime.get_ocr_sampler_cls()

    logits = torch.tensor([[4.0, -4.0, 3.0, -2.0]])
    metadata = SimpleNamespace(
        no_penalties=False,
        prompt_token_ids=torch.tensor([[0, 3, 4]]),
        repetition_penalties=torch.tensor([2.0]),
        frequency_penalties=torch.tensor([0.5]),
        presence_penalties=torch.tensor([1.0]),
    )
    result = sampler_cls.apply_penalties(logits, metadata, [[1, 1, 2, -1, 4]])
    # Prompt tokens receive repetition only; generated tokens additionally
    # receive count-based frequency and presence penalties. Padding is ignored.
    torch.testing.assert_close(result, torch.tensor([[2.0, -10.0, 0.0, -4.0]]))
    metadata.no_penalties = True
    assert sampler_cls.apply_penalties(result, metadata, [[]]) is result


# Execution audit


def test_execution_audit_disabled_and_exception_scope(runtime, monkeypatch, tmp_path):
    class FailingModel(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            raise ValueError("intentional forward failure")

    model = FailingModel()
    original = model.forward
    monkeypatch.delenv("VLLM_FL_OCR_AUDIT_OPS", raising=False)
    assert runtime.install_execution_audit(model) is None
    assert model.forward == original
    monkeypatch.setenv("VLLM_FL_OCR_AUDIT_OPS", "1")
    monkeypatch.setenv("VLLM_FL_OCR_AUDIT_DIR", str(tmp_path))
    recorder = runtime.install_execution_audit(model)
    assert runtime.install_execution_audit(model) is recorder
    with pytest.raises(ValueError, match="intentional"):
        model(torch.ones(2))
    assert any("aten::add" in schema for schema in recorder.schemas)
    saved = json.loads((tmp_path / f"execution-{os.getpid()}.json").read_text())
    assert saved["method_calls"] == {"forward": 1}
    assert saved["method_failures"] == {"forward": 1}
    assert saved["schemas"] == dict(recorder.schemas)
    assert recorder.depth == 0
    recorded = dict(recorder.schemas)
    torch.ones(2) + 2
    assert dict(recorder.schemas) == recorded


def test_execution_audit_multimodal_nested_entry(runtime, monkeypatch, tmp_path):
    class MultimodalModel(torch.nn.Module):
        def forward(self, x):
            return self.embed_multimodal(x)

        def embed_multimodal(self, x):
            return self.get_multimodal_embeddings(x)

        def get_multimodal_embeddings(self, x):
            return self.encoder_eager_forward(x)

        def encoder_eager_forward(self, x):
            return x + 1

    monkeypatch.setenv("VLLM_FL_OCR_AUDIT_OPS", "1")
    monkeypatch.setenv("VLLM_FL_OCR_AUDIT_DIR", str(tmp_path))
    model = MultimodalModel()
    recorder = runtime.install_execution_audit(model)
    x = torch.ones(2)
    result = model(x)
    assert recorder.depth == 0
    saved = json.loads((tmp_path / f"execution-{os.getpid()}.json").read_text())
    assert saved["method_calls"] == {
        "forward": 1,
        "embed_multimodal": 1,
        "get_multimodal_embeddings": 1,
        "encoder_eager_forward": 1,
    }
    assert (
        sum(
            count for schema, count in saved["schemas"].items() if "aten::add" in schema
        )
        == 1
    )
    assert not list(tmp_path.glob("*.tmp"))
    torch.testing.assert_close(result, torch.tensor([2.0, 2.0]))


def test_execution_audit_rejects_actual_forbidden_schema(runtime):
    # A separate FRAGMENT can coexist with a core's existing namespace.
    library = torch.library.Library("_C_ascend", "FRAGMENT")
    library.define("ocr_audit_test_forbidden(Tensor input) -> Tensor")
    library.impl("ocr_audit_test_forbidden", lambda x: x.clone(), "CPU")
    recorder = runtime.OCRExecutionAudit()
    with pytest.raises(RuntimeError, match="Forbidden OCR device operator"), recorder:
        torch.ops._C_ascend.ocr_audit_test_forbidden(torch.ones(2))
    recorded = dict(recorder.schemas)
    torch.ones(2) + 1
    assert dict(recorder.schemas) == recorded


# Image crop counting in the vLLM patch


PATCH = ROOT / "patches/unlimited_ocr/vllm-0.24-unlimited-ocr.patch"


def added_source(path):
    lines = PATCH.read_text(encoding="utf-8").splitlines()
    start = lines.index("+++ b/" + path) + 1
    result = []
    for line in lines[start:]:
        if line.startswith("--- "):
            break
        if line.startswith("+"):
            result.append(line[1:])
    return "\n".join(result)


class Images:
    def __init__(self, width=2560, height=5120, count=1):
        self.size = SimpleNamespace(width=width, height=height)
        self.count = count

    def __len__(self):
        return self.count

    def get_image_size(self, index):
        return self.size

    def get_items(self, *args):
        return self

    def get_count(self, *args, **kwargs):
        return self.count


class CachedBase:
    def _cached_apply_hf_processor(self, inputs, timing_ctx):
        return "cached"


def load_classes():
    tree = ast.parse(added_source("vllm/model_executor/models/unlimited_ocr.py"))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        in {"UnlimitedOCRProcessingInfo", "UnlimitedOCRMultiModalProcessor"}
    ]
    # Context resolution is the only substituted method. All token and cache
    # control flow comes directly from the complete added model source.
    for node in classes:
        node.decorator_list = []
        node.body = [
            method
            for method in node.body
            if not isinstance(method, ast.FunctionDef)
            or method.name not in {"get_hf_config", "get_hf_processor"}
        ]
    calls = []

    def count_tiles(width, height, *, max_num, image_size):
        calls.append(max_num)
        return max_num, 1

    namespace = dict(
        math=math,
        count_tiles=count_tiles,
        CROP_MODE=True,
        IMAGE_SIZE=640,
        BASE_SIZE=1024,
        _UNLIMITED_OCR_MAX_CROPS=32,
        DeepseekOCRProcessingInfo=object,
        DeepseekOCRMultiModalProcessor=CachedBase,
        ImageEmbeddingItems=type("Embeddings", (), {}),
        ImageProcessorItems=Images,
        ImageSize=SimpleNamespace,
        PromptReplacement=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *classes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(PATCH), "exec"), namespace)
    return namespace, calls


class CropCountTests(unittest.TestCase):
    def make_processor(self, server=None):
        namespace, calls = load_classes()
        info = namespace["UnlimitedOCRProcessingInfo"]()
        defaults = {
            "image_size": 640,
            "base_size": 1024,
            "crop_mode": True,
            "max_crops": 32,
        }
        defaults.update(server or {})
        info.ctx = SimpleNamespace(
            get_merged_mm_kwargs=lambda kwargs: {**defaults, **kwargs}
        )
        resolved = []

        def get_processor(**kwargs):
            result = SimpleNamespace(**{**defaults, **kwargs}, image_token_id=99)
            resolved.append(result)
            return result

        info.get_hf_processor = get_processor
        processor = namespace["UnlimitedOCRMultiModalProcessor"]()
        processor.info = info
        processor._apply_hf_processor = lambda *args: "fresh"
        return info, processor, calls, resolved

    def test_prompt_uses_resolved_processor_crop_limit(self):
        for server, request, expected in [
            ({}, {}, 32),
            ({"max_crops": 6}, {}, 6),
            ({"max_crops": 16}, {}, 16),
            ({"max_crops": 6}, {"max_crops": 64}, 64),
        ]:
            with self.subTest(server=server, request=request):
                info, processor, calls, resolved = self.make_processor(server)
                updates = processor._get_prompt_updates(Images(), request, {})
                tokens = updates[0].replacement(0)
                self.assertEqual(calls[-1], expected)
                self.assertEqual(len(tokens), 16 * 17 + 10 * (expected * 10 + 1) + 1)
                self.assertEqual(resolved[0].max_crops, expected)

    def test_direct_count_resolves_server_crop_limit(self):
        for limit in [6, 16, 32, 64]:
            info, _, calls, _ = self.make_processor({"max_crops": limit})
            info.get_num_image_tokens(image_width=2560, image_height=5120)
            self.assertEqual(calls, [limit])
            calls.clear()
            info.get_num_image_tokens(
                image_width=2560, image_height=5120, image_size=640, base_size=1024
            )
            self.assertEqual(calls, [limit])

    def test_request_limits_do_not_leak_into_later_cached_requests(self):
        _, processor, calls, _ = self.make_processor({"max_crops": 16})
        for kwargs, limit in [({}, 16), ({"max_crops": 64}, 64), ({}, 16)]:
            inputs = SimpleNamespace(mm_data_items=Images())
            self.assertEqual(
                processor._cached_apply_hf_processor(inputs, None), "cached"
            )
            tokens = processor._get_prompt_updates(inputs.mm_data_items, kwargs, {})[
                0
            ].replacement(0)
            self.assertEqual(calls[-1], limit)
            self.assertEqual(len(tokens), 273 + 10 * (limit * 10 + 1))

    def test_maximum_feature_size_uses_resolved_limit(self):
        for limit in [1, 6, 16, 32, 64]:
            info, _, _, _ = self.make_processor({"max_crops": limit})
            size = info.get_image_size_with_most_features()
            self.assertEqual(size.width, 640)
            self.assertEqual(size.height, 640 * limit)
        info, _, _, _ = self.make_processor({"max_crops": 64, "crop_mode": False})
        size = info.get_image_size_with_most_features()
        self.assertEqual((size.width, size.height), (640, 640))

    def test_crop_gates_and_cache_routing(self):
        for request, images in [
            ({"max_crops": 6}, Images(count=2)),
            ({"max_crops": 12, "crop_mode": False}, Images()),
            ({"max_crops": 12}, Images(640, 640)),
        ]:
            _, processor, calls, _ = self.make_processor()
            tokens = processor._get_prompt_updates(images, request, {})[0].replacement(
                0
            )
            self.assertEqual(len(tokens), 273)
            self.assertEqual(calls, [])
            route = processor._cached_apply_hf_processor(
                SimpleNamespace(mm_data_items=images), None
            )
            self.assertEqual(route, "fresh" if len(images) > 1 else "cached")
