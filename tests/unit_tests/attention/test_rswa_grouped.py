"""CPU-only R-SWA mask, metadata routing, and grouped attention tests, without importing vLLM or torch_npu."""

import ast
import importlib.util
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
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
    module = types.ModuleType("rswa_grouped_helpers_under_test")
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
    common = NS(
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

    fake_builder = NS(get_rswa_attn_mask=build_mask)
    decodes = 4 if phase == state.DecodeOnly else 2
    self = NS(
        device=torch.device("cpu"),
        model_config=NS(
            rswa_window=3,
            max_model_len=32,
            runner_type="generate",
            hf_config=NS(
                model_type="unlimited-ocr" if name.startswith("OCR") else "ordinary"
            ),
        ),
        vllm_config=NS(quant_config=None, speculative_config=speculative),
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


if __name__ == "__main__":
    unittest.main()
