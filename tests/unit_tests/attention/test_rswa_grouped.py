"""CPU-only grouped R-SWA tests, without importing vLLM or torch_npu."""

import ast
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[3]
ATTENTION_DIR = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl"
SPEC = importlib.util.spec_from_file_location(
    "rswa_grouped_standard_mask", ATTENTION_DIR / "attention_mask.py"
)
mask_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mask_module)


def load_helpers():
    """Compile only the pure helpers; backend imports must never run on CPU."""
    source = ast.parse((ATTENTION_DIR / "attention.py").read_text(encoding="utf-8"))
    names = {"RswaAttentionGroup", "_build_grouped_rswa_masks", "_run_grouped_rswa_fia"}
    nodes = [
        node
        for node in source.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    prelude = ast.parse(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\nfrom typing import *\nimport torch\n"
        "_SWA_INT_MAX = 2147483647\n"
    ).body
    module = types.ModuleType("rswa_grouped_helpers_under_test")
    # dataclass resolves annotations through sys.modules on recent Python.
    sys.modules[module.__name__] = module
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


if __name__ == "__main__":
    unittest.main()
