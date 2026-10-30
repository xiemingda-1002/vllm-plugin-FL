"""CPU regression: run with Python + torch; no vLLM/NPU import required."""

import ast
import importlib.util
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_fl"


class GraphMode(Enum):
    FULL = 1
    FULL_DECODE_ONLY = 2
    NONE = 3


def load_function(path, name, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    scope = tree.body
    if class_name:
        scope = next(
            n for n in scope if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = {
        "torch": torch,
        "AscendCommonAttentionMetadata": SimpleNamespace,
        "CUDAGraphMode": GraphMode,
    }
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


normalize = load_function("dispatch/backends/vendor/ascend/impl/attention/attention.py", "_rswa_mask_request_lengths")
unpadded = load_function(
    "dispatch/backends/vendor/ascend/impl/attention/utils.py", "unpadded", "AscendCommonAttentionMetadata"
)
pad_query_starts = load_function(
    "worker/model_runner.py", "_pad_query_start_loc_for_fia", "ModelRunnerFL"
)
spec = importlib.util.spec_from_file_location(
    "rswa_standard_mask", ROOT / "dispatch/backends/vendor/ascend/impl/attention/standard_mask.py"
)
mask_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mask_module)


class PaddingTests(unittest.TestCase):
    def mask(self, prefix, seq, query):
        return mask_module.AttentionMaskBuilder(torch.device("cpu")).get_rswa_attn_mask(
            prefix, seq, query, max_kv_len=32, max_query_len=1, rswa_window=4
        )

    def test_graph_padding_has_no_queries_and_real_mask_is_unchanged(self):
        # Three live requests rounded up to the four-request graph bucket.
        starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        query = starts[1:] - starts[:-1]
        prefix = torch.tensor([2, 0, 3, 99], dtype=torch.int32)
        seq = torch.tensor([8, 5, 9, 0], dtype=torch.int32)
        with self.assertRaisesRegex(AssertionError, "prefix_lens exceeds"):
            self.mask(prefix, seq, query)
        normalized_prefix, mask_query = normalize(prefix, query, 3)
        result = self.mask(normalized_prefix, seq, mask_query)
        self.assertTrue(
            torch.equal(result[:3], self.mask(prefix[:3], seq[:3], query[:3]))
        )
        self.assertTrue(torch.all(result[3] == 1).item())
        self.assertEqual(query.tolist(), [1, 1, 1, 1])
        self.assertEqual(prefix.tolist(), [2, 0, 3, 99])
        self.assertEqual(seq.tolist(), [8, 5, 9, 0])
        self.assertEqual(starts.tolist(), [0, 1, 2, 3, 4])

    def padded_starts(self, real, bucket, rswa_window=4):
        class Buffer:
            def __init__(self):
                self.np = np.zeros(18, dtype=np.int32)
                self.np[: real + 1] = np.arange(real + 1)
                self.copied = False

            def copy_to_gpu(self):
                self.copied = True

        buffer = Buffer()
        runner = SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=GraphMode.FULL_DECODE_ONLY
            ),
            model_config=SimpleNamespace(rswa_window=rswa_window),
            uniform_decode_query_len=1,
            arange_np=np.arange(18),
        )
        count = pad_query_starts(runner, buffer, bucket, real, GraphMode.FULL, None)
        self.assertTrue(buffer.copied)
        return torch.from_numpy(buffer.np[: count + 1].copy())

    def test_live_bucket_shapes_match_capture_and_keep_real_masks(self):
        for real, bucket in ((3, 4), (5, 8), (9, 16)):
            with self.subTest(real=real, bucket=bucket):
                starts = self.padded_starts(real, bucket)
                self.assertEqual(starts.tolist(), list(range(bucket + 1)))
                query = starts[1:] - starts[:-1]
                seq = torch.tensor([10] * real + [0] * (bucket - real))
                prefix = torch.tensor([3] * real + [0] * (bucket - real))
                prefix, mask_query = normalize(prefix, query, real)
                live = self.mask(prefix, seq, mask_query)
                capture = self.mask(
                    torch.zeros(bucket, dtype=torch.int32),
                    torch.ones(bucket, dtype=torch.int32),
                    torch.ones(bucket, dtype=torch.int32),
                )
                self.assertEqual(live.shape, capture.shape)
                self.assertEqual(tuple(live.shape), (bucket, 1, 1, 32))
                self.assertTrue(torch.all(live[real:] == 1).item())
                self.assertTrue(
                    torch.equal(
                        live[:real], self.mask(prefix[:real], seq[:real], query[:real])
                    )
                )
                self.assertEqual(query.tolist(), [1] * bucket)
                self.assertEqual(seq.tolist(), [10] * real + [0] * (bucket - real))

    def test_non_rswa_padding_keeps_existing_merged_dummy(self):
        starts = self.padded_starts(5, 8, rswa_window=None)
        self.assertEqual(starts.tolist(), [0, 1, 2, 3, 4, 5, 8])

    def test_original_zero_seq_padding_failure_is_reproduced(self):
        prefix, seq, query = (
            torch.tensor([4, 0]),
            torch.tensor([8, 0]),
            torch.tensor([1, 1]),
        )
        with self.assertRaisesRegex(AssertionError, "query_lens must be <= seq_lens"):
            self.mask(prefix, seq, query)
        prefix, query = normalize(prefix, query, 1)
        self.assertTrue(torch.all(self.mask(prefix, seq, query)[1] == 1).item())

    def test_invalid_real_zero_prefix_row_is_still_rejected(self):
        prefix, query = normalize(torch.tensor([0, 0]), torch.tensor([1, 1]), 1)
        with self.assertRaisesRegex(AssertionError, "query_lens must be <= seq_lens"):
            self.mask(prefix, torch.tensor([0, 0]), query)

    def test_missing_boundary_does_not_guess_padding(self):
        prefix, query = torch.tensor([0]), torch.tensor([1])
        output_prefix, output_query = normalize(prefix, query, None)
        self.assertIs(output_prefix, prefix)
        self.assertIs(output_query, query)
        with self.assertRaisesRegex(AssertionError, "query_lens must be <= seq_lens"):
            self.mask(output_prefix, torch.tensor([0]), output_query)

    def test_invalid_boundaries_are_rejected(self):
        for count in (-1, 3, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                normalize(torch.tensor([0, 0]), torch.tensor([1, 1]), count)

    def test_unpadded_preserves_and_clamps_real_boundary(self):
        class Metadata:
            def __getattr__(self, name):
                return torch.arange(8)

        for count, extent, expected in ((3, 2, 2), (3, 4, 3), (None, 2, None)):
            metadata = Metadata()
            metadata.rswa_num_actual_reqs = count
            result = unpadded(metadata, 2, extent)
            self.assertEqual(result.rswa_num_actual_reqs, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
