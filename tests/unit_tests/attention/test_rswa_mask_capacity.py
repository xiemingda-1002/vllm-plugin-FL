"""CPU regression tests; require torch, but do not import vLLM or NPU code."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import torch

MASK_PATH = (
    Path(__file__).resolve().parents[3] / "vllm_fl/dispatch/backends/"
    "vendor/ascend/impl/attention_mask.py"
)
SPEC = importlib.util.spec_from_file_location("rswa_native_mask_under_test", MASK_PATH)
mask_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mask_module)
ENV = "VLLM_FL_RSWA_MASK_MAX_MIB"


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
        for value in ("", "no", "1.5", "0", "-1", "8193", " 512", "+512", "²"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "integer from 1 to 8192"),
            ):
                self.build([0], [1], [1], 1, 1, 1, limit=value)


if __name__ == "__main__":
    unittest.main()
