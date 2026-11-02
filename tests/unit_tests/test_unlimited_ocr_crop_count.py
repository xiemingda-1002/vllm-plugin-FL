"""Exercise the shipped patch without importing torch, transformers, or vLLM."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

PATCH = (
    Path(__file__).resolve().parents[2]
    / "patches/unlimited_ocr/vllm-0.24-unlimited-ocr.patch"
)


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


if __name__ == "__main__":
    unittest.main()
