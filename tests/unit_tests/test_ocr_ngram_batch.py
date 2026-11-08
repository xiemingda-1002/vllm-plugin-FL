# Copyright (c) 2026 BAAI. All rights reserved.
"""Compare batch masks to actual core source without importing the model stack."""

import ast
import importlib.util
import os
import sys
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


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
    path = Path(__file__).resolve().parents[2] / "vllm_fl/models/ocr_ngram.py"
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
    path = Path(__file__).resolve().parents[2] / "vllm_fl/models/ocr_ngram.py"
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
