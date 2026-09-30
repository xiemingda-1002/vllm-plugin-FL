# Copyright (c) 2026 BAAI. All rights reserved.

"""Contracts and NPU numerical coverage for the rc1 Ascend penalty chain."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO = Path(__file__).parents[3]
SAMPLER = REPO / "vllm_fl/sample/sampler.py"
PENALTIES = REPO / "vllm_fl/sample/penalties.py"
BINCOUNT = (
    REPO
    / "vllm_fl/dispatch/backends/vendor/ascend/ops/triton/bincount.py"
)
KERNEL = REPO / "vllm_fl/dispatch/backends/vendor/ascend/ops/triton/penalty.py"


def _function_source(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            result = ast.get_source_segment(source, node)
            assert result is not None
            return result
    raise AssertionError(f"{name} missing from {path}")


def test_penalty_chain_is_vendor_scoped_and_has_upstream_fallback() -> None:
    sampler = SAMPLER.read_text()
    override = _function_source(SAMPLER, "apply_penalties")
    assert "from vllm.triton_utils import HAS_TRITON" in sampler
    assert "if not HAS_TRITON:" in override
    assert "Sampler.apply_penalties(" in override
    assert "sampling_metadata.no_penalties" in override
    assert "from vllm_fl.sample.penalties import apply_all_penalties" in override
    assert "vllm_ascend" not in sampler


def _compiled_sampler_apply_penalties(has_triton: bool):
    source = SAMPLER.read_text()
    tree = ast.parse(source)
    method = next(
        item for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSampler"
        for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "apply_penalties"
    )
    method.decorator_list = []
    fallback = SimpleNamespace(calls=[])
    def upstream(logits, metadata, output):
        fallback.calls.append((logits, metadata, output))
        return "upstream-result"
    namespace = {
        "torch": torch, "Any": object, "HAS_TRITON": has_triton,
        "logger": SimpleNamespace(warning_once=lambda *_: None),
        "Sampler": SimpleNamespace(apply_penalties=upstream),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SAMPLER), "exec"), namespace)
    sampler = SimpleNamespace(
        topk_topp_sampler=SimpleNamespace(enable_reduce_sample=False)
    )
    return lambda *args: namespace["apply_penalties"](sampler, *args), fallback


def test_penalty_override_executes_no_penalty_identity_and_no_triton_fallback() -> None:
    logits = torch.randn(2, 7)
    metadata = SimpleNamespace(no_penalties=True, prompt_token_ids=None)
    apply, fallback = _compiled_sampler_apply_penalties(False)
    assert apply(logits, metadata, [[], []]) == "upstream-result"
    assert fallback.calls == [(logits, metadata, [[], []])]

    apply, fallback = _compiled_sampler_apply_penalties(True)
    assert apply(logits, metadata, [[], []]) is logits
    assert fallback.calls == []


def test_penalty_wrapper_converts_minus_one_sentinels_for_empty_outputs() -> None:
    source = PENALTIES.read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_all_penalties"
    )
    captured = {}
    def kernel(*args):
        captured["output"] = args[2].clone()
        return args[0]
    namespace = {
        "torch": torch,
        "_convert_to_tensors": lambda *_: torch.tensor([[5, -1], [-1, -1]]),
        "apply_penalties_triton": kernel,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(PENALTIES), "exec"), namespace)
    logits = torch.zeros(2, 7)
    assert namespace["apply_all_penalties"](
        logits, torch.zeros(2, 1), torch.zeros(2), torch.zeros(2), torch.ones(2), [
            [5], []
        ]
    ) is logits
    assert torch.equal(captured["output"], torch.tensor([[5, 7], [7, 7]]))


def test_penalty_contract_preserves_padding_and_full_vocab_bincount() -> None:
    converter = _function_source(PENALTIES, "apply_all_penalties")
    bincount = BINCOUNT.read_text()
    assert "output_tokens_t == -1, padding_token_id" in converter
    assert "pad=vocab_size" in PENALTIES.read_text()
    assert "tp_rank" in bincount
    assert "tp_rank is None" in bincount
    assert "get_vectorcore_num()" in bincount
    assert "ascend.impl.triton_utils" in bincount + KERNEL.read_text()
    assert "vllm_ascend" not in (PENALTIES.read_text() + bincount + KERNEL.read_text())


def _reference_penalties(
    logits: torch.Tensor,
    prompt: torch.Tensor,
    output: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
    repetition: torch.Tensor,
) -> torch.Tensor:
    expected = logits.clone()
    vocab_size = logits.shape[1]
    for row in range(logits.shape[0]):
        prompt_seen = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
        output_counts = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
        for token in prompt[row]:
            if 0 <= token < vocab_size:
                prompt_seen[token] = True
        for token in output[row]:
            if 0 <= token < vocab_size:
                output_counts[token] += 1
        repeated = prompt_seen | output_counts.bool()
        values = expected[row]
        values[repeated & (values > 0)] /= repetition[row]
        values[repeated & (values <= 0)] *= repetition[row]
        values -= frequency[row] * output_counts.to(values.dtype)
        values -= presence[row] * output_counts.bool().to(values.dtype)
    return expected


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None,
    reason="requires an Ascend NPU runtime",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ascend_penalties_match_reference_for_edge_cases(dtype: torch.dtype) -> None:
    import torch_npu  # noqa: F401  # registers torch.npu before kernel setup

    from vllm_fl.dispatch.backends.vendor.ascend.ops.triton.bincount import (
        get_token_bin_counts_and_mask_triton,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.ops.triton.penalty import (
        apply_penalties_triton,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils import (
        init_device_properties_triton,
    )

    init_device_properties_triton()
    vocab_size = 257  # deliberately not a power of two
    logits = torch.linspace(-2, 2, 3 * vocab_size, device="npu", dtype=dtype).view(3, vocab_size)
    prompt = torch.tensor([[0, 2, 256, 257], [1, 1, 260, 257], [257, 257, 257, 257]], device="npu")
    # vocab_size is the sentinel and 260 is out of range; 5 is repeated.
    output = torch.tensor([[5, 5, 257, 257], [1, 5, 260, 257], [257, 257, 257, 257]], device="npu")
    presence = torch.tensor([0.5, 0.0, 1.25], device="npu")
    frequency = torch.tensor([0.25, -0.5, 0.75], device="npu")
    repetition = torch.tensor([1.2, 0.8, 1.5], device="npu")
    expected = _reference_penalties(logits, prompt, output, presence, frequency, repetition)
    actual = apply_penalties_triton(logits.clone(), prompt, output, presence, frequency, repetition)
    torch.npu.synchronize()
    rtol, atol = (1e-5, 1e-6) if dtype is torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)

    counts, mask = get_token_bin_counts_and_mask_triton(
        torch.empty((3, 0), dtype=torch.long, device="npu"), vocab_size
    )
    assert counts.shape == (3, vocab_size)
    assert not mask.any()
