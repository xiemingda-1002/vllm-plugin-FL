"""CPU mask semantics; imports neither vLLM nor torch-npu."""

import ast
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
IMPL = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl"
spec = importlib.util.spec_from_file_location("ocr_mask", IMPL / "attention_mask.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def mask(prefix, seq, query, qmax, kmax=32):
    return module.AttentionMaskBuilder(torch.device("cpu")).get_rswa_attn_mask(
        torch.tensor(prefix), torch.tensor(seq), torch.tensor(query), kmax, qmax, 4
    )


def test_chunk_and_mixed_mask_matches_scalar_reference():
    prefix, seq, query = [8, 3, 0], [5, 14, 0], [3, 1, 0]
    actual = mask(prefix, seq, query, 3)
    for b in range(3):
        for i in range(3):
            for j in range(32):
                pos = seq[b] - query[b] + i
                keep = (
                    i < query[b]
                    and j < seq[b]
                    and j <= pos
                    and (j < prefix[b] or j > pos - 4)
                )
                assert actual[b, 0, i, j].item() == int(not keep)


def test_request_reorder_is_equivariant():
    original = mask([3, 8], [14, 5], [1, 3], 3)
    reordered = mask([8, 3], [5, 14], [3, 1], 3)
    assert torch.equal(original[[1, 0]], reordered)


def test_full_padding_uses_explicit_real_request_boundary():
    tree = ast.parse((IMPL / "attention.py").read_text(encoding="utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_rswa_mask_request_lengths"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = {"torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[])),
            "helper",
            "exec",
        ),
        ns,
    )
    prefix, query = torch.tensor([2, 0, 99]), torch.tensor([1, 1, 1])
    p, q = ns[fn.name](prefix, query, 2)
    assert p.tolist() == [2, 0, 0]
    assert q.tolist() == [1, 1, 0]
    assert prefix.tolist() == [2, 0, 99]
    assert torch.all(mask(p.tolist(), [8, 5, 0], q.tolist(), 1)[2] == 1)


def test_invalid_and_oversized_masks_fail_before_allocation():
    with pytest.raises(AssertionError, match="query_lens must be <= seq_lens"):
        mask([1], [1], [2], 2)
    with pytest.raises(ValueError, match="too large"):
        mask([1], [1], [1], 32768, 32768)
