"""CPU numerical contracts for the OCR tensor implementations.

The Sampler framework base is stubbed only while loading the module: these
tests exercise FL numerics, not vLLM's forwarding/signature integration.
"""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load("ocr_runtime_contract", "vllm_fl/models/ocr_runtime.py")


@pytest.fixture
def sampler_module(monkeypatch):
    stub = types.ModuleType("vllm.v1.sample.sampler")
    stub.Sampler = torch.nn.Module
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "vllm.v1.sample.sampler", stub)
        module = load("ocr_sampler_contract", "vllm_fl/models/ocr_sampler.py")
    return module


@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("rotary_dim", [4, 8])
@pytest.mark.parametrize("key_heads", [None, 1, 3])
def test_rope_complex_reference(neox, rotary_dim, key_heads):
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


@pytest.mark.parametrize(
    "k,p,expected",
    [
        (None, None, [0.6, 0.3, 0.1]),
        (2, None, [2 / 3, 1 / 3, 0.0]),
        (None, 0.8, [2 / 3, 1 / 3, 0.0]),
        (2, 0.5, [1.0, 0.0, 0.0]),
    ],
)
def test_sampling_probability_contract(sampler_module, monkeypatch, k, p, expected):
    captured = []
    generator = torch.Generator().manual_seed(13)

    def multinomial(probs, count, generator=None):
        captured.append((probs.clone(), generator))
        return probs.argmax().reshape(1)

    monkeypatch.setattr(torch, "multinomial", multinomial)
    samples, logprobs = sampler_module.OCRTopKTopPSampler()(
        torch.tensor([[0.6, 0.3, 0.1]]).log(),
        {0: generator},
        None if k is None else torch.tensor([k]),
        None if p is None else torch.tensor([p]),
    )
    torch.testing.assert_close(captured[0][0], torch.tensor(expected))
    assert captured[0][1] is generator
    assert samples.tolist() == [0]
    assert logprobs is None


def test_penalties_prompt_output_and_padding(sampler_module):
    logits = torch.tensor([[4.0, -4.0, 3.0, -2.0]])
    metadata = SimpleNamespace(
        no_penalties=False,
        prompt_token_ids=torch.tensor([[0, 3, 4]]),
        repetition_penalties=torch.tensor([2.0]),
        frequency_penalties=torch.tensor([0.5]),
        presence_penalties=torch.tensor([1.0]),
    )
    result = sampler_module.OCRSampler.apply_penalties(
        logits, metadata, [[1, 1, 2, -1, 4]]
    )
    # Prompt tokens receive repetition only; generated tokens additionally
    # receive count-based frequency and presence penalties. Padding is ignored.
    torch.testing.assert_close(result, torch.tensor([[2.0, -10.0, 0.0, -4.0]]))
    metadata.no_penalties = True
    assert sampler_module.OCRSampler.apply_penalties(result, metadata, [[]]) is result


def test_execution_audit_disabled_and_exception_scope(monkeypatch, tmp_path):
    audit = load("ocr_audit_contract", "vllm_fl/models/ocr_execution_audit.py")

    class FailingModel(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            raise ValueError("intentional forward failure")

    model = FailingModel()
    original = model.forward
    monkeypatch.delenv("VLLM_FL_OCR_AUDIT_OPS", raising=False)
    assert audit.install_execution_audit(model) is None
    assert model.forward == original
    monkeypatch.setenv("VLLM_FL_OCR_AUDIT_OPS", "1")
    monkeypatch.setenv("VLLM_FL_OCR_AUDIT_DIR", str(tmp_path))
    recorder = audit.install_execution_audit(model)
    assert audit.install_execution_audit(model) is recorder
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


def test_execution_audit_multimodal_nested_entry(monkeypatch, tmp_path):
    audit = load("ocr_audit_nested_contract", "vllm_fl/models/ocr_execution_audit.py")

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
    recorder = audit.install_execution_audit(model)
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


def test_execution_audit_rejects_actual_forbidden_schema():
    audit = load(
        "ocr_audit_forbidden_contract", "vllm_fl/models/ocr_execution_audit.py"
    )
    # A separate FRAGMENT can coexist with a core's existing namespace.
    library = torch.library.Library("_C_ascend", "FRAGMENT")
    library.define("ocr_audit_test_forbidden(Tensor input) -> Tensor")
    library.impl("ocr_audit_test_forbidden", lambda x: x.clone(), "CPU")
    recorder = audit.OCRExecutionAudit()
    with pytest.raises(RuntimeError, match="Forbidden OCR device operator"), recorder:
        torch.ops._C_ascend.ocr_audit_test_forbidden(torch.ones(2))
    recorded = dict(recorder.schemas)
    torch.ones(2) + 1
    assert dict(recorder.schemas) == recorded
