# Copyright (c) 2026 BAAI. All rights reserved.
"""SAM dispatch contract; device accuracy is checked separately on NPU."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def runtime():
    path = Path(__file__).resolve().parents[2] / "vllm_fl/models/ocr_runtime.py"
    spec = importlib.util.spec_from_file_location("ocr_runtime_sam_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorContract:
    def __init__(self, shape=(2, 12, 196, 64), dtype="bf16", device="npu"):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.device = NS(type=device)
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
    fake_torch = NS(
        bfloat16="bf16",
        nn=NS(functional=NS(scaled_dot_product_attention=lambda *a, **kw: "sdpa")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        NS(
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
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    layer = NS(
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
        NS(add_decomposed_rel_pos=lambda *a: (rh, rw)),
    )
    q, k, v = layer.qkv(x).reshape(2, 6, 3, 3, 8).permute(2, 0, 3, 1, 4)
    bias = (rh + rw).reshape(2, 3, 6, 6) if use_rel_pos else None
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    expected = layer.proj(expected.transpose(1, 2).reshape(2, 2, 3, 24))
    torch.testing.assert_close(runtime._ocr_sam_forward(layer, x), expected)
