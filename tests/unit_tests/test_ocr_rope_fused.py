# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contract tests plus opt-in device verification of the public NPU API."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "ocr_rope_runtime",
    Path(__file__).resolve().parents[2] / "vllm_fl/models/ocr_runtime.py",
)
runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runtime)


def layer(head_size, rotary_dim, neox, device="cpu", dtype=torch.float32):
    phase = torch.arange(19)[:, None] * torch.linspace(0.01, 0.8, rotary_dim // 2)
    return SimpleNamespace(
        head_size=head_size,
        rotary_dim=rotary_dim,
        is_neox_style=neox,
        cos_sin_cache=torch.cat((phase.cos(), phase.sin()), -1).to(device, dtype),
    )


def reference(op, positions, x):
    if x is None:
        return None
    shaped = x.reshape(positions.numel(), -1, op.head_size)
    out = shaped.clone()
    coeff = op.cos_sin_cache[positions.flatten().long()].float()
    c, s = coeff.chunk(2, -1)
    if op.is_neox_style:
        a = shaped[..., : op.rotary_dim // 2].float()
        b = shaped[..., op.rotary_dim // 2 : op.rotary_dim].float()
        out[..., : op.rotary_dim // 2] = a * c[:, None] - b * s[:, None]
        out[..., op.rotary_dim // 2 : op.rotary_dim] = b * c[:, None] + a * s[:, None]
    else:
        a = shaped[..., : op.rotary_dim : 2].float()
        b = shaped[..., 1 : op.rotary_dim : 2].float()
        out[..., : op.rotary_dim : 2] = a * c[:, None] - b * s[:, None]
        out[..., 1 : op.rotary_dim : 2] = b * c[:, None] + a * s[:, None]
    return out.reshape(x.shape)


@pytest.mark.parametrize(
    "neox,rotary_dim,key_heads",
    [
        (True, 128, 2),
        (True, 64, 1),
        (False, 128, 2),
        (False, 64, None),
    ],
)
@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_cpu_reference(monkeypatch, neox, rotary_dim, key_heads, enabled):
    if enabled is None:
        monkeypatch.delenv("VLLM_FL_OCR_FUSED_ROPE", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_OCR_FUSED_ROPE", enabled)

    def forbidden(*args):
        raise AssertionError("CPU must never dispatch to torch_npu")

    monkeypatch.setattr(runtime, "_ocr_fused_rotary_mul", forbidden)
    op = layer(128, rotary_dim, neox)
    positions = torch.tensor([[7, 1], [7, 3]])
    query = torch.randn(4, 4 * 128)
    key = None if key_heads is None else torch.randn(4, key_heads, 128)
    q_before, k_before = query.clone(), None if key is None else key.clone()
    q, k = runtime._ocr_rotary_forward(op, positions, query, key)
    torch.testing.assert_close(q, reference(op, positions, query))
    torch.testing.assert_close(query, q_before)
    if key is None:
        assert k is None
    else:
        torch.testing.assert_close(k, reference(op, positions, key))
        torch.testing.assert_close(key, k_before)


@pytest.mark.parametrize("neox", [True, False])
def test_public_api_adapter(monkeypatch, neox):
    monkeypatch.setattr(runtime, "_ocr_can_audit", lambda: True)
    calls = []

    def api(x, c, s, rotary_mode="half", rotate=None):
        calls.append((x, c, s, rotary_mode, rotate))
        assert x.is_contiguous() and c.is_contiguous() and s.is_contiguous()
        return x.clone()

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=api))
    x = torch.randn(3, 2, 128)[..., :64]
    c, s = torch.ones(3, 1, 1, 64), torch.zeros(3, 1, 1, 64)
    result = runtime._ocr_fused_rotary_mul(x, c, s, neox)
    torch.testing.assert_close(result, x)
    assert calls[0][0].shape == (3, 1, 2, 64)
    assert calls[0][3] == ("half" if neox else "interleave")


def test_public_api_error_is_visible(monkeypatch):
    def unsupported(*args, **kwargs):
        raise RuntimeError("unsupported contract")

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=unsupported)
    )
    with pytest.raises(RuntimeError, match="unsupported contract"):
        runtime._ocr_fused_rotary_mul(
            torch.ones(1, 1, 128),
            torch.ones(1, 1, 1, 128),
            torch.zeros(1, 1, 1, 128),
            True,
        )


def test_fused_audit_follows_success_once(monkeypatch):
    monkeypatch.setattr(runtime, "_ocr_can_audit", lambda: True)
    evidence = []
    monkeypatch.setattr(runtime, "_OCR_FUSED_ROPE_AUDITED", False)
    monkeypatch.setattr(
        runtime, "_write_audit", lambda stage, data: evidence.append((stage, data))
    )

    def api(x, *args, **kwargs):
        return x.clone()

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=api))
    x = torch.ones(1, 1, 128)
    c, s = torch.ones(1, 1, 1, 128), torch.zeros(1, 1, 1, 128)
    runtime._ocr_fused_rotary_mul(x, c, s, True)
    runtime._ocr_fused_rotary_mul(x, c, s, True)
    assert len(evidence) == 1
    assert evidence[0][0] == "rope-fused"
    assert evidence[0][1]["event"] == "api_returned"


@pytest.mark.parametrize(
    "neox,rotary_dim,key_heads",
    [
        (True, 128, 2),
        (True, 64, 1),
        (False, 128, 2),
        (False, 64, None),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_fused_matches_reference(monkeypatch, neox, rotary_dim, key_heads, dtype):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU unavailable")
    monkeypatch.setenv("VLLM_FL_OCR_FUSED_ROPE", "1")
    op = layer(128, rotary_dim, neox, "npu", dtype)
    positions = torch.tensor([[7, 1], [7, 3]], device="npu")
    # Strided input exercises adapter contiguity and original shape preservation.
    query = torch.randn(4, 4, 256, device="npu", dtype=dtype)[..., ::2]
    key = (
        None
        if key_heads is None
        else torch.randn(4, key_heads * 128, device="npu", dtype=dtype)
    )
    before = query.clone()
    q, k = runtime._ocr_rotary_forward(op, positions, query, key)
    tolerance = 0.032 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(
        q, reference(op, positions, query), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(query, before)
    if key is None:
        assert k is None
    else:
        torch.testing.assert_close(
            k, reference(op, positions, key), atol=tolerance, rtol=tolerance
        )
    if rotary_dim < op.head_size:
        torch.testing.assert_close(
            q[..., rotary_dim:], query[..., rotary_dim:], atol=0, rtol=0
        )


@pytest.mark.parametrize("compiling,capturing", [(True, False), (False, True)])
def test_audit_skipped_during_compile_or_capture(monkeypatch, compiling, capturing):
    monkeypatch.setattr(runtime, "_OCR_FUSED_ROPE_AUDITED", False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    capture_checks = []

    def is_capturing():
        capture_checks.append(True)
        return capturing

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=is_capturing),
        raising=False,
    )

    def forbidden_audit(*args):
        raise AssertionError("audit must not run during compile or capture")

    monkeypatch.setattr(runtime, "_write_audit", forbidden_audit)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_rotary_mul=lambda x, *args, **kwargs: x.clone()),
    )
    x = torch.ones(1, 1, 128)
    c, s = torch.ones(1, 1, 1, 128), torch.zeros(1, 1, 1, 128)
    result = runtime._ocr_fused_rotary_mul(x, c, s, True)
    torch.testing.assert_close(result, x)
    assert runtime._OCR_FUSED_ROPE_AUDITED is False
    assert len(capture_checks) == (0 if compiling else 1)
