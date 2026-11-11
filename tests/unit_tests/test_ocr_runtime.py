# Copyright (c) 2026 BAAI. All rights reserved.
"""OCR runtime contracts and opt-in verification of the public NPU RoPE API."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def runtime():
    # Load independently for each test to isolate one-time audit state.
    path = Path(__file__).resolve().parents[2] / "vllm_fl/models/ocr_runtime.py"
    spec = importlib.util.spec_from_file_location("ocr_runtime_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Rotary embeddings


def rope_layer(head_size, rotary_dim, neox, device="cpu", dtype=torch.float32):
    phase = torch.arange(19)[:, None] * torch.linspace(0.01, 0.8, rotary_dim // 2)
    return SimpleNamespace(
        head_size=head_size,
        rotary_dim=rotary_dim,
        is_neox_style=neox,
        cos_sin_cache=torch.cat((phase.cos(), phase.sin()), -1).to(device, dtype),
    )


def rope_reference(op, positions, x):
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
def test_cpu_reference(runtime, monkeypatch, neox, rotary_dim, key_heads, enabled):
    if enabled is None:
        monkeypatch.delenv("VLLM_FL_OCR_FUSED_ROPE", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_OCR_FUSED_ROPE", enabled)

    def forbidden(*args):
        raise AssertionError("CPU must never dispatch to torch_npu")

    monkeypatch.setattr(runtime, "_ocr_fused_rotary_mul", forbidden)
    op = rope_layer(128, rotary_dim, neox)
    positions = torch.tensor([[7, 1], [7, 3]])
    query = torch.randn(4, 4 * 128)
    key = None if key_heads is None else torch.randn(4, key_heads, 128)
    q_before, k_before = query.clone(), None if key is None else key.clone()
    q, k = runtime._ocr_rotary_forward(op, positions, query, key)
    torch.testing.assert_close(q, rope_reference(op, positions, query))
    torch.testing.assert_close(query, q_before)
    if key is None:
        assert k is None
    else:
        torch.testing.assert_close(k, rope_reference(op, positions, key))
        torch.testing.assert_close(key, k_before)


@pytest.mark.parametrize("neox", [True, False])
def test_public_api_adapter(runtime, monkeypatch, neox):
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


def test_public_api_error_is_visible(runtime, monkeypatch):
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


def test_fused_audit_follows_success_once(runtime, monkeypatch):
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
def test_npu_fused_matches_reference(
    runtime, monkeypatch, neox, rotary_dim, key_heads, dtype
):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU unavailable")
    monkeypatch.setenv("VLLM_FL_OCR_FUSED_ROPE", "1")
    op = rope_layer(128, rotary_dim, neox, "npu", dtype)
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
        q, rope_reference(op, positions, query), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(query, before)
    if key is None:
        assert k is None
    else:
        torch.testing.assert_close(
            k, rope_reference(op, positions, key), atol=tolerance, rtol=tolerance
        )
    if rotary_dim < op.head_size:
        torch.testing.assert_close(
            q[..., rotary_dim:], query[..., rotary_dim:], atol=0, rtol=0
        )


@pytest.mark.parametrize("compiling,capturing", [(True, False), (False, True)])
def test_audit_skipped_during_compile_or_capture(
    runtime, monkeypatch, compiling, capturing
):
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


@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("rotary_dim", [4, 8])
@pytest.mark.parametrize("key_heads", [None, 1, 3])
def test_rope_complex_reference(runtime, neox, rotary_dim, key_heads):
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


# SAM attention


class TensorContract:
    def __init__(self, shape=(2, 12, 196, 64), dtype="bf16", device="npu"):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.device = SimpleNamespace(type=device)
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
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                scaled_dot_product_attention=lambda *a, **kw: "sdpa"
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
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
    torch.manual_seed(17)
    layer = SimpleNamespace(
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
        SimpleNamespace(add_decomposed_rel_pos=lambda *a: (rh, rw)),
    )
    q, k, v = layer.qkv(x).reshape(2, 6, 3, 3, 8).permute(2, 0, 3, 1, 4)
    bias = (rh + rw).reshape(2, 3, 6, 6) if use_rel_pos else None
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    expected = layer.proj(expected.transpose(1, 2).reshape(2, 2, 3, 24))
    torch.testing.assert_close(runtime._ocr_sam_forward(layer, x), expected)


# Slot mapping


@pytest.fixture
def slots_module(runtime, monkeypatch):
    monkeypatch.delenv("VLLM_FL_OCR_TORCH_SLOT_MAPPING", raising=False)
    monkeypatch.setattr(runtime, "ocr_runtime_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_ocr_can_audit", lambda: False)
    monkeypatch.setattr(runtime, "_supported_device", lambda device: True)
    return runtime


def slot_group(noncontiguous=False):
    table = torch.tensor([[8, 3, 12], [19, 2, 5]], dtype=torch.int32)
    if noncontiguous:
        storage = torch.empty((4, 3), dtype=table.dtype)
        storage[::2].copy_(table)
        table = storage[::2]
    return SimpleNamespace(
        block_size=4,
        blocks_per_kv_block=1,
        use_hybrid_blocks=False,
        pcp_world_size=1,
        dcp_world_size=1,
        max_num_batched_tokens=10,
        block_table=SimpleNamespace(gpu=table),
        slot_mapping=SimpleNamespace(gpu=torch.full((12,), 777, dtype=torch.int64)),
    )


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_values_padding_and_storage(slots_module, noncontiguous):
    groups = [slot_group(noncontiguous), slot_group()]
    positions = torch.tensor([0, 3, 4, 7, 8, 11], dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    pointers = [g.slot_mapping.gpu.data_ptr() for g in groups]
    assert slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=groups), 2, requests, positions
    )
    for g, pointer in zip(groups, pointers):
        assert g.slot_mapping.gpu.tolist() == [
            32,
            35,
            12,
            11,
            20,
            23,
            -1,
            -1,
            -1,
            -1,
            777,
            777,
        ]
        assert g.slot_mapping.gpu.data_ptr() == pointer


@pytest.mark.parametrize(
    "field,value",
    [
        ("pcp_world_size", 2),
        ("dcp_world_size", 2),
        ("blocks_per_kv_block", 2),
        ("use_hybrid_blocks", True),
        ("block_size", 0),
        ("kv_cache_block_size", 8),
        ("max_num_batched_tokens", 0),
        ("slot_mapping_mode", SimpleNamespace(name="NONE")),
    ],
)
def test_group_fallback_is_atomic(slots_module, field, value):
    groups = [slot_group(), slot_group()]
    setattr(groups[1], field, value)
    assert not slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=groups), 2, torch.tensor([0]), torch.tensor([0])
    )
    assert all(torch.all(g.slot_mapping.gpu == 777) for g in groups)


def test_empty_tokens(slots_module):
    g = slot_group()
    empty = torch.empty(0, dtype=torch.int64)
    assert slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 0, empty, empty
    )
    assert g.slot_mapping.gpu.tolist() == [-1] * 10 + [777, 777]


def test_opt_in_and_device_gates(slots_module, monkeypatch):
    g = slot_group()
    manager = SimpleNamespace(block_tables=[g])
    args = (manager, 1, torch.tensor([0]), torch.tensor([0]))
    monkeypatch.setenv("VLLM_FL_OCR_TORCH_SLOT_MAPPING", "0")
    assert not slots_module.compute_ocr_slot_mapping(*args)
    monkeypatch.setenv("VLLM_FL_OCR_TORCH_SLOT_MAPPING", "1")
    monkeypatch.setattr(slots_module, "ocr_runtime_enabled", lambda: False)
    assert not slots_module.compute_ocr_slot_mapping(*args)
    monkeypatch.setattr(slots_module, "ocr_runtime_enabled", lambda: True)
    monkeypatch.setattr(slots_module, "_supported_device", lambda device: False)
    assert not slots_module.compute_ocr_slot_mapping(*args)
    assert torch.all(g.slot_mapping.gpu == 777)


def test_invalid_tensor_contract(slots_module):
    g = slot_group()
    g.slot_mapping.gpu = g.slot_mapping.gpu.to(torch.int32)
    assert not slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 2, torch.tensor([0]), torch.tensor([0])
    )
    assert torch.all(g.slot_mapping.gpu == 777)


def test_column_strided_table_falls_back(slots_module):
    g = slot_group()
    g.block_table.gpu = g.block_table.gpu.T.contiguous().T
    assert not slots_module.compute_ocr_slot_mapping(
        SimpleNamespace(block_tables=[g]), 1, torch.tensor([0]), torch.tensor([0])
    )
    assert torch.all(g.slot_mapping.gpu == 777)


def test_audit_only_after_success_and_never_during_capture(slots_module, monkeypatch):
    events = []
    monkeypatch.setattr(slots_module, "_write_audit", lambda *args: events.append(args))
    monkeypatch.setattr(slots_module, "_ocr_can_audit", lambda: False)
    g = slot_group()
    args = (SimpleNamespace(block_tables=[g]), 1, torch.tensor([0]), torch.tensor([0]))
    assert slots_module.compute_ocr_slot_mapping(*args)
    assert not events and not slots_module._SLOT_AUDITED
    monkeypatch.setattr(slots_module, "_ocr_can_audit", lambda: True)
    assert slots_module.compute_ocr_slot_mapping(*args)
    assert slots_module.compute_ocr_slot_mapping(*args)
    assert len(events) == 1 and events[0][0] == "slot-mapping"
    assert events[0][1]["event"] == "torch_ops_enqueued"


# Sampling


@pytest.mark.parametrize(
    "k,p,expected",
    [
        (None, None, [0.6, 0.3, 0.1]),
        (2, None, [2 / 3, 1 / 3, 0.0]),
        (None, 0.8, [2 / 3, 1 / 3, 0.0]),
        (2, 0.5, [1.0, 0.0, 0.0]),
    ],
)
def test_sampling_probability_contract(runtime, monkeypatch, k, p, expected):
    captured = []
    generator = torch.Generator().manual_seed(13)

    def multinomial(probs, count, generator=None):
        captured.append((probs.clone(), generator))
        return probs.argmax().reshape(1)

    monkeypatch.setattr(torch, "multinomial", multinomial)
    samples, logprobs = runtime.OCRTopKTopPSampler()(
        torch.tensor([[0.6, 0.3, 0.1]]).log(),
        {0: generator},
        None if k is None else torch.tensor([k]),
        None if p is None else torch.tensor([p]),
    )
    torch.testing.assert_close(captured[0][0], torch.tensor(expected))
    assert captured[0][1] is generator
    assert samples.tolist() == [0]
    assert logprobs is None


def test_penalties_prompt_output_and_padding(runtime, monkeypatch):
    # Stub only the framework base to exercise the FL penalties implementation.
    stub = ModuleType("vllm.v1.sample.sampler")
    stub.Sampler = torch.nn.Module
    with monkeypatch.context() as context:
        context.setitem(sys.modules, stub.__name__, stub)
        sampler_cls = runtime.get_ocr_sampler_cls()

    logits = torch.tensor([[4.0, -4.0, 3.0, -2.0]])
    metadata = SimpleNamespace(
        no_penalties=False,
        prompt_token_ids=torch.tensor([[0, 3, 4]]),
        repetition_penalties=torch.tensor([2.0]),
        frequency_penalties=torch.tensor([0.5]),
        presence_penalties=torch.tensor([1.0]),
    )
    result = sampler_cls.apply_penalties(logits, metadata, [[1, 1, 2, -1, 4]])
    # Prompt tokens receive repetition only; generated tokens additionally
    # receive count-based frequency and presence penalties. Padding is ignored.
    torch.testing.assert_close(result, torch.tensor([[2.0, -10.0, 0.0, -4.0]]))
    metadata.no_penalties = True
    assert sampler_cls.apply_penalties(result, metadata, [[]]) is result


# Execution audit


def test_execution_audit_disabled_and_exception_scope(runtime, monkeypatch, tmp_path):
    class FailingModel(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            raise ValueError("intentional forward failure")

    model = FailingModel()
    original = model.forward
    monkeypatch.delenv("VLLM_FL_OCR_AUDIT_OPS", raising=False)
    assert runtime.install_execution_audit(model) is None
    assert model.forward == original
    monkeypatch.setenv("VLLM_FL_OCR_AUDIT_OPS", "1")
    monkeypatch.setenv("VLLM_FL_OCR_AUDIT_DIR", str(tmp_path))
    recorder = runtime.install_execution_audit(model)
    assert runtime.install_execution_audit(model) is recorder
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


def test_execution_audit_multimodal_nested_entry(runtime, monkeypatch, tmp_path):
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
    recorder = runtime.install_execution_audit(model)
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


def test_execution_audit_rejects_actual_forbidden_schema(runtime):
    # A separate FRAGMENT can coexist with a core's existing namespace.
    library = torch.library.Library("_C_ascend", "FRAGMENT")
    library.define("ocr_audit_test_forbidden(Tensor input) -> Tensor")
    library.impl("ocr_audit_test_forbidden", lambda x: x.clone(), "CPU")
    recorder = runtime.OCRExecutionAudit()
    with pytest.raises(RuntimeError, match="Forbidden OCR device operator"), recorder:
        torch.ops._C_ascend.ocr_audit_test_forbidden(torch.ones(2))
    recorded = dict(recorder.schemas)
    torch.ones(2) + 1
    assert dict(recorder.schemas) == recorded
