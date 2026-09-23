"""CPU-isolated contracts for AscendRMSNorm quantized bias handling."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn


LAYER_NORM_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/ascend/impl/layernorm.py"
)


def _ascend_rms_norm_definition() -> ast.ClassDef:
    tree = ast.parse(LAYER_NORM_PATH.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendRMSNorm"
    )


@pytest.fixture
def rms_norm_module() -> ModuleType:
    """Load only the production class; never import the FL package root."""
    class RMSNorm(nn.Module):
        def __init__(
            self, hidden_size, eps, var_hidden_size, has_weight, dtype
        ) -> None:
            super().__init__()
            self.has_weight = has_weight
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
            self.variance_epsilon = eps
            self.native_calls = 0

        def forward_native(self, x, residual=None):
            self.native_calls += 1
            return (x + 99, residual) if residual is not None else x + 99

    module = ModuleType("isolated_ascend_rms_norm")
    module.__dict__.update(
        torch=torch,
        RMSNorm=RMSNorm,
        get_current_vllm_config=lambda: SimpleNamespace(quant_config=None),
    )
    definition = _ascend_rms_norm_definition()
    exec(
        compile(ast.Module([definition], []), str(LAYER_NORM_PATH), "exec"),
        module.__dict__,
    )
    return module


@pytest.fixture
def fake_torch_npu(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    def rms_norm(x, weight, eps):
        del weight, eps
        return x * 2, None

    def add_rms_norm(x, residual, weight, eps):
        del weight, eps
        return x + residual, None, residual * 3

    fake = SimpleNamespace(
        npu_rms_norm=rms_norm,
        npu_add_rms_norm=add_rms_norm,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    return fake


def _set_quant_description(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    description: tuple[str, ...] | None,
) -> None:
    quant_config = (
        None
        if description is None
        else SimpleNamespace(quant_description=description)
    )
    monkeypatch.setattr(
        module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(quant_config=quant_config),
    )


@pytest.mark.parametrize("description", (None, ("weight",), ("some.norm.bias",)))
def test_quant_description_controls_norm_bias_parameter(
    monkeypatch: pytest.MonkeyPatch,
    rms_norm_module: ModuleType,
    description: tuple[str, ...] | None,
) -> None:
    _set_quant_description(monkeypatch, rms_norm_module, description)

    layer = rms_norm_module.AscendRMSNorm(4)

    if description == ("some.norm.bias",):
        assert isinstance(layer.bias, nn.Parameter)
        assert layer.bias.requires_grad is False
        assert layer.bias_loaded is False
        assert layer.bias.weight_loader.__self__ is layer
    else:
        assert layer.bias is None
        assert layer.bias_loaded is False


def test_loaded_quant_bias_is_applied_without_residual(
    monkeypatch: pytest.MonkeyPatch,
    rms_norm_module: ModuleType,
    fake_torch_npu: SimpleNamespace,
) -> None:
    del fake_torch_npu
    _set_quant_description(monkeypatch, rms_norm_module, ("q_a_layernorm.norm.bias",))
    layer = rms_norm_module.AscendRMSNorm(3)
    layer.bias.weight_loader(layer.bias, torch.tensor([1.0, -2.0, 3.0]))

    output = layer.forward_oot(torch.tensor([[2.0, 4.0, 6.0]]))

    assert layer.bias_loaded is True
    assert torch.equal(output, torch.tensor([[5.0, 6.0, 15.0]]))


def test_bias_loader_accepts_scalar_and_rejects_bad_shape(
    monkeypatch: pytest.MonkeyPatch, rms_norm_module: ModuleType
) -> None:
    _set_quant_description(monkeypatch, rms_norm_module, ("norm.bias",))
    scalar = rms_norm_module.AscendRMSNorm(1)
    scalar.bias.weight_loader(scalar.bias, torch.tensor(2.5))
    assert scalar.bias.item() == pytest.approx(2.5)
    assert scalar.bias_loaded is True

    vector = rms_norm_module.AscendRMSNorm(2)
    with pytest.raises(AssertionError, match="Attempted to load weight"):
        vector.bias.weight_loader(vector.bias, torch.ones(3))
    assert vector.bias_loaded is False


def test_residual_path_chunks_and_applies_allocated_bias(
    monkeypatch: pytest.MonkeyPatch,
    rms_norm_module: ModuleType,
    fake_torch_npu: SimpleNamespace,
) -> None:
    del fake_torch_npu
    _set_quant_description(monkeypatch, rms_norm_module, ("norm.bias",))
    layer = rms_norm_module.AscendRMSNorm(2)
    layer.bias.data.copy_(torch.tensor([0.5, 1.5]))
    calls = []
    monkeypatch.setattr(
        rms_norm_module,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(
                vllm=SimpleNamespace(
                    maybe_chunk_residual=lambda x, residual: calls.append(
                        (x, residual)
                    )
                    or residual + 10
                )
            )
        ),
    )

    output, residual = layer.forward_oot(
        torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])
    )

    assert len(calls) == 1
    assert torch.equal(output, torch.tensor([[14.5, 17.5]]))
    assert torch.equal(residual, torch.tensor([[39.0, 42.0]]))


def test_no_weight_uses_native_behavior_before_ascend_ops(
    monkeypatch: pytest.MonkeyPatch,
    rms_norm_module: ModuleType,
    fake_torch_npu: SimpleNamespace,
) -> None:
    fake_torch_npu.npu_rms_norm = lambda *_args: pytest.fail(
        "no-weight path must not call npu_rms_norm"
    )
    fake_torch_npu.npu_add_rms_norm = lambda *_args: pytest.fail(
        "no-weight path must not call npu_add_rms_norm"
    )
    _set_quant_description(monkeypatch, rms_norm_module, ("norm.bias",))
    layer = rms_norm_module.AscendRMSNorm(2, has_weight=False)

    assert torch.equal(layer.forward_oot(torch.ones(1, 2)), torch.full((1, 2), 100.0))
    assert layer.native_calls == 1
