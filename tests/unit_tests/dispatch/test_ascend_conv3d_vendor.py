from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import torch
import torch.nn.functional as F

SOURCE_ROOT = Path(__file__).resolve().parents[3]
CONV_PATH = SOURCE_ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/conv.py"
PATCH_PATH = SOURCE_ROOT / "vllm_fl/dispatch/backends/vendor/ascend/patch.py"


def _stub_module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_ascend_conv3d(monkeypatch):
    class Conv3dLayer:
        def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
            return F.conv3d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

    monkeypatch.setitem(sys.modules, "vllm", _stub_module("vllm"))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor", _stub_module("vllm.model_executor")
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers",
        _stub_module("vllm.model_executor.layers"),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.conv",
        _stub_module("vllm.model_executor.layers.conv", Conv3dLayer=Conv3dLayer),
    )
    spec = importlib.util.spec_from_file_location("test_ascend_conv", CONV_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ascend_conv3d_forwards_to_conv3d_kernel(monkeypatch) -> None:
    AscendConv3dLayer = _load_ascend_conv3d(monkeypatch).AscendConv3dLayer
    layer = AscendConv3dLayer.__new__(AscendConv3dLayer)
    layer.weight = torch.randn(3, 2, 2, 2, 2)
    layer.bias = torch.randn(3)
    layer.stride = (1, 1, 1)
    layer.padding = (0, 0, 0)
    layer.dilation = (1, 1, 1)
    layer.groups = 1
    x = torch.randn(2, 2, 3, 4, 4)

    torch.testing.assert_close(
        layer.forward_oot(x),
        F.conv3d(x, layer.weight, layer.bias, stride=layer.stride),
    )


def test_ascend_conv3d_selects_forward_conv(monkeypatch) -> None:
    AscendConv3dLayer = _load_ascend_conv3d(monkeypatch).AscendConv3dLayer
    layer = AscendConv3dLayer.__new__(AscendConv3dLayer)
    expected = torch.empty(0)
    layer._forward_conv = Mock(return_value=expected)
    x = torch.empty(1, 1, 1, 1, 1)

    assert layer.forward_oot(x) is expected
    layer._forward_conv.assert_called_once_with(x)


def test_ascend_patch_registers_conv3d_only_as_custom_op(monkeypatch) -> None:
    class CustomOp:
        register_oot = Mock()

    class PluggableLayer:
        register_oot = Mock()

    conv = _load_ascend_conv3d(monkeypatch)
    custom_op_module = _stub_module(
        "vllm.model_executor.custom_op",
        CustomOp=CustomOp,
        PluggableLayer=PluggableLayer,
    )
    monkeypatch.setitem(sys.modules, "vllm.model_executor.custom_op", custom_op_module)
    monkeypatch.setitem(sys.modules, "vllm_fl", _stub_module("vllm_fl"))
    monkeypatch.setitem(sys.modules, "vllm_fl.configs", _stub_module("vllm_fl.configs"))
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.configs.ascend_cache",
        _stub_module("vllm_fl.configs.ascend_cache", refresh_block_size=Mock()),
    )
    prefix = "vllm_fl.dispatch.backends.vendor.ascend"
    monkeypatch.setitem(
        sys.modules,
        f"{prefix}.impl.conv",
        _stub_module(
            f"{prefix}.impl.conv",
            AscendConv3dLayer=conv.AscendConv3dLayer,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{prefix}.impl.gdn",
        _stub_module(
            f"{prefix}.impl.gdn", AscendGatedDeltaNetAttention=type("GDN", (), {})
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{prefix}.impl.layernorm",
        _stub_module(
            f"{prefix}.impl.layernorm",
            AscendGemmaRMSNorm=type("Gemma", (), {}),
            AscendRMSNorm=type("RMS", (), {}),
            AscendRMSNormGated=type("Gated", (), {}),
            ensure_ascend_rms_norm_gated_registered=Mock(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{prefix}.impl.mm_encoder_attention",
        _stub_module(
            f"{prefix}.impl.mm_encoder_attention",
            AscendMMEncoderAttention=type("MM", (), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{prefix}.impl.vocab_parallel_embedding",
        _stub_module(
            f"{prefix}.impl.vocab_parallel_embedding",
            AscendParallelLMHead=type("Head", (), {}),
            AscendVocabParallelEmbedding=type("Embedding", (), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{prefix}.ops.mla",
        _stub_module(
            f"{prefix}.ops.mla",
            AscendMultiHeadLatentAttention=type("MLA", (), {}),
            ensure_mla_forward_registered=Mock(),
        ),
    )

    spec = importlib.util.spec_from_file_location("test_ascend_patch", PATCH_PATH)
    assert spec is not None and spec.loader is not None
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    patch.patch_op_cls()

    registrations = {
        call.kwargs["name"]: call.kwargs["_decorated_op_cls"]
        for call in CustomOp.register_oot.call_args_list
    }
    assert registrations["Conv3dLayer"] is conv.AscendConv3dLayer
    assert all(
        call.kwargs.get("name") != "Conv3dLayer"
        for call in PluggableLayer.register_oot.call_args_list
    )
