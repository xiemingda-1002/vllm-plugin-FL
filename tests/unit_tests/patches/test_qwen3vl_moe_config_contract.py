"""No-weight contract tests for the Ascend Qwen3-VL MoE config repair."""
from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch.nn as nn


ROOT = Path(__file__).parents[3]
QWEN_PATCH = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3vl.py"
MODEL_ENV_VAR = "FL_TEST_QWEN3VL_MODEL_DIR"


def _patch_module():
    pytest.importorskip("vllm")
    spec = importlib.util.spec_from_file_location(
        "fl_qwen3vl_moe_config_contract", QWEN_PATCH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_moe_init_preserves_constructor_config_and_signature():
    module = _patch_module()
    expected_model_config = object()
    vllm_config = SimpleNamespace(model_config=expected_model_config)

    class FakeMoE:
        def __init__(self, *, vllm_config, prefix=""):
            self.original_init_args = (vllm_config, prefix)

    original_signature = inspect.signature(FakeMoE.__init__)
    module._patch_qwen3vl_moe_model_config(FakeMoE)
    first_wrapped_init = FakeMoE.__init__
    module._patch_qwen3vl_moe_model_config(FakeMoE)

    instance = FakeMoE(vllm_config=vllm_config, prefix="visual")
    assert instance.model_config is expected_model_config
    assert instance.original_init_args == (vllm_config, "visual")
    assert FakeMoE.__init__ is first_wrapped_init
    assert inspect.signature(FakeMoE.__init__) == original_signature


def test_moe_init_does_not_replace_existing_model_config():
    module = _patch_module()
    existing_model_config = object()
    vllm_config = SimpleNamespace(model_config=object())

    class FakeMoE:
        def __init__(self, *, vllm_config, prefix=""):
            self.model_config = existing_model_config

    module._patch_qwen3vl_moe_model_config(FakeMoE)
    assert FakeMoE(vllm_config=vllm_config).model_config is existing_model_config


def test_preserved_config_unblocks_inherited_encoder_config_method(monkeypatch):
    module = _patch_module()
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    import vllm.model_executor.models.qwen3_vl as qwen3_vl

    model_config = SimpleNamespace(max_model_len=8192)
    multimodal_config = SimpleNamespace(get_limit_per_prompt=lambda modality: 1)
    vllm_config = SimpleNamespace(model_config=model_config)
    processing_info = SimpleNamespace(
        get_num_frames_with_most_features=lambda **kwargs: 6
    )
    monkeypatch.setattr(
        qwen3_vl.MULTIMODAL_REGISTRY,
        "get_processing_info",
        lambda received_model_config: (
            processing_info
            if received_model_config is model_config
            else pytest.fail("inherited method did not receive instance model_config")
        ),
    )

    class FakeMoE:
        get_max_frames_per_video = Qwen3VLForConditionalGeneration.get_max_frames_per_video
        get_encoder_cudagraph_config = Qwen3VLForConditionalGeneration.get_encoder_cudagraph_config

        def __init__(self, *, vllm_config, prefix=""):
            self.is_multimodal_pruning_enabled = False
            self.multimodal_config = multimodal_config
            self.visual = SimpleNamespace(out_hidden_size=2048)

    module._patch_qwen3vl_moe_model_config(FakeMoE)
    instance = FakeMoE(vllm_config=vllm_config)
    config = instance.get_encoder_cudagraph_config()

    assert config.modalities == ["image", "video"]
    assert config.out_hidden_size == 2048
    assert config.max_frames_per_video == 6


def test_real_moe_inherited_encoder_config_accepts_processor_model_config():
    """Exercise the real inherited method without running the MoE initializer.

    The object is intentionally created via ``__new__`` plus ``nn.Module``
    initialization, so this checks the configuration protocol only and neither
    allocates model layers nor opens model weights.
    """
    pytest.importorskip("vllm")
    from vllm.config import ModelConfig
    from vllm.model_executor.models.qwen3_vl_moe import (
        Qwen3VLMoeForConditionalGeneration,
    )

    model_path = os.environ.get(MODEL_ENV_VAR)
    if not model_path or not Path(model_path).is_dir():
        pytest.skip(f"set {MODEL_ENV_VAR} to a local Qwen3-VL model directory")

    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 2, "video": 0},
    )
    instance = object.__new__(Qwen3VLMoeForConditionalGeneration)
    nn.Module.__init__(instance)
    instance.model_config = model_config
    instance.multimodal_config = model_config.multimodal_config
    instance.is_multimodal_pruning_enabled = False
    instance.visual = SimpleNamespace(out_hidden_size=2048)

    config = instance.get_encoder_cudagraph_config()
    assert config.modalities == ["image", "video"]
    assert config.out_hidden_size == 2048
    assert config.max_frames_per_video >= 1
