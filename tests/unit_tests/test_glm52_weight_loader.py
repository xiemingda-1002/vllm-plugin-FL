"""Contracts for GLM-5.2's non-speculative target-model weight loader."""

from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("vllm")


def test_glm52_loader_skips_only_checkpoint_rot_weight() -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_glm52_weight_loader import (
        AscendGlmMoeDsaForCausalLM,
    )

    target = nn.Module()
    target.register_parameter("known", nn.Parameter(torch.zeros(1)))

    loaded = AscendGlmMoeDsaForCausalLM.load_weights(
        target,
        [("rot.weight", torch.ones(1))],
    )

    assert loaded == set()
    with pytest.raises(ValueError, match="named 'unexpected'"):
        AscendGlmMoeDsaForCausalLM.load_weights(
            target,
            [("unexpected.weight", torch.ones(1))],
        )


def test_glm52_loader_patch_is_idempotent_and_does_not_patch_mtp(
    monkeypatch,
) -> None:
    import vllm.model_executor.models.deepseek_mtp as deepseek_mtp
    import vllm.model_executor.models.deepseek_v2 as deepseek_v2
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_glm52_weight_loader import (
        AscendGlmMoeDsaForCausalLM,
        apply_glm52_weight_loader_patch,
    )

    original_glm = AscendGlmMoeDsaForCausalLM.__bases__[0]
    original_mtp = deepseek_mtp.DeepSeekMTP
    original_mtp_layer = deepseek_mtp.DeepSeekMultiTokenPredictorLayer
    original_spec_mapper = deepseek_mtp.get_spec_layer_idx_from_weight_name
    original_target_spec_mapper = deepseek_v2.get_spec_layer_idx_from_weight_name
    monkeypatch.setattr(deepseek_v2, "GlmMoeDsaForCausalLM", original_glm)

    apply_glm52_weight_loader_patch()
    apply_glm52_weight_loader_patch()

    assert deepseek_v2.GlmMoeDsaForCausalLM is AscendGlmMoeDsaForCausalLM
    assert deepseek_mtp.DeepSeekMTP is original_mtp
    assert deepseek_mtp.DeepSeekMultiTokenPredictorLayer is original_mtp_layer
    assert deepseek_mtp.get_spec_layer_idx_from_weight_name is original_spec_mapper
    assert deepseek_v2.get_spec_layer_idx_from_weight_name is original_target_spec_mapper


def test_ascend_startup_installs_glm52_loader_before_model_construction() -> None:
    source = (Path(__file__).resolve().parents[2] /
        "vllm_fl/dispatch/backends/vendor/ascend/patch.py"
    ).read_text()

    loader = source.index("apply_glm52_weight_loader_patch()")
    constructor = source.index("apply_glm52_shared_indexer_patch()")

    assert loader < constructor
