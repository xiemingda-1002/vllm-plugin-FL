# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend GLM-5.2 target-model weight-loader compatibility.

Current GLM-5.2 W8A8 checkpoints can contain ``rot.weight`` for the optional
MTP path even when speculative decoding is disabled.  The target causal model
does not own that tensor, so current vLLM-Ascend skips it before module
resolution.  Keep that target-model compatibility separate from the MTP
implementation, which is intentionally outside the current FL scope.
"""

import vllm.model_executor.models.deepseek_v2 as deepseek_v2
from vllm.model_executor.models.utils import AutoWeightsLoader

ROT_WEIGHT_NAME = "rot.weight"


class AscendGlmMoeDsaForCausalLM(deepseek_v2.GlmMoeDsaForCausalLM):
    """GLM target model that ignores the checkpoint's MTP-only rot tensor."""

    _fl_glm52_rot_weight_loader = True

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self, skip_prefixes=[ROT_WEIGHT_NAME])
        return loader.load_weights(weights)


def apply_glm52_weight_loader_patch() -> None:
    """Install the rc1 GLM target-model loader before lazy model resolution."""
    current = deepseek_v2.GlmMoeDsaForCausalLM
    if getattr(current, "_fl_glm52_rot_weight_loader", False):
        return
    deepseek_v2.GlmMoeDsaForCausalLM = AscendGlmMoeDsaForCausalLM
