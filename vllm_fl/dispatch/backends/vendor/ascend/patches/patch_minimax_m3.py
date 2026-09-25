# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-M3 Ascend adaptation: upstream injections + architecture registration.

Mirrors ``patch_deepseek_v4.py``: a thin, vendor-scoped entry point that the
parent ``patch`` module calls once per process.

Unlike DeepSeek-V4 or Qwen3.5, vLLM 0.24 **already ships** a complete
MiniMax-M3 implementation, so nothing is copied or subclassed here for its own
sake. What is missing on Ascend is platform behaviour, installed through a
small number of injections (see ``impl/minimax_m3_injections.py``):

* ``fused_minimax_m3_qknorm_rope_kv_insert`` -> this repository's CANN
  operators (QK-norm + partial RoPE + paged-cache insert),
* ``flashinfer.norm`` -> ``torch_npu`` (flashinfer does not exist on NPU),
* ``SiluAndMulWithClamp`` -> ``torch_npu.npu_clipped_swiglu``,
* ``fused_allreduce_gemma_rms_norm`` -> platform all-reduce + Gemma RMSNorm,
* the sparse-attention attend -> ``npu_sparse_attention_score``,
* the lightning indexer top-k -> the vendor decode split (see ``impl/indexer``),
* upstream's ``ApplyRotaryEmb`` -> a partial-rotary-correct override.

Additionally the vendor-owned ``models/minimax_m3_ascend.py`` classes are bound
in place of the upstream ones, because upstream's M3 carries no
``@support_torch_compile`` and vLLM routes such architectures to the
breakable-cudagraph path, which does not exist on NPU.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Architecture name -> the module:class vLLM should resolve.
_ARCHITECTURES = {
    "MiniMaxM3SparseForCausalLM": (
        "vllm.models.minimax_m3",
        "MiniMaxM3SparseForCausalLM",
    ),
    "MiniMaxM3SparseForConditionalGeneration": (
        "vllm.models.minimax_m3",
        "MiniMaxM3SparseForConditionalGeneration",
    ),
}

_installed = False


def apply_minimax_m3_patches() -> bool:
    """Install the Ascend adaptation and register the M3 architectures.

    Idempotent and safe to call in every plugin process (API server, engine
    core, spawned worker). Returns ``True`` when the upstream MiniMax-M3 runtime
    was present and adapted; ``False`` leaves the release untouched so non-M3
    models are never affected.
    """
    global _installed
    if _installed:
        return True

    from vllm_fl.dispatch.backends.vendor.ascend.impl.minimax_m3_injections import (
        install_minimax_m3_injections,
    )

    if not install_minimax_m3_injections():
        logger.info(
            "MiniMax-M3: upstream runtime not present in this vLLM build; "
            "no M3 adaptation installed"
        )
        return False

    from vllm_fl.dispatch.backends.vendor.ascend.impl.indexer.index_topk_ascend import (
        install_index_topk_replacement,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.minimax_m3_graph_replay import (
        install_minimax_m3_graph_conformance,
    )
    from vllm_fl.attention.ascend.sparse_attn_m3 import (
        install_sparse_attn_replacement,
    )
    from vllm_fl.models.minimax_m3_ascend import install_ascend_minimax_m3_model

    install_index_topk_replacement()
    install_sparse_attn_replacement()
    install_minimax_m3_graph_conformance()
    install_ascend_minimax_m3_model()

    from vllm.model_executor.models import registry as model_registry

    for architecture, (module, class_name) in _ARCHITECTURES.items():
        # Keep the source registries coherent for introspection, then pin the
        # already-materialised registry at the upstream path.
        model_registry._TEXT_GENERATION_MODELS.setdefault(
            architecture, (module.split(".")[-1], class_name)
        )
        model_registry._VLLM_MODELS.setdefault(
            architecture, (module.split(".")[-1], class_name)
        )
        model_registry.ModelRegistry.register_model(
            architecture, f"{module}:{class_name}"
        )

    _installed = True
    logger.info(
        "MiniMax-M3: Ascend adaptation installed and architectures registered"
    )
    return True


__all__ = ["apply_minimax_m3_patches"]
