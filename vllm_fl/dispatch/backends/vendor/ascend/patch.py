# Copyright (c) 2026 BAAI. All rights reserved.

import logging

import vllm
from vllm_fl.configs.ascend_cache import refresh_block_size

logger = logging.getLogger(__name__)
_patches_applied = False
_op_classes_patched = False
_flashcomm_ops_and_layers_registered = False

def apply_ascend_patches():
    """Apply all Ascend-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_distributed import (
        apply_ascend_distributed_patch,
    )

    apply_ascend_distributed_patch()
    from vllm_fl.scheduling.ascend_balance import apply_balance_scheduling_patch
    # Register the current-v0.24 Qwen kernels before importing the patch module
    # which installs model methods. These are required capabilities and must
    # fail visibly instead of leaving the CUDA/FlagGems implementation active.
    from vllm_fl.kv_cache.ascend.deepseek_v4_kv_cache import apply_deepseek_v4_kv_cache_patches
    from vllm_fl.dispatch.backends.vendor.ascend.impl.graph_fusion_ops import ensure_graph_fusion_ops_registered
    from vllm_fl.dispatch.backends.vendor.ascend.impl.linearnorm import split_qkv_rmsnorm_mrope  # noqa: F401
    from vllm_fl.dispatch.backends.vendor.ascend.impl.moe_custom_ops import ensure_ascend_moe_custom_ops_registered
    from vllm_fl.dispatch.backends.vendor.ascend.ops.dsa import ensure_dsa_forward_registered
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_glm52 import apply_glm52_shared_indexer_patch
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_glm52_weight_loader import (
        apply_glm52_weight_loader_patch,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_mla_prefill_backend import (
        apply_ascend_mla_prefill_backend_patch,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_deepseek_v4 import apply_deepseek_v4_patches
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_minimax_m3 import (
        apply_minimax_m3_patches,
    )

    ensure_graph_fusion_ops_registered()
    apply_balance_scheduling_patch()
    ensure_ascend_moe_custom_ops_registered()
    ensure_dsa_forward_registered()
    apply_deepseek_v4_patches()
    # MiniMax-M3: upstream M3 ships in vLLM 0.24; only Ascend behaviour is
    # installed here (injections + architecture registration).
    apply_minimax_m3_patches()
    # MLAAttention constructs this auxiliary object while loading every MLA
    # model. Install the Ascend boundary before any model constructor runs.
    apply_ascend_mla_prefill_backend_patch()
    # GLM W8A8 checkpoints may carry an MTP-only rot.weight even when the
    # target model runs without speculative decoding. Skip exactly that
    # tensor before vLLM attempts target-model module resolution.
    apply_glm52_weight_loader_patch()
    apply_glm52_shared_indexer_patch()
    apply_deepseek_v4_kv_cache_patches()
    register_flashcomm_ops_and_layers()

    enable_ascend_native_ops()
    # Install before model-runner construction creates MambaCopyBuffers.
    # The installer owns its own idempotency and keeps all heavy imports lazy.
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_mamba_utils import patch_mamba_batch_copy

    patch_mamba_batch_copy()
    patch_causal_conv1d()
    patch_fla_ops()
    from vllm_fl.dispatch.backends.vendor.ascend.patches import patch_qwen3_5  # noqa: F401
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_qwen3vl import apply_qwen3vl_patch

    apply_qwen3vl_patch()
    patch_op_cls()
    patch_fused_moe()
    _patches_applied = True


def register_flashcomm_ops_and_layers() -> None:
    """Install current-rc1 FlashComm1 ops/layers only for Ascend startup."""
    global _flashcomm_ops_and_layers_registered
    if _flashcomm_ops_and_layers_registered:
        return

    from vllm.model_executor.custom_op import PluggableLayer

    from vllm_fl.dispatch.backends.vendor.ascend.impl.flashcomm_custom_ops import (
        ensure_ascend_flashcomm_custom_ops_registered,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.linear import (
        AscendColumnParallelLinear,
        AscendMergedColumnParallelLinear,
        AscendQKVParallelLinear,
        AscendReplicatedLinear,
        AscendRowParallelLinear,
        ensure_ascend_linear_custom_ops_registered,
    )

    ensure_ascend_flashcomm_custom_ops_registered()
    ensure_ascend_linear_custom_ops_registered()
    for name, layer_cls in {
        "QKVParallelLinear": AscendQKVParallelLinear,
        "MergedColumnParallelLinear": AscendMergedColumnParallelLinear,
        "ColumnParallelLinear": AscendColumnParallelLinear,
        "RowParallelLinear": AscendRowParallelLinear,
        "ReplicatedLinear": AscendReplicatedLinear,
    }.items():
        PluggableLayer.register_oot(
            _decorated_layer_cls=layer_cls,
            name=name,
        )

    _flashcomm_ops_and_layers_registered = True
    logger.info("Registered Ascend FlashComm1 ops and linear layers")


def enable_ascend_native_ops() -> None:
    """Require FL's native Qwen GDN provider on an installed Ascend wheel."""
    from vllm_fl.ascend_custom_ops import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError(
            "FL Ascend Qwen GDN native payload is unavailable. Install with "
            "VLLM_VENDOR=ascend and a matching SOC_VERSION."
        )


def patch_mamba_config():
    """Patch HybridAttentionMambaModelConfig for Ascend."""
    from vllm_fl.configs.ascend_mamba import verify_and_update_config

    vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
    logger.info("Patched HybridAttentionMambaModelConfig for Ascend")


def patch_causal_conv1d():
    """Patch causal_conv1d ops with Ascend implementations."""
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
    import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

    from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_npu
    from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import causal_conv1d_update_npu

    _conv1d_lib.causal_conv1d_fn = causal_conv1d_fn_npu
    _conv1d_lib.causal_conv1d_update = causal_conv1d_update_npu
    _qwen3_next_lib.causal_conv1d_fn = causal_conv1d_fn_npu
    _qwen3_next_lib.causal_conv1d_update = causal_conv1d_update_npu
    logger.info("Patched causal_conv1d ops for Ascend")

def patch_fused_moe():
    """Patch fused MoE ops with Ascend implementations."""
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe import fused_experts_impl

    import vllm_fl.ops.fused_moe.fused_moe as fused_moe_lib

    fused_moe_lib.fused_experts_impl = fused_experts_impl
    logger.info("Patched fused_moe for Ascend")

def patch_fla_ops():
    """Patch FLA ops and fused_gdn_gating with Ascend implementations."""
    import vllm.model_executor.layers.fla.ops as _fla_ops_lib
    import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_lib
    import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

    from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.chunk import (
        chunk_gated_delta_rule,
        chunk_gated_delta_rule_fwd,
    )

    _fla_ops_lib.chunk_gated_delta_rule = chunk_gated_delta_rule
    _fla_ops_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
    _fla_chunk_lib.chunk_gated_delta_rule = chunk_gated_delta_rule
    _fla_chunk_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
    _qwen3_next_lib.chunk_gated_delta_rule = chunk_gated_delta_rule
    logger.info("Patched FL-owned current-v0.24 FLA ops for Ascend")

def patch_op_cls():
    """Register the model-layer implementations required by Ascend."""
    global _op_classes_patched
    if _op_classes_patched:
        return

    from vllm.model_executor.custom_op import CustomOp, PluggableLayer

    from vllm_fl.dispatch.backends.vendor.ascend.impl.gdn import AscendGatedDeltaNetAttention
    from vllm_fl.dispatch.backends.vendor.ascend.impl.layernorm import (
        AscendGemmaRMSNorm,
        AscendRMSNorm,
        AscendRMSNormGated,
        ensure_ascend_rms_norm_gated_registered,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.mm_encoder_attention import AscendMMEncoderAttention
    from vllm_fl.dispatch.backends.vendor.ascend.ops.mla import (
        AscendMultiHeadLatentAttention,
        ensure_mla_forward_registered,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.vocab_parallel_embedding import (
        AscendParallelLMHead,
        AscendVocabParallelEmbedding,
    )

    ensure_ascend_rms_norm_gated_registered()
    ensure_mla_forward_registered()

    for name, op_cls in {
        "MMEncoderAttention": AscendMMEncoderAttention,
        "GatedDeltaNetAttention": AscendGatedDeltaNetAttention,
        "RMSNorm": AscendRMSNorm,
        "GemmaRMSNorm": AscendGemmaRMSNorm,
        "RMSNormGated": AscendRMSNormGated,
    }.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)
    for name, layer_cls in {
        "VocabParallelEmbedding": AscendVocabParallelEmbedding,
        "ParallelLMHead": AscendParallelLMHead,
        # SFA uses the rc1 boundary; dense MLA delegates to its existing
        # upstream wrapper. DeepSeek-V4 DSA has a separate execution chain.
        "MultiHeadLatentAttentionWrapper": AscendMultiHeadLatentAttention,
    }.items():
        PluggableLayer.register_oot(_decorated_layer_cls=layer_cls, name=name)
    _op_classes_patched = True
    logger.info("Registered Ascend custom ops and pluggable layers")
