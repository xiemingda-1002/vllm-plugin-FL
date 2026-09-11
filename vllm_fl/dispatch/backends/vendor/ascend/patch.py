# Copyright (c) 2026 BAAI. All rights reserved.

import logging

import vllm

logger = logging.getLogger(__name__)
_patches_applied = False
_op_classes_patched = False
_flashcomm_ops_and_layers_registered = False

def apply_ascend_patches():
    """Apply all Ascend-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    # Register the current-v0.24 Qwen kernels before importing the patch module
    # which installs model methods. These are required capabilities and must
    # fail visibly instead of leaving the CUDA/FlagGems implementation active.
    from .impl.linearnorm import split_qkv_rmsnorm_mrope  # noqa: F401
    from .impl.graph_fusion_ops import ensure_graph_fusion_ops_registered
    from .impl.moe_custom_ops import ensure_ascend_moe_custom_ops_registered

    ensure_graph_fusion_ops_registered()
    ensure_ascend_moe_custom_ops_registered()
    register_flashcomm_ops_and_layers()

    enable_ascend_native_ops()
    # Install before model-runner construction creates MambaCopyBuffers.
    # The installer owns its own idempotency and keeps all heavy imports lazy.
    from .patches.patch_mamba_utils import patch_mamba_batch_copy

    patch_mamba_batch_copy()
    patch_causal_conv1d()
    patch_fla_ops()
    from .patches import patch_qwen3_5  # noqa: F401

    patch_op_cls()
    patch_fused_moe()
    _patches_applied = True


def register_flashcomm_ops_and_layers() -> None:
    """Install current-rc1 FlashComm1 ops/layers only for Ascend startup."""
    global _flashcomm_ops_and_layers_registered
    if _flashcomm_ops_and_layers_registered:
        return

    from vllm.model_executor.custom_op import PluggableLayer

    from .impl.flashcomm_custom_ops import (
        ensure_ascend_flashcomm_custom_ops_registered,
    )
    from .impl.linear import (
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
    from .patches.patch_mamba_config import verify_and_update_config

    vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
    logger.info("Patched HybridAttentionMambaModelConfig for Ascend")

def patch_causal_conv1d():
    """Patch causal_conv1d ops with Ascend implementations."""
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
    import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

    from .impl.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_npu
    from .impl.causal_conv1d import causal_conv1d_update_npu

    _conv1d_lib.causal_conv1d_fn = causal_conv1d_fn_npu
    _conv1d_lib.causal_conv1d_update = causal_conv1d_update_npu
    _qwen3_next_lib.causal_conv1d_fn = causal_conv1d_fn_npu
    _qwen3_next_lib.causal_conv1d_update = causal_conv1d_update_npu
    logger.info("Patched causal_conv1d ops for Ascend")

def patch_fused_moe():
    """Patch fused MoE ops with Ascend implementations."""
    from .impl.fused_moe import fused_experts_impl

    import vllm_fl.ops.fused_moe.fused_moe as fused_moe_lib

    fused_moe_lib.fused_experts_impl = fused_experts_impl
    logger.info("Patched fused_moe for Ascend")

def patch_fla_ops():
    """Patch FLA ops and fused_gdn_gating with Ascend implementations."""
    import vllm.model_executor.layers.fla.ops as _fla_ops_lib
    import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_lib
    import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

    from .impl.fla.chunk import (
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

    from .impl.gdn import AscendGatedDeltaNetAttention
    from .impl.layernorm import (
        AscendGemmaRMSNorm,
        AscendRMSNorm,
        AscendRMSNormGated,
        ensure_ascend_rms_norm_gated_registered,
    )
    from .impl.mm_encoder_attention import AscendMMEncoderAttention
    from .impl.vocab_parallel_embedding import (
        AscendParallelLMHead,
        AscendVocabParallelEmbedding,
    )

    ensure_ascend_rms_norm_gated_registered()

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
    }.items():
        PluggableLayer.register_oot(_decorated_layer_cls=layer_cls, name=name)
    _op_classes_patched = True
    logger.info("Registered required Ascend Qwen custom ops")

def refresh_block_size(vllm_config, block_size=128):
    """
    Refresh the block size in cache config.
    """
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    if not cache_config:
        return

    if cache_config.block_size is None:
        cache_config.block_size = block_size

    if not scheduler_config or not model_config:
        return

    if model_config.hf_config.model_type == "deepseek_v4":
        if cache_config.block_size not in (32, 64, 128):
            logger.warning(
                "For deepseek_v4 model, block size should be 32, 64 or "
                "128. Setting block size to 32 for better performance."
            )
            cache_config.block_size = 32
        return

    if model_config.is_hybrid:
        # Hybrid attention+Mamba models have already selected a block size
        # from their state/page geometry. Resetting it here makes the worker
        # disagree with the engine and breaks the contiguous Ascend layout.
        return

    if cache_config.block_size != block_size and (
        cache_config.enable_prefix_caching
        or scheduler_config.enable_chunked_prefill
    ):
        logger.info(
            "Block size is set to %d if prefix cache or chunked prefill "
            "is enabled.",
            block_size,
        )
        cache_config.block_size = block_size
