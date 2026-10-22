# Copyright (c) 2026 BAAI. All rights reserved.

import logging

logger = logging.getLogger(__name__)
_patches_applied = False

def apply_ascend_patches():
    """Apply all Ascend-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True
    # Patch modules for Ascend platform
    patch_causal_conv1d()
    patch_fla_ops()
    patch_op_cls()
    patch_fused_moe()
    patch_gdn()
    patch_vision_position_interpolation()
    patch_sampler_filter()
    patch_triton_ops()
    from .patches.patch_mamba_utils import apply_mamba_copy_patch

    apply_mamba_copy_patch()


def patch_sampler_filter():
    """Use the packaged A2/A3 filter while retaining vLLM's sampler contract."""
    import torch_npu
    from vllm.v1.sample.ops import topk_topp_sampler

    def apply_top_k_top_p(logits, k, p):
        if k is None and p is None:
            return logits
        # vLLM's metadata uses int64; CANN's filter accepts int32 only.
        if k is not None:
            import torch

            k = k.to(torch.int32)
        return torch_npu.npu_top_k_top_p(logits, k=k, p=p)

    # vLLM chooses its CUDA-oriented Triton filter for batches >=8 even on
    # OOT platforms. Match rc1's built-in A2/A3 filtering path instead.
    topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p


def patch_vision_position_interpolation():
    """Use vLLM's tensor reference instead of the CUDA-oriented launch grid."""
    from vllm.model_executor.models import qwen3_vl

    # The Triton implementation launches one block per spatial patch. Vision
    # profiling can exceed Ascend's 65535-block limit; the upstream tensor
    # implementation has the same interpolation/spatial-merge contract.
    qwen3_vl.triton_pos_embed_interpolate = qwen3_vl.pos_embed_interpolate_native


def patch_gdn():
    """Keep the current upstream GDN model, replacing its numerical core."""
    from vllm.model_executor.custom_op import PluggableLayer

    from .impl.gdn import AscendQwenGatedDeltaNetAttention

    PluggableLayer.register_oot(
        _decorated_layer_cls=AscendQwenGatedDeltaNetAttention,
        name="QwenGatedDeltaNetAttention",
    )

def patch_causal_conv1d():
    """Patch causal_conv1d ops with Ascend implementations."""
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

        from .impl.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_npu
        from .impl.causal_conv1d import causal_conv1d_update_npu

        _conv1d_lib.causal_conv1d_fn = causal_conv1d_fn_npu
        _conv1d_lib.causal_conv1d_update = causal_conv1d_update_npu
        _qwen3_next_lib.causal_conv1d_fn = causal_conv1d_fn_npu
        _qwen3_next_lib.causal_conv1d_update = causal_conv1d_update_npu
        logger.info("Patched causal_conv1d ops for Ascend")
    except Exception as e:
        logger.warning("Failed to patch causal_conv1d ops: %s", e)

def patch_fused_moe():
    """Patch fused MoE ops with Ascend implementations."""
    # TODO ops' triton implementation is not ready yet
    from .impl.fused_moe import fused_experts_impl
    try:
        import vllm_fl.ops.fused_moe.fused_moe as fused_moe_lib

        fused_moe_lib.fused_experts_impl = fused_experts_impl

        logger.info("Patched fused_moe for Ascend")
    except Exception as e:
        logger.warning("Failed to patch fused_moe ops: %s", e)

def patch_fla_ops():
    """Patch FLA ops and fused_gdn_gating with Ascend implementations."""
    try:
        import vllm.model_executor.layers.fla.ops as _fla_ops_lib
        import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_lib
        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_lib
        import vllm.model_executor.layers.fla.ops.layernorm_guard as _fla_layernorm_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib
        from flag_gems.runtime.backend._ascend.fla import (
            chunk_gated_delta_rule_fwd,
            fused_recurrent_gated_delta_rule_fwd,
        )
        from flag_gems.runtime.backend._ascend.fla.layernorm_guard import (
            LayerNormFn as ascend_LayerNormFn,
        )

        from .impl.fla import chunk_gated_delta_rule_npu

        _fla_ops_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
        _fla_chunk_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
        _fla_chunk_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        _fla_recurrent_lib.fused_recurrent_gated_delta_rule_fwd = fused_recurrent_gated_delta_rule_fwd
        _fla_layernorm_lib.LayerNormFn = ascend_LayerNormFn
        _qwen3_next_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        logger.info("Patched FLA ops for Ascend")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)

def patch_op_cls():
    """Patch MMEncoderAttention to use manual matmul attention on NPU.

    The NPU npu_fused_infer_attention_score kernel only supports head_dim
    in {64, 128, 192}. The vision encoder may have non-standard head_dim
    (e.g. 72 for Qwen3.5). F.scaled_dot_product_attention on NPU may also
    dispatch to the same problematic kernel. Use pure-PyTorch matmul
    attention instead.
    """
    try:
        from vllm.model_executor.custom_op import CustomOp

        from .impl.mm_encoder_attention import AscendMMEncoderAttention
        from .impl.vocab_parallel_embedding import AscendVocabParallelEmbedding
        REGISTERED_ASCEND_OPS = {
            "VocabParallelEmbedding": AscendVocabParallelEmbedding,
            "MMEncoderAttention": AscendMMEncoderAttention,
        }
        for name, op_cls in REGISTERED_ASCEND_OPS.items():
            CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)
        logger.info("Patched MMEncoderAttention for NPU (matmul attention)")
    except Exception as e:
        logger.warning("Failed to patch MMEncoderAttention: %s", e)

def refresh_block_size(vllm_config, block_size = 128):
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

    # Hybrid state pages are aligned by the Mamba configuration verifier.
    if not model_config.is_hybrid and cache_config.block_size != block_size:
        if cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill:
            logger.info(f"Block size is set to {block_size} if prefix cache or chunked prefill is enabled.")
            cache_config.block_size = block_size


def patch_triton_ops():
    """Route torch_npu.npu_* names to the Triton implementations under .impl.

    Note: ``npu_format_cast`` is deliberately NOT routed.  It is a storage-format
    cast (NZ fractal layout), so a Triton port has no meaningful semantics, and going
    through the proxy breaks torch_npu's lazy AclOp initialisation: dsv4's weight
    loading hits it via ``impl/quantization/linear_utils.py:111`` and the process dies
    with ``SetPrecisionMode ... AclSetCompileopt(ACL_PRECISION_MODE)``.

    Each target is replaced by a proxy object rather than a plain function.  A
    plain function breaks call sites that use an op's ``out=`` variant, e.g.
    ``torch_npu.npu_fused_infer_attention_score.out(...)`` in
    ``impl/attention.py`` L681/L922: the native object exposes ``.out`` while a
    function does not, which surfaced as
    ``AttributeError: 'function' object has no attribute 'out'`` and made graph
    capture fail with ``AclmdlRICaptureEnd ... 107033``.  The proxy forwards any
    attribute access to the native op and only intercepts ``__call__``, where it
    tries the Triton implementation first and falls back to native on exception.

    The proxy is installed on two independent lookup paths:
    ``torch_npu.<name>`` (the module attribute) and ``torch.ops.npu.<name>``
    (the dispatcher namespace entry).  Call sites use both spellings; e.g. the
    MiniMax-M3 injection helpers call ``torch.ops.npu.npu_rms_norm``,
    ``torch.ops.npu.npu_add_rms_norm`` and ``torch.ops.npu.npu_clipped_swiglu``
    exclusively, so patching the module attribute alone never saw them.
    """
    import importlib
    import logging

    import torch
    import torch_npu

    _log = logging.getLogger(__name__)

    # Overriding torch.ops.npu.* also perturbs torch_npu's LazyInitAclops ordering:
    # with these nine overrides installed, DeepSeek-V4's SFA attention dies on
    #   SetPrecisionMode ... AclSetCompileopt(ACL_PRECISION_MODE)
    # while the 25 torch_npu.* targets alone run fine (measured: SETP 8 -> 0).
    # Flip this to True only when a model actually calls the torch.ops.npu spelling
    # (MiniMax-M3 does) and has been validated on the box.
    _ENABLE_OPS_NPU_OVERRIDES = False


    targets = [
        ("npu_swiglu", "impl.swiglu", "swiglu"),
        ("npu_clipped_swiglu", "impl.clipped_swiglu", "clipped_swiglu"),
        ("npu_gelu_mul", "impl.gelu_mul", "gelu_mul"),
        ("npu_grouped_matmul", "impl.grouped_matmul", "grouped_matmul"),
        ("npu_moe_token_unpermute", "impl.moe_token_unpermute", "moe_token_unpermute"),
        ("npu_moe_token_permute", "impl.moe_token_permute", "moe_token_permute"),
        ("npu_moe_init_routing_v2", "impl.moe_init_routing_v2", "moe_init_routing_v2"),
        ("npu_rms_norm", "impl.rms_norm", "npu_rms_norm"),
        ("npu_kv_rmsnorm_rope_cache", "impl.kv_rmsnorm_rope_cache", "npu_kv_rmsnorm_rope_cache"),
        ("npu_dynamic_quant", "impl.dynamic_quant", "dynamic_quant"),
        ("npu_dynamic_mx_quant", "impl.dynamic_mx_quant", "dynamic_mx_quant"),
        ("npu_dynamic_block_quant", "impl.dynamic_block_quant", "dynamic_block_quant"),
        ("npu_quant_matmul", "impl.quant_matmul", "quant_matmul"),
        ("npu_quantize", "impl.quantize", "quantize"),
        ("npu_fused_infer_attention_score", "impl.fused_infer_attention_score",
         "fused_infer_attention_score"),
        ("npu_fusion_attention", "impl.fusion_attention", "fusion_attention"),
        ("npu_gather_pa_kv_cache", "impl.gather_pa_kv_cache", "gather_pa_kv_cache"),
        ("npu_kv_quant_sparse_flash_attention", "impl.kv_quant_sparse_flash_attention",
         "kv_quant_sparse_flash_attention"),
        ("npu_scatter_nd_update", "impl.scatter_nd_update", "scatter_nd_update"),
        ("npu_transpose_batchmatmul", "impl.transpose_batchmatmul", "transpose_batchmatmul"),
        ("npu_transpose_quant_batchmatmul", "impl.transpose_quant_batchmatmul",
         "transpose_quant_batchmatmul"),
        # GLM-5 / GLM-5.2: sparse indexer and the paged-KV scatter.  The
        # trailing underscore in npu_scatter_nd_update_ is part of the name
        # used by sfa_v1.py L1780/L1854/L1905/L1923; the non-underscored op
        # above is a different torch_npu entry point.
        ("npu_lightning_indexer", "impl.lightning_indexer", "lightning_indexer"),
        ("npu_quant_lightning_indexer", "impl.lightning_indexer_quant",
         "lightning_indexer_quant"),
        ("npu_scatter_nd_update_", "impl.scatter_nd_update", "scatter_nd_update"),
        # MiniMax-M3: torch.ops.npu spelling, see the dispatcher block below.
        ("npu_add_rms_norm", "impl.add_rms_norm", "npu_add_rms_norm"),
    ]

    class _TritonOpProxy:
        """Call the Triton implementation, keep the native op's attributes."""

        __slots__ = ("_native", "_impl", "_name")

        def __init__(self, native, impl, name):
            object.__setattr__(self, "_native", native)
            object.__setattr__(self, "_impl", impl)
            object.__setattr__(self, "_name", name)

        def __call__(self, *args, **kwargs):
            # Inside a torch.compile region Dynamo cannot inline a Python wrapper
            # that carries try/except + logging: it marks the frame "skipped" and
            # the enclosing graph fails with
            #   RuntimeError: Attempted to inline function marked as skipped
            # (seen on the Qwen3-VL profile_run path, vllm_fl/compilation/graph.py
            # :203).  Handing the compiled region straight to the native op keeps
            # the graph valid; the Triton implementation is used on the eager
            # paths, which is where these operators are actually dispatched today.
            if torch.compiler.is_compiling():
                return self._native(*args, **kwargs)
            try:
                return self._impl(*args, **kwargs)
            except Exception as exc:
                _log.warning("[TRITON-OPS] %s fallback: %r", self._name, exc)
                return self._native(*args, **kwargs)

        def __getattr__(self, item):
            return getattr(self._native, item)

    installed = []
    active, failed = [], []
    for attr, modname, fnname in targets:
        native = getattr(torch_npu, attr, None)
        if native is None:
            failed.append("%s(no such attr)" % attr)
            continue
        try:
            mod = importlib.import_module("." + modname, package=__package__)
            impl = getattr(mod, fnname)
        except Exception as exc:
            failed.append("%s(import:%r)" % (attr, exc))
            continue
        setattr(torch_npu, attr, _TritonOpProxy(native, impl, attr))
        active.append(attr)
        installed.append((attr, impl))

    # ``torch.ops.npu`` is a separate lookup path from ``torch_npu.<name>``:
    # it is a ``torch._ops._OpNamespace`` whose ``__getattr__`` builds an
    # ``OpOverloadPacket`` from the dispatcher and caches it with ``setattr``
    # (torch/_ops.py, ``_OpNamespace.__getattr__``).  Shadow those cached
    # namespace entries with the same proxy.  Assigning to them is supported
    # because ``_OpNamespace`` is a module subclass without ``__slots__``, and
    # the proxy keeps forwarding ``.default`` / ``.out`` to the native packet.
    shadowed, absent = [], []
    if _ENABLE_OPS_NPU_OVERRIDES:
        npu_ops = getattr(torch.ops, "npu", None)
        if npu_ops is None:
            failed.append("torch.ops.npu(no such namespace)")
        else:
            for attr, impl in installed:
                packet = getattr(npu_ops, attr, None)
                if packet is None:
                    # Not every torch_npu Python entry point is also registered in
                    # the ``npu`` dispatcher namespace; only shadow what exists.
                    absent.append(attr)
                    continue
                try:
                    setattr(npu_ops, attr, _TritonOpProxy(packet, impl,
                                                          "torch.ops.npu." + attr))
                except Exception as exc:
                    failed.append("torch.ops.npu.%s(setattr:%r)" % (attr, exc))
                    continue
                shadowed.append(attr)

    _log.info("[TRITON-OPS] ACTIVE=%d %s", len(active), ",".join(active))
    _log.info("[TRITON-OPS] DISPATCHER=%d %s", len(shadowed), ",".join(shadowed))
    if absent:
        _log.info("[TRITON-OPS] torch.ops.npu missing %d: %s",
                  len(absent), ",".join(absent))
    if failed:
        _log.warning("[TRITON-OPS] FAILED=%d %s", len(failed), "; ".join(failed))
