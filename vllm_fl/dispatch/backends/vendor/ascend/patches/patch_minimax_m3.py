# Copyright (c) 2026 BAAI. All rights reserved.
"""Atomic, model-scoped MiniMax-M3 adaptation for Ascend."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from threading import RLock

logger = logging.getLogger(__name__)
_installed = False
_install_lock = RLock()
_ARCHITECTURES = {
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
}
_VENDOR = "vllm_fl.dispatch.backends.vendor.ascend.impl"


class _PatchTransaction:
    """Restore precisely the prepared Python namespaces if installation fails."""

    def __init__(self, objects):
        self.snapshots = [(obj, dict(vars(obj))) for obj in objects]

    def rollback(self):
        for obj, before in reversed(self.snapshots):
            for name in set(vars(obj)) - set(before):
                delattr(obj, name)
            for name, value in before.items():
                if vars(obj).get(name) is not value:
                    setattr(obj, name, value)


def _require_call(function, *args, **kwargs):
    """Reject incompatible upstream call contracts before patching anything."""
    try:
        inspect.signature(function).bind(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"MiniMax-M3 incompatible API: {function.__module__}.{function.__qualname__}"
        ) from exc


def _prepare_patches():
    """Resolve dependencies and all mutation targets before changing upstream."""
    names = (
        "vllm._custom_ops",
        "vllm.models.minimax_m3.nvidia.model",
        "vllm.models.minimax_m3.common.vision_tower",
        "vllm.models.minimax_m3.common.sparse_attention",
        "vllm.models.minimax_m3.common.indexer",
        "vllm.models.minimax_m3.common.ops.index_topk",
        f"{_VENDOR}.minimax_m3_injections",
        f"{_VENDOR}.minimax_m3_rotary",
        f"{_VENDOR}.minimax_m3_graph_replay",
        f"{_VENDOR}.attention.sparse_attn_m3",
        f"{_VENDOR}.indexer.index_topk_ascend",
        f"{_VENDOR}.indexer.index_vendor",
        f"{_VENDOR}.indexer.index_decode_vendor",
        "vllm_fl.models.minimax_m3_ascend",
    )
    modules = {name: importlib.import_module(name) for name in names}
    required = {
        names[0]: ("fused_minimax_m3_qknorm_rope_kv_insert",),
        names[1]: (
            "MiniMAXGemmaRMSNorm",
            "SiluAndMulWithClamp",
            "fused_allreduce_gemma_rms_norm",
            "MiniMaxM3Attention",
            "MiniMaxM3SparseAttention",
            "MiniMaxM3Model",
        ),
        names[2]: ("ApplyRotaryEmb",),
        names[3]: (
            "MiniMaxM3SparseBackend",
            "MiniMaxM3SparseImpl",
            "MiniMaxM3SparseTritonImpl",
        ),
        names[4]: ("MiniMaxM3IndexerImpl",),
        names[5]: (
            "minimax_m3_index_topk",
            "minimax_m3_index_decode",
            "minimax_m3_index_score",
        ),
    }
    for name, attributes in required.items():
        for attribute in attributes:
            if not callable(getattr(modules[name], attribute, None)):
                raise RuntimeError(f"MiniMax-M3 requires {name}.{attribute}")

    model = modules[names[1]]
    _require_call(
        model.SiluAndMulWithClamp.__init__, None, swiglu_limit=7.0, alpha=1.0, beta=0.0
    )
    _require_call(
        modules[names[2]].ApplyRotaryEmb.__init__,
        None,
        enforce_enable=True,
        enable_fp32_compute=True,
    )
    norm_layer = importlib.import_module("vllm.model_executor.layers.layernorm")
    _require_call(norm_layer.GemmaRMSNorm.__init__, None, 128, eps=1e-6)
    _require_call(
        modules[names[3]].MiniMaxM3SparseTritonImpl.forward,
        None,
        layer=None,
        query=None,
        kv_cache=None,
        output=None,
    )
    qkv = importlib.import_module(f"{_VENDOR}.linearnorm.minimax_m3_qknorm_rope")
    expected = inspect.signature(qkv.fused_minimax_m3_qknorm_rope_kv_insert)
    upstream = inspect.signature(
        modules[names[0]].fused_minimax_m3_qknorm_rope_kv_insert
    )
    for name, parameter in expected.parameters.items():
        actual = upstream.parameters.get(name)
        if actual is None or (actual.kind, actual.default) != (
            parameter.kind,
            parameter.default,
        ):
            raise RuntimeError(
                f"MiniMax-M3 incompatible QK-norm/RoPE parameter: {name}"
            )
    _require_call(
        modules[names[0]].fused_minimax_m3_qknorm_rope_kv_insert,
        **dict.fromkeys(expected.parameters),
    )

    selector = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.ascend.attention_selector"
    )
    kernels = selector.get_minimax_sparse_kernels()
    _require_call(kernels.qknorm_rope_cache, **dict.fromkeys(expected.parameters))
    _require_call(kernels.rotary, None, None, None, True)
    _require_call(kernels.gemma_weight, None)
    _require_call(kernels.attend, *([None] * 10), plane_major=False)
    # Schemas are inert, process-lifetime capabilities, prepared before any
    # upstream wrapper changes. A schema-registration failure propagates here.
    modules["vllm_fl.models.minimax_m3_ascend"]._register_ops()

    # Capture all already-imported MiniMax consumers that may hold copied names.
    objects = list(modules.values())
    objects.extend(
        module
        for name, module in list(sys.modules.items())
        if name.startswith("vllm.models.minimax_m3") and module is not None
    )
    sparse = modules[names[3]]
    objects.extend(
        (
            sparse.MiniMaxM3SparseBackend,
            sparse.MiniMaxM3SparseImpl,
            sparse.MiniMaxM3SparseTritonImpl,
            modules[names[4]].MiniMaxM3IndexerImpl,
        )
    )
    # Each target is captured once, preserving class/module identity on rollback.
    objects = list({id(obj): obj for obj in objects}.values())
    return modules, objects


def apply_minimax_m3_patches(vllm_config) -> bool:
    """Install only for an Ascend MiniMax config; propagate every failure.

    vLLM already owns architecture registration. Rebind only MiniMax's actual
    construction modules and preserve the upstream registry and other models.
    """
    from vllm.platforms import current_platform

    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or ()
    if getattr(
        current_platform, "vendor_name", None
    ) != "ascend" or not _ARCHITECTURES.intersection(architectures):
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    if getattr(cache_config, "cache_dtype", "auto") != "auto":
        raise ValueError("MiniMax-M3 Ascend supports only cache_dtype='auto'")

    global _installed
    with _install_lock:
        if _installed:
            return True
        modules, objects = _prepare_patches()
        transaction = _PatchTransaction(objects)
        try:
            injections = modules[f"{_VENDOR}.minimax_m3_injections"]
            indexer = modules[f"{_VENDOR}.indexer.index_topk_ascend"]
            sparse = modules[f"{_VENDOR}.attention.sparse_attn_m3"]
            graph = modules[f"{_VENDOR}.minimax_m3_graph_replay"]
            model = modules["vllm_fl.models.minimax_m3_ascend"]
            installers = [
                model._install_official_gemma_rmsnorm,
                injections.install_minimax_m3_injections,
                indexer.install_index_topk_replacement,
            ]
            if os.environ.get("VLLM_FL_M3_NPU_SPARSE_ATTN", "1") == "1":
                installers.append(sparse.install_sparse_attn_replacement)
            installers.extend(
                (
                    graph.install_minimax_m3_graph_conformance,
                    model.install_ascend_minimax_m3_model,
                )
            )
            for install in installers:
                if not install():
                    raise RuntimeError(
                        f"MiniMax-M3 required adapter failed: {install.__name__}"
                    )
        except BaseException:
            transaction.rollback()
            raise
        _installed = True
        logger.info("MiniMax-M3: installed Ascend adapters for the selected model")
        return True


__all__ = ["apply_minimax_m3_patches"]
