# Copyright (c) 2026 BAAI. All rights reserved.
"""Install MiniMax-local Ascend adapters through the vendor patch entry.

Gemma norms reuse vLLM's existing CustomOp layer. No FlashInfer package is
fabricated and no global activation/rotary OOT registration is changed.
"""

from __future__ import annotations

import importlib
import logging

import torch

logger = logging.getLogger(__name__)

_installed = False

# Upstream modules we patch attributes on.
_UPSTREAM_MODEL_MODULES = (
    "vllm.models.minimax_m3.nvidia.model",
    "vllm.models.minimax_m3.amd.model",
)


# ---------------------------------------------------------------------------
# Injection 1: fused QK-norm + RoPE + KV-insert
# ---------------------------------------------------------------------------
def _install_qknorm_rope_injection() -> bool:
    from vllm_fl.dispatch.backends.vendor.ascend.impl.linearnorm.minimax_m3_qknorm_rope import (
        fused_minimax_m3_qknorm_rope_kv_insert,
    )

    replacement = fused_minimax_m3_qknorm_rope_kv_insert

    try:
        custom_ops = importlib.import_module("vllm._custom_ops")
    except ImportError as exc:  # pragma: no cover - non-upstream wheels
        logger.warning("MiniMax-M3: cannot import vllm._custom_ops (%s)", exc)
        return False

    original = getattr(custom_ops, "fused_minimax_m3_qknorm_rope_kv_insert", None)
    if original is None:
        # Upstream M3 is absent or was refactored: nothing to adapt.
        logger.warning(
            "MiniMax-M3: vllm._custom_ops.fused_minimax_m3_qknorm_rope_kv_insert "
            "not found; skipping injection 1"
        )
        return False
    if original is replacement:
        return True

    custom_ops.fused_minimax_m3_qknorm_rope_kv_insert = replacement
    logger.info("MiniMax-M3: injected Ascend QK-norm/RoPE/KV-insert operator")
    return True


def _install_apply_rotary_emb_injection() -> bool:
    """Partial-rotary correctness for the vision tower (see the module docstring
    of ``apply_rotary_emb.py`` for the measured failure it fixes)."""
    from .minimax_m3_rotary import install_apply_rotary_emb_override

    return install_apply_rotary_emb_override()


def _install_activation_injection() -> bool:
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.models.minimax_m3.nvidia import model

    class AscendSiluAndMulWithClamp(SiluAndMulWithClamp):
        """MiniMax-local SwiGLU adapter; other models retain their layer."""

        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            from vllm_fl.dispatch.backends.vendor.ascend.attention_selector import (
                get_minimax_sparse_kernels,
            )

            return get_minimax_sparse_kernels().activation(
                x,
                alpha=self.alpha,
                limit=self.swiglu_limit,
                bias=self.beta,
            )

    model.SiluAndMulWithClamp = AscendSiluAndMulWithClamp
    return True


# ---------------------------------------------------------------------------
# Injection 4: fused_allreduce_gemma_rms_norm
# ---------------------------------------------------------------------------
def _fused_allreduce_gemma_rms_norm_ascend(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """all-reduce + Gemma RMSNorm without the CUDA-only fused kernel.

    Mirrors the upstream (CUDA) fusion: the preceding dense MLP runs with
    ``reduce_results=False`` and this function completes its all-reduce while
    normalising. On Ascend we reuse the platform's padded all-reduce so that
    the sequence-parallel padding contract is preserved.
    """
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() > 1:
        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states)
    hidden_states, residual = norm(hidden_states, residual)
    return hidden_states, residual


def _install_allreduce_norm_injection() -> bool:
    # Only rebind MiniMax consumers, never the shared layer implementation.
    model_module = importlib.import_module(_UPSTREAM_MODEL_MODULES[0])
    if not callable(getattr(model_module, "fused_allreduce_gemma_rms_norm", None)):
        raise RuntimeError("MiniMax-M3 requires fused_allreduce_gemma_rms_norm")
    model_module.fused_allreduce_gemma_rms_norm = _fused_allreduce_gemma_rms_norm_ascend
    logger.info("MiniMax-M3: injected Ascend fused_allreduce_gemma_rms_norm")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def install_minimax_m3_injections() -> bool:
    """Install all MiniMax-M3 Ascend injections. Idempotent.

    Called only by the atomic vendor entry after dependency preparation.
    Every required adapter must succeed; partial installation raises.
    """
    global _installed
    if _installed:
        return True

    results = {
        "qknorm_rope": _install_qknorm_rope_injection(),
        "activation": _install_activation_injection(),
        "allreduce_norm": _install_allreduce_norm_injection(),
        "apply_rotary_emb": _install_apply_rotary_emb_injection(),
    }
    logger.info(
        "MiniMax-M3 Ascend injections: %s",
        ", ".join(f"{k}={'ok' if v else 'skip'}" for k, v in results.items()),
    )
    if not all(results.values()):
        # Not an upstream-M3 runtime: leave the flag clear so a later call in a
        # different process (spawn) can retry.
        raise RuntimeError("MiniMax-M3 required Ascend injections are incomplete")
    _installed = True
    return True


__all__ = ["install_minimax_m3_injections"]
