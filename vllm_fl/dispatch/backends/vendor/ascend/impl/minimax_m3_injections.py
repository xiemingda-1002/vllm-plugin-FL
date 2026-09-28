# Copyright (c) 2026 BAAI. All rights reserved.
"""Runtime injection points that adapt upstream MiniMax-M3 to Ascend.

Why injection instead of a fork
-------------------------------
``vllm-ascend`` carries a 1194-line copy of the upstream 1176-line model file
and rewrites 15 of its 30 symbols (measured by AST diff), which makes every
future upstream change a merge conflict and silently drops upstream bug fixes.
We instead leave the upstream files untouched and replace the four places where
they reach for NVIDIA-only functionality.

The four injection points (verified by reading the upstream source)
------------------------------------------------------------------

1. ``model.py:364, 591`` calls
   ``ops.fused_minimax_m3_qknorm_rope_kv_insert(...)`` where ``ops`` is the
   ``vllm._custom_ops`` module object. Because it is resolved as a module
   attribute on every call, replacing the attribute is enough.

2. ``model.py:130`` imports ``gemma_rmsnorm`` / ``gemma_fused_add_rmsnorm``
   from ``flashinfer.norm`` **inside the function body**. We register a
   ``flashinfer.norm`` stub in ``sys.modules`` that routes to torch_npu.

3. ``model.py:173`` builds ``SiluAndMulWithClamp``, a ``CustomOp`` subclass
   with **no NPU branch** (measured). It therefore fell through to eager
   PyTorch. We register an Ascend ``forward_oot`` via ``CustomOp.register_oot``.

4. ``model.py:723, 737`` call ``fused_allreduce_gemma_rms_norm(...)`` imported
   at module scope, so the *model module's* attribute must be patched (patching
   the defining module alone would have no effect).

Every injection is idempotent and guarded so that a missing upstream symbol
degrades to a warning rather than breaking non-M3 models.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import logging
import os
import sys
import types

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
    if getattr(original, "_fl_ascend_replacement", False):
        return True

    try:
        replacement._fl_ascend_original = original
        replacement._fl_ascend_replacement = True
    except AttributeError:
        pass  # torch custom ops reject arbitrary attribute assignment
    custom_ops.fused_minimax_m3_qknorm_rope_kv_insert = replacement
    logger.info("MiniMax-M3: injected Ascend QK-norm/RoPE/KV-insert operator")
    return True


# ---------------------------------------------------------------------------
# Injection 2: FlashInfer Gemma RMSNorm -> torch_npu
# ---------------------------------------------------------------------------
def _gemma_rmsnorm_ascend(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``x * (1 + w) * rsqrt(mean(x^2) + eps)`` over the last dim."""
    gamma = (1.0 + weight.to(torch.float32)).to(x.dtype)
    out, _ = torch.ops.npu.npu_rms_norm(x, gamma, eps)
    return out


def _gemma_fused_add_rmsnorm_ascend(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """In-place fused add + Gemma RMSNorm, mirroring the FlashInfer contract.

    Upstream discards the return value and relies on ``x`` and ``residual``
    being updated in place, i.e. ``residual += x`` then ``x = norm(residual)``.

    Measured semantics of ``torch.ops.npu.npu_add_rms_norm(x1, x2, gamma, eps)``:

        returns (normed, rstd, x1 + x2)

    and it mutates **neither** input (verified on hardware: with x1=1, x2=2 the
    third output is 3 while both inputs stay unchanged). So both results must be
    copied back — an earlier version assigned the tuple element to the local
    name ``residual``, which updated nothing and corrupted every residual stream
    (measured: residual rel-error 0.59 even in fp32, vs 0.0 for the x path).
    """
    gamma = (1.0 + weight.to(torch.float32)).to(residual.dtype)
    normed, _rstd, added = torch.ops.npu.npu_add_rms_norm(x, residual, gamma, eps)
    x.copy_(normed)
    residual.copy_(added)


_FL_ADD_RMSNORM_OP = "minimax_m3_gemma_fused_add_rmsnorm"
_add_rmsnorm_op = None


def _gemma_fused_add_rmsnorm_op(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """Opaque body: the plain shim, with both in-place targets declared."""
    _gemma_fused_add_rmsnorm_ascend(x, residual, weight, eps)


def _gemma_fused_add_rmsnorm_op_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    return None


def _ensure_add_rmsnorm_op():
    """Register the opaque fused add+RMSNorm op once."""
    global _add_rmsnorm_op
    if _add_rmsnorm_op is not None:
        return _add_rmsnorm_op
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_FL_ADD_RMSNORM_OP,
        op_func=_gemma_fused_add_rmsnorm_op,
        mutates_args=["x", "residual"],
        fake_impl=_gemma_fused_add_rmsnorm_op_fake,
        dispatch_key="PrivateUse1",
    )
    _add_rmsnorm_op = getattr(torch.ops.vllm, _FL_ADD_RMSNORM_OP)
    return _add_rmsnorm_op


def _gemma_fused_add_rmsnorm_ascend_opaque(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """Installed into the flashinfer shim.

    Goes through the registered op for NPU tensors so compilation keeps the
    in-place update; upstream discards the return value and relies on ``x`` and
    ``residual`` being rewritten. As a plain function those two ``copy_`` calls
    are invisible to torch.compile, which corrupts the residual stream in every
    layer -- measured with compilation on, the model emits well-formed but
    meaningless text with finite logprobs.
    """
    if x.device.type != "npu" or _add_rmsnorm_op is None:
        _gemma_fused_add_rmsnorm_ascend(x, residual, weight, eps)
        return
    _add_rmsnorm_op(x, residual, weight, eps)


def _install_flashinfer_injection() -> bool:
    try:
        existing = importlib.import_module("flashinfer.norm")
    except ImportError:
        existing = None

    if existing is not None and not getattr(existing, "_fl_ascend_stub", False):
        # A real FlashInfer install on Ascend would be unexpected; do not
        # silently clobber a genuine dependency.
        logger.warning(
            "MiniMax-M3: flashinfer.norm is importable and not our stub; "
            "leaving it alone"
        )
        return False
    if existing is not None:
        return True

    def _make_package_stub(name: str) -> types.ModuleType:
        """Create an importable package stub.

        ``ModuleType`` alone is not enough: vLLM's allreduce-RMS fusion pass
        calls ``importlib.util.find_spec("flashinfer")``, which raises
        ``ValueError`` when ``__spec__`` is missing. A real ``ModuleSpec`` with
        a loader keeps such probes well-defined.
        """
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        return module

    package = sys.modules.get("flashinfer")
    if package is None:
        package = _make_package_stub("flashinfer")
        sys.modules["flashinfer"] = package

    norm = types.ModuleType("flashinfer.norm")
    norm.__spec__ = importlib.machinery.ModuleSpec("flashinfer.norm", loader=None)
    norm.gemma_rmsnorm = _gemma_rmsnorm_ascend
    # Register before any tracing can begin: registering lazily inside the
    # forward makes Dynamo see a "function marked as skipped" and abort.
    _ensure_add_rmsnorm_op()
    norm.gemma_fused_add_rmsnorm = _gemma_fused_add_rmsnorm_ascend_opaque
    norm._fl_ascend_stub = True  # type: ignore[attr-defined]
    sys.modules["flashinfer.norm"] = norm
    package.norm = norm  # type: ignore[attr-defined]
    logger.info("MiniMax-M3: installed flashinfer.norm -> torch_npu shim")
    return True


# ---------------------------------------------------------------------------
# Injection 3: SiluAndMulWithClamp (no NPU branch upstream)
# ---------------------------------------------------------------------------
def _install_bind_kv_cache_injection() -> bool:
    """Allow a decoder layer to own several attention caches on OOT platforms.

    MiniMax-M3 registers both a main K/V cache and an indexer cache per sparse
    layer; upstream ``bind_kv_cache`` raises NotImplementedError for that case
    on any non-CUDA/XPU/CPU platform.
    """
    from .minimax_m3_kv_bind import install_bind_kv_cache_patch

    return install_bind_kv_cache_patch()


def _install_apply_rotary_emb_injection() -> bool:
    """Partial-rotary correctness for the vision tower (see the module docstring
    of ``apply_rotary_emb.py`` for the measured failure it fixes)."""
    from .minimax_m3_rotary import install_apply_rotary_emb_override

    return install_apply_rotary_emb_override()


def _install_activation_injection() -> bool:
    try:
        from vllm.model_executor.layers.activation import SiluAndMulWithClamp
        from vllm.model_executor.custom_op import CustomOp
    except ImportError as exc:  # pragma: no cover
        logger.warning("MiniMax-M3: cannot import SiluAndMulWithClamp (%s)", exc)
        return False

    if getattr(SiluAndMulWithClamp, "_fl_ascend_oot", False):
        return True

    class AscendSiluAndMulWithClamp(SiluAndMulWithClamp):
        """SwiGLU-OAI via the torch_npu fused operator."""

        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ops.npu.npu_clipped_swiglu(
                x,
                alpha=self.alpha,
                limit=self.swiglu_limit,
                bias=self.beta,
                interleaved=False,
            )

    AscendSiluAndMulWithClamp.__name__ = "AscendSiluAndMulWithClamp"
    AscendSiluAndMulWithClamp._fl_ascend_oot = True  # type: ignore[attr-defined]
    try:
        CustomOp.register_oot(
            _decorated_op_cls=AscendSiluAndMulWithClamp,
            name="SiluAndMulWithClamp",
        )
    except Exception as exc:  # already registered or upstream refactor
        logger.warning("MiniMax-M3: SiluAndMulWithClamp OOT registration failed: %s", exc)
        return False
    logger.info("MiniMax-M3: registered Ascend SiluAndMulWithClamp OOT")
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
    module_name = "vllm.model_executor.layers.fused_allreduce_gemma_rms_norm"
    try:
        owner = importlib.import_module(module_name)
    except ImportError:
        logger.warning("MiniMax-M3: %s not importable; skipping injection 4", module_name)
        return False

    if getattr(owner.fused_allreduce_gemma_rms_norm, "_fl_ascend_replacement", False):
        return True

    _fused_allreduce_gemma_rms_norm_ascend._fl_ascend_replacement = True  # type: ignore[attr-defined]
    owner.fused_allreduce_gemma_rms_norm = _fused_allreduce_gemma_rms_norm_ascend

    # The model modules import the function at module scope, so their own
    # attributes must be rebound as well.
    for name in _UPSTREAM_MODEL_MODULES:
        try:
            model_module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(model_module, "fused_allreduce_gemma_rms_norm"):
            model_module.fused_allreduce_gemma_rms_norm = (
                _fused_allreduce_gemma_rms_norm_ascend
            )
    logger.info("MiniMax-M3: injected Ascend fused_allreduce_gemma_rms_norm")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def install_minimax_m3_injections() -> bool:
    """Install all MiniMax-M3 Ascend injections. Idempotent.

    Returns ``True`` when at least the primary (operator) injection is in
    place, i.e. when this vLLM release carries the upstream M3 runtime.
    """
    global _installed
    if _installed:
        return True

    results = {
        "qknorm_rope": _install_qknorm_rope_injection(),
        "flashinfer_norm": _install_flashinfer_injection(),
        "activation": _install_activation_injection(),
        "allreduce_norm": _install_allreduce_norm_injection(),
        "apply_rotary_emb": _install_apply_rotary_emb_injection(),
        "bind_kv_cache": _install_bind_kv_cache_injection(),
    }
    logger.info(
        "MiniMax-M3 Ascend injections: %s",
        ", ".join(f"{k}={'ok' if v else 'skip'}" for k, v in results.items()),
    )
    if not results["qknorm_rope"]:
        # Not an upstream-M3 runtime: leave the flag clear so a later call in a
        # different process (spawn) can retry.
        return False
    _installed = True
    return True


__all__ = ["install_minimax_m3_injections"]
