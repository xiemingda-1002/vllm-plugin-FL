# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in OCR compute contract. Framework Python APIs remain available."""

import functools
import inspect
import json
import logging
import os
import sys
import threading
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger(__name__)

_OCR_FUSED_ROPE_AUDITED = False
_OCR_FUSED_SAM_AUDITED = False
_SLOT_AUDITED = False


def _ocr_sam_attention(q, k, v, attn_bias=None):
    """SAM's floating additive bias, using the public torch-npu API."""
    import torch

    supported = (
        os.environ.get("VLLM_FL_OCR_FUSED_SAM", "1") == "1"
        and q.device.type == "npu"
        and q.dtype == torch.bfloat16
        and q.ndim == 4
        and q.shape[1] == 12
        and q.shape[-1] == 64
        and k.shape == q.shape
        and v.shape == q.shape
        and k.device == q.device == v.device
        and k.dtype == q.dtype == v.dtype
        and (
            attn_bias is None
            or (
                attn_bias.device == q.device
                and attn_bias.dtype == q.dtype
                and attn_bias.shape == (*q.shape[:3], k.shape[2])
            )
        )
    )
    if not supported:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias
        )
    import torch_npu

    result = torch_npu.npu_fused_infer_attention_score(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        pse_shift=None if attn_bias is None else attn_bias.contiguous(),
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        input_layout="BNSD",
        scale=q.shape[-1] ** -0.5,
    )[0]
    global _OCR_FUSED_SAM_AUDITED
    if not _OCR_FUSED_SAM_AUDITED and _ocr_can_audit():
        _write_audit(
            "sam-fused",
            {
                "api": "torch_npu.npu_fused_infer_attention_score",
                "source": getattr(
                    torch_npu.npu_fused_infer_attention_score, "__module__", "unknown"
                ),
                "event": "api_returned",
                "input_shape": list(q.shape),
                "dtype": str(q.dtype),
                "has_bias": attn_bias is not None,
                "input_layout": "BNSD",
                "scale": q.shape[-1] ** -0.5,
            },
        )
        _OCR_FUSED_SAM_AUDITED = True
    return result


def _ocr_sam_forward(self, x):
    """FL-owned copy of core SAM RelPosAttention, preserving its geometry."""
    from vllm.model_executor.models.deepencoder import add_decomposed_rel_pos

    B, H, W, _ = x.shape
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
    rel_h, rel_w = None, None
    if self.use_rel_pos:
        rel_h, rel_w = add_decomposed_rel_pos(
            q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
        )
    q = q.view(B, self.num_heads, H * W, -1)
    k = k.view(B, self.num_heads, H * W, -1)
    v = v.view(B, self.num_heads, H * W, -1)
    attn_bias = None
    if self.use_rel_pos:
        rel_h = rel_h.view(
            B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3)
        )
        rel_w = rel_w.view(
            B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3)
        )
        attn_bias = (rel_h + rel_w).view(
            B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4)
        )
    x = _ocr_sam_attention(q, k, v, attn_bias)
    x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    return self.proj(x)


def _write_audit(stage, evidence):
    directory = os.environ.get("VLLM_FL_OCR_AUDIT_DIR")
    if directory:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{stage}-{os.getpid()}.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )


def ocr_runtime_enabled():
    return os.environ.get("VLLM_FL_OCR_TORCH_NPU", "0") == "1"


def _ocr_rotary_init(
    self, head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
):
    # Bypass RotaryEmbeddingFL's ATB registration; preserve the core cache builder.
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    RotaryEmbedding.__init__(
        self, head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
    )


def _ocr_rotary_forward(self, positions, query, key=None):
    """FL tensor RoPE: no ATB, vLLM kernel, or half-style assumptions."""
    import torch

    positions = positions.flatten().long()
    cache = self.cos_sin_cache.to(device=query.device, dtype=query.dtype)
    cos, sin = cache.index_select(0, positions).chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    fused = (
        os.environ.get("VLLM_FL_OCR_FUSED_ROPE", "1") == "1"
        and query.device.type == "npu"
    )
    if fused:
        # npu_rotary_mul consumes full-width coefficients in [S, B, N, D].
        if self.is_neox_style:
            fused_cos = torch.cat((cos, cos), dim=-1)
            fused_sin = torch.cat((sin, sin), dim=-1)
        else:
            fused_cos = cos.repeat_interleave(2, dim=-1)
            fused_sin = sin.repeat_interleave(2, dim=-1)
        fused_cos, fused_sin = fused_cos.unsqueeze(1), fused_sin.unsqueeze(1)

    def rotate(tensor):
        shape = tensor.shape
        x = tensor.reshape(positions.numel(), -1, self.head_size)
        rotary, tail = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        if fused:
            transformed = _ocr_fused_rotary_mul(
                rotary, fused_cos, fused_sin, self.is_neox_style
            )
        elif self.is_neox_style:
            first, second = rotary.chunk(2, dim=-1)
            transformed = torch.cat(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            )
        else:
            first, second = rotary[..., 0::2], rotary[..., 1::2]
            transformed = torch.stack(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            ).flatten(-2)
        if self.rotary_dim == self.head_size:
            return transformed.reshape(shape)
        return torch.cat((transformed, tail), dim=-1).reshape(shape)

    return rotate(query), None if key is None else rotate(key)


def _ocr_can_audit():
    """Keep filesystem I/O and audit state mutations outside compile/capture."""
    import torch

    if torch.compiler.is_compiling():
        return False
    npu = getattr(torch, "npu", None)
    return npu is None or not npu.is_current_stream_capturing()


def _ocr_fused_rotary_mul(rotary, cos, sin, is_neox_style):
    """Public torch-npu API; unsupported contracts deliberately raise."""
    import torch_npu

    result = torch_npu.npu_rotary_mul(
        rotary.unsqueeze(1).contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        rotary_mode="half" if is_neox_style else "interleave",
    ).squeeze(1)
    # Only eager execution outside capture may record an actual API return.
    # Compiled/captured execution must not write files or mutate audit state.
    global _OCR_FUSED_ROPE_AUDITED
    if not _OCR_FUSED_ROPE_AUDITED and _ocr_can_audit():
        _write_audit(
            "rope-fused",
            {
                "api": "torch_npu.npu_rotary_mul",
                "source": getattr(torch_npu.npu_rotary_mul, "__module__", "unknown"),
                "event": "api_returned",
                "rotary_mode": "half" if is_neox_style else "interleave",
                "input_shape": list(rotary.shape),
                "dtype": str(rotary.dtype),
            },
        )
        _OCR_FUSED_ROPE_AUDITED = True
    return result


def configure_ocr_runtime(config):
    if not ocr_runtime_enabled():
        return
    if any(x == "vllm_ascend" or x.startswith("vllm_ascend.") for x in sys.modules):
        raise RuntimeError("OCR torch-npu mode cannot load vllm-ascend")
    model = config.model_config
    if getattr(model.hf_config, "model_type", None) not in (
        "unlimited-ocr",
        "unlimited_ocr",
    ):
        raise ValueError("OCR torch-npu mode requires Unlimited-OCR")
    if config.speculative_config is not None or config.quant_config is not None:
        raise ValueError("OCR torch-npu mode requires non-speculative unquantized BF16")
    if str(model.dtype) != "torch.bfloat16":
        raise ValueError("OCR torch-npu mode requires BF16")
    from vllm_fl.utils import get_oot_blacklist, get_oot_whitelist, is_oot_enabled

    required = {"rms_norm", "silu_and_mul", "rotary_embedding", "fused_moe"}
    white, black = get_oot_whitelist(), set(get_oot_blacklist() or [])
    if (
        not is_oot_enabled()
        or required & black
        or (white is not None and not required <= set(white))
    ):
        raise ValueError("OCR requires FL RMSNorm/Silu/RoPE/FusedMoE registration")
    # CustomOp.enabled reads the concrete class registration name, which can
    # differ from its public snake_case control in an OOT replacement.
    aliases = (
        "rms_norm",
        "RMSNorm",
        "silu_and_mul",
        "SiluAndMul",
        "rotary_embedding",
        "RotaryEmbedding",
    )
    custom_ops = config.compilation_config.custom_ops
    if any("-" + name in custom_ops for name in aliases):
        raise ValueError("OCR required custom ops were explicitly disabled")
    for name in aliases:
        if "+" + name not in custom_ops:
            custom_ops.append("+" + name)
    from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

    RotaryEmbeddingFL.__init__ = _ocr_rotary_init
    RotaryEmbeddingFL.forward_oot = _ocr_rotary_forward

    if os.environ.get("VLLM_FL_OCR_FUSED_SAM", "1") == "1":
        from vllm.model_executor.models.deepencoder import RelPosAttention

        RelPosAttention.forward = _ocr_sam_forward

    # Own BF16 GEMM numerics instead of vLLM dispatch_unquantized_gemm.
    import torch.nn.functional as functional

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    def linear_apply(self, layer, x, bias=None):
        return functional.linear(x, layer.weight, bias)

    UnquantizedLinearMethod.apply = linear_apply
    UnquantizedEmbeddingMethod.apply = linear_apply
    text = model.hf_config.text_config
    if getattr(text, "qk_nope_head_dim", 0) or getattr(text, "qk_rope_head_dim", 0):
        raise ValueError("OCR torch-npu mode supports plain MHA only")
    # Fail rather than dispatching bias routing to the upstream custom kernel.
    if getattr(text, "topk_method", None) == "noaux_tc":
        raise ValueError("OCR torch-npu mode does not support correction-bias routing")
    if os.environ.get("USE_FLAGGEMS", "0") != "0":
        raise ValueError("OCR torch-npu mode requires USE_FLAGGEMS=0")


def audit_ocr_model(model):
    """Check instantiated layers, not just registry entries. Return a manifest."""
    if not ocr_runtime_enabled():
        return {}
    manifest = {}
    for name, layer in model.named_modules():
        kind = type(layer).__name__
        module = type(layer).__module__
        if "vllm_ascend" in module:
            raise RuntimeError(f"Forbidden OCR layer {name}: {module}.{kind}")
        expected = {
            "RMSNorm": "RMSNormFL",
            "SiluAndMul": "SiluAndMulFL",
            "RotaryEmbedding": "RotaryEmbeddingFL",
        }
        if kind in expected:
            raise RuntimeError(f"OCR layer {name} escaped FL registration: {kind}")
        if kind in expected.values():
            bound = getattr(layer, "_forward_method", None)
            if bound is None or not getattr(bound, "__module__", "").startswith(
                "vllm_fl."
            ):
                raise RuntimeError(f"OCR layer {name} is not executing its FL forward")
        if kind in expected.values() or kind in (
            "AscendMMEncoderAttention",
            "RelPosAttention",
            "MoERunner",
        ):
            manifest[name] = f"{module}.{kind}"
        if kind == "MMEncoderAttention":
            raise RuntimeError(f"OCR visual attention {name} escaped FL registration")
        if kind == "RelPosAttention":
            bound = layer.forward
            actual = getattr(bound, "__func__", bound)
            enabled = os.environ.get("VLLM_FL_OCR_FUSED_SAM", "1") == "1"
            if enabled and actual is not _ocr_sam_forward:
                raise RuntimeError(f"OCR SAM {name} is not executing its FL forward")
            if not enabled and actual is _ocr_sam_forward:
                raise RuntimeError(
                    "OCR SAM forward patched while fused SAM is disabled"
                )
            source = inspect.getsource(actual)
            if not enabled and "scaled_dot_product_attention" not in source:
                raise RuntimeError("Unexpected SAM attention implementation")
            manifest[name] = {
                "layer": f"{module}.{kind}",
                "fused_sam_enabled": enabled,
                "forward": f"{actual.__module__}.{actual.__qualname__}",
            }
        if kind == "MoERunner":
            method = getattr(layer, "_quant_method", None)
            if not type(method).__module__.startswith("vllm_fl."):
                raise RuntimeError(f"OCR MoE {name} has non-FL method {type(method)}")
    if not manifest:
        raise RuntimeError("No OCR compute layers were audited")
    if any(x == "vllm_ascend" or x.startswith("vllm_ascend.") for x in sys.modules):
        raise RuntimeError("vllm-ascend imported during OCR model construction")
    _write_audit("model", manifest)
    return manifest


def audit_ocr_dispatch():
    """Resolve and check the required numerical FL implementations."""
    if not ocr_runtime_enabled():
        return {}
    from vllm_fl.dispatch import get_default_manager, get_policy

    if not get_policy().strict:
        raise RuntimeError("OCR requires strict dispatch without fallback")
    manifest = {}
    manager = get_default_manager()
    for name in ("rms_norm", "silu_and_mul", "fused_experts"):
        impl = manager._resolve_impl(name)
        source = getattr(impl.fn, "__module__", "")
        if impl.vendor != "ascend" or not source.startswith(
            "vllm_fl.dispatch.backends.vendor.ascend."
        ):
            raise RuntimeError(
                f"OCR {name} selected an unsupported implementation: {source}"
            )
        manifest[name] = {"implementation": impl.impl_id, "source": source}
    _write_audit("dispatch", manifest)
    return manifest


def _supported_device(device):
    return device.type in ("npu", "privateuseone")


def compute_ocr_slot_mapping(block_table_manager, num_reqs, req_indices, positions):
    """Return False without writes when any cache group is unsupported.

    Request indices and positions must already satisfy the runner's indexing
    contract. No device values are read back or validated on the host.
    """
    if (
        os.environ.get("VLLM_FL_OCR_TORCH_SLOT_MAPPING", "1") != "1"
        or not ocr_runtime_enabled()
    ):
        return False
    if not isinstance(req_indices, torch.Tensor) or not isinstance(
        positions, torch.Tensor
    ):
        return False
    if (
        req_indices.ndim != 1
        or positions.ndim != 1
        or req_indices.shape != positions.shape
        or req_indices.dtype not in (torch.int32, torch.int64)
        or positions.dtype != torch.int64
        or req_indices.device != positions.device
        or not _supported_device(positions.device)
        or not isinstance(num_reqs, int)
        or num_reqs < 0
    ):
        return False
    tables = getattr(block_table_manager, "block_tables", None)
    if not isinstance(tables, (list, tuple)) or not tables:
        return False
    num_tokens = positions.numel()
    contracts = []
    for group in tables:
        mode = getattr(group, "slot_mapping_mode", None)
        # Older core BlockTable versions have only ordinary token slot mapping.
        if mode is not None and getattr(mode, "name", None) != "TOKEN_TO_KV_SLOT":
            return False
        block_size = getattr(group, "block_size", None)
        capacity = getattr(group, "max_num_batched_tokens", None)
        table = getattr(getattr(group, "block_table", None), "gpu", None)
        slots = getattr(getattr(group, "slot_mapping", None), "gpu", None)
        if (
            getattr(group, "pcp_world_size", None) != 1
            or getattr(group, "dcp_world_size", None) != 1
            or getattr(group, "blocks_per_kv_block", None) != 1
            or getattr(group, "use_hybrid_blocks", None) is not False
            or not isinstance(block_size, int)
            or block_size <= 0
            or getattr(group, "kv_cache_block_size", block_size) != block_size
            or not isinstance(capacity, int)
            or capacity < num_tokens
            or not isinstance(table, torch.Tensor)
            or not isinstance(slots, torch.Tensor)
        ):
            return False
        if (
            table.ndim != 2
            or table.shape[0] < num_reqs
            or table.stride(1) != 1
            or table.dtype not in (torch.int32, torch.int64)
            or table.device != positions.device
            or slots.ndim != 1
            or slots.numel() < capacity
            or slots.dtype != torch.int64
            or slots.device != positions.device
            or not slots.is_contiguous()
        ):
            return False
        contracts.append((table, slots, block_size, capacity))
    # Preflight all groups before touching any graph-visible output storage.
    for table, slots, block_size, capacity in contracts:
        if num_tokens:
            block_indices = positions // block_size
            physical_blocks = table[req_indices, block_indices].to(torch.int64)
            slots[:num_tokens].copy_(
                physical_blocks * block_size + positions % block_size
            )
        slots[num_tokens:capacity].fill_(-1)
    global _SLOT_AUDITED
    if not _SLOT_AUDITED and _ocr_can_audit():
        _write_audit(
            "slot-mapping",
            {
                "implementation": "vllm_fl.models.ocr_runtime.compute_ocr_slot_mapping",
                "event": "torch_ops_enqueued",
                "num_tokens": num_tokens,
                "num_groups": len(contracts),
                "block_sizes": [item[2] for item in contracts],
                "device": str(positions.device),
                "core_triton_slot_kernel_called": False,
            },
        )
        _SLOT_AUDITED = True
    return True


FORBIDDEN = frozenset({"_C", "_moe_C", "_C_cache_ops", "_C_ascend", "vllm_ascend"})


class OCRExecutionAudit(TorchDispatchMode):
    """Count actually intercepted schemas, rejecting forbidden namespaces."""

    def __init__(self):
        super().__init__()
        self.schemas = Counter()
        self.method_calls = Counter()
        self.method_failures = Counter()
        self.depth = 0
        self.lock = threading.RLock()

    def persist(self):
        directory = os.environ.get("VLLM_FL_OCR_AUDIT_DIR")
        if not directory:
            return
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"execution-{os.getpid()}.json"
        temporary = target.with_suffix(".json.tmp")
        payload = {
            "pid": os.getpid(),
            "method_calls": dict(self.method_calls),
            "method_failures": dict(self.method_failures),
            "schemas": dict(self.schemas),
            "scope": (
                "Observed eager dispatch only; opaque operators and graph replay "
                "internals may be hidden. This is not full compute closure proof."
            ),
        }
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        schema = str(func._schema)
        namespace = func._schema.name.split("::", 1)[0]
        self.schemas[schema] += 1
        if self.schemas[schema] == 1:
            logger.info("FL OCR observed dispatch schema: %s", schema)
        if namespace in FORBIDDEN:
            raise RuntimeError(f"Forbidden OCR device operator namespace: {schema}")
        # vllm namespace wrappers remain visible in the evidence. Their nested
        # execution is not necessarily exposed to TorchDispatchMode.
        return func(*args, **(kwargs or {}))


def install_execution_audit(model):
    """Wrap compute entry points with exception-safe mode scope; return recorder.

    Disabled by default. No hooks or global dispatch modes remain after a
    forward returns or raises. The wrapper is for diagnostic eager runs only.
    """
    if os.environ.get("VLLM_FL_OCR_AUDIT_OPS", "0") != "1":
        return None
    existing = getattr(model, "_ocr_execution_audit", None)
    if existing is not None:
        return existing
    recorder = OCRExecutionAudit()

    def wrap(original, method_name):
        @functools.wraps(original)
        def audited(*args, **kwargs):
            # Serialize diagnostic calls, and enter the mode once when entry
            # points call each other. Re-entering the same mode duplicates ops.
            with recorder.lock:
                outer = recorder.depth == 0
                recorder.depth += 1
                recorder.method_calls[method_name] += 1
                try:
                    if outer:
                        with recorder:
                            return original(*args, **kwargs)
                    return original(*args, **kwargs)
                except BaseException:
                    recorder.method_failures[method_name] += 1
                    raise
                finally:
                    recorder.depth -= 1
                    if outer:
                        # Persistence failures must not replace a model error.
                        try:
                            recorder.persist()
                        except Exception:
                            logger.exception("Could not persist OCR dispatch evidence")

        return audited

    for method_name in (
        "forward",
        "embed_multimodal",
        "get_multimodal_embeddings",
        "encoder_eager_forward",
        "get_input_embeddings",
        "embed_input_ids",
    ):
        original = getattr(model, method_name, None)
        if callable(original):
            setattr(model, method_name, wrap(original, method_name))
    model._ocr_execution_audit = recorder
    logger.warning(
        "FL OCR dispatch audit enabled for eager diagnostics; observed schemas "
        "alone do not prove full compute closure or graph replay internals"
    )
    return recorder


class OCRTopKTopPSampler(nn.Module):
    def forward(self, logits, generators, k, p):
        values, ids = logits.sort(dim=-1, descending=True)
        if k is not None:
            positions = torch.arange(values.shape[-1], device=values.device)
            values.masked_fill_(positions[None] >= k[:, None], -float("inf"))
        if p is not None:
            cumulative = values.softmax(-1).cumsum(-1)
            mask = cumulative - values.softmax(-1) >= p[:, None]
            mask[:, 0] = False
            values.masked_fill_(mask, -float("inf"))
        filtered = torch.full_like(logits, -float("inf")).scatter_(1, ids, values)
        probs = filtered.softmax(-1)
        samples = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
        for row in range(logits.shape[0]):
            samples[row] = torch.multinomial(
                probs[row], 1, generator=generators.get(row)
            )[0]
        return samples, None


def get_ocr_sampler_cls():
    """Load the vLLM sampler base only when the OCR runtime is enabled."""
    from vllm.v1.sample.sampler import Sampler

    class OCRSampler(Sampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.topk_topp_sampler = OCRTopKTopPSampler()

        def forward(
            self,
            logits,
            sampling_metadata,
            predict_bonus_token=False,
            logprobs_mode_override=None,
        ):
            if (
                sampling_metadata.max_num_logprobs is not None
                or sampling_metadata.logprob_token_ids
            ):
                raise ValueError(
                    "OCR torch-only sampler currently requires logprobs disabled"
                )
            return super().forward(
                logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
            )

        @staticmethod
        def apply_penalties(logits, sampling_metadata, output_token_ids):
            if sampling_metadata.no_penalties:
                return logits
            vocab = logits.shape[-1]
            prompt = sampling_metadata.prompt_token_ids
            for row, output in enumerate(output_token_ids):
                counts = torch.zeros(
                    vocab + 1, dtype=logits.dtype, device=logits.device
                )
                ids = torch.tensor(
                    [x if 0 <= x < vocab else vocab for x in output],
                    device=logits.device,
                    dtype=torch.int64,
                )
                counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
                counts = counts[:vocab]
                seen = counts > 0
                if prompt is not None:
                    pids = prompt[row].long()
                    pmask = torch.zeros(
                        vocab + 1, dtype=torch.bool, device=logits.device
                    )
                    pmask.scatter_(
                        0, torch.where((pids >= 0) & (pids < vocab), pids, vocab), True
                    )
                    seen = seen | pmask[:vocab]
                penalty = sampling_metadata.repetition_penalties[row]
                penalized = torch.where(
                    logits[row] > 0, logits[row] / penalty, logits[row] * penalty
                )
                logits[row] = torch.where(seen, penalized, logits[row])
                logits[row] -= counts * sampling_metadata.frequency_penalties[row]
                logits[row] -= (counts > 0).to(
                    logits.dtype
                ) * sampling_metadata.presence_penalties[row]
            return logits

    return OCRSampler
