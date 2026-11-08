# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in OCR compute contract. Framework Python APIs remain available."""

import inspect
import os
import sys

_OCR_FUSED_ROPE_AUDITED = False
_OCR_FUSED_SAM_AUDITED = False


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
        import json
        from pathlib import Path

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
