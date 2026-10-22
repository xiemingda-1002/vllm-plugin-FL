# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 model-owned composition with dispatch-selected math providers.

These building blocks retain checkpoint names and explicit norm placement.
They do not register a model or own cache allocation and scheduler metadata.
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_fl.dispatch import CachedOp
from vllm_fl.ops.layernorm import RMSNormFL as RMSNorm

_project = CachedOp("deepseek_v4_project")
_save_state = CachedOp("deepseek_v4_save_state")
_compress = CachedOp("deepseek_v4_compress")
_hc_pre = CachedOp("deepseek_v4_hc_pre")
_hc_post = CachedOp("deepseek_v4_hc_post")


class DeepseekV4Compressor(nn.Module):
    def __init__(
        self,
        hidden_size,
        head_dim,
        compress_ratio,
        *,
        rms_eps=1e-6,
        params_dtype=torch.bfloat16,
        prefix="",
    ):
        super().__init__()
        if compress_ratio not in (4, 128):
            raise ValueError("Compression ratio must be 4 or 128")
        self.compress_ratio = compress_ratio
        self.norm_eps = rms_eps
        coff = 2 if compress_ratio == 4 else 1
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32),
            requires_grad=False,
        )
        for name in ("wkv", "wgate"):
            layer = ReplicatedLinear(
                hidden_size,
                coff * head_dim,
                bias=False,
                params_dtype=params_dtype,
                quant_config=None,
                prefix=f"{prefix}.{name}" if prefix else name,
                return_bias=False,
                disable_tp=True,
            )
            setattr(self, name, layer)
        self.norm = RMSNorm(head_dim, rms_eps, dtype=params_dtype)

    def forward(
        self,
        x,
        positions,
        slot_mapping,
        state_cache,
        state_block_table,
        output_request_ids,
        output_positions,
        cos_sin_cache,
    ):
        # Calling the normal BF16 Linear forward would round the projections
        # before they reach persistent FP32 state. The provider owns that choice.
        kv, gate = _project(x, self.wkv.weight, self.wgate.weight)
        _save_state(
            kv,
            gate,
            self.ape,
            positions,
            slot_mapping,
            state_cache,
            self.compress_ratio,
        )
        return _compress(
            state_cache,
            state_block_table,
            output_request_ids,
            output_positions,
            self.norm.weight,
            cos_sin_cache,
            compress_ratio=self.compress_ratio,
            rms_eps=self.norm_eps,
            output_dtype=x.dtype,
        )

    def forward_from_cache(
        self,
        x,
        state_leaf,
        compact_leaf,
        state_metadata,
        compact_metadata,
        cos_sin_cache,
    ):
        if (
            state_leaf.spec.compress_ratio != 1
            or compact_leaf.spec.compress_ratio != self.compress_ratio
        ):
            raise ValueError("Compressor cache ratios do not match the model")
        output = self(
            x,
            state_metadata.positions,
            state_metadata.slot_mapping,
            state_leaf.kv_cache,
            state_metadata.block_table,
            compact_metadata.output_request_ids,
            compact_metadata.output_positions,
            cos_sin_cache,
        )
        compact_leaf.store(output, compact_metadata)
        return output


class DeepseekV4DecoderLayer(nn.Module):
    """Compose injected attention/MoE modules without binding their math backend."""

    def __init__(
        self,
        hidden_size,
        hc_mult,
        hc_sinkhorn_iters,
        hc_eps,
        rms_eps,
        self_attn,
        mlp,
        *,
        params_dtype=torch.bfloat16,
    ):
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise NotImplementedError("The current HC decoder requires BF16 residuals")
        self.self_attn = self_attn
        self.mlp = mlp
        self.norm_eps = rms_eps
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.input_layernorm = RMSNorm(hidden_size, rms_eps, dtype=params_dtype)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, rms_eps, dtype=params_dtype
        )
        mix = (2 + hc_mult) * hc_mult
        for block in ("attn", "ffn"):
            for name, shape in (
                ("fn", (mix, hc_mult * hidden_size)),
                ("base", (mix,)),
                ("scale", (3,)),
            ):
                self.register_parameter(
                    f"hc_{block}_{name}",
                    nn.Parameter(
                        torch.empty(shape, dtype=torch.float32), requires_grad=False
                    ),
                )

    def hc_pre(self, x, fn, scale, base):
        return _hc_pre(
            x, fn, scale, base, self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters
        )

    def hc_post(self, x, residual, post, comb):
        return _hc_post(x, residual, post, comb)

    def forward(self, positions, hidden_states, input_ids=None, llama_4_scaling=None):
        residual = hidden_states
        x, post, comb = self.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        x = self.self_attn(
            positions=positions, hidden_states=x, llama_4_scaling=llama_4_scaling
        )
        hidden_states = self.hc_post(x, residual, post, comb)
        residual = hidden_states
        x, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, input_ids=input_ids)
        return self.hc_post(x, residual, post, comb)
