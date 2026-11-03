# Copyright (c) 2026 BAAI. All rights reserved.
"""GLM-5.2 model hooks; keep the upstream model and checkpoint loader."""

from functools import cache

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.models import deepseek_v2


def _is_glm52(config):
    return getattr(config, "model_type", None) == "glm_moe_dsa"


class Glm52Indexer(nn.Module):
    """rc1 BF16 indexer projections with the standard checkpoint parameter names."""

    def __init__(
        self,
        vllm_config,
        config,
        hidden_size,
        q_lora_rank,
        quant_config,
        cache_config,
        topk_indices_buffer,
        prefix="",
        is_inplace_rope=False,
    ):
        super().__init__()
        from vllm_fl.attention.sparse_mla import SparseIndexerCache

        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.topk_indices_buffer = topk_indices_buffer
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # wk and weights_proj are FLOAT checkpoint shards; upstream's loader
        # merges both into this unquantized, replicated parameter.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.k_cache = SparseIndexerCache(
            prefix=f"{prefix}.k_cache",
            head_size=self.head_dim,
            cache_config=cache_config,
        )

    def forward(self, hidden_states, qr, positions, rotary_emb):
        q = self.wq_b(qr)[0].view(-1, self.n_head, self.head_dim)
        kw = self.wk_weights_proj(hidden_states)[0]
        k = self.k_norm(kw[:, : self.head_dim])
        weights = kw[:, self.head_dim :]
        q_pe, q_nope = q.split(
            [self.rope_dim, self.head_dim - self.rope_dim],
            dim=-1,
        )
        k_pe, k_nope = k.split(
            [self.rope_dim, self.head_dim - self.rope_dim],
            dim=-1,
        )
        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q = (
            torch.cat(
                [q_pe.reshape(-1, self.n_head, self.rope_dim), q_nope],
                dim=-1,
            )
            .to(torch.bfloat16)
            .contiguous()
        )
        k = (
            torch.cat(
                [k_pe.reshape(-1, self.rope_dim), k_nope],
                dim=-1,
            )
            .unsqueeze(1)
            .to(torch.bfloat16)
            .contiguous()
        )
        # Unlike upstream's FP8 path, rc1 native BF16 lightning indexer
        # receives raw projection weights (no q_scale/softmax rescaling).
        return self.k_cache(
            q,
            k,
            weights.to(torch.bfloat16).contiguous(),
            self.topk_indices_buffer,
        )


class Glm52MLAWrapper(MultiHeadLatentAttentionWrapper):
    """Reuse upstream preprocessing/rope/o_proj with a native sparse core."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        scale,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        q_lora_rank,
        kv_lora_rank,
        mla_modules,
        cache_config=None,
        quant_config=None,
        prefix="",
        skip_topk=False,
    ):
        config = get_current_vllm_config().model_config.hf_config
        if not (_is_glm52(config) and mla_modules.is_sparse):
            super().__init__(
                hidden_size,
                num_heads,
                scale,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                q_lora_rank,
                kv_lora_rank,
                mla_modules,
                cache_config,
                quant_config,
                prefix,
                skip_topk,
            )
            return
        nn.Module.__init__(self)
        from vllm_fl.attention.sparse_mla import SparseMLAAttention

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        for name in (
            "fused_qkv_a_proj",
            "kv_a_proj_with_mqa",
            "q_a_layernorm",
            "q_b_proj",
            "q_proj",
            "kv_a_layernorm",
            "kv_b_proj",
            "rotary_emb",
            "o_proj",
            "indexer",
        ):
            setattr(self, name, getattr(mla_modules, name))
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = True
        self.skip_topk = skip_topk
        self.topk_indices_buffer = mla_modules.topk_indices_buffer
        if self.indexer is not None:
            self.topk_tokens = self.indexer.topk_tokens
        self.prefix = prefix
        self.mla_attn = SparseMLAAttention(
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=True,
            indexer=self.indexer,
            topk_indices_buffer=mla_modules.topk_indices_buffer,
        )


@cache
def _install_glm52_hooks():
    upstream_indexer = deepseek_v2.Indexer

    def scoped_indexer(vllm_config, config, *args, **kwargs):
        cls = Glm52Indexer if _is_glm52(config) else upstream_indexer
        return cls(vllm_config, config, *args, **kwargs)

    deepseek_v2.Indexer = scoped_indexer
    # PluggableLayer dispatches by class name, not its public op alias.
    PluggableLayer.register_oot(
        _decorated_layer_cls=Glm52MLAWrapper,
        name="MultiHeadLatentAttentionWrapper",
    )


class GlmMoeDsaForCausalLM(deepseek_v2.GlmMoeDsaForCausalLM):
    # The loader passes this dictionary to the quantizer by reference. Keep a
    # GLM-owned dictionary so upstream initialization cannot mutate other models.
    packed_modules_mapping = {
        **deepseek_v2.GlmMoeDsaForCausalLM.packed_modules_mapping,
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_types is not None:
            # Convert checkpoint sharing into upstream's existing constructor
            # gate and shared top-k buffer protocol; do not copy its ctor.
            if len(indexer_types) != config.num_hidden_layers:
                raise ValueError("GLM indexer_types must describe every backbone layer")
            config.index_topk_pattern = [
                "S" if isinstance(kind, str) and kind.lower() == "shared" else "F"
                for kind in indexer_types
            ]
            if config.index_topk_pattern[0] == "S":
                raise ValueError("GLM first backbone layer must produce indexer top-k")
        _install_glm52_hooks()
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        # These optional speculative-only tensors do not belong to target LM.
        return super().load_weights(
            (name, weight)
            for name, weight in weights
            if name not in {"tinyrot.weight", "rot.weight"}
        )
