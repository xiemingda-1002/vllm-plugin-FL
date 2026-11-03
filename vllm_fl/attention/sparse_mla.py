# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA orchestration with independent, scheduler-owned cache leaves."""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_fl.compat.kv_cache import register_kv_cache_owner
from vllm_fl.dispatch import CachedOp

_write = CachedOp("paged_cache_write")
_attention = CachedOp("sparse_mla_attention")
_indexer = CachedOp("sparse_mla_indexer")


class SparseMLACacheSpec(MLAAttentionSpec):
    """Preserve each physical leaf width in the shared logical cache group."""

    @classmethod
    def merge(cls, specs):
        # The base MLA merge checks quantization/compression but permits
        # different widths. These leaves need UniformType's per-layer specs.
        assert all(spec == specs[0] for spec in specs), (
            "Sparse MLA leaves require identical layouts to merge"
        )
        return super().merge(specs)


@dataclass
class SparseMLAMetadata:
    block_table: torch.Tensor
    query_ends: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    num_actual_tokens: int


class SparseMLABackend(AttentionBackend):
    requires_graph_task_updates = False

    @staticmethod
    def get_name():
        return "FL_SPARSE_MLA"

    @staticmethod
    def is_mla():
        return True

    @staticmethod
    def is_sparse():
        return True

    @staticmethod
    def get_impl_cls():
        return SparseMLAAttention

    @staticmethod
    def get_builder_cls():
        return SparseMLAMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]

    @classmethod
    def get_preferred_block_size(cls, default_block_size):
        return 128

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        if num_kv_heads != 1:
            raise ValueError("Sparse MLA cache requires one KV head")
        return num_blocks, block_size, 1, head_size

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return tuple(range(5 if include_num_layers_dimension else 4))


class SparseMLAMetadataBuilder(AttentionMetadataBuilder[SparseMLAMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        parallel = vllm_config.parallel_config
        if (
            vllm_config.speculative_config is not None
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise NotImplementedError("Sparse MLA requires non-speculative CP1")

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        nreq = common.num_reqs
        # The runner updates these fixed device buffers before each replay.
        # No CPU lengths or captured Python lists participate in native SFA.
        return SparseMLAMetadata(
            common.block_table_tensor[:nreq],
            common.query_start_loc[1 : nreq + 1],
            common.seq_lens[:nreq],
            common.slot_mapping,
            common.num_actual_tokens,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs):
        return self.build(0, common_attn_metadata)


@register_kv_cache_owner
class SparseMLACache(nn.Module, AttentionLayerBase):
    """One contiguous tensor and one ordinary planner spec per cache leaf."""

    _fl_sparse_mla_cache = True

    def __init__(self, prefix, head_size, cache_config):
        super().__init__()
        config = get_current_vllm_config()
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("Sparse MLA currently requires BF16 cache")
        if config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Sparse MLA currently requires BF16 activations")
        if cache_config.block_size != 128:
            raise ValueError("Sparse MLA native cache requires 128-token pages")
        self.layer_name = prefix
        self.head_size = head_size
        self.kv_cache = torch.tensor([])
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate cache layer: {prefix}")
        context[prefix] = self

    def get_attn_backend(self):
        return SparseMLABackend

    def get_kv_cache_spec(self, vllm_config):
        return SparseMLACacheSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=torch.bfloat16,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
        )

    def validate_kv_cache(self, cache):
        if not isinstance(cache, torch.Tensor):
            raise TypeError("Sparse MLA cache must be a Tensor")
        if (
            cache.dtype != torch.bfloat16
            or cache.ndim != 4
            or cache.shape[1:] != (128, 1, self.head_size)
            or not cache.is_contiguous()
        ):
            raise ValueError(
                f"{self.layer_name}: expected contiguous BF16 [P,128,1,{self.head_size}], "
                f"got shape={tuple(cache.shape)}, dtype={cache.dtype}, "
                f"stride={cache.stride()}"
            )

    def bind_kv_cache(self, cache):
        self.kv_cache = cache


def _metadata(layer_name):
    metadata = get_forward_context().attn_metadata
    return metadata.get(layer_name) if isinstance(metadata, dict) else None


def _store(layer, values, metadata):
    n = metadata.num_actual_tokens
    slots = metadata.slot_mapping
    _write(
        values[:n].reshape(n, layer.head_size).contiguous(),
        slots[:n],
        layer.kv_cache.squeeze(2),
    )


def _sparse_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    metadata = _metadata(layer_name)
    output.fill_(-1)
    if metadata is None:
        return
    _store(layer, k, metadata)
    n = metadata.num_actual_tokens
    indices = _indexer(
        q[:n],
        layer.kv_cache,
        weights[:n],
        metadata.block_table,
        metadata.query_ends,
        metadata.seq_lens,
        output.shape[-1],
    )
    output[:n].copy_(indices.reshape(n, -1))


def _sparse_indexer_fake(q, k, weights, output, layer_name):
    return None


direct_register_custom_op(
    op_name="fl_sparse_mla_indexer",
    op_func=_sparse_indexer,
    mutates_args=["output"],
    fake_impl=_sparse_indexer_fake,
)


class SparseIndexerCache(SparseMLACache):
    def forward(self, q, k, weights, topk_indices_buffer):
        output = topk_indices_buffer[: q.shape[0]]
        torch.ops.vllm.fl_sparse_mla_indexer(
            q,
            k,
            weights,
            output,
            self.layer_name,
        )
        return output


def _sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    rope: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    metadata = _metadata(layer_name)
    output.zero_()
    if metadata is None:
        return
    rope_metadata = _metadata(layer.rope_cache.layer_name)
    if rope_metadata is None or (
        rope_metadata.block_table.data_ptr() != metadata.block_table.data_ptr()
    ):
        raise ValueError("Sparse MLA latent/RoPE leaves must share a planner group")
    _store(layer, kv, metadata)
    _store(layer.rope_cache, rope, rope_metadata)
    n = metadata.num_actual_tokens
    q_nope, q_rope = q[:n].split(
        [layer.qk_nope_head_dim, layer.qk_rope_head_dim],
        dim=-1,
    )
    latent_q = (
        torch.bmm(
            q_nope.transpose(0, 1),
            layer.W_UK_T,
        )
        .transpose(0, 1)
        .contiguous()
    )
    latent_output = _attention(
        latent_q,
        q_rope.contiguous(),
        layer.kv_cache,
        layer.rope_cache.kv_cache,
        indices[:n],
        metadata.block_table,
        metadata.query_ends,
        metadata.seq_lens,
        layer.scale,
    )
    projected = torch.bmm(
        latent_output.transpose(0, 1),
        layer.W_UV,
    ).transpose(0, 1)
    output[:n].copy_(projected.reshape(n, -1))


def _sparse_attention_fake(q, kv, rope, indices, output, layer_name):
    return None


direct_register_custom_op(
    op_name="fl_sparse_mla_attention",
    op_func=_sparse_attention,
    mutates_args=["output"],
    fake_impl=_sparse_attention_fake,
)


@register_kv_cache_owner
class SparseMLAAttention(MLAAttention, SparseMLACache):
    """Reuse upstream projection math with latent and RoPE stored separately."""

    def __init__(
        self,
        num_heads,
        scale,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        q_lora_rank,
        kv_lora_rank,
        kv_b_proj,
        cache_config,
        quant_config=None,
        prefix="",
        use_sparse=True,
        indexer=None,
        topk_indices_buffer=None,
    ):
        # MLAAttention's generic CUDA prefill/workspace setup is unnecessary
        # here. Subclass identity preserves the loader's post-load callback.
        SparseMLACache.__init__(self, prefix, kv_lora_rank, cache_config)
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.kv_b_proj = kv_b_proj
        self.topk_indices_buffer = topk_indices_buffer
        if topk_indices_buffer is None:
            raise ValueError("Sparse MLA requires the model's shared top-k buffer")
        self.rope_cache = SparseMLACache(
            f"{prefix}.rope_cache",
            qk_rope_head_dim,
            cache_config,
        )

    get_attn_backend = SparseMLACache.get_attn_backend
    get_kv_cache_spec = SparseMLACache.get_kv_cache_spec

    def process_weights_after_loading(self, act_dtype):
        method = getattr(self.kv_b_proj, "quant_method", None)
        if method is not None and not isinstance(method, UnquantizedLinearMethod):
            raise NotImplementedError("Sparse MLA requires unquantized kv_b_proj")
        weight = get_and_maybe_dequant_weights(
            self.kv_b_proj,
            out_dtype=act_dtype,
        ).T.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key, value = weight.split(
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        self.W_UK_T = key.permute(1, 2, 0).contiguous()
        self.W_UV = value.transpose(0, 1).contiguous()

    def forward(self, q, kv_c_normed, k_pe, output_shape=None):
        output = q.new_empty(output_shape)
        torch.ops.vllm.fl_sparse_mla_attention(
            q,
            kv_c_normed,
            k_pe,
            self.topk_indices_buffer[: q.shape[0]],
            output,
            self.layer_name,
        )
        return output
