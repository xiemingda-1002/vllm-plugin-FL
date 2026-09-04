# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Ascend KV-cache contracts used by DeepSeek V4 sparse attention."""

from dataclasses import dataclass

import torch
from typing_extensions import Self
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

try:
    from vllm.model_executor.layers.deepseek_compressor import (
        CompressorStateCache,
    )
    from vllm.model_executor.layers.deepseek_v4_attention import (
        DeepseekV4IndexerCache,
    )
except ImportError:
    from vllm.models.deepseek_v4.attention import DeepseekV4IndexerCache
    from vllm.models.deepseek_v4.compressor import CompressorStateCache


@dataclass(frozen=True, kw_only=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLA cache spec with an optional per-token quantization scale."""

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8

    @property
    def page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * (
            self.head_size * get_dtype_size(self.dtype)
            + self.scale_dim * get_dtype_size(self.scale_dtype)
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        parallel_config = vllm_config.parallel_config
        context_parallel_size = (
            parallel_config.decode_context_parallel_size
            * parallel_config.prefill_context_parallel_size
        )
        if context_parallel_size > 1:
            max_model_len = cdiv(max_model_len, context_parallel_size)
        return (
            cdiv(max_model_len, self.block_size * self.compress_ratio)
            * self.page_size_bytes
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendMLAAttentionSpec) for spec in specs)
        cache_dtype_strs = {spec.cache_dtype_str for spec in specs}
        compress_ratios = {spec.compress_ratio for spec in specs}
        model_versions = {spec.model_version for spec in specs}
        scale_dims = {spec.scale_dim for spec in specs}
        scale_dtypes = {spec.scale_dtype for spec in specs}
        assert all(
            len(values) == 1
            for values in (
                cache_dtype_strs,
                compress_ratios,
                model_versions,
                scale_dims,
                scale_dtypes,
            )
        ), "All layers in a merged Ascend MLA cache must use one layout."
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            cache_dtype_str=cache_dtype_strs.pop(),
            compress_ratio=compress_ratios.pop(),
            model_version=model_versions.pop(),
            scale_dim=scale_dims.pop(),
            scale_dtype=scale_dtypes.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class AscendSlidingWindowMLASpec(SlidingWindowMLASpec):
    """Sliding-window MLA spec whose storage follows the logical dtype."""

    @property
    def real_page_size_bytes(self) -> int:
        return (
            self.storage_block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(
            isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs
        )
        cache_dtype_strs = {spec.cache_dtype_str for spec in specs}
        compress_ratios = {spec.compress_ratio for spec in specs}
        model_versions = {spec.model_version for spec in specs}
        sliding_windows = {spec.sliding_window for spec in specs}
        assert all(
            len(values) == 1
            for values in (
                cache_dtype_strs,
                compress_ratios,
                model_versions,
                sliding_windows,
            )
        ), "All layers in a merged Ascend sliding-window cache must match."
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=sliding_windows.pop(),
            cache_dtype_str=cache_dtype_strs.pop(),
            compress_ratio=compress_ratios.pop(),
            model_version=model_versions.pop(),
        )


class AscendCompressorStateCache(CompressorStateCache):
    """Compressor state cache using the block layout required by DSA ops."""

    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        block_size: int,
        prefix: str,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert dtype == torch.float32
        assert compress_ratio in (4, 128)
        self.compress_ratio = compress_ratio
        self.sliding_window = (1 + (compress_ratio == 4)) * compress_ratio
        self.block_size = block_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        page_size_padded = (
            16640
            if self.state_dim == 2 * 256 and self.compress_ratio == 4
            else 131072
        )
        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=None,
            page_size_padded=page_size_padded,
        )

    def forward(self): ...

    def get_attn_backend(self):
        from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa import (
            AscendDSABackend,
        )

        return AscendDSABackend


class AscendDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    """Quantized indexer cache storing int8 keys and fp16 scales."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            model_version="deepseek_v4",
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.cache_config.cache_dtype,
            scale_dim=1 if self.head_dim == 128 else 0,
            scale_dtype=torch.float16,
        )

    def forward(self): ...

    def get_attn_backend(self):
        from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa import (
            AscendDSABackend,
        )

        return AscendDSABackend


class AscendDeepseekV4SWACache(DeepseekV4SWACache):
    """DeepSeek V4 sliding-window cache using the Ascend DSA layout."""

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__(head_dim, window_size, torch.uint8, prefix, cache_config)
        self.dtype = dtype
        self.block_size = 128

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            model_version="deepseek_v4",
            alignment=None,
        )

    def forward(self): ...

    def get_attn_backend(self):
        from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa import (
            AscendDSABackend,
        )

        return AscendDSABackend


def reshape_dsa_kv_cache(
    raw_tensor: torch.Tensor,
    kv_cache_spec: KVCacheSpec,
    num_blocks: int,
) -> list[torch.Tensor] | None:
    """Create page-strided views for DSA attention and state caches."""
    if not isinstance(
        kv_cache_spec,
        (AscendMLAAttentionSpec, AscendSlidingWindowMLASpec),
    ):
        return None

    shapes = [
        (
            num_blocks,
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
        )
    ]
    dtypes = [kv_cache_spec.dtype]
    if (
        isinstance(kv_cache_spec, AscendMLAAttentionSpec)
        and kv_cache_spec.scale_dim > 0
    ):
        shapes.append(
            (
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.scale_dim,
            )
        )
        dtypes.append(kv_cache_spec.scale_dtype)

    views: list[torch.Tensor] = []
    storage_offset_bytes = 0
    for shape, dtype in zip(shapes, dtypes):
        dtype_size = get_dtype_size(dtype)
        contiguous_stride = torch.empty(shape).stride()
        page_stride = kv_cache_spec.page_size_bytes // dtype_size
        assert storage_offset_bytes % dtype_size == 0
        views.append(
            torch.as_strided(
                raw_tensor.view(dtype),
                size=shape,
                stride=(page_stride, *contiguous_stride[1:]),
                storage_offset=storage_offset_bytes // dtype_size,
            )
        )
        storage_offset_bytes += contiguous_stride[0] * dtype_size
    assert storage_offset_bytes <= kv_cache_spec.page_size_bytes
    return views
