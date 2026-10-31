# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend backend implementation.

This backend provides operator implementations for Huawei Ascend NPUs.
"""

from __future__ import annotations

import torch

from vllm_fl.dispatch.backends.base import Backend


def _prefer_v53(query: torch.Tensor, compact_cache: torch.Tensor | None) -> bool:
    """Shape-level tie-break between the V5.3 module and the upstream kernel.

    Two call shapes are structurally faster on the upstream kernel (measured on
    910_9382, adapter-level NPUGraph replay median, bf16 / D=512 / N1=8):

    * pure sliding-window attention, ``compact_cache is None``: 0.348 ms
      upstream vs 0.485 ms V5.3 at T=264.  There is no compact branch, so the
      module's per-call plan/gather host work is never amortised.
    * single-token decode, ``T <= 1``: 0.126 ms vs 0.313 ms.  The same host
      work cannot be amortised over a single row.

    Everything else keeps the V5.3 module when ``supports()`` accepts it.
    Metadata only -- no device read, no host sync -- so the caller stays
    graph-capturable.
    """
    if compact_cache is None:
        return False
    if not isinstance(query, torch.Tensor) or query.ndim < 1:
        return False
    return int(query.shape[0]) > 1


class AscendBackend(Backend):
    """
    Ascend backend for operator implementations.

    This backend uses Ascend CANN libraries to provide high-performance
    operator implementations for Huawei Ascend NPUs.
    """

    _available: bool | None = None

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def vendor(self) -> str | None:
        return "ascend"

    def is_available(self) -> bool:
        """Check if Ascend hardware and libraries are available."""
        if AscendBackend._available is None:
            # Check if NPU device is available
            if torch.npu.is_available() and torch.npu.device_count() > 0:
                AscendBackend._available = True
            else:
                AscendBackend._available = False
        return AscendBackend._available

    # ==================== Operator Implementations ====================
    def deepseek_v4_compressor(self, *args, **kwargs):
        from .impl.compressor import deepseek_v4_compressor

        return deepseek_v4_compressor(*args, **kwargs)

    def deepseek_v4_sparse_attention(
        self,
        query: torch.Tensor,
        swa_cache: torch.Tensor,
        compact_cache: torch.Tensor | None,
        swa_block_table: torch.Tensor,
        compact_block_table: torch.Tensor | None,
        request_ids: torch.Tensor,
        positions: torch.Tensor,
        swa_lengths: torch.Tensor,
        compact_lengths: torch.Tensor | None,
        compact_indices: torch.Tensor | None,
        window_size: int,
        compress_ratio: int = 4,
        scale: float | None = None,
        attn_sink: torch.Tensor | None = None,
        value_dim: int | None = None,
        unique_indices: bool | None = None,
    ) -> torch.Tensor:
        """Joint SWA/compact normalization: V5.3 module or upstream kernel.

        Dispatch is by call metadata only (``supports()`` reads shapes, dtypes
        and strides, never the device), so it stays graph-capturable:

        * inside the V5.3 envelope -- production window 128, fp16/bf16,
          D<=512, H<=16, compress ratio 0/4/128, contiguous metadata, paged
          caches spanning < 2**31 elements -- and not one of the two degenerate
          shapes below, the grouped-window Triton module
          ``impl/sparse_attn_sharedkv.py`` serves the call;
        * two degenerate shapes go back to the upstream kernel because it is
          measurably faster there: pure sliding window (``compact_cache is
          None``, 0.348 vs 0.485 ms) and single-token decode (``T <= 1``,
          0.126 vs 0.313 ms).  See ``_prefer_v53``;
        * everything else keeps ``impl/sparse_attn_sharedkv_upstream``, the
          kernel this backend shipped before, byte for byte: fp32, small or
          odd windows, wide (int64) views, strided metadata.

        ``unique_indices`` belongs to the V5.3 path only (see the interface
        section of that module): ``True``/``None`` = the caller guarantees no repeated slot in
        a ``compact_indices`` row (production indexer top-K, C128 ``arange``),
        so when the kernel's host-metadata coverage rule sees a row that spans
        every candidate it may take the dense "limit" mask, which is exact
        there; ``False`` = repeats are possible and the multiplicity mask is
        pinned.  It is never forwarded to the upstream signature.
        """
        if unique_indices is None:
            unique_indices = True
        from .impl.sparse_attn_sharedkv import (
            deepseek_v4_sparse_attention as v53_forward,
            supports as v53_supports,
        )

        if _prefer_v53(query, compact_cache) and v53_supports(
            query,
            swa_cache,
            compact_cache,
            swa_block_table,
            compact_block_table,
            request_ids,
            positions,
            swa_lengths,
            compact_lengths,
            compact_indices,
            window_size,
            compress_ratio,
            scale,
            attn_sink,
            value_dim,
        ):
            return v53_forward(
                query,
                swa_cache,
                compact_cache,
                swa_block_table,
                compact_block_table,
                request_ids,
                positions,
                swa_lengths,
                compact_lengths,
                compact_indices,
                window_size,
                compress_ratio,
                scale,
                attn_sink,
                value_dim,
                unique_indices=bool(unique_indices),
            )
        from .impl.sparse_attn_sharedkv_upstream import (
            deepseek_v4_sparse_attention as upstream_impl,
        )

        return upstream_impl(
            query,
            swa_cache,
            compact_cache,
            swa_block_table,
            compact_block_table,
            request_ids,
            positions,
            swa_lengths,
            compact_lengths,
            compact_indices,
            window_size,
            compress_ratio=compress_ratio,
            scale=scale,
            attn_sink=attn_sink,
            value_dim=value_dim,
        )

    def deepseek_v4_indexer(self, *args, **kwargs):
        from .impl.quant_lightning_indexer import deepseek_v4_indexer

        return deepseek_v4_indexer(*args, **kwargs)

    def deepseek_v4_index_quant(self, *args, **kwargs):
        from .impl.quantization import deepseek_v4_index_quant

        return deepseek_v4_index_quant(*args, **kwargs)

    def paged_cache_write(self, *args, **kwargs):
        """Write token rows into a paged cache using Triton."""
        from .impl.scatter_nd_update_v2 import scatter_nd_update_v2

        return scatter_nd_update_v2(*args, **kwargs)

    def modelslim_w8a8_prepare_experts(self, *args, **kwargs):
        from .impl.w8a8_moe import modelslim_w8a8_prepare_experts

        return modelslim_w8a8_prepare_experts(*args, **kwargs)

    def modelslim_w8a8_experts(self, *args, **kwargs):
        from .impl.w8a8_moe import modelslim_w8a8_experts

        return modelslim_w8a8_experts(*args, **kwargs)

    def modelslim_w8a8_prepare_weight(self, *args, **kwargs):
        from .impl.quantization import modelslim_w8a8_prepare_weight

        return modelslim_w8a8_prepare_weight(*args, **kwargs)

    def modelslim_w8a8_dynamic_linear(self, *args, **kwargs):
        from .impl.quantization import modelslim_w8a8_dynamic_linear

        return modelslim_w8a8_dynamic_linear(*args, **kwargs)

    def gdn_decode(self, *args, **kwargs):
        from .impl.gdn_reference import gdn_decode

        return gdn_decode(*args, **kwargs)

    def gdn_prefill(self, *args, **kwargs):
        from .impl.gdn_reference import gdn_prefill

        return gdn_prefill(*args, **kwargs)

    def fused_experts(self, *args, **kwargs):
        """Stable expert-call seam; implementation owns routing and math."""
        from .impl.fused_moe import fused_experts_impl

        return fused_experts_impl(*args, **kwargs)

    def topk_softmax(
        self,
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize: bool,
    ):
        from .impl.fused_moe import topk_softmax

        return topk_softmax(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_ascend

        return silu_and_mul_ascend(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        RMS normalization.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_ascend

        return rms_norm_ascend(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            obj: The calling obj (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_ascend

        return rotary_embedding_ascend(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for Ascend NPU.

        This method returns the native Ascend attention backend that uses
        torch_npu operators (npu_fused_infer_attention_score, etc.)
        instead of flag_gems operators.

        Uses vllm_fl's native Ascend implementation which directly calls
        torch_npu operators without depending on vllm-ascend package.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            if use_sparse:
                raise NotImplementedError(
                    "MLA with sparse attention is not implemented for Ascend yet."
                )
            return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendMLABackend"
        return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendAttentionBackend"
