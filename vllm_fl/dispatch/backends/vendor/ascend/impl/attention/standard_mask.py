# Copyright (c) 2026 BAAI. All rights reserved.

"""
Attention mask builder for Ascend NPU backend.

This module provides utilities for creating and caching attention masks
required by Ascend NPU attention operators.
"""

from typing import ClassVar, Optional

import torch


class AttentionMaskBuilder:
    """
    Builder for creating and caching attention masks.

    This class manages attention mask creation with caching to avoid
    redundant tensor allocations. The masks are created lazily and
    cached at the class level for reuse across instances.

    Attributes:
        device: The device to create masks on.
    """

    # Class-level cache for masks (shared across all instances)
    _chunked_prefill_mask: ClassVar[Optional[torch.Tensor]] = None
    _chunked_prefill_mask_device: ClassVar[Optional[torch.device]] = None
    _mla_mask: ClassVar[Optional[torch.Tensor]] = None
    _mla_mask_dtype: ClassVar[Optional[torch.dtype]] = None
    _pcp_mla_mask: ClassVar[Optional[torch.Tensor]] = None
    _pcp_mla_mask_dtype: ClassVar[Optional[torch.dtype]] = None

    def __init__(self, device: torch.device):
        """
        Initialize the attention mask builder.

        Args:
            device: The device to create masks on.
        """
        self.device = device

    def get_splitfuse_attn_mask(self) -> torch.Tensor:
        """
        Get attention mask for split-fuse (chunked prefill) attention.

        Creates a 2048x2048 upper triangular mask with int8 dtype.
        The mask is cached and reused for subsequent calls.

        Returns:
            Upper triangular attention mask tensor.
        """
        cls = AttentionMaskBuilder
        if (cls._chunked_prefill_mask is None or
            cls._chunked_prefill_mask_device != self.device):
            cls._chunked_prefill_mask = torch.triu(
                torch.ones(2048, 2048), diagonal=1
            ).to(torch.int8).to(self.device)
            cls._chunked_prefill_mask_device = self.device
        return cls._chunked_prefill_mask

    def get_rswa_attn_mask(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        max_kv_len: int,
        max_query_len: int,
        rswa_window: int,
    ) -> torch.Tensor:
        if not isinstance(rswa_window, int) or isinstance(rswa_window, bool) or rswa_window <= 0:
            raise ValueError(f"rswa_window must be a positive integer, got {rswa_window!r}")
        if not isinstance(max_kv_len, int) or max_kv_len <= 0:
            raise ValueError(f"max_kv_len must be a positive integer, got {max_kv_len!r}")
        if not isinstance(max_query_len, int) or max_query_len <= 0:
            raise ValueError(f"max_query_len must be a positive integer, got {max_query_len!r}")

        prefix_lens = prefix_lens.to(device=self.device, dtype=torch.int64, non_blocking=True).reshape(-1)
        seq_lens = seq_lens.to(device=self.device, dtype=torch.int64, non_blocking=True).reshape(-1)
        query_lens = query_lens.to(device=self.device, dtype=torch.int64, non_blocking=True).reshape(-1)
        num_reqs = prefix_lens.numel()
        if seq_lens.numel() != num_reqs or query_lens.numel() != num_reqs:
            raise ValueError(
                "R-SWA metadata tensors must have the same number of requests: "
                f"prefix={num_reqs}, seq={seq_lens.numel()}, query={query_lens.numel()}"
            )

        # An int8 mask still becomes very large for long-context batches. Fail
        # before allocating a multi-gigabyte temporary instead of surfacing an
        # opaque device OOM from one of the arange/broadcast operations.
        mask_elements = num_reqs * max_query_len * max_kv_len
        if mask_elements > (1 << 29):
            raise ValueError(
                f"R-SWA mask is too large: {num_reqs}*{max_query_len}*{max_kv_len}={mask_elements} elements"
            )

        # The full prompt can be longer than the KV processed by an early prefill
        # chunk. valid_keys and causal below still mask unprocessed/future tokens.
        torch._assert(torch.all(prefix_lens >= 0), "R-SWA prefix_lens must be non-negative")
        torch._assert(torch.all(seq_lens >= 0), "R-SWA seq_lens must be non-negative")
        torch._assert(torch.all(query_lens >= 0), "R-SWA query_lens must be non-negative")
        torch._assert(torch.all(prefix_lens <= max_kv_len), "R-SWA prefix_lens exceeds max_kv_len")
        torch._assert(torch.all(query_lens <= seq_lens), "R-SWA query_lens must be <= seq_lens")
        torch._assert(torch.all(seq_lens <= max_kv_len), "R-SWA seq_lens exceeds max_kv_len")
        torch._assert(torch.all(query_lens <= max_query_len), "R-SWA query_lens exceeds max_query_len")

        if num_reqs == 0:
            return torch.ones(
                (0, 1, max_query_len, max_kv_len),
                dtype=torch.int8,
                device=self.device,
            )

        prefix_lens = prefix_lens.view(-1, 1, 1)
        seq_lens = seq_lens.view(-1, 1, 1)
        query_lens = query_lens.view(-1, 1, 1)

        query_offsets = torch.arange(
            max_query_len,
            device=self.device,
            dtype=torch.int64,
        ).view(1, -1, 1)
        kv_positions = torch.arange(
            max_kv_len,
            device=self.device,
            dtype=torch.int64,
        ).view(1, 1, -1)
        query_positions = seq_lens - query_lens + query_offsets

        valid_queries = query_offsets < query_lens
        valid_keys = kv_positions < seq_lens
        causal = kv_positions <= query_positions
        in_prefix = kv_positions < prefix_lens
        in_window = (query_positions - kv_positions) < rswa_window
        keep = valid_queries & valid_keys & causal & (in_prefix | in_window)

        return (~keep).to(torch.int8).unsqueeze(1).contiguous()

    def get_mla_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """
        Get attention mask for MLA (Multi-head Latent Attention).

        Creates a 512x512 upper triangular mask. For fp16, uses
        float32 min value as mask; otherwise uses 1.

        Args:
            dtype: The dtype for the mask tensor.

        Returns:
            MLA attention mask tensor.
        """
        cls = AttentionMaskBuilder
        if cls._mla_mask is None or cls._mla_mask_dtype != dtype:
            if dtype == torch.float16:
                mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            prefill_mask = torch.triu(
                torch.ones(512, 512, device=self.device, dtype=dtype), 1
            )
            cls._mla_mask = torch.where(prefill_mask == 1, mask_value, 0).to(dtype)
            cls._mla_mask_dtype = dtype
        return cls._mla_mask

    def get_pcp_mla_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """
        Get attention mask for PCP (Prefill Context Parallel) MLA.

        Creates a 512x512 upper triangular mask.

        Args:
            dtype: The dtype for the mask tensor.

        Returns:
            PCP MLA attention mask tensor.
        """
        cls = AttentionMaskBuilder
        if cls._pcp_mla_mask is None or cls._pcp_mla_mask_dtype != dtype:
            cls._pcp_mla_mask = torch.triu(
                torch.ones(512, 512, device=self.device, dtype=dtype), 1
            )
            cls._pcp_mla_mask_dtype = dtype
        return cls._pcp_mla_mask

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Get a general attention mask for given sequence length.

        Creates a causal mask (lower triangular) for the given sequence length.

        Args:
            max_seq_len: Maximum sequence length.
            dtype: The dtype for the mask tensor.

        Returns:
            Causal attention mask tensor.
        """
        # Create lower triangle matrix (True for valid positions)
        mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
        # Invert to get mask positions (True for masked positions)
        mask_flag = ~mask_flag
        # For fp16, use -inf; otherwise use 1
        mask_value = float("-inf") if dtype == torch.float16 else 1
        attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype).masked_fill_(
            mask_flag, mask_value
        )
        return attn_mask.contiguous().to(self.device)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached masks. Useful for testing or memory cleanup."""
        cls._chunked_prefill_mask = None
        cls._chunked_prefill_mask_device = None
        cls._mla_mask = None
        cls._mla_mask_dtype = None
        cls._pcp_mla_mask = None
        cls._pcp_mla_mask_dtype = None


# Global obj cache for convenience
_builder_instance: Optional[AttentionMaskBuilder] = None
_builder_device: Optional[torch.device] = None


def get_attention_mask_builder(device: torch.device) -> AttentionMaskBuilder:
    """
    Get or create a global AttentionMaskBuilder obj.

    This function provides a convenient way to access the mask builder
    without managing obj lifecycle.

    Args:
        device: The device for the mask builder.

    Returns:
        AttentionMaskBuilder obj.
    """
    global _builder_instance, _builder_device
    if _builder_instance is None or _builder_device != device:
        _builder_instance = AttentionMaskBuilder(device)
        _builder_device = device
    return _builder_instance


__all__ = [
    "AttentionMaskBuilder",
    "get_attention_mask_builder",
]
