"""Ascend-specific sampling integration for FL."""

from .sampler import AscendSampler, global_stream

__all__ = ["AscendSampler", "global_stream"]
