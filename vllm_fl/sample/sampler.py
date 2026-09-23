# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM-Ascend v0.24.0rc1.

"""Ascend sampler paths that avoid host synchronization.

This module is imported by the model runner only for the Ascend vendor.  It
keeps the current upstream Sampler contract while applying the current
vLLM-Ascend global-stream random sampling and NPU top-k/top-p implementation.
"""

from __future__ import annotations

from typing import Any

import torch
import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

logger = init_logger(__name__)

_GLOBAL_STREAM: Any | None = None


def _additional_config() -> dict[str, Any]:
    try:
        return get_current_vllm_config().additional_config or {}
    except (AssertionError, AttributeError):
        return {}


def async_exponential_enabled() -> bool:
    """Match rc1 configuration, including batch-invariant incompatibility."""
    return bool(_additional_config().get("enable_async_exponential", False)) and not (
        envs.VLLM_BATCH_INVARIANT
    )


def global_stream():
    """Return the process-local stream shared by Ascend sampling/state work."""
    global _GLOBAL_STREAM
    if _GLOBAL_STREAM is None:
        _GLOBAL_STREAM = torch.npu.Stream()
    return _GLOBAL_STREAM


def _empty_exponential_noise_like(
    probs: torch.Tensor, use_fp64_gumbel: bool
) -> torch.Tensor:
    dtype = torch.float64 if use_fp64_gumbel else probs.dtype
    return torch.empty(probs.shape, dtype=dtype, device=probs.device)


def _sample_with_exponential_noise(
    probs: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    if noise.dtype == probs.dtype:
        scores = probs.div_(noise)
    else:
        scores = noise.reciprocal_().mul_(probs)
    return scores.argmax(dim=-1).view(-1)


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Sample on the global NPU stream without a CPU/NPU synchronization."""
    stream = global_stream()
    with torch.npu.stream(stream):
        noise = _empty_exponential_noise_like(probs, use_fp64_gumbel)
        if len(generators) != probs.shape[0]:
            noise.exponential_()
        for row, generator in generators.items():
            noise[row].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(stream)
    return _sample_with_exponential_noise(probs, noise)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Use the A2/A3 torch-npu fused filter used by current vLLM-Ascend."""
    if p is None and k is None:
        return logits
    import torch_npu

    return torch_npu.npu_top_k_top_p(logits, k=k, p=p)


class AscendTopKTopPSampler(TopKTopPSampler):
    """Current rc1 native top-k/top-p sampling path for Ascend."""

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if envs.VLLM_BATCH_INVARIANT:
            return super().forward_native(logits, generators, k, p)

        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if async_exponential_enabled() and hasattr(self, "_async_noise"):
            self._async_noise_event.synchronize()
            sampled = _sample_with_exponential_noise(probs, self._async_noise)
            del self._async_noise
            return sampled, logits_to_return
        return (
            random_sample(probs, generators, self.use_fp64_gumbel),
            logits_to_return,
        )

    def set_async_noise(self, noise: torch.Tensor, event: Any) -> None:
        self._async_noise = noise
        self._async_noise_event = event


class AscendSampler(Sampler):
    """Upstream-compatible Sampler with rc1 Ascend sampling primitives."""

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ) -> None:
        super().__init__(
            logprobs_mode=logprobs_mode,
            use_fp64_gumbel=use_fp64_gumbel,
        )
        self.topk_topp_sampler = AscendTopKTopPSampler(
            logprobs_mode=logprobs_mode,
            use_fp64_gumbel=use_fp64_gumbel,
        )
        self.async_exponential_event = torch.npu.Event()

    def do_async_exponential(
        self,
        batch_size: int,
        vocab_size: int,
        generators: dict[int, torch.Generator],
    ) -> None:
        """Overlap exponential-noise generation with the model forward."""
        current_stream = torch.npu.current_stream()
        stream = global_stream()
        with torch.npu.stream(stream):
            stream.wait_stream(current_stream)
            dtype = torch.float64 if self.use_fp64_gumbel else torch.float32
            noise = torch.empty(
                (batch_size, vocab_size), device="npu", dtype=dtype
            )
            if len(generators) != batch_size:
                noise.exponential_()
            for row, generator in generators.items():
                noise[row].exponential_(generator=generator)
            self.async_exponential_event.record()
        self.topk_topp_sampler.set_async_noise(
            noise, self.async_exponential_event
        )
