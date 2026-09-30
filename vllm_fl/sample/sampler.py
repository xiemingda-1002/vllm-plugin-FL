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
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

logger = init_logger(__name__)

_GLOBAL_STREAM: Any | None = None


def async_exponential_enabled(enable_async_exponential: bool = False) -> bool:
    """Apply the async-exponential gate without relying on a scoped config."""
    return bool(enable_async_exponential) and not envs.VLLM_BATCH_INVARIANT


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


def reduce_sample_greedy(logits: torch.Tensor, tp_group: Any) -> torch.Tensor:
    """Select a global argmax from uniform TP-local vocabulary shards."""
    _, local_vocab = logits.shape
    local_value, local_index = logits.max(dim=-1)
    global_index = local_index + tp_group.rank_in_group * local_vocab
    values = tp_group.all_gather(local_value.unsqueeze(-1), dim=-1)
    indices = tp_group.all_gather(global_index.unsqueeze(-1), dim=-1)
    winner = values.argmax(dim=-1, keepdim=True)
    return indices.gather(-1, winner).squeeze(-1)


def reduce_sample_candidates(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    max_top_k: int,
    tp_group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only per-shard top-k candidates, then filter globally.

    ``max_top_k`` is supplied by the runner after excluding full-vocabulary
    rows.  This contract avoids a hidden full-vocab collective.
    """
    _, local_vocab = logits.shape
    local_k = min(max_top_k, local_vocab)
    values, indices = torch.topk(logits, k=local_k, dim=-1)
    indices = indices + tp_group.rank_in_group * local_vocab
    values = tp_group.all_gather(values, dim=-1)
    indices = tp_group.all_gather(indices, dim=-1)
    if k is not None or p is not None:
        import torch_npu

        values = torch_npu.npu_top_k_top_p(values, k=k, p=p)
    return values, indices


class AscendTopKTopPSampler(TopKTopPSampler):
    """Current rc1 native top-k/top-p sampling path for Ascend."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # AscendSampler replaces this with its runner-initialized value.
        self.enable_async_exponential = False
        self.enable_reduce_sample = False
        self.max_top_k: int | None = None

    def prepare_sampling(self, max_top_k: int | None) -> None:
        self.max_top_k = max_top_k

    def set_reduce_sample_enabled(self, enabled: bool) -> None:
        self.enable_reduce_sample = bool(enabled)

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if envs.VLLM_BATCH_INVARIANT:
            return super().forward_native(logits, generators, k, p)

        if self.enable_reduce_sample:
            if self.max_top_k is None:
                raise RuntimeError("reduce-sample requires a finite prepared top-k")
            from vllm.distributed.parallel_state import get_tp_group

            candidate_logits, candidate_ids = reduce_sample_candidates(
                logits, k, p, self.max_top_k, get_tp_group()
            )
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = candidate_logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = candidate_logits.log_softmax(
                    dim=-1, dtype=torch.float32
                )
            probs = candidate_logits.softmax(dim=-1, dtype=torch.float32)
            position = random_sample(probs, generators, self.use_fp64_gumbel)
            token_ids = candidate_ids.gather(1, position.unsqueeze(1)).squeeze(1)
            return token_ids, logits_to_return

        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if self.enable_async_exponential and hasattr(self, "_async_noise"):
            logger.debug_once(
                "[sample/sampler] Using async-exponential sampling path. "
                "Pre-computed exponential randoms from separate stream will be used."
            )
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
        enable_async_exponential: bool = False,
        enable_reduce_sample: bool = False,
    ) -> None:
        super().__init__(
            logprobs_mode=logprobs_mode,
            use_fp64_gumbel=use_fp64_gumbel,
        )
        self.topk_topp_sampler = AscendTopKTopPSampler(
            logprobs_mode=logprobs_mode,
            use_fp64_gumbel=use_fp64_gumbel,
        )
        self.enable_async_exponential = async_exponential_enabled(
            enable_async_exponential
        )
        self.topk_topp_sampler.enable_async_exponential = (
            self.enable_async_exponential
        )
        self.enable_reduce_sample = bool(enable_reduce_sample)
        self.topk_topp_sampler.set_reduce_sample_enabled(self.enable_reduce_sample)
        self.async_exponential_event = torch.npu.Event()

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """Apply the rc1 Triton penalties, or retain the upstream fallback.

        The Triton path is deliberately selected here rather than at module
        import time: non-Ascend workers continue to use :class:`Sampler`, and
        Ascend installations without Triton retain upstream semantics.
        """
        if not HAS_TRITON:
            logger.warning_once(
                "[sample/sampler] Triton not available; falling back to "
                "vLLM's penalty implementation."
            )
            return Sampler.apply_penalties(
                logits, sampling_metadata, output_token_ids
            )
        if sampling_metadata.no_penalties:
            return logits
        assert sampling_metadata.prompt_token_ids is not None
        from vllm_fl.sample.penalties import apply_all_penalties

        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
            reduce_sample=self.topk_topp_sampler.enable_reduce_sample,
        )

    def do_async_exponential(
        self,
        batch_size: int,
        vocab_size: int,
        generators: dict[int, torch.Generator],
    ) -> None:
        """Overlap exponential-noise generation with the model forward."""
        # Native rc1 also bypasses consumption in reduce mode.  Avoid spending
        # a full-vocab buffer on an intentionally separate candidate path.
        if self.enable_reduce_sample:
            return
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

    def prepare_sampling(self, max_top_k: int | None, enabled: bool) -> None:
        """Set request-batch effective state without mutating global config."""
        self.topk_topp_sampler.set_reduce_sample_enabled(
            self.enable_reduce_sample and enabled
        )
        self.topk_topp_sampler.prepare_sampling(max_top_k)

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Use the same active TP group as random candidate sampling."""
        if self.topk_topp_sampler.enable_reduce_sample:
            from vllm.distributed.parallel_state import get_tp_group

            return reduce_sample_greedy(logits, get_tp_group())
        return logits.argmax(dim=-1).view(-1)
