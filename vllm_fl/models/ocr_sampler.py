# Copyright (c) 2026 BAAI. All rights reserved.
"""Torch-only sampling numerics under the vLLM sampling metadata protocol."""

import torch
from torch import nn

from vllm.v1.sample.sampler import Sampler


class OCRTopKTopPSampler(nn.Module):
    def forward(self, logits, generators, k, p):
        values, ids = logits.sort(dim=-1, descending=True)
        if k is not None:
            positions = torch.arange(values.shape[-1], device=values.device)
            values.masked_fill_(positions[None] >= k[:, None], -float("inf"))
        if p is not None:
            cumulative = values.softmax(-1).cumsum(-1)
            mask = cumulative - values.softmax(-1) >= p[:, None]
            mask[:, 0] = False
            values.masked_fill_(mask, -float("inf"))
        filtered = torch.full_like(logits, -float("inf")).scatter_(1, ids, values)
        probs = filtered.softmax(-1)
        samples = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
        for row in range(logits.shape[0]):
            samples[row] = torch.multinomial(
                probs[row], 1, generator=generators.get(row)
            )[0]
        return samples, None


class OCRSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk_topp_sampler = OCRTopKTopPSampler()

    def forward(
        self,
        logits,
        sampling_metadata,
        predict_bonus_token=False,
        logprobs_mode_override=None,
    ):
        if (
            sampling_metadata.max_num_logprobs is not None
            or sampling_metadata.logprob_token_ids
        ):
            raise ValueError(
                "OCR torch-only sampler currently requires logprobs disabled"
            )
        return super().forward(
            logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )

    @staticmethod
    def apply_penalties(logits, sampling_metadata, output_token_ids):
        if sampling_metadata.no_penalties:
            return logits
        vocab = logits.shape[-1]
        prompt = sampling_metadata.prompt_token_ids
        for row, output in enumerate(output_token_ids):
            counts = torch.zeros(vocab + 1, dtype=logits.dtype, device=logits.device)
            ids = torch.tensor(
                [x if 0 <= x < vocab else vocab for x in output],
                device=logits.device,
                dtype=torch.int64,
            )
            counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
            counts = counts[:vocab]
            seen = counts > 0
            if prompt is not None:
                pids = prompt[row].long()
                pmask = torch.zeros(vocab + 1, dtype=torch.bool, device=logits.device)
                pmask.scatter_(
                    0, torch.where((pids >= 0) & (pids < vocab), pids, vocab), True
                )
                seen = seen | pmask[:vocab]
            penalty = sampling_metadata.repetition_penalties[row]
            penalized = torch.where(
                logits[row] > 0, logits[row] / penalty, logits[row] * penalty
            )
            logits[row] = torch.where(seen, penalized, logits[row])
            logits[row] -= counts * sampling_metadata.frequency_penalties[row]
            logits[row] -= (counts > 0).to(
                logits.dtype
            ) * sampling_metadata.presence_penalties[row]
        return logits
