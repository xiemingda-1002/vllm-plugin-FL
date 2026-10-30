# SPDX-License-Identifier: Apache-2.0
"""DSv4 MTP contracts and tensor helpers; no vLLM/Ascend import side effects."""
from __future__ import annotations

import torch


def mtp_sequence_partition(value, rank, world_size, pad_size):
    """Match the embedding's padded token-row reduce-scatter partition."""
    if pad_size:
        value = torch.cat((value, value.new_zeros((pad_size,) + value.shape[1:])), dim=0)
    if value.shape[0] % world_size:
        raise ValueError("MTP sequence extent must be divisible by tensor parallel size")
    return value.chunk(world_size, dim=0)[rank]


def validate_dsv4_mtp_config(config):
    spec = config.speculative_config
    import os
    for name in ("MTP_PREPARE_STEPS", "MTP_FUSED_INPUTS", "MTP_LOCAL_ARGMAX"):
        if os.getenv(name, "1") not in ("0", "1"):
            raise ValueError(name + " must be 0 or 1")
    if not (config.additional_config or {}).get("enable_dsv4_mtp", False):
        raise NotImplementedError("DSv4 MTP requires additional_config.enable_dsv4_mtp=true")
    if (config.additional_config or {}).get("enable_dsa_cp", False):
        raise NotImplementedError("DSv4 MTP currently requires enable_dsa_cp=false")
    if spec.method != "mtp" or not 1 <= spec.num_speculative_tokens <= 3:
        raise NotImplementedError("DSv4 supports sequential method=mtp with 1..3 draft tokens")
    if getattr(spec, "parallel_drafting", False):
        raise NotImplementedError("DSv4 parallel/tree drafting is not supported")
    if getattr(spec, "num_speculative_tokens_per_batch_size", None):
        raise NotImplementedError("DSv4 MTP currently requires a fixed draft token count")
    if getattr(spec, "draft_sample_method", "greedy") != "greedy" or getattr(spec, "rejection_sample_method", "standard") != "standard":
        raise NotImplementedError("DSv4 MTP requires greedy draft sampling and standard verification")
    if not getattr(spec, "enforce_eager", False):
        raise NotImplementedError("DSv4 MTP draft requires speculative_config.enforce_eager=true")
    parallel = config.parallel_config
    for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise NotImplementedError("DSv4 MTP currently requires PP=PCP=DCP=1")
    if (getattr(config.scheduler_config, "async_scheduling", False)
            and getattr(spec, "disable_padded_drafter_batch", False)):
        raise NotImplementedError("DSv4 async MTP requires the padded GPU drafter batch")
    if getattr(getattr(config, "cache_config", None), "enable_prefix_caching", False):
        raise NotImplementedError("DSv4 MTP currently requires --no-enable-prefix-caching")
    if getattr(config, "lora_config", None) is not None or getattr(config, "kv_transfer_config", None) is not None:
        raise NotImplementedError("DSv4 MTP LoRA and disaggregated KV transfer are not supported")
    if getattr(config.model_config.hf_config, "num_nextn_predict_layers", 0) != 1:
        raise NotImplementedError("DSv4 MTP requires one exported MTP layer")


def add_mtp_quant_aliases(quant_config):
    """Map ModelSlim's raw MTP description to FL's decoder module prefixes.

    Only add aliases for mtp.* keys; keep target-model and other-family keys.
    The actual checkpoint names are mapped independently by the weight loader.
    """
    description = getattr(quant_config, "quant_description", None)
    if description is None:
        return
    replacements = ((".attn.", ".self_attn."), (".ffn.", ".mlp."),
                    (".w1.", ".gate_proj."), (".w2.", ".down_proj."),
                    (".w3.", ".up_proj."), (".ffn_norm.", ".post_attention_layernorm."),
                    (".attn_norm.", ".input_layernorm."))
    for name, value in list(description.items()):
        if not name.startswith("mtp."):
            continue
        alias = name
        for before, after in replacements:
            alias = alias.replace(before, after)
        if alias in description and description[alias] != value:
            raise ValueError(f"Conflicting MTP quantization description for {alias}")
        description[alias] = value


def resolve_next_tokens(sampled, discard, backup, vocab_size):
    if sampled.shape[0] == 0 or sampled.shape[1] == 0:
        return backup.to(torch.int32).clone(), torch.zeros_like(backup, dtype=torch.int32)
    valid = (sampled >= 0) & (sampled < vocab_size)
    counts = valid.sum(dim=1).to(torch.int32)
    offsets = torch.arange(sampled.shape[1], device=sampled.device).expand_as(sampled)
    last = torch.where(valid, offsets, -1).max(dim=1).values
    values = sampled.gather(1, last.clamp_min(0).unsqueeze(1)).squeeze(1)
    values = torch.where((last >= 0) & ~discard, values, backup)
    return values.to(torch.int32), torch.where(discard, 0, counts).to(torch.int32)


def padded_draft_indices(cu_draft, valid_counts, query_start):
    counts = cu_draft - torch.cat((cu_draft.new_zeros(1), cu_draft[:-1]))
    rejected = torch.where(counts > 0, counts + 1 - valid_counts, 0)
    return (query_start[1:] - 1 - rejected).to(torch.int32), rejected.to(torch.int32)


def advance_draft_positions(positions, block_table, seq_lens, block_size, max_model_len, active_mask=None):
    """Advance only active rows; never turn max-position padding into slot zero."""
    if positions.numel() == 0 or block_table.shape[1] == 0:
        return torch.zeros_like(positions), torch.full_like(positions, -1), torch.ones_like(seq_lens)
    next_pos = positions.to(torch.int64) + 1
    block_indices = next_pos // block_size
    valid = (next_pos >= 0) & (next_pos < max_model_len) & (block_indices < block_table.shape[1])
    if active_mask is not None:
        valid = valid & active_mask
    safe = next_pos.clamp(min=0, max=max_model_len - 1)
    columns = (safe // block_size).clamp(max=block_table.shape[1] - 1)
    blocks = block_table.gather(1, columns.unsqueeze(1)).squeeze(1).to(torch.int64)
    valid = valid & (blocks >= 0)
    slots = torch.where(valid, blocks * block_size + safe % block_size, -1)
    clamped = torch.where(valid, next_pos, 0)
    lengths = torch.where(valid, (seq_lens + 1).clamp(max=max_model_len), 1)
    return clamped, slots, lengths


def greedy_rejection_tokens(draft_ids, counts, target_argmax, bonus_ids, max_spec_len):
    """Exact greedy verification, including zero draft/first rejection/bonus.

    No vocabulary-sized recovery-sampling Triton launch is needed. Every row
    contains the accepted prefix and precisely one target token, then -1.
    """
    device = bonus_ids.device
    batch = len(counts)
    out = torch.full((batch, max_spec_len + 1), -1, dtype=torch.int32, device=device)
    if batch == 0:
        return out
    count = torch.tensor(counts, dtype=torch.int64, device=device)
    if draft_ids.numel() == 0:
        out[:, 0] = bonus_ids.flatten().to(torch.int32)
        return out
    starts = torch.cat((count.new_zeros(1), count.cumsum(0)[:-1]))
    columns = torch.arange(max_spec_len, device=device).unsqueeze(0)
    valid = columns < count.unsqueeze(1)
    indices = (starts.unsqueeze(1) + columns).clamp(max=draft_ids.numel() - 1)
    equal = draft_ids[indices] == target_argmax[indices]
    accepted = (torch.where(valid, equal, True).to(torch.int32).cumprod(1).bool() & valid).sum(1)
    out[:, :max_spec_len] = torch.where(columns < accepted.unsqueeze(1), draft_ids[indices], -1).to(torch.int32)
    next_index = (starts + accepted).clamp(max=target_argmax.numel() - 1)
    target = torch.where(accepted < count, target_argmax[next_index], bonus_ids.flatten())
    out[torch.arange(batch, device=device), accepted] = target.to(torch.int32)
    return out



def prepare_async_dsv4_host_metadata(cpu_computed, query_lengths, prev_positions,
                                     valid_counts, prev_draft_counts,
                                     prev_computed):
    """Mirror vLLM GPU rejection correction without modifying scheduler state.

    ``prev_computed`` is the previous real target batch's private snapshot;
    row moves use the exact mapping passed to the GPU correction function.
    Deferred CPU bookkeeping still owns input_batch/request counters.
    """
    import numpy as np
    computed = np.asarray(cpu_computed, dtype=np.int32).copy()
    queries = np.asarray(query_lengths, dtype=np.int32)
    previous_rows = np.asarray(prev_positions, dtype=np.int32)
    if computed.ndim != 1 or queries.shape != computed.shape or previous_rows.shape != computed.shape:
        raise ValueError("DSv4 async host metadata must have one row per active request")
    if np.any(queries < 0):
        raise ValueError("Negative scheduled query length")
    if valid_counts is not None:
        drafts = np.asarray(prev_draft_counts, dtype=np.int32)
        mask = previous_rows >= 0
        rows = previous_rows[mask]
        if np.any(rows >= len(drafts)):
            raise ValueError("Previous draft row outside copied counts")
        mask[np.flatnonzero(mask)] = drafts[rows] > 0
        rows = previous_rows[mask]
        if rows.size:
            if prev_computed is None:
                raise RuntimeError("Missing previous DSv4 target snapshot for async correction")
            previous = np.asarray(prev_computed, dtype=np.int32)
            counts = np.asarray(valid_counts, dtype=np.int32)
            if np.any(rows >= len(previous)) or np.any(rows >= len(counts)):
                raise ValueError("Previous row outside DSv4 target/acceptance snapshot")
            if np.any(counts[rows] < 0) or np.any(counts[rows] > drafts[rows] + 1):
                raise ValueError("Accepted token count outside previous draft-plus-bonus extent")
            # Same formula as update_num_computed_tokens_for_batch_change:
            # previous actual computed + current accepted-prefix/bonus count.
            computed[mask] = previous[rows] + counts[rows]
    seq_lens = computed + queries
    starts = np.cumsum(queries, dtype=np.int64) - queries
    positions = (np.repeat(computed.astype(np.int64), queries)
                 + np.arange(int(queries.sum()), dtype=np.int64)
                 - np.repeat(starts, queries))
    return computed, seq_lens, positions
