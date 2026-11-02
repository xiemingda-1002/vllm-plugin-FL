# SPDX-License-Identifier: Apache-2.0
"""DSv4-only eager MTP proposer and exact greedy verifier for FL Ascend."""
import os
from copy import copy
from dataclasses import fields, replace

import torch
import numpy as np
from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor, MinTokensLogitsProcessor,
)
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.eagle import EagleProposer

from .dsv4_mtp_utils import (
    advance_draft_positions, greedy_rejection_tokens, padded_draft_indices,
    resolve_next_tokens, validate_dsv4_mtp_config,
)


logger = init_logger(__name__)


def _plain_greedy_sampling(metadata):
    """Host-only guard: bypass no stateful or argmax-changing processing."""
    if (not metadata.all_greedy or metadata.all_random
            or metadata.max_num_logprobs is not None
            or metadata.logprob_token_ids
            or not metadata.no_penalties
            or metadata.allowed_token_ids_mask is not None
            or metadata.bad_words_token_ids):
        return False
    holder = metadata.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        return False
    for processor in metadata.logitsprocs.non_argmax_invariant:
        # Builtins are present even when inactive. Unknown implementations and
        # subclasses stay on the original path, including their side effects.
        if type(processor) is MinTokensLogitsProcessor and not processor.min_toks:
            continue
        if type(processor) is LogitBiasLogitsProcessor and not processor.biases:
            continue
        return False
    return True


class Dsv4MTPProposer(EagleProposer):
    def __init__(self, vllm_config, device, runner=None):
        validate_dsv4_mtp_config(vllm_config)
        super().__init__(vllm_config, device, runner)
        # The installed generic SpecDecodeBaseProposer accepts runner but
        # does not retain it (unlike AscendSpecDecodeBaseProposer).
        self.runner = runner
        self._fast_mtp_metadata = bool(
            (vllm_config.additional_config or {}).get("dsv4_mtp_fast_metadata", False))
        self._host_metadata_active = False
        self._pending_draft_counts = None
        self._prepare_steps = (os.getenv("MTP_PREPARE_STEPS", "1") == "1"
                               and self._fast_mtp_metadata)
        self._fused_inputs = os.getenv("MTP_FUSED_INPUTS", "1") == "1"
        self.use_local_argmax_reduction = os.getenv("MTP_LOCAL_ARGMAX", "1") == "1"
        flag = os.getenv("MTP_OVERLAP_SECOND_PREP", "1")
        if flag not in ("0", "1"):
            raise ValueError("MTP_OVERLAP_SECOND_PREP must be 0 or 1")
        self._overlap_second_prep = flag == "1"
        flag = os.getenv("MTP_DIRECT_DRAFT_HIDDEN", "1")
        if flag not in ("0", "1"):
            raise ValueError("MTP_DIRECT_DRAFT_HIDDEN must be 0 or 1")
        self._direct_draft_hidden = flag == "1"
        self._step_buffers = [
            (torch.empty(self.max_batch_size, dtype=torch.int64, device=device),
             torch.empty(self.max_batch_size, dtype=self._slot_mapping_buffer.dtype, device=device),
             torch.empty(self.max_batch_size, dtype=torch.int32, device=device),
             torch.empty(self.max_batch_size, dtype=torch.bool, device=device))
            for _ in range(2)
        ]

    def _get_model(self):
        # Installed vLLM copies the target's eager flag when constructing the
        # draft ModelConfig. Keep this draft-only override explicit.
        self.speculative_config.draft_model_config.enforce_eager = True
        return super()._get_model()

    def _maybe_share_embeddings(self, target_language_model):
        # DSv4 exports mtp.0.emb.tok_emb separately from target embeddings.
        pass

    def _maybe_share_lm_head(self, target_language_model):
        self._share_mtp_indices = False

    def _determine_batch_execution_and_padding(self, num_tokens, use_cudagraphs=True):
        # Keep upstream DP rank coordination even though the draft runs eager.
        return super()._determine_batch_execution_and_padding(num_tokens, False)

    def dummy_run(self, num_tokens, use_cudagraphs=True,
                  is_graph_capturing=False, slot_mappings=None):
        if is_graph_capturing:
            return
        if not self._prepare_steps:
            return super().dummy_run(num_tokens, False, False, slot_mappings)
        # Idle DP ranks must use exactly the same packed collective and K
        # forwards as real ranks. Startup/profile also use this path.
        first, follow = self._coordinate_step_sizes(num_tokens, num_tokens)
        for step in range(self.num_speculative_tokens):
            mode, count, across = first if step == 0 else follow
            if (self._draft_attn_layer_names and slot_mappings
                    and next(iter(self._draft_attn_layer_names)) in slot_mappings):
                slots = self._get_slot_mapping(count)
            else:
                slots = slot_mappings or {}
            with set_forward_context(
                None, self.vllm_config, num_tokens=count,
                num_tokens_across_dp=across, cudagraph_runtime_mode=mode,
                slot_mapping=slots):
                self.model(input_ids=self.input_ids[:count],
                           positions=self._get_positions(count),
                           hidden_states=self.hidden_states[:count],
                           inputs_embeds=None)

    def propose(self, *args, **kwargs):
        sampling = kwargs.get("sampling_metadata", args[7] if len(args) > 7 else None)
        if sampling is None or not sampling.all_greedy or sampling.max_num_logprobs is not None:
            raise NotImplementedError("DSv4 MTP requires temperature=0 and no logprobs")
        args = list(args)
        self._draft_active_mask = None
        cad = copy(kwargs["common_attn_metadata"] if "common_attn_metadata" in kwargs else args[6])
        cad.seq_lens = cad.seq_lens.clone()
        lengths = cad._seq_lens_cpu
        if lengths is None:
            lengths = cad.seq_lens_cpu
        if lengths is None or cad.positions_cpu is None:
            raise RuntimeError("DSv4 MTP requires rejection-corrected target CPU metadata")
        if lengths.device.type != "cpu" or cad.positions_cpu.device.type != "cpu":
            raise RuntimeError("DSv4 MTP host metadata must already reside on CPU")
        self._host_lengths = lengths[:cad.num_reqs].clone()
        self._host_positions = cad.positions_cpu[:cad.num_actual_tokens].clone()
        self._host_query_starts = cad.query_start_loc_cpu[:cad.num_reqs + 1].clone()
        rejected = kwargs.get("num_rejected_tokens_gpu", args[9] if len(args) > 9 else None)
        self._host_needs_rejection = rejected is not None
        self._host_sampling_positions = None
        self._host_active_mask = None
        self._host_metadata_active = True
        if "common_attn_metadata" in kwargs:
            kwargs["common_attn_metadata"] = cad
        else:
            args[6] = cad
        try:
            if self._prepare_steps:
                return self._propose_prepared(*args, **kwargs)
            return super().propose(*args, **kwargs)
        finally:
            # Dummy/profile forwards must never reuse a previous real batch.
            self._host_metadata_active = False
            self._pending_draft_counts = None

    def _coordinate_step_sizes(self, first_tokens, follow_tokens):
        """One eager DP collective carrying both forward extents.

        dummy_run uses the same layout/order so idle DP ranks participate.
        Graph and microbatch modes stay disabled for this draft-only path.
        """
        parallel = self.vllm_config.parallel_config
        if parallel.data_parallel_size == 1:
            return ((CUDAGraphMode.NONE, first_tokens, None),
                    (CUDAGraphMode.NONE, follow_tokens, None))
        from vllm.v1.worker.dp_utils import _get_device_and_group
        import torch.distributed as dist
        device, group = _get_device_and_group(parallel)
        sizes = torch.zeros((2, parallel.data_parallel_size), dtype=torch.int32)
        sizes[:, parallel.data_parallel_rank] = torch.tensor([first_tokens, follow_tokens])
        packed = sizes.to(device, non_blocking=True)
        dist.all_reduce(packed, group=group)
        sizes = packed.cpu()
        return ((CUDAGraphMode.NONE, first_tokens, sizes[0]),
                (CUDAGraphMode.NONE, follow_tokens, sizes[1]))

    def _propose_prepared(self, num_speculative_tokens, target_token_ids,
                          target_positions, target_hidden_states, next_token_ids,
                          token_indices_to_sample, common_attn_metadata,
                          sampling_metadata, mm_embed_inputs=None,
                          num_rejected_tokens_gpu=None, slot_mappings=None):
        """Prepare all K metadata snapshots before the first draft forward."""
        if self.needs_extra_input_slots or self.uses_mrope or self.supports_mm_inputs:
            raise NotImplementedError("DSv4 prepared steps require plain sequential text MTP")
        if not 1 <= num_speculative_tokens <= 3:
            raise ValueError("DSv4 prepared steps require K1..3")
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        batch = common_attn_metadata.batch_size()
        num_tokens, indices, cad = self.set_inputs_first_pass(
            target_token_ids, next_token_ids, target_positions,
            target_hidden_states, token_indices_to_sample,
            common_attn_metadata, num_rejected_tokens_gpu)
        _, first_metadata = self.build_per_group_and_layer_attn_metadata(cad)
        first_size, follow_size = self._coordinate_step_sizes(num_tokens, batch)
        first_inputs, slot_size = self.build_model_inputs_first_pass(
            num_tokens, first_size[1], mm_embed_inputs)

        # Keep K1/empty and unsupported extents on the exact old order.
        # Both branches retain the same single packed DP coordination above.
        if (self._overlap_second_prep and num_speculative_tokens > 1
                and batch > 0 and batch == cad.num_reqs
                and self._fast_mtp_metadata and self._host_metadata_active):
            return self._propose_with_second_prep_overlap(
                num_speculative_tokens, batch, indices, cad,
                first_metadata, first_size, follow_size, first_inputs,
                slot_size, num_rejected_tokens_gpu)

        plans = []
        if num_speculative_tokens > 1:
            follow = copy(cad)
            follow.num_actual_tokens = batch
            follow.num_input_tokens = batch
            follow.max_query_len = 1
            follow.query_start_loc = self.arange[:batch + 1]
            follow.query_start_loc_cpu = torch.from_numpy(self.token_arange_np[:batch + 1])
            if num_rejected_tokens_gpu is not None:
                follow.seq_lens = cad.seq_lens - num_rejected_tokens_gpu
            positions = self.positions[indices]
            active = None
            for step in range(1, num_speculative_tokens):
                # This waits on the existing accepted-count event once, before
                # any draft forward, and keeps CPU/GPU correction identical.
                self._advance_host_metadata(follow, batch, self.block_size)
                if self._fused_inputs:
                    from .dsv4_mtp_kernels import advance_positions
                    positions, slots, lengths, active = advance_positions(
                        positions, follow.block_table_tensor, follow.seq_lens,
                        self.block_size, self.max_model_len, active,
                        self._step_buffers[step - 1])
                else:
                    positions, slots, lengths = advance_draft_positions(
                        positions, follow.block_table_tensor, follow.seq_lens,
                        self.block_size, self.max_model_len, active)
                    active = slots >= 0
                follow = copy(follow)
                follow.slot_mapping = slots
                follow.seq_lens = lengths
                _, metadata = self.build_per_group_and_layer_attn_metadata(
                    follow, draft_index=step, positions=positions)
                plans.append((positions, slots, metadata))

        # Input/position/hidden buffers for step zero have not been overwritten
        # by metadata preparation. Each later step owns its metadata storage.
        mode, count, across = first_size
        with set_forward_context(
            first_metadata, self.vllm_config, num_tokens=count,
            num_tokens_across_dp=across, cudagraph_runtime_mode=mode,
            slot_mapping=self._get_slot_mapping(slot_size, cad.slot_mapping)):
            hidden = self.model(**first_inputs)
        if self.model_returns_tuple():
            raise NotImplementedError("DSv4 draft returns its single pre-HC hidden tensor")
        selected = hidden[indices]
        ids = self._greedy_sample(selected)
        output = torch.empty((batch, num_speculative_tokens), dtype=ids.dtype, device=ids.device)
        output[:, 0].copy_(ids)
        hidden = selected
        mode, count, across = follow_size
        for step, (positions, slots, metadata) in enumerate(plans, 1):
            self.input_ids[:batch].copy_(ids)
            self.positions[:batch].copy_(positions)
            self.hidden_states[:batch].copy_(hidden)
            with set_forward_context(
                metadata, self.vllm_config, num_tokens=count,
                num_tokens_across_dp=across, cudagraph_runtime_mode=mode,
                slot_mapping=self._get_slot_mapping(count, slots)):
                hidden = self.model(input_ids=self.input_ids[:count],
                                    positions=self._get_positions(count),
                                    hidden_states=self.hidden_states[:count],
                                    inputs_embeds=None)
            hidden = hidden[:batch]
            ids = self._greedy_sample(hidden)
            output[:, step].copy_(ids)
        return output

    def _propose_with_second_prep_overlap(
            self, num_speculative_tokens, batch, indices, cad,
            first_metadata, first_size, follow_size, first_inputs, slot_size,
            num_rejected_tokens_gpu):
        """Overlap follow-up host preparation with an already submitted step.

        No new stream/event/context is introduced. Follow-up device writes and
        later input/mask reuse stay ordered after first-step consumers on the
        current stream. DSA auxiliary/shared-expert streams join that stream
        before model return. Each follow-up fast metadata step already owns
        its slot/SAS/RoPE and advance-position storage.
        """
        # Submit all first-step consumers, including its head, before the
        # CPU consumes this proposal's accepted-count D2H event.
        mode, count, across = first_size
        with set_forward_context(
            first_metadata, self.vllm_config, num_tokens=count,
            num_tokens_across_dp=across, cudagraph_runtime_mode=mode,
            slot_mapping=self._get_slot_mapping(slot_size, cad.slot_mapping)):
            hidden = self.model(**first_inputs)
        if self.model_returns_tuple():
            raise NotImplementedError("DSv4 draft returns its single pre-HC hidden tensor")
        selected = hidden[indices]
        ids = self._greedy_sample(selected)
        output = torch.empty((batch, num_speculative_tokens), dtype=ids.dtype, device=ids.device)
        output[:, 0].copy_(ids)
        hidden = selected
        plans = []
        if num_speculative_tokens > 1:
            follow = copy(cad)
            follow.num_actual_tokens = batch
            follow.num_input_tokens = batch
            follow.max_query_len = 1
            follow.query_start_loc = self.arange[:batch + 1]
            follow.query_start_loc_cpu = torch.from_numpy(self.token_arange_np[:batch + 1])
            if num_rejected_tokens_gpu is not None:
                follow.seq_lens = cad.seq_lens - num_rejected_tokens_gpu
            positions = self.positions[indices]
            active = None
            for step in range(1, num_speculative_tokens):
                # Same current-generation event and correction as the old
                # path; the first model/head is now already submitted.
                self._advance_host_metadata(follow, batch, self.block_size)
                if self._fused_inputs:
                    from .dsv4_mtp_kernels import advance_positions
                    positions, slots, lengths, active = advance_positions(
                        positions, follow.block_table_tensor, follow.seq_lens,
                        self.block_size, self.max_model_len, active,
                        self._step_buffers[step - 1])
                else:
                    positions, slots, lengths = advance_draft_positions(
                        positions, follow.block_table_tensor, follow.seq_lens,
                        self.block_size, self.max_model_len, active)
                    active = slots >= 0
                follow = copy(follow)
                follow.slot_mapping = slots
                follow.seq_lens = lengths
                _, metadata = self.build_per_group_and_layer_attn_metadata(
                    follow, draft_index=step, positions=positions)
                plans.append((positions, slots, metadata))

        mode, count, across = follow_size
        for step, (positions, slots, metadata) in enumerate(plans, 1):
            self.input_ids[:batch].copy_(ids)
            self.positions[:batch].copy_(positions)
            # Only bypass the hidden copy for the exact unpadded layout.
            # Inspect metadata only: no device read, conversion or allocation.
            direct_hidden = (
                self._direct_draft_hidden and count == batch
                and hidden.ndim == 2 and self.hidden_states.ndim == 2
                and hidden.shape[0] == count
                and count <= self.hidden_states.shape[0]
                and hidden.shape[1] == self.hidden_states.shape[1]
                and hidden.dtype == self.hidden_states.dtype
                and hidden.device == self.hidden_states.device
                and hidden.is_contiguous()
                and self.hidden_states.is_contiguous()
            )
            if direct_hidden:
                draft_hidden = hidden
            else:
                # Preserve the old tail/padding contents when count != batch.
                self.hidden_states[:batch].copy_(hidden)
                draft_hidden = self.hidden_states[:count]
            with set_forward_context(
                metadata, self.vllm_config, num_tokens=count,
                num_tokens_across_dp=across, cudagraph_runtime_mode=mode,
                slot_mapping=self._get_slot_mapping(count, slots)):
                hidden = self.model(input_ids=self.input_ids[:count],
                                    positions=self._get_positions(count),
                                    hidden_states=draft_hidden,
                                    inputs_embeds=None)
            hidden = hidden[:batch]
            ids = self._greedy_sample(hidden)
            output[:, step].copy_(ids)
        return output

    def prepare_next_token_ids_padded(self, sampled_token_ids, requests,
                                      gpu_input_batch, discard_request_mask):
        batch = gpu_input_batch.num_reqs
        for i in range(batch):
            self.backup_next_token_ids.np[i] = requests[gpu_input_batch.req_ids[i]].get_token_id(
                gpu_input_batch.num_tokens_no_spec[i] - 1)
        self.backup_next_token_ids.copy_to_gpu(batch)
        # The runner passes a max_num_reqs-sized mask, while the sampled
        # token tensor contains only active requests. Slice before torch.where
        # to prevent a one-request batch broadcasting to the full capacity.
        if self._fused_inputs and sampled_token_ids.device.type in ("npu", "privateuseone"):
            from .dsv4_mtp_kernels import next_tokens
            return next_tokens(sampled_token_ids[:batch], discard_request_mask[:batch],
                               self.backup_next_token_ids.gpu[:batch], gpu_input_batch.vocab_size)
        return resolve_next_tokens(sampled_token_ids[:batch], discard_request_mask[:batch],
                                   self.backup_next_token_ids.gpu[:batch],
                                   gpu_input_batch.vocab_size)

    def prepare_inputs_padded(self, common_attn_metadata, spec_decode_metadata,
                              valid_sampled_tokens_count):
        prepare = padded_draft_indices
        if self._fused_inputs and valid_sampled_tokens_count.device.type in ("npu", "privateuseone"):
            from .dsv4_mtp_kernels import padded_indices
            prepare = padded_indices
        indices, rejected = prepare(
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count, common_attn_metadata.query_start_loc)
        # Preserve Ascend positions, phase and CPU mirrors, not just base fields.
        self._pending_draft_counts = tuple(spec_decode_metadata.num_draft_tokens)
        return copy(common_attn_metadata), indices, rejected

    def prepare_inputs(self, common_attn_metadata, sampled_token_ids, num_draft_tokens):
        base, indices = super().prepare_inputs(
            common_attn_metadata, sampled_token_ids, num_draft_tokens)
        result = copy(common_attn_metadata)
        for field in fields(base):
            setattr(result, field.name, getattr(base, field.name))
        result.positions = common_attn_metadata.positions[indices]
        result.positions_cpu = result.positions.cpu()
        return result, indices

    def _update_positions_dependent_metadata(self, positions, common_attn_metadata,
                                            batch_size, input_batch_size, block_size):
        if self._host_metadata_active:
            self._advance_host_metadata(common_attn_metadata, batch_size, block_size)
        new_pos, slots, lengths = advance_draft_positions(
            positions, common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens, block_size, self.max_model_len,
            getattr(self, "_draft_active_mask", None))
        self._draft_active_mask = slots >= 0
        self.positions[:batch_size].copy_(new_pos)
        self._slot_mapping_buffer[:input_batch_size].fill_(-1)
        self._slot_mapping_buffer[:batch_size].copy_(slots)
        common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
        # Do not mutate the target runner's lengths via an aliased tensor.
        common_attn_metadata.seq_lens = lengths
        return self.positions[:batch_size]

    def _advance_host_metadata(self, cad, batch_size, block_size):
        if self._host_sampling_positions is None:
            ends = self._host_query_starts[1:] - 1
            if self._host_needs_rejection:
                # The runner enqueued this count copy before propose(). Wait
                # once per proposal, only when K>1 actually needs a new step.
                # Reuse its event instead of copying positions/lengths per step.
                if self.runner is None or self._pending_draft_counts is None:
                    raise RuntimeError("Missing DSv4 rejection-count owner")
                counts = self.runner._get_valid_sampled_token_count()
                drafts = np.asarray(self._pending_draft_counts, dtype=np.int32)
                valid = np.asarray(counts, dtype=np.int32)
                if valid.shape != drafts.shape or valid.size != batch_size:
                    raise RuntimeError("DSv4 acceptance-count rows do not match the draft batch")
                if np.any((valid < 0) | (valid > drafts + 1)):
                    raise RuntimeError("DSv4 acceptance count exceeds the previous draft extent")
                rejected = np.where(drafts > 0, drafts + 1 - valid, 0)
                ends = ends - torch.from_numpy(rejected)
                self._host_lengths = self._host_lengths - torch.from_numpy(rejected)
            self._host_sampling_positions = self._host_positions[ends.long()]

        if self.runner is None:
            raise RuntimeError("DSv4 MTP requires the runner's CPU block table")
        table = self.runner.input_batch.block_table[self.kv_cache_gid]
        if table.block_size != block_size:
            raise RuntimeError("DSv4 host/GPU kernel block sizes differ")
        host_table = table.get_cpu_tensor()[:batch_size, :cad.block_table_tensor.shape[1]].numpy()
        pos = self._host_sampling_positions.numpy().astype(np.int64, copy=False) + 1
        valid = (pos >= 0) & (pos < self.max_model_len)
        if self._host_active_mask is not None:
            valid &= self._host_active_mask.numpy()
        cols = host_table.shape[1]
        if cols:
            column = np.minimum(np.maximum(pos, 0) // block_size, cols - 1)
            valid &= (pos // block_size < cols)
            valid &= host_table[np.arange(batch_size), column] >= 0
        else:
            valid[:] = False
        positions = np.where(valid, pos, 0)
        lengths = np.where(valid, np.minimum(self._host_lengths.numpy() + 1, self.max_model_len), 1)
        self._host_sampling_positions = torch.from_numpy(positions)
        self._host_positions = self._host_sampling_positions
        self._host_lengths = torch.from_numpy(lengths.astype(np.int32, copy=False))
        self._host_active_mask = torch.from_numpy(valid)

    def build_per_group_and_layer_attn_metadata(self, common_attn_metadata, draft_index=0,
                                               positions=None):
        cad = common_attn_metadata
        count = cad.num_actual_tokens
        cad.positions = self.positions[:count] if positions is None else positions[:count]
        if self._host_metadata_active:
            cad.positions_cpu = self._host_positions[:count]
            lengths = self._host_lengths
        else:
            # Startup dummy/profile only; no real target host snapshot exists.
            cad.positions_cpu = cad.positions.cpu()
            lengths = cad.seq_lens.cpu()
        queries = cad.query_start_loc_cpu[1:] - cad.query_start_loc_cpu[:-1]
        cad._seq_lens_cpu = cad.seq_lens_cpu = lengths
        cad._num_computed_tokens_cpu = cad.num_computed_tokens_cpu = lengths - queries
        cad.seq_lens_cpu_upper_bound = lengths.clone()
        cad.max_seq_len = int(lengths.max()) if lengths.numel() else 0
        cad.num_input_tokens = count
        cad.graph_pad_size = -1
        groups, layers = [], {}
        for group in self.draft_attn_groups:
            builder = group.get_metadata_builder()
            if (getattr(builder, "ascend_metadata_protocol", None) != "dsa_v1"
                    or getattr(builder, "compressor_ratio", 2) > 1):
                raise NotImplementedError("DSv4 MTP requires FL DSA SWA-only draft attention")
            fast_decode = (
                getattr(self, "_fast_mtp_metadata", False) and self._host_metadata_active
                and draft_index > 0 and cad.max_query_len == 1
                and cad.num_actual_tokens == cad.num_reqs and cad.num_reqs > 0
                and cad.graph_pad_size == -1)
            if fast_decode and callable(getattr(builder, "build_for_dsv4_mtp_decode", None)):
                metadata = builder.build_for_dsv4_mtp_decode(cad, draft_index)
            else:
                metadata = builder.build(
                    common_prefix_len=0, common_attn_metadata=cad,
                    num_reqs_actual=cad.num_reqs,
                    prefill_ratio_to_sas_metadata={}, decode_ratio_to_sas_metadata={},
                    common_ratio_to_sas_metadata={}, block_size=group.kv_cache_spec.block_size)
            groups.append(metadata)
            layers.update(dict.fromkeys(group.layer_names, metadata))
        return groups, layers


class Dsv4GreedyRejectionSampler(RejectionSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flag = os.getenv("MTP_FUSED_GREEDY", "1")
        if flag not in ("0", "1"):
            raise ValueError("MTP_FUSED_GREEDY must be 0 or 1")
        self._fused_greedy = flag == "1"
        flag = os.getenv("MTP_VERIFY_ARGMAX", "1")
        if flag not in ("0", "1"):
            raise ValueError("MTP_VERIFY_ARGMAX must be 0 or 1")
        self._verify_argmax = flag == "1"

    def forward(self, metadata, draft_probs, logits, sampling_metadata):
        if not sampling_metadata.all_greedy or sampling_metadata.max_num_logprobs is not None:
            raise NotImplementedError("DSv4 MTP requires temperature=0 and no logprobs")
        if (self._verify_argmax
                and logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and _plain_greedy_sampling(sampling_metadata)):
            # Argmax commutes with row selection. FP16/BF16 values are exactly
            # representable in FP32, so the original conversion cannot change
            # their ordering. Keep the same torch argmax and first-index ties.
            token_ids = logits.argmax(dim=-1)
            target_argmax = token_ids[metadata.target_logits_indices]
            bonus = token_ids[metadata.bonus_logits_indices].to(torch.int32).unsqueeze(-1)
            logger.info_once(
                "DSv4 MTP verification uses shared argmax before row selection.",
                scope="global",
            )
        else:
            bonus = self.sampler(
                logits=logits[metadata.bonus_logits_indices],
                sampling_metadata=replace(sampling_metadata, max_num_logprobs=None),
                predict_bonus_token=True).sampled_token_ids
            target = self.apply_logits_processors(
                logits[metadata.target_logits_indices].float(), sampling_metadata, metadata)
            target_argmax = target.argmax(-1)
        if self._fused_greedy and logits.device.type in ("npu", "privateuseone"):
            from vllm.triton_utils import HAS_TRITON
            if not HAS_TRITON:
                raise RuntimeError("MTP_FUSED_GREEDY=1 requires Ascend Triton; use 0 for the previous path")
            from .dsv4_mtp_greedy import fused_greedy_rejection_tokens
            tokens = fused_greedy_rejection_tokens(metadata, target_argmax, bonus)
        else:
            tokens = greedy_rejection_tokens(
                metadata.draft_token_ids, metadata.num_draft_tokens,
                target_argmax, bonus, metadata.max_spec_len)
        return SamplerOutput(sampled_token_ids=tokens, logprobs_tensors=None)
