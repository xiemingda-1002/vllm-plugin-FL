# R-SWA performance settings

The launch script defaults to a 2048 MiB final int8 mask limit and grouped mixed prefill/decode (`RSWA_MASK_MAX_MIB=2048 RSWA_GROUPED_MASK=1`). These settings accompany 64 sequences and a 1024-token batch budget.

The limit controls the final mask, not total peak device memory. Large masks are filled in chunks to reduce intermediates. Grouping applies only to eligible mixed batches without speculative decoding or graph/request/token padding; decode must precede prefill. Other batches use the fallback. FULL_DECODE_ONLY decode remains available.

The launcher checks `max_num_seqs <= max_num_batched_tokens` and `max_num_seqs * max_num_batched_tokens <= mask_MiB * 32` against the ungrouped fallback. TP2 does not divide each rank's mask allocation. Full KV is retained by the compatibility manager; memory grows with sequence length.

`FL R-SWA grouped mask active` indicates a qualifying batch actually used grouping. Configuration alone is not activation evidence. Set `RSWA_GROUPED_MASK=0` to compare, keeping the batch shape within its mask budget. Preserve full accuracy coverage and use the [performance procedure](perf_h100/README.md) for comparable measurements. MTP remains disabled.
