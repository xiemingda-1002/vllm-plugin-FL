# Common attention metadata graphs

The producer writes slot mappings, clears padded block-table rows, and updates
GPU computed-token counts using persistent input and output buffers. Capture
warms the kernel outside the graph and replays the new graph before returning,
so attention builders always consume current metadata, including with zero
model warmups. Dummy model forwards still use PAD_SLOT_ID for every token.

FULL and PIECEWISE model modes both capture this independent metadata graph.
PIECEWISE uses the runner's maximum request extent because its model graph
shapes describe tokens, not request counts. Both real and dummy steps initialize
the complete query-start and sequence-length tails. Changing actual request
counts or lengths therefore reuses the same metadata graph without capturing
on the request path. FULL retains request-count keys. Microbatching and model
eager mode use eager metadata generation. Profiling cleanup drops cached graphs.

`VLLM_FL_COMMON_ATTENTION_METADATA=0` restores the original per-group
`BlockTable.compute_slot_mapping` path, builder padding, and computed-token cache
behavior. NVIDIA enables the new producer by default. Other platforms retain
the original producer until validated; `VLLM_FL_COMMON_ATTENTION_METADATA=1`
opts into the new pointer-table Triton kernel for validation. Graph support is
checked separately through the platform graph API. An unavailable graph uses
the new producer eagerly only when that producer is enabled.

The new kernel uses uint64 device pointer tables. Exposing a graph API alone
does not establish support for that kernel on a different compiler or device.
MUSA PIECEWISE and other vendor hardware require their own device validation;
CUDA tests cannot establish their numerical or runtime compatibility.

The producer's functional GPU tests follow the same enable policy. On other
vendors, explicitly set `VLLM_FL_COMMON_ATTENTION_METADATA=1` to run those
tests while validating the new kernel; the default suite retains the original
producer's platform contract.

## Ascend attention phases

The Ascend runner determines the attention phase from the current batch's
computed, scheduled, and valid token counts, following vLLM-Ascend 0.24.0rc1.
`AscendCommonAttentionMetadata` carries that phase to the attention builders.
Other vendors keep their existing metadata contract.

A pure prefill batch can be either an initial prompt or a continuation with
cached history. The ordinary Ascend builder must preserve the runner's phase
instead of inferring `PrefillNoCache` from the absence of decode requests.
Initial prompts, including a one-token prompt, use the current K/V tensors;
chunked continuations and prefix-cache hits read the paged KV cache with
cumulative sequence lengths. Disabling prefix caching does not remove the
need to read earlier chunks of the same request.

Capture, warmup, and idle dummy forwards supply an explicit phase so they never
inherit a previous real batch's state. The upstream microbatch slicer drops
vendor fields and can split a request across token ranges. FL therefore restores
the Ascend phase from each real slice's query and computed lengths, using the
same classifier; dummy slices retain their explicit dummy phase. This adapter
does not establish new speculative-decoding or distributed execution support.

The focused regression suite is:

```bash
pytest -q \
  tests/unit_tests/dispatch/test_ascend_prefill_state_lifecycle.py \
  tests/unit_tests/dispatch/test_ascend_attention_builder_contract.py \
  tests/unit_tests/ascend/test_dsa_attention_state_lifecycle.py \
  tests/unit_tests/worker/test_glm52_sfa_metadata_lifecycle.py
```
