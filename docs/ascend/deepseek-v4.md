# DeepSeek V4 Flash W8A8 checkpoint

This migration targets vLLM 0.24.0 and uses the matching vLLM-Ascend
0.24.0rc1 implementation as its source baseline. FL owns the compressed DSA
attention/cache lifecycle, ModelSlim W8A8 operators, hash routing, and ordinary
MoE communication paths. The installed runtime does not require the
vllm-ascend distribution.

Build the native extension in the matching Ascend image with the correct
`SOC_VERSION` for the target hardware:

```bash
VLLM_VENDOR=ascend pip install --no-build-isolation .
```

The A3 validation uses `VLLM_PLUGINS=fl` and `USE_FLAGGEMS=0`; FL detects the
Ascend runtime automatically. Its topology is TP4, DP4, effective EP16. The tested graph
configuration is `FULL_DECODE_ONLY`, capture sizes `[1, 2, 4]`, async scheduling,
block size 128, maximum model length 133120, maximum batched tokens 8192, and
maximum sequences 32. Prefix caching, FlashComm1, fused MC2, and MTP are disabled.
Fused MC2 is distinct from the ordinary MC2 communication path.

For model loading, the tested configuration is:

```bash
--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}'
```

Do not combine this setting with `--safetensors-load-strategy prefetch`.

## Validation and remaining limits

- A3 TP8 eager and graph request evidence, plus DP4/TP4 eager request evidence,
  are retained in the migration Task.
- After correcting compressed DSA dummy actual/padded request counts, short
  concurrency rounds of 16, 16, and 32 and a roughly 4K-input concurrency-8
  round completed without the previously observed device address error.
- Two long-document requests with 88,054 and 88,057 input tokens correctly
  answered project/person/location questions and summaries. A subsequent
  concurrency-16 short round passed. Response logprobs were finite.
- A separate 61,423-token numeric retrieval request consistently omitted the
  final two digits. This remains a known observation; the long-document tests
  do not establish that numeric retrieval is repaired.
- The complete historical WikiText workload (64 requests each at 1K, 4K, 16K,
  and 64K input, 1024 output tokens) is still under validation at this
  checkpoint. Full per-DP graph replay evidence and current-version native
  performance parity are not claimed by this checkpoint.

Evidence is recorded in Task `fl-vllm024-ascend-migration-20260909`, particularly
Loops `deepseek-v4-a3-dsa-dummy-metadata-v1`,
`deepseek-v4-a3-loaded-final-gate-v1`,
`deepseek-v4-a3-long-semantic-gate-v1`, and
`deepseek-v4-a3-historical-corpus-v1`. A2 and A3 conclusions remain separate.
