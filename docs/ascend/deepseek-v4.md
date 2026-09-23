# DeepSeek V4 Flash W8A8

This migration uses vLLM 0.24.0 and matching vLLM-Ascend 0.24.0rc1 source. FL
owns compressed DSA attention/cache lifecycle, ModelSlim W8A8 operators, hash
routing, communication and graph integration. Runtime installation does not
require the vllm-ascend distribution or an importable upstream checkout.

Build separately in the matching hardware image with verified `SOC_VERSION`:

```bash
VLLM_VENDOR=ascend pip install --no-build-isolation .
```

## Latest A3 integrated configuration

The accepted A3 run uses TP4, DP4, effective EP16 across 16 logical devices.
Its wheel was built for `ascend910_9391`; do not reuse it on A2. Verify standalone
installation from neutral `/run` with `PYTHONPATH` unset: a source checkout in an
image default workdir can remain importable after `pip uninstall`.

This is the accepted service setting (diagnostic profiler switches omitted).
`MODEL` points to ModelSlim W8A8 weights.

```bash
export VLLM_PLUGINS=fl USE_FLAGGEMS=0
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV
vllm serve "$MODEL" \
  --tensor-parallel-size 4 --data-parallel-size 4 --enable-expert-parallel \
  --max-model-len 1048576 --max-num-batched-tokens 10240 --max-num-seqs 64 \
  --block-size 32 --gpu-memory-utilization 0.9 --quantization ascend \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --api-server-count 1 \
  --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16,32,64]}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --additional-config '{"enable_flashcomm1":true,"enable_dsa_cp":true,"enable_shared_expert_dp":true,"enable_fused_mc2":1,"enable_cpu_binding":true,"multistream_overlap_shared_expert":false}'
```

Validation used host networking, privileged device access, 512G shared memory,
unlimited memlock, and image jemalloc at `/usr/lib/aarch64-linux-gnu/libjemalloc.so.2`
through `LD_PRELOAD`; verify that library before reuse. Map the target driver/device
files and reserve `ASCEND_RT_VISIBLE_DEVICES`. Debug logging/profiler collection
were evidence controls, not performance settings. Do not combine tested multithread
loading with safetensors prefetch. FlashComm1 is `additional_config`, not an old
environment switch; DSA context parallelism is separate. MTP/speculative decoding
is not migrated or enabled.

Fused MC2 is distinct from ordinary MC2. Support is A3 W8A8 with EP≤32; it does
not establish BF16/W4A8 fused support, EPLB, hierarchy or mixed placement.
Shared-expert multistream overlap is disabled for fused execution. Ordinary MC2
with overlap is a separate later performance candidate, not an interchangeable
setting.

## Current acceptance

- Fifteen strict semantic cases, including a 65,536-input-token case, and a
  separate concurrency-16 round passed. Main review checked 31 raw responses:
  all naturally stopped, were nonempty, had finite logprobs, and no replacement
  characters.
- All 16 worker traces contain real `DispatchFFNCombine` kernel/API. Replay
  identity links process, device and graph execution for every worker, covering
  all four DP replicas; configuration logs alone were not accepted.
- CPU-binding startup evidence covers all 16 workers. Accepted 1,048,576 model
  length is startup capacity, not a 1M-token accuracy result.
- The same A3 wheel passes bounded Qwen text graph regressions. Separately, the
  accepted A2 wheel passed four bounded Qwen text graph cases: 35B/27B each had
  four natural-stop correct “4/Blue” outputs (FULL counts 2/7), and 0.6B FG0/FG1
  each had two correct outputs (FULL1, replay18). This does not establish A2
  DeepSeek, long-context, distributed, or performance acceptance. A2 QwenVL
  separately passed a bounded image/static-video graph regression, not general
  visual/video acceptance; A3 QwenVL was not tested.
- Diagnostic traces are not performance results. Native matching-version
  comparison, throughput/latency reporting, and best-stable selection remain
  pending; no native performance parity is claimed.

Evidence: Task `fl-vllm024-ascend-migration-20260909`, Loops
`deepseek-v4-fused-mc2-a3-runtime-v2` and `integrated-qwen-a3-post-fused-v2`.

## Historical observations retained

Earlier A3 TP8 eager/graph and DP4/TP4 eager tests remain in the Task. After
compressed DSA dummy metadata correction, short concurrency 16, 16 and 32 and
roughly 4K-input concurrency-8 completed without the previous device address
error. Two long-document requests with 88,054 and 88,057 input tokens answered
project/person/location questions and summaries, followed by a passing short
concurrency-16 round.

A separate historical 61,423-token numeric retrieval request repeatedly omitted
the final two digits. Later semantic gates do not prove this limitation fixed.
These are not claims of general factual accuracy. Historical WikiText workloads
must remain distinct from the current candidate's bounded correctness acceptance.
Evidence remains in `deepseek-v4-a3-dsa-dummy-metadata-v1`,
`deepseek-v4-a3-loaded-final-gate-v1`, `deepseek-v4-a3-long-semantic-gate-v1`,
and `deepseek-v4-a3-historical-corpus-v1`; keep source, wheel, hardware and
request identity attached to each result.
