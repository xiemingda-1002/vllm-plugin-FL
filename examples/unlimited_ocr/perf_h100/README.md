# Comparable synthetic OCR performance

Start the service using [the example instructions](../README.md). Install EvalScope 1.14.1 and its dependencies into the active Python environment. This script neither installs dependencies nor starts a server.

```bash
MODEL=/path/to/Unlimited-OCR bash examples/unlimited_ocr/perf_h100/run_perf.sh 1024
MODEL=/path/to/Unlimited-OCR bash examples/unlimited_ocr/perf_h100/run_perf.sh 4096
MODEL=/path/to/Unlimited-OCR bash examples/unlimited_ocr/perf_h100/run_perf.sh 16384
```

`BASE_URL` defaults to `http://127.0.0.1:${PORT:-10000}` and excludes `/v1`. `MODEL` must be a local model directory for native tokenizer export. `PERF_OUT` can select a new output directory. `ROUNDS=1` runs a smoke test; the default is three rounds.

Each case uses random_vl, one 2048¡Á2048 RGB image, concurrency 64, 1024 requests, seed 42 and exactly 1024 output tokens. Input text lengths exclude server BOS/template/image tokens. The client identity template avoids adding text tokens. Ngram and Gundam settings match the accuracy client. This is not an exact reproduction of an H100 paper workload.

Each round must return 1024 successful requests, no failures and average output length 1024. For three rounds, report rounds 2/3 mean output throughput only if `abs(r2-r3)/mean < 1%`; otherwise retain the results and repeat. Keep TTFT, TPOT and all latency distributions in raw EvalScope reports alongside throughput. Smoke runs have no accepted three-round mean. Full OmniDocBench accuracy remains a separate required check.

Outputs contain environment metadata, tokenizer export, per-round logs and raw EvalScope reports. Three-round runs also produce `summary.json`. Preparation and each round check `/v1/models` without inference; readiness does not establish scheduler, graph, TP or optimization activation. Set `OCR_SERVER_CONFIG` to the verified launch configuration to record operator-provided evidence.

Environment metadata records FL commit, dirty status, diff hash, source hashes, tokenizer hash and EvalScope version. Disk hashes do not prove that the service loaded those files. `FL_ROOT` defaults to this repository; `FL_CANDIDATE_MANIFEST` optionally records an explicit manifest. Generated caches and results are excluded from source hashes. No historical host or candidate paths are assumed.
