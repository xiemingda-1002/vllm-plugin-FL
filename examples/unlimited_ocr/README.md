# Unlimited-OCR on FL Ascend

This example serves Unlimited-OCR through FL and torch-npu. Apply the [vLLM core patch](../../patches/unlimited_ocr/README.md) first, then install this repository using the normal Ascend build instructions. The adaptation targets vLLM 0.24 (`ee0da84ab9e04ac7610e28580af62c365e898389`). Use compatible Torch, torch-npu and CANN versions; an existing compiled extension can only be reused with matching ABI.

From the repository root:

```bash
MODEL=/path/to/Unlimited-OCR bash examples/unlimited_ocr/run_server.sh
```

`MODEL` is required: use your model directory or the model's Hugging Face ID. Performance tokenizer export requires a local directory. The script uses the installed vLLM and puts this repository first on `PYTHONPATH`; `FL_SOURCE_DIR` can override that root. CANN setup is sourced only when `${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}` exists.

Defaults: `DEVICES=0,1`, `TP_SIZE=2`, `PORT=10000`, `MAX_NUM_SEQS=64`, `MAX_NUM_BATCHED_TOKENS=1024`, `GRAPH_MODE=full`. The server binds to `0.0.0.0`. FULL_DECODE_ONLY graphs cover decode; vision and prefill remain outside FULL graphs. `GRAPH_MODE=eager` is available for comparison. Stop the existing service before reusing its devices or port.

All performance paths are enabled by default: `FUSED_SAM=1`, `FUSED_ROPE=1`, `BATCH_NGRAM=1`, `TORCH_SLOT_MAPPING=1`, `RSWA_GROUPED_MASK=1`, `RSWA_MASK_MAX_MIB=2048`. Each switch can be overridden. MTP/speculative decoding is not enabled. The single-BOS template, BF16, full-attention CLIP, R-SWA window, native tokenizer and Gundam image processing remain unchanged. See [R-SWA details](PERFORMANCE_RSWA.md).

Run requests from another terminal:

```bash
python3 examples/unlimited_ocr/test_ocr.py /path/to/sample.png --base http://127.0.0.1:10000 --image-mode gundam
bash examples/unlimited_ocr/run_accuracy.sh /path/to/omnidocbench_v1_5/images
bash examples/unlimited_ocr/run_benchmark.sh /path/to/omnidocbench_v1_5/images
```

Both wrappers accept `IMAGES` instead of a positional directory, plus `PORT`, `BASE_URL`, `CONCURRENCY` (default 64), `MAX_TOKENS` (8192), and additional Python CLI options. Accuracy uses every image by default; `--samples 10` is a smoke check. Benchmark uses every image by default; `--limit 100` limits a quick run. Clients send ngram size 35/window 128 and Gundam processing (`image_size=640`, `base_size=1024`, `crop_mode=true`). The server registers FL's ngram processor.

Use the official OmniDocBench v1.5 evaluator and its dependencies in the active Python environment:

```bash
bash examples/unlimited_ocr/run_score.sh /path/to/results/RUN \
  --dataset /path/to/OmniDocBench.json --evaluator /path/to/evaluator_v1_5
```

`OCR_DATASET` and `OCR_EVALUATOR` can supply these paths. Full evaluation requires all 1355 pages; `--subset` explicitly selects partial scoring. Inspect failed requests, empty output, truncation and coverage alongside official scores. A smoke test does not establish full accuracy.

For synthetic throughput, TTFT and TPOT, follow [the performance procedure](perf_h100/README.md). Keep raw EvalScope records and latency metrics with throughput and full accuracy results. Source installation, requested configuration and a readiness response alone do not establish device correctness or actual optimization activation. `audit_runtime.py` and device checks are optional diagnostics.
