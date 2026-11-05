# Ascend FP32 review validation

These standalone checks require the matching Ascend/vLLM 0.24 environment. They run on the visible NPU and do not download model weights or install dependencies.

## Attention components and graph lifetime

From the FL repository root:

```bash
export FL_SOURCE_DIR="$PWD"
export EVIDENCE_DIR=/tmp/fl-fp32-review
mkdir -p "$EVIDENCE_DIR"
ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH="$FL_SOURCE_DIR:$PYTHONPATH" \
  VLLM_VENDOR=ascend VLLM_FL_VENDOR=ascend VLLM_PLUGINS=fl USE_FLAGGEMS=0 \
  python tools/validation/check_fp32_graph.py
```

The check retains `fp32_boundaries.json`. It uses CPU sequence-length metadata, as the real model does, while checking that captured metadata buffers reside with the NPU KV cache. It covers 4096/8192 contexts, a non-multiple context length (4097), block sizes 64/128, outer BF16/FP16 autocast, actual graph replay after changing Q/K/V and metadata, cache replacement, changing an existing owner's max length/table geometry, and `GraphWrapper.clear_graphs()`. Unsupported TP/PCP/DCP, noncausal and cross-attention configurations must raise before either eager or FULL dispatch.

The numerical comparison uses tolerance `atol=rtol=2e-5`; HF32 is explicitly disabled in this component check. Device metadata buffers are replaced after invalidating graph entries; they are never resized in place while a captured graph can reference them.

## CosyVoice3 LLM eager/FULL equivalence

Prepare the original FP32 speech-LLM export using the external [CosyVoice integration](https://github.com/fanacio/CosyVoice/tree/vllm_fl_ascend). This check covers the LLM only, not frontend or Flow/HiFT execution.

```bash
export MODEL_DIR=/models/exported_cosyvoice3_llm_fp32
for mode in eager full; do
  ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH="$FL_SOURCE_DIR:$PYTHONPATH" \
    VLLM_VENDOR=ascend VLLM_FL_VENDOR=ascend VLLM_PLUGINS=fl USE_FLAGGEMS=0 \
    VLLM_FL_ASCEND_SLOT_MAPPING=torch \
    python tools/validation/check_cosyvoice_llm.py \
      --model "$MODEL_DIR" --execution "$mode" --output "$EVIDENCE_DIR/llm-$mode.json"
done
```

Both runs use the same CPU random seed and prompt embeddings (lengths 4 and 9), greedy generation of 16 speech tokens, top logprobs, and two repetitions. Compare token IDs and logprobs in the retained JSON, and inspect the actual replay log (`FL graph replay completed`) alongside the output comparison. The optional worker diagnostics count successful `NPUGraph.replay()` calls; the output comparison is required, rather than treating a counter alone as correctness evidence. HF32 policy is recorded separately in each diagnostic report.

The legacy `vllm_fl.dispatch.backends.vendor.ascend.impl.fp32_diagnostics.FP32Diagnostics` import remains a compatibility shim. Diagnostic counters and opt-in replay instrumentation now live in `vllm_fl.validation.fp32_diagnostics`; normal numerical kernels do not maintain global debug counters.

See `docs/ASCEND_FP32_REVIEW_20261105.md` for model regression scope, results, and retained evidence. The 2026-11-03 validation remains historical evidence; it does not replace the checks on this revision.
