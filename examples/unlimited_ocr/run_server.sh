#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${OCR_WORKDIR:-$SCRIPT_DIR}"
FL_SOURCE_DIR="${FL_SOURCE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export VLLM_FL_OCR_TORCH_SLOT_MAPPING="${TORCH_SLOT_MAPPING:-1}"
[[ "$VLLM_FL_OCR_TORCH_SLOT_MAPPING" =~ ^[01]$ ]] || { echo "TORCH_SLOT_MAPPING must be 0 or 1." >&2; exit 1; }
echo "FL torch slot mapping: $VLLM_FL_OCR_TORCH_SLOT_MAPPING"
export VLLM_FL_OCR_BATCH_NGRAM="${BATCH_NGRAM:-1}"
[[ "$VLLM_FL_OCR_BATCH_NGRAM" =~ ^[01]$ ]] || { echo "BATCH_NGRAM must be 0 or 1." >&2; exit 1; }
echo "Batch NGram ban writes: $VLLM_FL_OCR_BATCH_NGRAM"
export VLLM_FL_OCR_FUSED_SAM="${FUSED_SAM:-1}"
[[ "$VLLM_FL_OCR_FUSED_SAM" =~ ^[01]$ ]] || { echo "FUSED_SAM must be 0 or 1." >&2; exit 1; }
echo "SAM fused attention: $VLLM_FL_OCR_FUSED_SAM"
export VLLM_FL_OCR_FUSED_ROPE="${FUSED_ROPE:-1}"
[[ "$VLLM_FL_OCR_FUSED_ROPE" =~ ^[01]$ ]] || { echo "FUSED_ROPE must be 0 or 1." >&2; exit 1; }
echo "FL source: $FL_SOURCE_DIR; fused RoPE: $VLLM_FL_OCR_FUSED_ROPE"
set +u
if [[ -f "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}" ]]; then
  # shellcheck source=/dev/null
  source "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
fi
set -u
export VLLM_PLUGINS=fl VLLM_FL_VENDOR=ascend VLLM_VENDOR=ascend USE_FLAGGEMS=0
export VLLM_FL_OCR_TORCH_NPU=1 VLLM_FL_PREFER_ENABLED=1
export VLLM_FL_OCR_AUDIT_DIR="${AUDIT_DIR:-$PWD/results/runtime-audit-${PORT:-10000}}"
export VLLM_FL_PREFER=vendor VLLM_FL_STRICT=1 VLLM_FL_ALLOW_VENDORS=ascend
# An inherited OOT filter must not bypass the strict OCR implementations.
unset VLLM_FL_OOT_BLACKLIST VLLM_FL_OOT_WHITELIST
export PYTHONPATH="${FL_SOURCE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES:-0,1}"
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
TP_SIZE="${TP_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
RSWA_MASK_MAX_MIB="${RSWA_MASK_MAX_MIB:-2048}"
RSWA_GROUPED_MASK="${RSWA_GROUPED_MASK:-1}"
if [[ ! "$RSWA_MASK_MAX_MIB" =~ ^[1-9][0-9]{0,3}$ ]] || (( RSWA_MASK_MAX_MIB > 8192 )); then
  echo 'RSWA_MASK_MAX_MIB must be 1..8192.' >&2; exit 1
fi
[[ "$RSWA_GROUPED_MASK" =~ ^[01]$ ]] || { echo 'RSWA_GROUPED_MASK must be 0 or 1.' >&2; exit 1; }
export VLLM_FL_RSWA_MASK_MAX_MIB="$RSWA_MASK_MAX_MIB"
export VLLM_FL_RSWA_GROUPED_MASK="$RSWA_GROUPED_MASK"
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'DEVICES must be comma-separated IDs.' >&2; exit 1; }
[[ "$TP_SIZE" =~ ^[12]$ ]] || { echo 'TP_SIZE must be 1 or 2.' >&2; exit 1; }
IFS=, read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
(( ${#devices[@]} >= TP_SIZE )) || { echo 'Not enough devices for TP_SIZE.' >&2; exit 1; }
if (( TP_SIZE == 2 )) && [[ "${devices[0]}" == "${devices[1]}" ]]; then echo 'TP2 needs different devices.' >&2; exit 1; fi
for value in "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS"; do
  [[ "$value" =~ ^[1-9][0-9]{0,4}$ ]] || { echo 'Batch settings must be positive integers <= 99999.' >&2; exit 1; }
done
# Fixed K=32768 and int8 masks: one MiB accommodates 32 batch-query rows.
# Keep the fallback safe even when grouping cannot apply (e.g. graph capture).
(( MAX_NUM_SEQS <= MAX_NUM_BATCHED_TOKENS && MAX_NUM_SEQS * MAX_NUM_BATCHED_TOKENS <= RSWA_MASK_MAX_MIB * 32 )) || { echo "R-SWA requires seqs <= token budget and seqs * budget <= $((RSWA_MASK_MAX_MIB * 32))." >&2; exit 1; }
MODEL="${MODEL:?Set MODEL to the local Unlimited-OCR model directory or its Hugging Face model ID}"
[[ -f "$SCRIPT_DIR/chat_template_bos.jinja" ]] || { echo 'BOS template missing.' >&2; exit 1; }
GRAPH_MODE="${GRAPH_MODE:-full}"
case "$GRAPH_MODE" in
  eager) graph_args=(--enforce-eager --compilation-config '{"custom_ops":["+rms_norm","+silu_and_mul","+rotary_embedding"]}') ;;
  full)
    export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    [[ "${ASCEND_LAUNCH_BLOCKING:-0}" != 1 ]] || { echo 'FULL graphs require ASCEND_LAUNCH_BLOCKING=0.' >&2; exit 1; }
    graph_config="$(python3 -c 'import json,sys; n=int(sys.argv[1]); sizes=sorted({1,n,*[2**k for k in range(n.bit_length()) if 2**k <= n]}); print(json.dumps(dict(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=sizes,custom_ops=["+rms_norm","+silu_and_mul","+rotary_embedding"])))' "$MAX_NUM_SEQS")"
    graph_args=(--compilation-config "$graph_config"
      --additional-config '{"graph_replay_log":true,"ascend_compilation_config":{"fuse_norm_quant":false,"fuse_qknorm_rope":false,"fuse_muls_add":false}}') ;;
  *) echo 'GRAPH_MODE must be eager or full (FULL_DECODE_ONLY).' >&2; exit 1 ;;
esac
echo "FL Unlimited-OCR: devices=$ASCEND_RT_VISIBLE_DEVICES tp=$TP_SIZE port=${PORT:-10000} seqs=$MAX_NUM_SEQS tokens=$MAX_NUM_BATCHED_TOKENS"
echo "R-SWA mask limit: ${RSWA_MASK_MAX_MIB} MiB; grouped prefill/decode: $RSWA_GROUPED_MASK"
echo "Graph mode: $GRAPH_MODE${graph_config:+ $graph_config}"
exec vllm serve "$MODEL" --served-model-name Unlimited-OCR \
  --chat-template "$SCRIPT_DIR/chat_template_bos.jinja" --hf-overrides '{"model_type":"unlimited-ocr"}' \
  --host 0.0.0.0 --port "${PORT:-10000}" --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size "$TP_SIZE" --disable-custom-all-reduce --max-model-len 32768 \
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --no-async-scheduling --enable-chunked-prefill --gpu-memory-utilization 0.80 "${graph_args[@]}" \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"image_size":640,"base_size":1024,"crop_mode":true}' \
  --logits-processors vllm_fl.models.ocr_ngram:NGramPerReqLogitsProcessorFL
