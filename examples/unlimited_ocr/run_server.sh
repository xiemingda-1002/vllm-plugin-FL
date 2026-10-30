#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${OCR_WORKDIR:-$SCRIPT_DIR}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-/vllm-workspace/vllm}"
FL_SOURCE_DIR="${FL_SOURCE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export VLLM_PLUGINS=fl VLLM_FL_VENDOR=ascend VLLM_VENDOR=ascend USE_FLAGGEMS=0
export PYTHONPATH="${VLLM_SOURCE_DIR}:${FL_SOURCE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES:-2}"
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
TP_SIZE="${TP_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'DEVICES must be comma-separated IDs.' >&2; exit 1; }
[[ "$TP_SIZE" =~ ^[12]$ ]] || { echo 'TP_SIZE must be 1 or 2.' >&2; exit 1; }
IFS=, read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
(( ${#devices[@]} >= TP_SIZE )) || { echo 'Not enough devices for TP_SIZE.' >&2; exit 1; }
if (( TP_SIZE == 2 )) && [[ "${devices[0]}" == "${devices[1]}" ]]; then echo 'TP2 needs different devices.' >&2; exit 1; fi
for value in "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS"; do
  [[ "$value" =~ ^[1-9][0-9]{0,4}$ ]] || { echo 'Batch settings must be positive integers <= 99999.' >&2; exit 1; }
done
(( MAX_NUM_SEQS <= MAX_NUM_BATCHED_TOKENS && MAX_NUM_SEQS * MAX_NUM_BATCHED_TOKENS <= 16384 )) || { echo 'R-SWA requires seqs <= token budget and seqs * budget <= 16384.' >&2; exit 1; }
MODEL="${MODEL:-/public-flash/models/q4_models/Unlimited-OCR}"
[[ -f "$MODEL/config.json" && -f chat_template_bos.jinja ]] || { echo 'Model config or BOS template missing.' >&2; exit 1; }
GRAPH_MODE="${GRAPH_MODE:-eager}"
case "$GRAPH_MODE" in
  eager) graph_args=(--enforce-eager) ;;
  full)
    export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    [[ "${ASCEND_LAUNCH_BLOCKING:-0}" != 1 ]] || { echo 'FULL graphs require ASCEND_LAUNCH_BLOCKING=0.' >&2; exit 1; }
    graph_config="$(python3 -c 'import json,sys; n=int(sys.argv[1]); sizes=sorted({1,n,*[2**k for k in range(n.bit_length()) if 2**k <= n]}); print(json.dumps(dict(cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=sizes)))' "$MAX_NUM_SEQS")"
    graph_args=(--compilation-config "$graph_config"
      --additional-config '{"graph_replay_log":true,"ascend_compilation_config":{"fuse_norm_quant":false,"fuse_qknorm_rope":false,"fuse_muls_add":false}}') ;;
  *) echo 'GRAPH_MODE must be eager or full (FULL_DECODE_ONLY).' >&2; exit 1 ;;
esac
echo "FL Unlimited-OCR: devices=$ASCEND_RT_VISIBLE_DEVICES tp=$TP_SIZE port=${PORT:-10002} seqs=$MAX_NUM_SEQS tokens=$MAX_NUM_BATCHED_TOKENS"
echo "Graph mode: $GRAPH_MODE${graph_config:+ $graph_config}"
exec vllm serve "$MODEL" --served-model-name Unlimited-OCR \
  --chat-template chat_template_bos.jinja --hf-overrides '{"model_type":"unlimited-ocr"}' \
  --host 0.0.0.0 --port "${PORT:-10002}" --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size "$TP_SIZE" --max-model-len 32768 \
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --enable-chunked-prefill --gpu-memory-utilization 0.80 "${graph_args[@]}" \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"image_size":640,"base_size":1024,"crop_mode":true}' \
  --logits-processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor
