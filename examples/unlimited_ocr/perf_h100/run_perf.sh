#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT:-10000}}"
BASE_URL="${BASE_URL%/}"
INPUT_LEN="${1:-4096}"
[[ "$INPUT_LEN" =~ ^(1024|4096|16384)$ ]] || { echo 'Input length must be 1024, 4096 or 16384.' >&2; exit 1; }
ROUNDS="${ROUNDS:-3}"
[[ "$ROUNDS" =~ ^(1|3)$ ]] || { echo "ROUNDS must be 1 (smoke) or 3 (measurement)." >&2; exit 1; }
OUT="${PERF_OUT:-results/${INPUT_LEN}-$(date +%Y%m%d-%H%M%S)-$$}"
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
export TOKENIZER_DIR="$OUT/tokenizer_export"
python3 prepare.py --output "$OUT/environment.json"
for (( ROUND=1; ROUND<=ROUNDS; ROUND++ )); do
  python3 prepare.py --check-server
  evalscope perf \
    --model Unlimited-OCR \
    --url "${BASE_URL:-http://127.0.0.1:${PORT:-10000}}/v1/chat/completions" \
    --api openai --api-key EMPTY \
    --dataset random_vl --tokenizer-path "$TOKENIZER_DIR" \
    --prefix-length 0 --apply-chat-template \
    --min-prompt-length "$INPUT_LEN" --max-prompt-length "$INPUT_LEN" \
    --image-width 2048 --image-height 2048 --image-format RGB --image-num 1 \
    --parallel 64 --number 1024 --no-test-connection --stream \
    --seed 42 --temperature 0 --min-tokens 1024 --max-tokens 1024 \
    --extra-args '{"ignore_eos":true,"stop":[],"stop_token_ids":[],"skip_special_tokens":false,"vllm_xargs":{"ngram_size":35,"window_size":128},"mm_processor_kwargs":{"image_size":640,"base_size":1024,"crop_mode":true}}' \
    --total-timeout 1800 --outputs-dir "$OUT/round$ROUND" \
    2>&1 | tee "$OUT/round$ROUND.log"
  python3 summarize.py "$OUT" --check-round "$ROUND"
done
if (( ROUNDS == 3 )); then python3 summarize.py "$OUT"; fi
