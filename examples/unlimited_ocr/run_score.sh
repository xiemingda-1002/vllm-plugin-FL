#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT="${1:?Usage: run_score.sh RESULT_DIRECTORY [score_ocr.py options]}"
shift
# CLI options after these defaults take precedence in argparse.
args=()
scope=--full
has_dataset="${OCR_DATASET:-}"
has_evaluator="${OCR_EVALUATOR:-}"
summarize=0
for arg in "$@"; do
  case "$arg" in
    --subset|--full) scope= ;;
    --dataset|--dataset=*) has_dataset=cli ;;
    --evaluator|--evaluator=*) has_evaluator=cli ;;
    --summarize-only) summarize=1 ;;
  esac
done
if (( summarize )); then scope=; fi
if (( ! summarize )) && [[ -z "$has_dataset" || -z "$has_evaluator" ]]; then
  echo "Set OCR_DATASET/OCR_EVALUATOR or pass --dataset/--evaluator." >&2; exit 1
fi
if [[ -n "$scope" ]]; then args+=("$scope"); fi
if [[ -n "${OCR_DATASET:-}" ]]; then args+=(--dataset "$OCR_DATASET"); fi
if [[ -n "${OCR_EVALUATOR:-}" ]]; then args+=(--evaluator "$OCR_EVALUATOR"); fi
exec python3 "$SCRIPT_DIR/score_ocr.py" "$RESULT" "${args[@]}" "$@"
