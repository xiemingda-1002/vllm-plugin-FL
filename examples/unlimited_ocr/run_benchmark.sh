#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGES="${IMAGES:-${1:?Pass the image directory as the first argument or set IMAGES}}"
if [[ "${1:-}" == "$IMAGES" ]]; then shift; fi
[[ -d "$IMAGES" ]] || { echo "Missing dataset: $IMAGES" >&2; exit 1; }
exec python3 "$SCRIPT_DIR/bench_ocr_gundam.py" "$IMAGES" --base "${BASE_URL:-http://127.0.0.1:${PORT:-10000}}" \
  --concurrency "${CONCURRENCY:-64}" --max-tokens "${MAX_TOKENS:-8192}" --warmup "${WARMUP:-0}" "$@"
