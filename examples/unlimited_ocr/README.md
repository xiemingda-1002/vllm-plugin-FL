# Unlimited-OCR on FL Ascend

## 安装 vLLM core 补丁

准备支持 Ascend 的 vLLM、FL 和 CANN 运行环境，使用 vLLM 0.24 的源码提交 `ee0da84ab9e04ac7610e28580af62c365e898389`。历史适配环境使用 Python 3.12.13、PyTorch 2.10.0+cpu、torch-npu 2.10.0.post2、Transformers 5.13.0、torchvision 0.25.0+cpu 和 CANN/ATB 9.0.1。

以下命令均从 FL 仓库根目录在 Bash 中执行。将 `/path/to/vllm` 换成独立的 vLLM 源码目录。补丁工具按 [manifest](../../patches/unlimited_ocr/manifest.json) 校验提交、补丁校验和及 18 个文件的 SHA256。

```bash
python3 patches/unlimited_ocr/apply_vllm_patch.py /path/to/vllm --check
python3 patches/unlimited_ocr/apply_vllm_patch.py /path/to/vllm
```

让运行环境从应用了补丁的源码目录加载 vLLM，例如使用该目录的 editable 安装。FL 也应安装到同一 Python 环境，使 `vllm serve` 可以发现 FL 插件。复用已构建的 FL 扩展时，确认 Python、PyTorch、torch-npu、CANN 和扩展 ABI 匹配，然后从 FL 仓库根目录更新 editable 安装。新环境按仓库的 Ascend 构建流程安装原生扩展。

```bash
VLLM_VENDOR= python3 -m pip install --no-deps --no-build-isolation -e .
```

如果源码已应用其他版本的 OCR 补丁，先停止使用该源码的服务，再用旧版补丁工具和 manifest 执行 `--reverse`，然后检查和应用当前版本。撤销当前补丁时，停止使用该源码的服务，再执行以下命令。

```bash
python3 patches/unlimited_ocr/apply_vllm_patch.py /path/to/vllm --reverse
```

## 启动服务

复制本节的 `cat` 代码块，在 FL 仓库根目录生成启动脚本。后续各节可按需用同样方式生成对应脚本。脚本通过 `BASH_SOURCE` 定位模板、Python 客户端和仓库根目录，生成时请保持代码块中的目标路径。

设置 `MODEL` 为本地 Unlimited-OCR 模型目录或 Hugging Face 模型 ID。服务默认监听 `0.0.0.0:10000`，使用 Ascend 设备 `0,1`、TP2、BF16、64 个并发序列和 1024 个批处理 token。默认图模式为 `full`（`FULL_DECODE_ONLY`），视觉编码和 Prefill 按非 FULL 路径执行。服务使用 V1 runner 和 chunked prefill，关闭异步调度、prefix caching 和多模态处理器缓存，使用单 BOS 模板和 FL NGram logits processor。示例以 MTP 关闭运行。

| 变量 | 默认值和约束 |
| --- | --- |
| `MODEL` | 必填 |
| `PORT` | `10000` |
| `DEVICES`、`TP_SIZE` | `0,1` 和 `2`。TP 支持 1 或 2，TP2 需要两个不同的设备 ID |
| `MAX_NUM_SEQS`、`MAX_NUM_BATCHED_TOKENS` | `64` 和 `1024`。序列数须不超过 token 预算，两者乘积须不超过 `RSWA_MASK_MAX_MIB × 32` |
| `GRAPH_MODE` | `full`。设为 `eager` 切换到 Eager。FULL 模式要求 `ASCEND_LAUNCH_BLOCKING` 不为 `1` |
| `RSWA_MASK_MAX_MIB` | `2048`，范围 1–8192 MiB，每个 rank 单独分配 |
| `RSWA_GROUPED_MASK` | `1`，设为 `0` 关闭分组 mask |
| `TORCH_SLOT_MAPPING`、`BATCH_NGRAM`、`FUSED_SAM`、`FUSED_ROPE` | 均为 `1`，可各自设为 `0` |
| `FL_SOURCE_DIR` | FL 仓库根目录，加入 `PYTHONPATH` |
| `OCR_WORKDIR` | 脚本所在目录；`chat_template_bos.jinja` 始终从脚本所在目录加载 |
| `CANN_ENV` | `/usr/local/Ascend/ascend-toolkit/set_env.sh`，文件存在时加载 |
| `AUDIT_DIR` | 工作目录下的 `results/runtime-audit-${PORT}` |

```bash
cat > examples/unlimited_ocr/run_server.sh <<'OCR_RUN_SERVER'
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
OCR_RUN_SERVER
```

```bash
MODEL=/path/to/Unlimited-OCR bash examples/unlimited_ocr/run_server.sh

# Optional single-device Eager launch. Stop the first server before reusing its port.
MODEL=/path/to/Unlimited-OCR DEVICES=0 TP_SIZE=1 GRAPH_MODE=eager \
  bash examples/unlimited_ocr/run_server.sh
```

服务就绪后，在另一个终端运行客户端。以下客户端示例默认使用 `http://127.0.0.1:10000`。可以通过 `PORT` 设置其他本机端口，或通过 `BASE_URL` 指定服务地址，例如 `http://ocr-host:10000`。

## 精度推理

将 OmniDocBench v1.5 图片目录作为第一个参数传入，或设置 `IMAGES`。默认并发 64（`CONCURRENCY`），输出上限 8192 token（`MAX_TOKENS`）。客户端显式使用 Gundam 图像参数（`image_size=640`、`base_size=1024`、`crop_mode=true`）和 NGram 参数（`ngram_size=35`、`window_size=128`）。默认每张图片请求一次，结果写入 `--output` 指定的目录，导出的 Markdown 位于该目录的 `predictions/`。可用 `--samples` 抽样，或通过 `--image-list` 重用指定的图片顺序。额外的参数传给 [test_ocr.py](test_ocr.py)。

```bash
cat > examples/unlimited_ocr/run_accuracy.sh <<'OCR_RUN_ACCURACY'
#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGES="${IMAGES:-${1:?Pass the image directory as the first argument or set IMAGES}}"
if [[ "${1:-}" == "$IMAGES" ]]; then shift; fi
[[ -d "$IMAGES" ]] || { echo "Missing dataset: $IMAGES" >&2; exit 1; }
exec python3 "$SCRIPT_DIR/test_ocr.py" "$IMAGES" --base "${BASE_URL:-http://127.0.0.1:${PORT:-10000}}" \
  --image-mode gundam --concurrency "${CONCURRENCY:-64}" --max-tokens "${MAX_TOKENS:-8192}" "$@"
OCR_RUN_ACCURACY
```

```bash
bash examples/unlimited_ocr/run_accuracy.sh /path/to/OmniDocBench-v1.5/images \
  --output /path/to/ocr-results

# Single-image request
python3 examples/unlimited_ocr/test_ocr.py /path/to/sample.png \
  --base http://127.0.0.1:10000 --image-mode gundam
```

## OmniDocBench 官方评分

在安装了官方 v1.5 evaluator 及 CDM 依赖的 CPU 环境中运行。评分使用精度推理的输出目录或平铺的 Markdown 预测目录，以及原生 `OmniDocBench.json`。通过 `OCR_DATASET` 和 `OCR_EVALUATOR` 或 `--dataset` 和 `--evaluator` 指定路径。默认 `--full`，要求全部 1355 页。抽样结果使用 `--subset`。可对已有的评分目录使用 `--summarize-only`，导出已生成的官方指标。其他参数见 [score_ocr.py](score_ocr.py)。

```bash
cat > examples/unlimited_ocr/run_score.sh <<'OCR_RUN_SCORE'
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
OCR_RUN_SCORE
```

```bash
bash examples/unlimited_ocr/run_score.sh /path/to/ocr-results \
  --dataset /path/to/OmniDocBench-v1.5/OmniDocBench.json \
  --evaluator /path/to/evaluator_v1_5 --output /path/to/ocr-scores

bash examples/unlimited_ocr/run_score.sh /path/to/ocr-scores --summarize-only
```

## 真实图片性能

通过第一个参数或 `IMAGES` 指定图片目录。默认并发 64（`CONCURRENCY`）、输出上限 8192 token（`MAX_TOKENS`）、顺序预热请求数 0（`WARMUP`）。设置 `--output` 指定 JSON 报告路径，可用 `--limit` 限制按大小排序后参与测量的图片数，或用 `--save-text` 在计时结束后保存 OCR 文本。其他参数见 [bench_ocr_gundam.py](bench_ocr_gundam.py)。

```bash
cat > examples/unlimited_ocr/run_benchmark.sh <<'OCR_RUN_BENCHMARK'
#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGES="${IMAGES:-${1:?Pass the image directory as the first argument or set IMAGES}}"
if [[ "${1:-}" == "$IMAGES" ]]; then shift; fi
[[ -d "$IMAGES" ]] || { echo "Missing dataset: $IMAGES" >&2; exit 1; }
exec python3 "$SCRIPT_DIR/bench_ocr_gundam.py" "$IMAGES" --base "${BASE_URL:-http://127.0.0.1:${PORT:-10000}}" \
  --concurrency "${CONCURRENCY:-64}" --max-tokens "${MAX_TOKENS:-8192}" --warmup "${WARMUP:-0}" "$@"
OCR_RUN_BENCHMARK
```

```bash
bash examples/unlimited_ocr/run_benchmark.sh /path/to/OmniDocBench-v1.5/images \
  --output /path/to/ocr-real-image-benchmark.json
```

## 合成多模态性能

在客户端环境安装支持 `random_vl` 的 EvalScope，并准备 Transformers、tokenizers 和模型的本地 tokenizer 文件。设置 `MODEL` 为本地模型的绝对路径。脚本进入 `examples/unlimited_ocr/perf_h100` 后运行，`PERF_OUT` 如使用相对路径，将以该目录为基准，目标必须是新目录。可用 `OCR_SERVER_CONFIG` 记录服务配置、`FL_ROOT` 指定待记录的 FL 仓库、`FL_CANDIDATE_MANIFEST` 记录候选配置文件的哈希。

第一个参数为合成文本 token 数，支持 1024、4096（默认）和 16384，服务端还会添加模板和图像 token。每轮以 64 并发发送 1024 个请求，每个请求包含一张 2048×2048 RGB 图片，要求生成 1024 token。默认跑 3 轮，校验各轮请求数、成功数、平均输出长度和吞吐后，将第 2、3 轮的平均输出吞吐写入 `summary.json`。`ROUNDS=1` 运行一轮并生成该轮结果。运行目录还保存 tokenizer、`environment.json`、每轮原始报告及日志。环境记录包含磁盘上的 FL 源码身份和操作者填写的服务配置。

```bash
cat > examples/unlimited_ocr/perf_h100/run_perf.sh <<'OCR_RUN_PERF'
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
python3 perf_tools.py prepare --output "$OUT/environment.json"
for (( ROUND=1; ROUND<=ROUNDS; ROUND++ )); do
  python3 perf_tools.py prepare --check-server
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
  python3 perf_tools.py summarize "$OUT" --check-round "$ROUND"
done
if (( ROUNDS == 3 )); then python3 perf_tools.py summarize "$OUT"; fi
OCR_RUN_PERF
```

```bash
MODEL=/path/to/Unlimited-OCR ROUNDS=1 \
  bash examples/unlimited_ocr/perf_h100/run_perf.sh 1024

MODEL=/path/to/Unlimited-OCR \
  OCR_SERVER_CONFIG='TP2, FULL_DECODE_ONLY, max_num_seqs=64, max_num_batched_tokens=1024' \
  bash examples/unlimited_ocr/perf_h100/run_perf.sh 4096
```
