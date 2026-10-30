# Unlimited-OCR on FL Ascend

本目录提供 Unlimited-OCR 的启动模板、HTTP 客户端和官方评分入口。FL 的 R-SWA、FULL Decode padding 和图回放日志修改直接位于 `vllm_fl/`；模型、配置、processor 和 tokenizer 所需的 18 个 vLLM core 文件通过 [外部 core 补丁](../../patches/unlimited_ocr/README.md)交付，必须先应用。

本次以最新 upstream 重建并合入 MTP 后移植 OCR，仅完成静态/CPU 检查，尚未进行 NPU、模型前向、精度或性能验收。下述设备记录属于 OCR 来源版本，不作为本次移植验收。

## 基线与安装

适配基于 FL `7e68bd246bffa3be5d77e56c556a9876e61ba04b` 和 vLLM 0.24 `ee0da84ab9e04ac7610e28580af62c365e898389`。历史验证环境为 Python 3.12.13、Torch 2.10.0+cpu、torch-npu 2.10.0.post2、Transformers 5.13.0、torchvision 0.25.0+cpu、CANN/ATB 9.0.1。未整体升级 core 或 Python 依赖。准备与该环境兼容的 Ascend 运行环境、FL 原生扩展和 vLLM 安装后，在 FL 仓库根目录执行：

```bash
python3 patches/unlimited_ocr/apply_vllm_patch.py /path/to/vllm --check
python3 patches/unlimited_ocr/apply_vllm_patch.py /path/to/vllm
```

core 必须由该源码目录加载，可沿用已有的 editable 安装。已有 FL 编译扩展仅在 Python、Torch、torch-npu、CANN 和扩展 ABI 一致时复用；这次 Python 适配本身不要求重编译。复用已编译扩展的环境中，可按原部署方式更新 FL editable 安装：

```bash
VLLM_VENDOR= python3 -m pip install --no-deps --no-build-isolation -e .
```

空 `VLLM_VENDOR` 只用于这次安装。新环境或 ABI 不一致时，应按仓库正常 Ascend 构建流程编译和安装扩展；上述命令不能替代缺失的原生构建。运行时脚本显式选择 `VLLM_PLUGINS=fl`、`VLLM_VENDOR=ascend` 和 `VLLM_FL_VENDOR=ascend`。

## 启动与请求

启动脚本默认以自身目录为工作目录，默认端口 10002、单芯、TP1、1 个序列和 2048 token 的批处理预算。通过 `MODEL` 指定本机模型目录：

```bash
MODEL=/path/to/Unlimited-OCR \
VLLM_SOURCE_DIR=/path/to/vllm \
DEVICES=0 bash examples/unlimited_ocr/run_server.sh
```

`VLLM_SOURCE_DIR` 默认 `/vllm-workspace/vllm`，`FL_SOURCE_DIR` 默认本仓库根目录；`OCR_WORKDIR` 可指定另一工作目录，该目录须包含 `chat_template_bos.jinja`。`MODEL`、`PORT`、`DEVICES`、`TP_SIZE`、`MAX_NUM_SEQS` 和 `MAX_NUM_BATCHED_TOKENS` 均可覆盖。以下双芯配置用于已有的短请求图模式检查；请按本机空闲设备调整：

```bash
MODEL=/path/to/Unlimited-OCR VLLM_SOURCE_DIR=/path/to/vllm \
GRAPH_MODE=full DEVICES=0,1 TP_SIZE=2 \
MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=1024 \
bash examples/unlimited_ocr/run_server.sh
```

`GRAPH_MODE` 默认 `eager`。`full` 使用 `FULL_DECODE_ONLY`：Decode 使用 FULL NPU 图，视觉编码和 Prefill 不启用 FULL。它关闭 breakable graph 和脚本指定的额外 FX 融合；每个捕获桶首次实际回放后输出 `FL graph replay completed` 日志。切换模式前先停止占用同一端口的服务。

在另一终端从仓库根目录发送请求，显式指定本机图片路径和服务端口：

```bash
python3 examples/unlimited_ocr/test_ocr.py /path/to/sample.png \
  --base http://127.0.0.1:10002 --image-mode gundam

python3 examples/unlimited_ocr/test_ocr.py /path/to/omnidocbench_v1_5/images \
  --base http://127.0.0.1:10002 --image-mode gundam --concurrency 16
```

`test_ocr.py` 为每个请求携带 ngram size 35、window size 128；`--image-mode gundam` 显式携带 `image_size=640, base_size=1024, crop_mode=true`。服务注册 `NGramPerReqLogitsProcessor`，使用单 BOS 的模板和 `model_type=unlimited-ocr` override。core tokenizer 白名单使该模型使用 tokenizer.json 的原生 ByteLevel 图，避免错误的 Llama tokenizer 类改写编码及解码。

R-SWA 保留整个 prompt/image 前缀，生成历史窗口为 128；视觉 CLIP 仍为全注意力。0.24 的兼容实现通过 FullAttentionManager 保留完整 KV，未迁入新版 decode-gap 回收，KV 内存随序列增长。启动脚本要求 `MAX_NUM_SEQS <= MAX_NUM_BATCHED_TOKENS`，两者乘积不超过 16384；mask 还受最多 `2**29` 元素的运行时保护。TP2 不分摊每个 rank 的 mask。

## 官方评分与已验证范围

模型权重、OmniDocBench 数据及第三方官方 evaluator 不随仓库分发。`score_ocr.py` 在独立 CPU 评分环境运行，不调用 OCR 服务；需要官方 evaluator/CDM 的依赖，包括 PyYAML、Node.js、TeX、ImageMagick 和 Ghostscript。显式指定原生 v1.5 数据集 JSON 和官方 evaluator 源码目录：

```bash
python3 examples/unlimited_ocr/score_ocr.py /path/to/generated-run \
  --dataset /path/to/omnidocbench_v1_5/OmniDocBench.json \
  --evaluator /path/to/evaluator_v1_5 --full
```

评分脚本校验原生数据集和官方评分前置条件。`--full` 要求全部 1355 页；抽样 run 使用 `--subset`，抽样指标不能直接与全量结果比较。

修复 tokenizer 后，已有 Eager 全量 1355 页的记录如下；这些是此前运行的记录，不代表每个新部署都会得到相同数值：

| 指标 | Eager 结果 |
|---|---:|
| Overall ↑ | 92.8437846690 |
| Text Edit ↓ | 0.0392336755 |
| Formula CDM ↑ | 91.6744121 |
| Table TEDS ↑ | 90.7803095 |
| Table TEDS-S ↑ | 94.2323010 |
| Reading Order Edit ↓ | 0.0436247637 |

FULL Decode 已通过 5 张独立图片的 21 个短请求：全部 stop、0 失败、0 截断，官方 `remove_det` 后的正文与 Eager 21/21 一致；坐标 token 可能不同。两个 TP rank 均确认 1/2/4/8/16 五个桶实际回放。FULL 的全量精度及正式性能尚未验收；上述短请求不能替代 1355 页评分或吞吐对照。V2、量化、context parallel 和更高并发不在此次验收范围内。

## 回退与交付边界

先停止当前服务，再设置 `GRAPH_MODE=eager` 即可回到 Eager。若要撤销 OCR core 适配，使用 [补丁工具的回退流程](../../patches/unlimited_ocr/README.md)，同时将 FL 源码切回不含此次适配的版本；不再使用本目录 OCR 启动入口。

仓库保留运行时代码、补丁、manifest、启动模板和可复现检查脚本。模型、数据集、第三方 evaluator、结果、服务日志、PID、编译缓存、容器镜像和二进制扩展属于部署或运行产物，不纳入此次提交。
