# Ascend vLLM 0.24 构建、部署与复现

本流程使用与目标硬件匹配的 vLLM-Ascend 基础镜像提供 CANN、PyTorch、torch-npu 与 vLLM
依赖；FL 的目标运行环境不安装 `vllm-ascend`。镜像、CANN 和具体版本应按部署场景确认，
本文不绑定私人镜像或主机地址。

选择与 A2 或 A3 分别匹配的基础镜像；创建持久验证容器时注意该基线镜像的默认
`ENTRYPOINT` 是 `vllm serve`，需要 shell 时显式使用 `--entrypoint /bin/bash`。启动前按目标
环境核对真实卡的设备映射以及 driver/firmware 挂载，不要假设不同机器具有相同的物理卡号或
挂载布局。

基础镜像可能预装 `vllm-ascend`。安装 FL wheel 前先卸载该运行时插件；CANN、torch、
torch-npu 和 vLLM 仍由基础镜像保留：

```bash
python -m pip uninstall -y vllm-ascend
python -c "import importlib.util; assert importlib.util.find_spec('vllm_ascend') is None"
```

## 构建

```bash
git clone <FL_REPOSITORY_URL> vllm-plugin-FL
cd vllm-plugin-FL

# 使用基础镜像提供的 SOC_VERSION；A2/A3 必须分别在相应环境中构建。
VLLM_VENDOR=ascend pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps dist/<wheel-file>.whl
```

`setup.py` 接受 `VLLM_VENDOR=ascend`，并从 `SOC_VERSION` 选择 A2
(`ascend910b*`) 或 A3 (`ascend910_93*`) 构建族；未设置时默认 `ascend910_93`。因此不要将
A2 生成的 wheel 当作 A3 wheel 使用，反之亦然。仅改动 Python 文件时可以使用 wheel 重打包；
若改动 native 源码或构建元数据，应在目标 SoC 环境重新执行：

```bash
VLLM_VENDOR=ascend pip install --no-build-isolation .
```

当前 CMake 源码的最小 native 闭包是基础镜像中的 CANN（含 Ascend CMake/op 构建工具）、
`torch_npu` 和其头文件/库，以及 CMake（至少 3.26）与 C++17 编译器。
`csrc/ascend/build_opp.sh` 将 CATLASS 作为构建期外部依赖：默认在 `.deps/catlass` 获取
`41bf90da655bba3c66d0acd7e00abe33960ecfd6`，也可通过 `CATLASS_PATH` 提供已准备好的
checkout。离线构建应先准备该固定版本并核对 commit；已有 checkout 的版本由部署者负责
确认。CATLASS 整仓不随 FL 源码提交，不要依赖旧工作区中残留的 third_party 目录。

上面的 `--no-deps` 安装要求 Python 依赖已准备好。本轮使用容器内 FlagGems 5.3.4，且不修改
FlagGems 源码；即使关闭其算子派发，FL 的导入链仍可能需要该包及依赖。准备匹配版本的
FlagGems wheel 和项目依赖后再做安装自检。无网络环境还应准备完整离线依赖；本轮 A3 曾补充
SQLAlchemy 2.0.48 及 greenlet，不能把“wheel 已安装”当成依赖闭包完整。

## 安装后自检

在非源码目录执行，避免 `PYTHONPATH` 掩盖打包遗漏：

```bash
cd /tmp
python -c "import vllm_fl; print(vllm_fl.__file__)"
python -m pip show -f vllm-plugin-FL
python -c "from vllm_fl.ascend_custom_ops import enable_custom_op; print(enable_custom_op())"
python -c "import importlib.util; print(importlib.util.find_spec('vllm_ascend'))"
```

最后一条在目标部署环境应显示 `None`。`enable_custom_op()` 必须在真实 Ascend runtime 执行；
CPU-only 环境不能代替动态算子注册验证。

## A3 TP2 启动与固定请求

以下是已接受 A3 Qwen3.6-35B-A3B TP2 短请求的复现基线。`<MODEL_DIR>`、`<NPU0>` 与
`<NPU1>` 是部署者替换的本地模型路径和两张物理卡；其余服务参数保持一致。该命令显式开启
async scheduling 和 CPU binding，关闭 FlashComm1、shared-expert overlap、prefix cache 与
npugraph_ex；这些开关是该正确性 Case 的条件，不是通用性能推荐。

```bash
export ASCEND_RT_VISIBLE_DEVICES=<NPU0>,<NPU1>
export VLLM_PLUGINS=fl USE_FLAGGEMS=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=512
export VLLM_LOGGING_LEVEL=DEBUG VLLM_LOG_STATS_INTERVAL=1

# FULL_DECODE_ONLY graph case
vllm serve <MODEL_DIR> \
  --served-model-name qwen36-a3-review \
  --data-parallel-size 1 --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --max-num-seqs 4 --max-model-len 4096 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.90 --disable-custom-all-reduce \
  --no-enable-prefix-caching --async-scheduling \
  --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":false,"multistream_overlap_shared_expert":false,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --cudagraph-metrics --generation-config vllm --trust-remote-code --seed 0 \
  --enable-log-requests --host 127.0.0.1 --port 19432
```

FL 会通过运行时设备探测选择 Ascend，并在平台 kernel 导入阶段发布 wheel
内置的 OPP；普通启动不需要设置 `VLLM_FL_PLATFORM=ascend`。该变量仅保留为
平台自动探测不可用时的显式覆盖选项。

eager 基线使用相同的环境变量、模型和并行参数，只替换图配置并追加 `--enforce-eager`：

```bash
# 将上例中的 --compilation-config 替换为下行，并在命令末尾追加 --enforce-eager。
--compilation-config '{"cudagraph_mode":"NONE"}'
```

服务健康后，以固定 chat 请求验证响应与 logprob；可按需要重复该请求以覆盖 replay：

```bash
curl --fail --silent --show-error http://127.0.0.1:19432/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-a3-review","messages":[{"role":"user","content":"State the value of 2 plus 2."}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":128,"temperature":0,"seed":0,"logprobs":true}'
```

保存服务日志、请求响应和退出/设备清理信息。FULL 必须由日志确认 capture 与 replay 实际发生；
仅启动成功或单个请求成功不足以证明图路径。

## DP2 图观测限定

DP2×TP2×effective EP4 需要四张卡、`--data-parallel-size 2`、
`--data-parallel-size-local 2`、`--data-parallel-backend mp`、`--enable-expert-parallel` 和
`--all2all-backend allgather_reducescatter`。已接受 Case 使用 `--api-server-count 1`，并通过
`VLLM_LOGGING_CONFIG_PATH` 指向测试专用的既有 FL 日志 handler，才能在一个 API 前端下观察到
两个 DP 副本的 capture/FULL 记录。默认的两个 API server 会抑制这类文本统计，不能据此宣称
两个 DP 都已 replay。该 logging 配置仅用于测试证据，不应进入产品源码。

DP2 Case 同时使用 `--no-async-scheduling`、`--max-num-batched-tokens 4096`、
`--gpu-memory-utilization 0.85` 和 `HCCL_BUFFSIZE=1024`，其余限制见验收 Run。
不要仅在上述 DP1 命令上增加 DP 参数，却沿用不同的调度和批次配置。

`USE_FLAGGEMS` 是可见环境开关，已测 A3 Qwen 模型运行使用 `USE_FLAGGEMS=0`；安装时对 0/1
的开关核验不是 `USE_FLAGGEMS=1` 的模型正确性验收。更多能力范围和验收限制见
[概览](./overview.md)与[迁移步骤](./migration.md)。
