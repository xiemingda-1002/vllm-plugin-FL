# Ascend vLLM 0.24 构建、部署与复现

本流程使用匹配硬件的 vLLM-Ascend 基础镜像提供 CANN、PyTorch、torch-npu 与 vLLM 依赖；FL
目标运行环境不安装 `vllm-ascend`。镜像、CANN 和具体版本按部署场景确认；本文不绑定私人镜像或
主机地址，也不把某次验收命令解释为性能推荐。

选择分别匹配 A2/A3 的基础镜像。若默认 `ENTRYPOINT` 是 `vllm serve`，创建 shell 时显式使用
`--entrypoint /bin/bash --workdir /run`。镜像可能将默认工作目录设为原生源码；即使卸载 wheel，
源码仍可能被导入。因此自检和服务均在中立目录执行，并从继承的 `PYTHONPATH` 中移除源码仓路径，
但保留镜像提供的 CANN SDK Python/OPP 路径；完全清空可能导致 worker 无法导入 `acl`。启动前核对
真实设备映射和 driver/firmware 挂载，不得假设不同机器有相同卡号、布局或工具安装路径。

## 容器启动与 NPU 工具门禁

容器命令以对应模型、对应 vLLM-Ascend 版本的官方教程为基础。官方示例通常同时挂载
Davinci 设备、`davinci_manager`、`devmm_svm`、`hisi_hdc`、DCMI、`hccn_tool`、driver
动态库和版本信息以及 `npu-smi`。实际执行前逐项核对宿主路径：例如有的教程使用
`/usr/local/bin/npu-smi`，而某些宿主实际安装在 `/usr/local/sbin/npu-smi`；应挂载真实文件，
不能仅依赖 `--privileged`。

以宿主实际路径 `/usr/local/sbin/npu-smi` 为例，容器参数至少包含：

```bash
--privileged --network host --ipc host \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
-v /usr/local/dcmi:/usr/local/dcmi:ro \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
-v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro
```

还需按目标卡集合逐个增加 `--device /dev/davinci<N>`。若宿主某个官方示例路径不存在，先定位
等价的实际安装路径并记录差异，不能把不存在的路径静默省略。模型启动前执行硬门禁：

```bash
command -v npu-smi
npu-smi info
test -e /dev/davinci_manager
test -e /dev/devmm_svm
test -e /dev/hisi_hdc
```

任一命令失败就停止启动并修正容器挂载。Run 证据应保存完整 `docker run` 命令、
`docker inspect`、容器内门禁输出及设备映射。只有宿主执行 `npu-smi` 不足以通过该门禁；
否则运行时 CPU binding、拓扑探测等逻辑可能退化或被跳过。

基础镜像可能预装 `vllm-ascend`。安装 FL wheel 前先移除运行时插件，保留 CANN、torch、
torch-npu 与 vLLM：

```bash
mkdir -p /run/fl-validation
cd /run/fl-validation
# 仅保留基础镜像中已核对的 /usr/local/Ascend CANN SDK 路径，不能加入源码仓路径或空项。
export PYTHONPATH=<CANN_SDK_PYTHON_AND_OPP_PATHS>
python -m pip uninstall -y vllm-ascend
python -c "import importlib.util; assert importlib.util.find_spec('vllm_ascend') is None"
```

## 构建

```bash
git clone <FL_REPOSITORY_URL> vllm-plugin-FL
cd vllm-plugin-FL

# 使用基础镜像提供并已核对的 SOC_VERSION；A2/A3 分别在相应环境构建。
VLLM_VENDOR=ascend pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps dist/<wheel-file>.whl
```

`setup.py` 接受 `VLLM_VENDOR=ascend`，并通过 `SOC_VERSION` 选择构建族；部署前必须以目标 commit
的 build 脚本确认值。当前已验证 A2 clean build 使用 `ascend910b1`，A3 使用
`ascend910_9391`。不要互用 wheel。只改 Python 文件可按已审计 native payload 重打包；改 native
源码或构建元数据时，在匹配 SoC 环境重新执行：

```bash
VLLM_VENDOR=ascend pip install --no-build-isolation .
```

可选的构建并发控制（仅影响编译调度，不改变算子计算）：`MAX_JOBS` 控制外层构建任务；
`TILINGKEY_PARALLEL_JOB` 为正整数时，显式启用 CANN 的单算子 tiling-key 内部并行
（本轮依据 CANN 9.0.1 工具实现验证）。
不设置后者时保持原构建行为。两层并发会叠加，应根据 CPU 和内存预算共同限制；
例如资源充足的构建机可尝试 `MAX_JOBS=8 TILINGKEY_PARALLEL_JOB=8`，并观察实际负载，
不能把它视为所有机器的推荐值。不要为提速裁剪所需算子或降低编译优化等级。
长时间构建应持久化整个源码构建目录，且不要对构建容器使用 `--rm`。

最小 native 闭包为基础镜像 CANN（含 Ascend CMake/op 工具）、`torch_npu` 及头文件/库、CMake
（至少 3.26）和 C++17 编译器。`csrc/ascend/build_opp.sh` 的 CATLASS 构建期依赖默认在
`.deps/catlass` 获取 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`，也可由 `CATLASS_PATH`
提供；离线构建先准备并核对该 commit。CATLASS 整仓不随 FL 源码提交，不应依赖旧工作区残留。

`--no-deps` 安装要求 Python 依赖已就绪。本轮容器使用 FlagGems 5.3.4，且不修改其源码；即使关闭
算子派发，FL 导入链仍可能需要该包及依赖。无网络环境应准备匹配 FlagGems wheel 和完整离线依赖。
本轮 A3 曾补充 SQLAlchemy 2.0.48 与 greenlet，不能把“wheel 已安装”当作依赖闭包完整。

## 安装后自检

```bash
cd /run/fl-validation
# 与容器创建阶段相同：只保留已核对的 CANN SDK Python/OPP 路径。
export PYTHONPATH=<CANN_SDK_PYTHON_AND_OPP_PATHS>
python -c "import vllm_fl; print(vllm_fl.__file__)"
python -m pip show -f vllm-plugin-FL
python -c "from vllm_fl.ascend_custom_ops import enable_custom_op; print(enable_custom_op())"
python -c "import importlib.util; print(importlib.util.find_spec('vllm_ascend'))"
```

最后一条在部署环境必须显示 `None`。`enable_custom_op()` 必须在真实 Ascend runtime 执行；CPU-only
不能替代动态算子注册验证。

## A3 TP2 Qwen 启动与固定请求

以下为已接受 A3 Qwen3.6-35B-A3B TP2 短请求的正确性复现基线。`<MODEL_DIR>`、`<NPU0>`、
`<NPU1>` 由部署者替换为模型路径与两个运行时可见 device ID。它显式开启 async scheduling 和
CPU binding，关闭 FlashComm1、shared-expert overlap、prefix cache 与 npugraph_ex；不是性能推荐。

```bash
export ASCEND_RT_VISIBLE_DEVICES=<NPU0>,<NPU1>
export VLLM_PLUGINS=fl USE_FLAGGEMS=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=512
export VLLM_LOGGING_LEVEL=DEBUG VLLM_LOG_STATS_INTERVAL=1
vllm serve <MODEL_DIR> \
  --served-model-name qwen36-a3-review \
  --data-parallel-size 1 --tensor-parallel-size 2 --distributed-executor-backend mp \
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

eager 基线使用相同环境、模型和并行参数，将图配置替换为
`--compilation-config '{"cudagraph_mode":"NONE"}'`，并追加 `--enforce-eager`。服务健康后可用：

```bash
curl --fail --silent --show-error http://127.0.0.1:19432/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-a3-review","messages":[{"role":"user","content":"State the value of 2 plus 2."}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":128,"temperature":0,"seed":0,"logprobs":true}'
```

保存服务日志、请求响应和退出/设备清理信息。FULL 必须由日志证明 capture/replay 实际发生；启动或
单请求成功不足以证明图路径。

## DP2 图观测限定

DP2×TP2×effective EP4 需要四张卡、`--data-parallel-size 2`、`--data-parallel-size-local 2`、
`--data-parallel-backend mp`、`--enable-expert-parallel` 和
`--all2all-backend allgather_reducescatter`。接受 Case 使用 `--api-server-count 1`，并通过
`VLLM_LOGGING_CONFIG_PATH` 指向测试专用、既有 FL 日志 handler，才能在一个 API 前端看到两个 DP
副本 capture/FULL。默认两个 API server 会抑制这类文本统计，不能据此宣称两个 DP 都 replay；该
logging 配置仅用于测试证据，不能进入产品源码。

DP2 Case 还使用 `--no-async-scheduling`、`--max-num-batched-tokens 4096`、
`--gpu-memory-utilization 0.85` 和 `HCCL_BUFFSIZE=1024`。不要仅在上述 DP1 命令增加 DP 参数，
却沿用不同调度/批次配置。

Qwen3-0.6B 的 A3 集成回归在 FlagGems 5.3.4 关闭/开启均通过；开启时追加
`VLLM_FL_FLAGOS_BLACKLIST_APPEND=pow_scalar,lift_fresh`，不改 FlagGems。35B/27B 本轮使用
`USE_FLAGGEMS=0`，不能外推小模型结果。DeepSeek DP4/TP4/EP16 配置见
[DeepSeek V4 复现](./deepseek-v4.md)。A2 已在独立 wheel 上接受四个有界 Qwen 文本图回归（35B/27B
各四个自然 stop “4/Blue” 输出；0.6B 的 FG0/FG1 各两个正确输出）；这不使本 A3 命令成为 A2 部署
配方，也不外推 A2 DeepSeek、长上下文、分布式或性能。A2 QwenVL 已接受独立的有界图像/静态视频
图回归，仍不构成一般视觉/视频部署结论；A3 QwenVL 未测。
