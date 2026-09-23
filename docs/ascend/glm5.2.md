# GLM-5.2 W8A8（A3）

实现基线为 vLLM 0.24.0 / vLLM-Ascend 0.24.0rc1，复用上游模型主体。
此处是 W8A8 配置，不是社区 W4A8C8 配方；不启用 C8、推测解码或 GLM DSA-CP。
不影响 DeepSeek 既有的 DSA-CP 实现。

## 迁移闭包

- Ascend SFA、W8A8 量化与 MoE 路径；共享专家多流复用现有 Ascend 实现。
- `enable_mlapo`：SFA 权重整理、量化 RMSNorm bias 加载、`mla_preprocess`
  schema/PrivateUse1/Meta、AscendC kernel 及 wheel 中的同级共享库。
- `enable_balance_scheduling`：Ascend 专用调度器与 DP 负载同步；显式配置优先于
  兼容环境变量 `VLLM_ASCEND_BALANCE_SCHEDULING`，默认关闭。
- FULL_DECODE_ONLY、异步调度、128 线程权重加载复用现有运行路径。

MLAPO native 文件来自匹配 rc1 的 `csrc/mla_preprocess`，辅助头来自
`csrc/kernels/types.h`。保留原始文件头；除适配 wrapper 声明外不修改 kernel 语义。
RMSNorm 保留 rc1 的 bias 初始化、权重加载及原生 torch_npu fallback 语义。

## 构建

使用匹配 A3 CANN/torch_npu 的基础环境；不要复用 A2 二进制。首次或 native 源码变化时：

```bash
VLLM_VENDOR=ascend pip install --no-build-isolation .
```

需要正确的 `SOC_VERSION`。A3 本轮为 `ascend910_9391`。
CANN 9.0.1 的 direct-kernel host stub 对 Ninja object path 存在兼容问题，Ascend
构建选择 Unix Makefiles；其他 vendor 的 generator 选择不变。
wheel 必须同时含 `_C_ascend*.so`、`libvllm_fl_ascend_kernels.so` 和完整
`_cann_ops_custom/`。共享库通过 `$ORIGIN` 解析，不依赖安装 `vllm_ascend`。
仅 Python 修改可复用同源码、同 SoC 的 native 产物重打包，无须重复编译 OPP。

## 验证配方

在按[部署指南](./deployment.md)准备好的 A3 16-device 容器中执行。
容器内先确认 `npu-smi info` 成功，使用实际模型路径；本轮 `/dev/shm` 为 512 GiB。
不要把宿主机不匹配的 CANN 安装目录覆盖到镜像内。

```bash
unset VLLM_VERSION LD_PRELOAD VLLM_ASCEND_ENABLE_FLASHCOMM1
unset VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE ASCEND_LAUNCH_BLOCKING
unset VLLM_ASCEND_ENABLE_MLAPO VLLM_ASCEND_BALANCE_SCHEDULING
unset VLLM_PLUGINS VLLM_FL_PLATFORM
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export USE_FLAGGEMS=0
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=200
export OMP_PROC_BIND=false OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /public-flash/models/GLM-5.2-w8a8 \
  --host 0.0.0.0 --port 8000 --served-model-name glm52-w8a8 \
  --trust-remote-code --seed 1024 \
  --data-parallel-size 2 --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 67000 --max-num-seqs 48 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 --quantization ascend --async-scheduling \
  --additional-config '{"enable_mlapo":true,"enable_balance_scheduling":true,"multistream_overlap_shared_expert":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":96}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --no-enable-prefix-caching --no-enable-log-requests
```

图观测验收额外使用 `--api-server-count 1 --cudagraph-metrics`：上游在 API 进程数
大于 1 时关闭文本统计日志。这不改变 DP2/TP8/EP16 拓扑，但做性能比较时也要对齐
API 进程数。普通 FL 自动识别 Ascend，不要求 `VLLM_FL_PLATFORM`。

## 当前证据与限制

Task `fl-vllm024-ascend-migration-20260909`，Loop `glm52-mlapo-native-package-v2`：

- A3 native direct build、PrivateUse1/Meta 及五输出 alias 检查通过。
- 保留 606 个 OPP 文件的 wheel 重打包、隔离安装及 native 加载通过。
- DP2/TP8/EP16 FULL 捕获与服务启动通过；4 个并发短问答正确自然结束。
- 32,791 / 65,541 输入 token 的问答正确，logprob 有限，无替换字符。
- 4 路长文本生成各输出 1,232–1,296 tokens，均正常结束、无替换字符。

最终安装与分 DP 重放验收记录在 `glm52-final-wheel-delivery-v1`：清理 GLM DSA-CP
尝试后安装的 wheel 再次通过全部语义门禁；两个 DP 均有独立的 FULL 重放统计。
例如同一日志窗口 Engine 000/001 分别记录 FULL count=218；这不是请求数或性能指标。
这些是有界正确性证据，不是完整精度数据集或性能对齐结论。
未验证 A2 GLM、GLM DSA-CP、C8、推测解码及其他并行拓扑；压测报告单独记录。
