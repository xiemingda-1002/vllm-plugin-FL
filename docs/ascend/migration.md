# Ascend vLLM 0.24 迁移步骤与验收

## 实施基线与原则

以 vLLM 0.24.0 与匹配的 `vllm-ascend` 0.24.0rc1 为源码基线。历史 v0.2 笔记仅描述集成经验，
不能替代当前源码合同。迁移必须覆盖平台注册、runner、forward context、通信、算子和图生命周期
的完整调用链；硬件特定改动保持在 Ascend 范围，并保留其他 FL vendor 行为。安装后的运行时不得
import `vllm_ascend`。

A2/A3 按各自匹配的 `SOC_VERSION` 独立构建和验证。安装、import 与服务测试均在两个源码
checkout 之外进行：仅卸载 distribution 不会阻止当前目录中的 checkout 被导入。只改 Python
文件时可重用已审计 native 成员；改 native 源码或构建元数据时必须在匹配 SoC 重新构建。

## 迁移步骤

1. 从匹配源码审计注册、平台选择、Worker/ModelRunner、模型 forward、KV cache、通信、算子注册和
   graph capture/replay 的完整闭包，不用 v0.2 实现取代当前路径。
2. 优先保留 vLLM 模型主体，按 FL 现有职责目录放置扩展；Ascend 专用实现使用独立子目录或
   `ascend_*` 模块，不能将所有平台、模型和图运行时代码集中到算子后端目录。
   wheel 必须闭合所需 Python、native library、OPP/配置等部署依赖。
3. 先以 `--enforce-eager` 真实请求建立正确性，再在同模型/拓扑、未启用 eager 下验证
   `FULL_DECODE_ONLY` capture 与 decode replay。启动配置、注册表或单个 leader 日志不能代替图证据。
4. 分布式从 active runtime group 导出参与规模和组内 rank，不得硬编码 global rank。功能验收拓扑为
   DP2×TP2×effective EP4；两个 DP 副本都必须有 capture 和 FULL decode replay 证据。
5. 在干净容器构建 wheel，在中立工作目录安装并验证 FL native op，确认没有 `vllm-ascend`。A2/A3
   的 build、wheel 与 runtime 证据相互独立。

## 目录职责与多厂商隔离

FL 是多厂商插件。目录归属按职责判断，不能将公共入口整体替换成 Ascend 实现。

| FL 目录或入口 | 职责与 Ascend 扩展边界 |
| --- | --- |
| `__init__.py`、`platform.py`、`utils.py` | 公共插件注册、平台选择与设备发现；保留其他厂商分支与默认行为。 |
| `dispatch/`、`dispatch/backends/vendor/` | 算子注册、选择、缓存及厂商后端；只加载当前厂商，不遍历导入所有后端。Ascend 算子与 MoE/量化算子实现仍留在其后端。 |
| `attention/` | 注意力后端协议及 metadata；Ascend 实现在 `attention/ascend/`，通用工具不变。 |
| `compilation/` | 通用图封装及编译接口；Ascend 编译器、图运行时和辅助模块按平台选择，不要求其他厂商安装 `torch_npu`。 |
| `configs/`、`models/` | 模型/运行配置与模型实现；Ascend 配置、模型扩展使用独立命名，注册仍受原有平台条件控制。 |
| `patches/` | 公共兼容补丁；Ascend 专属补丁位于 `dispatch/backends/vendor/ascend/patches/`，由同厂商 `patch.py` 按既有顺序安装，不在公共包初始化时执行。 |
| `worker/` | 公共 worker、runner 和调度入口；保留各厂商执行路径，仅在相应分支接入 Ascend 能力。 |
| `distributed/`、`ops/`、`quantization/` | 公共通信、OOT 层和量化接入；保留 FlagCX、其他厂商 OOT/量化行为。Ascend 专用通信另置独立模块。 |
| `platforms/ascend/`、`profiler/ascend/` | Ascend 硬件/运行环境检查及 profiler；不替代公共 `platform.py`，不改变原有加载条件。 |
| `kv_cache/ascend/`、`scheduling/ascend_balance.py` | Ascend cache 与调度扩展，仅在既有 Ascend 执行链中接入。 |

厂商隔离以实际依赖和副作用为准：`ascend_flashcomm.py` 是纯配置辅助模块，顶层仅依赖标准库，
可由公共 worker/runner 导入。`torch_npu`、原生库加载、全局模型补丁与设备通信则必须维持原有
平台门控和调用时机。公共包的 `__init__.py` 不应为目录搬迁新增这些副作用。

目录搬迁需同时更新绝对导入、相对导入和字符串类路径；用安装后的 wheel 检查导入闭包，
再运行公共接口及非 Ascend 分支的 mock/源码合同回归。这些检查不等同于其他厂商的实机验证。

## 已实现路径与边界

Qwen 的非量化 BF16/FP16 AllGather 路径保持可用。DeepSeek 扩展包括 ModelSlim W8A8、普通
MC2/AllToAll、压缩 DSA cache 与图生命周期、FlashComm1、DSA context parallelism、
shared-expert DP 和 A3 W8A8 fused MC2。配置所有权贯穿 worker 生命周期；短暂 forward context
不得重置已配置优化。

fused MC2 不等同于普通 MC2。fused 路径仅覆盖支持的 A3 W8A8 且 EP≤32；BF16/W4A8 fused、EPLB、
层次通信和 mixed expert placement 未覆盖。fused 执行显式禁用 shared-expert overlap；不能从单个
已接受拓扑外推任意组合。static-kernel 与 compiler-pass sequence parallelism 不在本次验收，
FlashComm1 是独立路径。配置使用 rc1 `additional_config` 语义；不支持 opt-in 必须显式失败，
不能静默进入未迁移代码。

### HCCL 通信组内存配置

Ascend 通信组沿用 rc1 的显式 `pg_options`：普通组使用 200 MiB，DP 组按原生公式计算
且至少 50 MiB，dynamic EPLB 组使用 100 MiB；MC2 组不覆盖缓冲大小，保留
`HCCL_BUFFSIZE` 的作用。因此不能仅比较两边环境变量，就认为实际通信内存相同。
组注册表保留完整 options（包括 `group_name`）、引用计数和销毁/恢复语义；
MC2/EPLB 使用独立复用域，未知非默认 options 禁止复用。
此能力只在 Ascend vendor 初始化时安装，不改变公共 KV 预算公式或非 HCCL 的原有生命周期。
减少通信缓冲占用不代表任意长序列并发下均不会发生 KV 抢占。

## 验收证据与当前状态

- 历史 eager 证据保留在原 Run；最新 Qwen 集成回归是有界图模式回归，不重新覆盖完整 eager、视觉或
  长上下文矩阵。
- A3 Qwen3-0.6B 在 FlagGems 5.3.4 的 `USE_FLAGGEMS=0`/`1` 均完成图模式语义请求；`1` 仅增加
  `pow_scalar,lift_fresh` 规避项，不修改 FlagGems。35B/27B 在同一 A3 wheel 上完成有界文本图回归，
  不覆盖多模态或长上下文。
- A3 DeepSeek DP4×TP4×effective EP16 通过 strict 15/15（含 64K 输入）和独立 C16 16/16。主审
  复核 31 个原始响应均自然结束、非空、无替换字符且 logprob 有限；16 worker 都有直接
  `DispatchFFNCombine` kernel/API 与 replay 身份证据，覆盖四个 DP 副本。
- A2 clean build 已接受，SoC `ascend910b1`，wheel SHA-256 为
  `0a52a955d8c4d1098c5f8b50627fe07063034073df0457f86d9f76b2ff91cd92`；此结论只覆盖构建/打包，
  未执行 NPU runtime。
- A2 runtime v1 在非特权、6→0 映射配置下，FL 模型加载前发生 `aclInit` 507899 而无效。特权、原始节点、
  visible6 的 pure torch probe 已通过，只证明该组合容器配置可用，不能拆分归因；probe 不含 FL
  import/模型。随后 runtime v2 的 `regression-002` 在同一独立 A2 wheel 上接受四个有界文本图 case：
  35B/27B 各四个自然 stop、正确 “4/Blue” 输出，FULL count 为 2/7；0.6B FG0/FG1 各两个正确自然 stop
  输出，均 FULL1、replay18。它不改变 v1 的环境观察，也不证明 A2 DeepSeek、视觉、长上下文、分布式或
  性能；A2 QwenVL 在独立 Loop 已接受有界图像/静态视频图回归，仍不证明一般视觉/视频能力；A3 QwenVL
  未测。
- 自然 stop 输出必须可读、非空、有限 logprob、无替换字符。强制长度性能输出单独标识，不能替代
  语义验收。当前没有同 Case native vLLM-Ascend 对照；FL 指标、统计行和 profiler 均不证明性能
  parity 或优化收益。

可恢复证据位于 Task `fl-vllm024-ascend-migration-20260909`，重点为
`deepseek-v4-fused-mc2-a3-runtime-v2`、`integrated-qwen-a3-post-fused-v2`、
`integrated-ascend-a2-clean-build-v1`、`integrated-qwen-a2-post-fused-v1`、
`a2-container-device-enumeration-v1` 与 `integrated-qwen-a2-post-fused-v2`。历史矩阵见
[概览](./overview.md)，DeepSeek 配置见 [DeepSeek V4](./deepseek-v4.md)。
