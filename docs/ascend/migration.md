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
2. 保持 vLLM 模型主体不变；在 `vllm_fl` Ascend vendor 层放置平台注册、patch、算子派发、通信和
   图适配。wheel 必须闭合所需 Python、native library、OPP/配置等部署依赖。
3. 先以 `--enforce-eager` 真实请求建立正确性，再在同模型/拓扑、未启用 eager 下验证
   `FULL_DECODE_ONLY` capture 与 decode replay。启动配置、注册表或单个 leader 日志不能代替图证据。
4. 分布式从 active runtime group 导出参与规模和组内 rank，不得硬编码 global rank。功能验收拓扑为
   DP2×TP2×effective EP4；两个 DP 副本都必须有 capture 和 FULL decode replay 证据。
5. 在干净容器构建 wheel，在中立工作目录安装并验证 FL native op，确认没有 `vllm-ascend`。A2/A3
   的 build、wheel 与 runtime 证据相互独立。

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
