# Ascend vLLM 0.24 Qwen 迁移步骤与验收

## 实施步骤

1. 以匹配版本的 vLLM 0.24.0 与 `vllm-ascend` 0.24.0rc1 审计完整调用闭包，而不是复用旧
   v0.2 实现。将平台注册、runner、forward context、通信、算子和图生命周期一并迁入。
2. 保持 vLLM 模型主体不变，在 `vllm_fl` 的 Ascend vendor 层实现平台注册、patch、算子
   派发、通信和图适配；部署时不得 import `vllm_ascend`。
3. 先以 `--enforce-eager` 建立真实请求正确性，再在未启用 eager 的同一模型/拓扑验证
   `FULL_DECODE_ONLY` 的 capture 和 decode replay。日志中的启动配置或单个 leader 的进度
   不能替代运行时图证据。
4. 分布式验证从实际 runtime group 派生通信规模与组内 rank。功能验收拓扑为
   DP2×TP2×effective EP4，且两个 DP 副本都必须记录 capture 和 FULL decode replay。
5. 独立构建 wheel 后，在非源码工作目录安装并验证 FL 自有 native op；运行时环境不得安装
   `vllm-ascend`。A2/A3 各自构建、安装和验收。

## 当前能力边界

Qwen MoE 的已验证默认路径是未量化 BF16/FP16 AllGather。配置读取遵循当前版本的
`additional_config` 嵌套结构，例如 `eplb_config`、`ascend_compilation_config` 和
`ascend_fusion_config`；不支持的 opt-in 必须显式失败，不能静默回退为默认路径。

FlashComm1 与 shared-expert multistream overlap 是独立路径，当前验收默认关闭。MC2、EPLB、
量化 MoE、mixed expert placement、shared-expert DP、static-kernel compilation 和
compiler-pass sequence parallelism 不在已支持范围。

## 验收口径

- 短请求：eager 与 FULL 都要求完整、可读、非空的 stop 输出和有限 logprob。
- 图模式：要求实际 capture 完成和 decode replay 记录；执行次数与请求数不必一一对应。
- 分布式：要求两个 DP 副本都有 capture/replay 证据，不能只以 leader 证明通过。
- 长上下文：当前 A3 覆盖并发 2 的 64K 输入 + 1K 输出；这是有界正确性门禁，不是事实准确率
  或完整报告质量认证。
- 性能：当前没有同版本原生 vLLM-Ascend 对照；不得以已有 A2 FL 矩阵宣称 A3 或原生性能
  parity。

具体已测项与 Run 索引见[概览](./overview.md)。
