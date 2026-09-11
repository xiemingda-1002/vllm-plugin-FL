# Ascend vLLM 0.24 Qwen 迁移概览

本目录记录 `vllm-plugin-FL` 在 vLLM 0.24 上的 Ascend Qwen 迁移交付。实现以
同代 `vllm-ascend` 0.24.0rc1 为源码基线；它只用于比对和迁移，交付运行环境不依赖
`vllm_ascend` 包。

迁移沿着平台注册、Worker/ModelRunner、Ascend 算子与通信、图生命周期和 wheel 打包的
实际调用链进行。模型主体复用上游 vLLM；Ascend 差异保持在 `vllm_fl` 的 vendor 路径。
先完成 eager 正确性，再验证 `FULL_DECODE_ONLY` capture/replay，随后以真实
DP2×TP2×effective EP4 证明两个 DP 副本均进入图路径。

## 已测矩阵

下表只陈述对应 Run 的覆盖范围；“通过”表示可读非空输出、有限 token/logprob、无运行时
错误，图模式另有 capture/replay 日志。它不表示逐 token 对齐、性能结论或未列拓扑可用。

| 硬件 | 模型/拓扑 | 模式与结果 | 证据入口 |
|---|---|---|---|
| A2 | Qwen3.6-35B-A3B，TP2 | true eager 与 FULL 各 4 个完整 stop 输出；FULL capture 3/3、实际 FULL 30 次 | `current024-a2-replay-a3-build-v1/a2_replay/a2-replay-002` |
| A3 | Qwen3.6-35B-A3B、Qwen3.6-27B，分别 TP2 | 每模型 eager/FULL 各 4 个完整 stop 输出；每个 FULL capture 3/3、实际 FULL 30 次 | `current024-a3-qwen-runtime-v1` |
| A3 | Qwen3.6-35B-A3B，DP2×TP2×effective EP4 | 两个 DP 副本均完成 capture；FULL 执行 11/17；10 请求各 DP 5 个成功输出 | `current024-a3-dp2-graph-v1/a3-dp2-002` |
| A3 | Qwen3.6-35B-A3B，TP2，长上下文 | 并发 2；每请求 65,536 输入 token + 1,024 输出 token；FULL 1,023，输出及 logprob 通过有限性/可读性门禁 | `current024-a3-longctx-v1/a3-longctx-001` |

这些证据保存在任务目录
`tasks/fl-vllm024-ascend-migration-20260909/loops/`，并由其中的 `decision.json` 和 Run
原始 artifacts 索引。A3 独立 wheel 已在真实 A3 验证：构建标记 `ascend910_9391`、运行时
SoC 253、7 个 FL 自有 native op 动态注册，且未安装 `vllm-ascend`。

## 范围与限制

- A2 与 A3 是独立环境和证据，不能互相外推；A2 的 Qwen3-0.6B FlagGems 0/1 图模式、以及
  A2 DP2 证据保留在既有 Run，未在本轮重新执行。
- A3 Qwen 模型运行使用 `USE_FLAGGEMS=0`；已核对安装开关 0/1，不代表 FlagGems 1 的模型
  正确性。
- 当前 MoE 支持未量化 BF16/FP16 AllGather。MC2、EPLB、量化 MoE、shared-expert DP、
  static kernel、compiler-pass sequence parallelism 和 MoE LoRA 不在本轮验收范围。
  FlashComm1 和 shared-expert multistream overlap 有实现/路径证据，但默认关闭，未承诺稳定
  性能收益。
- 本阶段不做与同版本原生 vLLM-Ascend 的性能比较。既有 A2 FL 性能矩阵只保留其原代码、
  平台和配置范围，不能外推 A3 或原生对照。

详见[迁移步骤与验收](./migration.md)和[构建、部署与复现](./deployment.md)。
