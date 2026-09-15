# Ascend vLLM 0.24 模型迁移概览

本目录记录 `vllm-plugin-FL` 在 vLLM 0.24 上的 Ascend 迁移交付。实现以匹配的
`vllm-ascend` 0.24.0rc1 为源码基线；它只用于比对和迁移，交付运行环境不依赖
`vllm_ascend` 包。模型主体复用上游 vLLM，Ascend 差异保持在 `vllm_fl` vendor 路径。

迁移按平台注册、Worker/ModelRunner、Ascend 算子与通信、图生命周期和 wheel 打包的实际调用链
进行：先完成 eager 正确性，再验证 `FULL_DECODE_ONLY` capture/replay，分布式再以真实
DP2×TP2×effective EP4 证明两个 DP 副本均进入图路径。

`integrated-qwenvl-a2-post-fused-v2` 已接受 A2 Qwen3-VL-30B-A3B-Instruct 的有界图像/静态视频
回归；范围、证据与限制见 [Qwen3-VL-30B-A3B-Instruct（A2）](./qwen3-vl.md)。A3 QwenVL 未测。

## 历史 Qwen 验收矩阵

以下记录保留对应历史 Run 的版本范围，不代表后续所有优化组合均经过重测。“通过”表示可读非空
输出、有限 token/logprob、无运行时错误，图模式另有 capture/replay 日志；不表示逐 token 对齐、
性能结论或未列拓扑可用。

| 硬件 | 模型/拓扑 | 模式与结果 | 证据入口 |
|---|---|---|---|
| A2 | Qwen3.6-35B-A3B，TP2 | true eager 与 FULL 各 4 个完整 stop 输出；FULL capture 3/3、实际 FULL 30 次 | `current024-a2-replay-a3-build-v1/a2_replay/a2-replay-002` |
| A3 | Qwen3.6-35B-A3B、Qwen3.6-27B，分别 TP2 | 每模型 eager/FULL 各 4 个完整 stop 输出；每个 FULL capture 3/3、实际 FULL 30 次 | `current024-a3-qwen-runtime-v1` |
| A3 | Qwen3.6-35B-A3B，DP2×TP2×effective EP4 | 两个 DP 副本均完成 capture；FULL 执行 11/17；10 请求各 DP 5 个成功输出 | `current024-a3-dp2-graph-v1/a3-dp2-002` |
| A3 | Qwen3.6-35B-A3B，TP2，长上下文 | 并发 2；每请求 65,536 输入 token + 1,024 输出 token；FULL 1,023，输出及 logprob 通过有限性/可读性门禁 | `current024-a3-longctx-v1/a3-longctx-001` |

这些证据保存在 `tasks/fl-vllm024-ascend-migration-20260909/loops/`，由对应的
`decision.json` 和 Run 原始 artifacts 索引。A3 独立 wheel 已在真实 A3 验证：构建标记
`ascend910_9391`、运行时 SoC 253、7 个 FL 自有 native op 动态注册，且未安装
`vllm-ascend`。

## 当前 A3 集成回归与 DeepSeek 验收

当前 A3 集成 wheel 在独立容器（未安装或导入 `vllm_ascend`）完成以下有界验收：

| 模型/拓扑 | 本轮覆盖 | 证据 Loop |
|---|---|---|
| Qwen3-0.6B，TP1 | FlagGems 5.3.4，`USE_FLAGGEMS=0` 后 `1`；各 2 个自然结束语义请求、有限 logprob，FULL graph 1、请求 replay 18 | `integrated-qwen-a3-post-fused-v2` |
| Qwen3.6-35B-A3B，TP2 | 图模式 4 请求通过；保存的 FULL 统计行 count=22 | 同上 |
| Qwen3.6-27B，TP2 | 图模式 4 请求通过；保存的 FULL 统计行 count=2 | 同上 |
| DeepSeek V4 Flash W8A8，DP4×TP4×effective EP16 | strict 15/15（含 65,536 输入 token）及 C16 16/16；16 worker 均有 fused MC2 kernel/API 与图重放证据 | `deepseek-v4-fused-mc2-a3-runtime-v2` |

FULL 统计行与请求数不是同一指标，不同日志窗口不可相加或用于性能比较。Qwen3-0.6B 的
FlagGems 开启用例仅使用 `pow_scalar,lift_fresh` 规避项，不修改 FlagGems，也不自动扩大
黑名单；该结论不能外推其他模型。

本轮 Qwen 是文本图模式回归，不代表重新完成视觉、长序列或 DP2 验收；A3 Qwen3-VL 明确不测。
DeepSeek 配置、通信分支与边界见 [DeepSeek V4](./deepseek-v4.md)。

## A2 当前状态

A2 clean build 已接受：目标 SoC 为 `ascend910b1`，wheel SHA-256 为
`0a52a955d8c4d1098c5f8b50627fe07063034073df0457f86d9f76b2ff91cd92`。该 Run 的构建、wheel
审计及 A2 native 闭包通过，但没有执行 NPU runtime。

A2 集成 runtime v1 在非特权容器、可见卡 6 映射为 0 的配置下，FL 模型加载前的
`torch.npu.set_device` 报 `aclInit` 507899 `Resource_Busy`，故没有模型正确性 verdict。随后同镜像采用特权容器、
原始 0..7 device node、`ASCEND_RT_VISIBLE_DEVICES=6` 的组合配置，pure `torch_npu` probe
通过（SOC220、可见一张卡、tiny tensor）；它没有 import FL 或加载模型。这只证明该组合容器
配置可用，不能将可用性分别归因于特权或原始节点映射。

`integrated-qwen-a2-post-fused-v2` 的 `regression-002` 已接受。它在同一独立 A2 wheel 上完成四个
有界文本图回归，driver exit 0：Qwen3.6-35B-A3B 与 Qwen3.5-27B-A3B 各有四个自然 stop、可读且
logprob 有限的 “4/Blue” 输出，记录 FULL count 分别为 2 与 7；Qwen3-0.6B 在 FG0/FG1 各有两个正确
自然 stop 输出，均为 FULL1、replay18。该结论仅为文本图模式回归；不外推 A2 DeepSeek、视觉、长上下文、
分布式或性能，也不能用 A3 结果替代 A2。A2 QwenVL 的有界图像/静态视频回归已在独立 Loop 接受，
不扩大为一般视觉或视频能力；A3 QwenVL 未测。

## 范围与限制

- MoE 不再限于未量化 AllGather：普通 MC2/AllToAll 与 ModelSlim W8A8 路径已迁移；A3 W8A8
  fused MC2 仅在 DP4×TP4×EP16 实际验证，其他拓扑须单独验证。
- fused MC2 与 shared-expert multistream overlap 不能同时开启。实现、配置接受、实际执行和性能
  收益是不同结论，不能仅凭启动参数宣称收益。
- EPLB、mixed expert placement、MC2 hierarchy、static kernel、compiler-pass sequence
  parallelism、MoE LoRA 及 fused MC2 的 BF16/W4A8 分支不在本轮交付范围。
- 当前没有本集成版本对同版本原生 vLLM-Ascend 的正式性能对照。代码验收与提交后，才可在同一
  A3 硬件、工作负载和参数下比较 FL/native；既有性能记录不能外推为本版本已对齐。
- A2/A3 native payload 不混用；原生来源文件保留各自许可，不能把全部 native 代码称为
  Apache-only。

详见 [迁移步骤与验收](./migration.md) 和 [构建、部署与复现](./deployment.md)。
