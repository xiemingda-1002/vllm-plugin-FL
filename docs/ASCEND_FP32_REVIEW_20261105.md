# Ascend FP32 评审修复验证（2026-11-05）

图状态随 GraphWrapper 和 batch descriptor 管理。清图时释放状态，KV cache 或表几何变化时先使旧图失效，再重建 NPU 缓冲区。FULL 与 eager 均禁用 attention 数值计算的 autocast，共用 TP、PCP、DCP 及 causal decoder 限制。调试统计移至可选验证工具。

在 Ascend 910C、CANN 9.0.1、vLLM 0.24.0 环境中对照修复前提交 `84fbb9d`，使用独立源码目录，保留 Unlimited-OCR core 补丁，`USE_FLAGGEMS=0`。性能测量期间各模型使用独立卡组，共享 CPU 和 I/O。

| 验证 | 修复前后结果 |
|---|---|
| NPU 组件 | 6 组外层 BF16/FP16 autocast、不同上下文和 block size，对照 18 次真实回放，最大误差 2.38e-7。覆盖 CPU 长度元数据、缓存替换、清图、重建和不支持配置 |
| BGE-M3 | 混合长度 13/13 均通过 CPU 对照。前后 eager 输出相同，graph 最大差 1.37e-7。四种配置 graph P50 波动约 1.5%–5.8%，均快于 eager |
| CosyVoice3 LLM | 固定 embeddings 的 eager/FULL token 一致，最大 logprob 差 1.36e-5，FULL 实际回放 30 次。修复前后各模式 token 与 logprob 一致，HF32 关闭 |
| CosyVoice3 HTTP | 四组中文精度均覆盖 12/12，CER 1.1544%。graph SIM×100 81.7377 → 81.7288，并发 1/2/4 吞吐变化约 ±1% |
| Unlimited-OCR | 固定 9 页均成功，overall 均为 93.3086，吞吐 643.83 → 654.06 token/s |
| Qwen3.6-27B / 35B / Qwen3-VL | 各模式性能请求 64/64 成功，吞吐变化 -0.35% / -0.66% / +1.12%。每模型 10 个文本用例各两遍，前后输出逐条一致。Qwen-VL 图像读字前后各 2/2 |

通用 Qwen 检查器在修复前后均将 greedy 探针达到输出上限判为失败。27B 和 35B 两轮均有同一算术误答。测试范围为抽样回归，DSv4F 因设备占用跳过。本 PR 的 CosyVoice3 改动范围是 LLM 侧，端到端测试使用外部集成。

复现入口见 `tools/validation/README.md`。精简指标与固定输入的逐 token 记录见 `docs/validation/pr10-1105/`。测试源码与提交源码已做 AST 对照。
