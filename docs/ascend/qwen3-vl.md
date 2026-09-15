# Qwen3-VL-30B-A3B-Instruct（A2）

本 checkpoint 的 Qwen3-VL 迁移以匹配的 `vllm-ascend` 0.24.0rc1 为当前源码基线，交付运行时仅使用
`vllm-plugin-FL`。本次接受范围为 A2、BF16、TP2、DP1、effective EP2、`USE_FLAGGEMS=0` 的有界
图像/静态视频图模式回归。

`integrated-qwenvl-a2-post-fused-v2/runs/visual_graph/visual-002` 已接受：driver exit 0，主审的固定
文本、单图、双图及静态视频请求为 8/8 自然 stop、可读，且 client 的有限 logprob 校验通过。两个 TP worker
均记录 decoder `FULL=3`、replay=194；encoder 在图像请求后 hits=1、静态视频后 hits=3、misses=0。
它证明该冻结请求集的 image/static-video encoder 与 decoder 图调用链实际执行，不证明一般视觉准确性、
运动理解、MP4/URL 解码或广泛视频能力。

静态视频仍是同一原始 logo 的未改像素重复帧，以显式 fps/帧元数据提交；其目的仅为验证视频解析、temporal
MRoPE/encoder 图调用链。A3 QwenVL 未测。A2 DeepSeek、长上下文、DP/多机、量化、性能和同版本原生
vLLM-Ascend 性能对比均未由本 Run 验收；性能须在已提交 revision 后另作同参数比较。

历史 `qwen3vl-a2-encoder-image-video-v2/runs/image_video/multimodal-001` 的 image/video 观察仍保留，
包括其当时的 encoder graph/decoder replay 统计；这些历史数字不与本轮 visual-002 的 FULL/replay
计数相加，也不替代本轮的 8-request 证据。

实现保持 Ascend vendor 边界：Qwen3-VL 的视觉/DeepStack/MRoPE 和 encoder ACLGraph 位于 Ascend
路径。上游 Qwen3-VL MoE 初始化未保留 inherited encoder 所需的实例 `model_config`，FL 在 Ascend patch
中以幂等、实例范围的 wrapper 修复该协议；Qwen MoE factory 的预导入别名仅在仍绑定确切上游 factory 时
替换，避免覆盖调用方自定义 factory。

本次不修改运行时代码或 native payload，也不构成新的 clean build。证据位于 Task
`fl-vllm024-ascend-migration-20260909` 的 `integrated-qwenvl-a2-post-fused-v2`，以及前置
`qwen3vl-a2-encoder-image-video-v2`、`qwen3vl-current024-visual-closure-v1`、
`qwen3vl-moe-factory-lifecycle-v1` 和 `qwen3vl-moe-encoder-config-closure-v1`。通用部署边界见
[构建、部署与复现](./deployment.md)。
