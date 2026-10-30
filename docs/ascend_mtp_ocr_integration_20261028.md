# Ascend MTP 与 Unlimited-OCR 重建移植

- upstream 基线：`7e68bd246bffa3be5d77e56c556a9876e61ba04b`。
- MTP 来源：`8e72b5d9d7e4bef0c4f2a7965f30de4c77c68bdb`；只移植 MTP 所需改动。
- OCR 来源：`d4f94d5e30773e85fd8a1c43349bae07c7d02a77`，提取相对上述 MTP 来源的 OCR 增量。

代码保留 upstream pluggable runtime、dispatch 和编译架构；Ascend attention 适配位于 `vllm_fl/dispatch/backends/vendor/ascend/impl/attention/`。OCR 增量包括 R-SWA mask 和 prompt 元数据、FULL Decode padding、捕获缓冲区原位更新、可选 graph replay 日志、CPU 回归测试及示例工具。FULL padding 特例限于单 token decode，不改变 MTP 多 token decode 的原有 padding 规则。未叠加旧 fork 的其他架构改动。

vLLM core 的 18 文件补丁单独位于 `patches/unlimited_ocr/`，必须依其 README 对 `ee0da84ab9e04ac7610e28580af62c365e898389` 单独检查及应用；FL 源码本身不包含这些 core 修改。manifest 中 FL 基线为本次 upstream 基线，OCR 来源另行记录；core SHA256 保持来源交付物一致。

本次组合移植仅进行 AST、diff 和可运行的 CPU stub 检查。当前主机无 torch，不执行需要 torch 的 OCR mask 测试。未进行 NPU、远端、模型前向、完整精度或性能测试。OCR 示例中保留的 Eager 全量及 FULL 短请求指标是历史来源记录，不作为此次移植验收；MTP 与 OCR 的组合运行仍须后续设备验证。
