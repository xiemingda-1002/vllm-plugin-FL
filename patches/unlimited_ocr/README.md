# vLLM 0.24 Unlimited-OCR core 补丁

`vllm-0.24-unlimited-ocr.patch` 将 Unlimited-OCR 所需的 18 个 core 文件回迁到 vLLM 提交 `ee0da84ab9e04ac7610e28580af62c365e898389`。FL 基线为 `7e68bd246bffa3be5d77e56c556a9876e61ba04b`；5 个 FL 运行时文件已直接纳入本分支，不需要再应用 FL 补丁。

本补丁包含 native 模型/config/registry、ModelScope alias、32-crop processor、动态图像尺寸、R-SWA metadata/spec、DeepSeek-V2 MHA 接入及 tokenizer ByteLevel 修复。`unlimited-ocr` 已加入错误 tokenizer_class 的白名单，避免模型配置中的 Llama 类覆盖 tokenizer.json 的原生编码解码图。历史 tokenizer 增量已合并，不能再次叠加。

`max_crops` 默认为 32。服务端或请求级覆盖时，占位 token 计数使用实际已解析 processor 的 crop 上限；profiling 的最大特征尺寸也使用同一配置，不再固定为 4×8 裁图。

平台校验要求 Ascend NPU：FL 使用 `vendor_name=ascend` 和 `device_type=npu`；原生 vLLM-Ascend 使用 NPU 与 `ASCEND_RT_VISIBLE_DEVICES` 平台标识。其他 OOT vendor 会在配置阶段被拒绝，OOT 本身不代表支持 R-SWA。

本次移植仅完成静态/CPU 检查，尚无 NPU 验证；历史设备环境和指标不代表本次重建代码通过设备验收。

## 检查、应用与验证

在 FL 仓库根目录执行，参数为独立的 vLLM 源码目录：

```bash
python3 patches/unlimited_ocr/apply_vllm_patch.py /path/to/vllm --check
python3 patches/unlimited_ocr/apply_vllm_patch.py /path/to/vllm
```

工具严格检查 vLLM 基线提交，并依据 `manifest.json` 校验修改前后的 SHA256。遇到基线或文件哈希不匹配时，应先查明源码差异，不能强制覆盖现有修改。补丁只改变 Python 源码，不升级依赖或安装模型。运行环境必须实际加载该目录中的 core，例如通过既有 editable 安装和启动脚本的 `VLLM_SOURCE_DIR`。

如果已经应用旧版 backport，升级前先停止使用该 core 的服务，并使用旧版工具、manifest 和 patch 执行 `--reverse`；随后再用新版工具执行 `--check` 和应用。新版工具不会自动覆盖旧版已应用状态。

OCR 来源版本的历史验证环境为 Python 3.12.13、Torch 2.10.0+cpu、torch-npu 2.10.0.post2、Transformers 5.13.0、torchvision 0.25.0+cpu、CANN/ATB 9.0.1。已有编译扩展只有在 ABI 一致时才可复用；新环境需依照各仓库正常流程构建和安装原生扩展。模型权重、数据集、第三方 evaluator 和二进制不随补丁提供。

启动、HTTP 请求、CPU 官方评分和验收限制见 [Unlimited-OCR 示例](../../examples/unlimited_ocr/README.md)。CPU/静态检查不代表模型前向、完整精度或性能验收；FULL 模式仅为 `FULL_DECODE_ONLY`，视觉编码及 Prefill 不启用 FULL。R-SWA 保留整个 prompt/image 前缀和 128 个生成历史窗口，兼容实现仍保存完整 KV。

## 回退

先停止所有加载该 core 目录的 OCR 服务，再执行：

```bash
python3 patches/unlimited_ocr/apply_vllm_patch.py /path/to/vllm --reverse
```

回退同样校验文件哈希，拒绝覆盖补丁应用后的其他编辑，并恢复修改前文件、移除补丁新增文件。回退不会停止服务、切换 FL 版本、修改权重或清理运行结果。撤销整套适配时，还需将 FL 源码切回不含此次适配的版本，并停止使用 Unlimited-OCR 示例入口。

本目录的 patch 和 manifest 是外部 core 依赖的交付物。原始源码快照、安装 payload、文件备份、测试日志、模型输出、镜像和缓存留在各自部署环境中，不纳入本次源码发布。
