# Unlimited-OCR runtime scope

Unlimited-OCR uses FL dispatch and PyTorch/torch-npu device operations on Ascend NPU. The strict OCR mode rejects vllm-ascend imports, unsupported quantization, speculative decoding and unsupported sampling options. It retains the vLLM Python serving, scheduling, multimodal processing and model interfaces.

The required external core patch is documented in [patches/unlimited_ocr](patches/unlimited_ocr/README.md). It targets vLLM commit `ee0da84ab9e04ac7610e28580af62c365e898389`; installing FL alone does not add the core model and metadata interfaces.

The OCR runtime bypasses the generic ATB RoPE path and optional FLA/Mamba imports. Drivers, CANN, HCCL, PyTorch, torch-npu, Transformers, tokenizers and image-processing dependencies remain required. Other FL models and backends are outside this dependency claim.

Defaults enable fused RoPE, SAM attention, batched NGram updates, tensor slot mapping and grouped R-SWA construction. Explicit environment overrides retain comparison paths. R-SWA uses a 2048 MiB mask budget in strict OCR mode; unrelated models retain their existing defaults. See [examples/unlimited_ocr](examples/unlimited_ocr/README.md) for startup, accuracy and performance commands.

CPU contracts and small NPU checks validate individual changes. Eager dispatch auditing cannot inspect opaque operator internals or graph replay; it is supporting evidence, not proof of every internal kernel. Full accuracy and performance results depend on the model, dataset, software versions and hardware.
