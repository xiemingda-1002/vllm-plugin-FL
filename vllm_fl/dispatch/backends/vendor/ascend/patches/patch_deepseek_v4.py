"""Vendor-scoped DeepSeek-V4 registration and fail-closed feature gates."""
def apply_deepseek_v4_patches() -> None:
    from vllm import ModelRegistry
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_fl.models.deepseek_v4:AscendDeepseekV4ForCausalLM")
