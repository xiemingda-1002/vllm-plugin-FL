"""FL Ascend DeepSeek-V4 non-MTP model entry point."""

from vllm_fl.dispatch.backends.vendor.ascend.models.deepseek_v4 import (
    AscendDeepseekV4ForCausalLM,
)

__all__ = ["AscendDeepseekV4ForCausalLM"]
