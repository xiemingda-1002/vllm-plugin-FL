"""FL Ascend DeepSeek-V4 non-MTP model entry point."""

from vllm_fl.models.deepseek_v4_ascend import (
    AscendDeepseekV4ForCausalLM,
)

__all__ = ["AscendDeepseekV4ForCausalLM"]
