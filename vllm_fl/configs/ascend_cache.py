"""Ascend cache policy, kept separate from patch registration."""

import logging


logger = logging.getLogger(__name__)


def refresh_block_size(vllm_config, block_size=128):
    """
    Refresh the block size in cache config.
    """
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    if not cache_config:
        return

    if cache_config.block_size is None:
        cache_config.block_size = block_size

    if not scheduler_config or not model_config:
        return

    if model_config.hf_config.model_type == "deepseek_v4":
        if cache_config.block_size not in (32, 64, 128):
            logger.warning(
                "For deepseek_v4 model, block size should be 32, 64 or "
                "128. Setting block size to 32 for better performance."
            )
            cache_config.block_size = 32
        return

    if model_config.is_hybrid:
        # Hybrid attention+Mamba models have already selected a block size
        # from their state/page geometry. Resetting it here makes the worker
        # disagree with the engine and breaks the contiguous Ascend layout.
        return

    if cache_config.block_size != block_size and (
        cache_config.enable_prefix_caching
        or scheduler_config.enable_chunked_prefill
    ):
        logger.info(
            "Block size is set to %d if prefix cache or chunked prefill "
            "is enabled.",
            block_size,
        )
        cache_config.block_size = block_size
