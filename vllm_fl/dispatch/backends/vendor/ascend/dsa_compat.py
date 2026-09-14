"""FL-owned rc1 DSA feature helpers and explicit migration boundaries."""
from __future__ import annotations

from functools import wraps

import torch
import torch_npu

from .hardware import AscendDeviceType, get_ascend_device_type


def get_ascend_config():
    from .impl.moe.compat import get_ascend_config as _get_ascend_config

    return _get_ascend_config()


def enable_dsa_cp() -> bool:
    try:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
    except AssertionError:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is None or not hasattr(text_config, "index_topk"):
        return False
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if bool(additional_config.get("enable_dsa_cp", False)):
        raise NotImplementedError(
            "FL Ascend DeepSeek V4 DSA-CP is not migrated yet; "
            "remove additional_config.enable_dsa_cp until the context-parallel "
            "execution chain is installed"
        )
    return False


def get_dsv4_compress_ratio(config, layer_idx: int) -> int:
    compress_ratios = getattr(config, "compress_ratios", None)
    if compress_ratios is None or layer_idx >= len(compress_ratios):
        return 0
    return compress_ratios[layer_idx]


def extract_dsv4_layer_index(config, prefix: str) -> int:
    from vllm.model_executor.models.utils import extract_layer_index

    layer_idx = extract_layer_index(prefix)
    if ".mtp." in f".{prefix}." and layer_idx < config.num_hidden_layers:
        return config.num_hidden_layers + layer_idx
    return layer_idx


def get_potential_max_tokens() -> int:
    from vllm.config import get_current_vllm_config
    return get_current_vllm_config().scheduler_config.max_num_batched_tokens


def is_pd_decode_recompute_scheduler_enabled() -> bool:
    try:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
    except AssertionError:
        return False
    kv_config = getattr(vllm_config, "kv_transfer_config", None)
    is_decode_consumer = bool(
        kv_config is not None
        and getattr(kv_config, "is_kv_consumer", False)
        and not getattr(kv_config, "is_kv_producer", False)
    )
    if is_decode_consumer and get_ascend_config().recompute_scheduler_enable:
        raise NotImplementedError(
            "FL Ascend PD decode recompute scheduling is not migrated"
        )
    return False


def npu_stream_switch(*args, **kwargs):
    from .impl.moe.compat import npu_stream_switch as _npu_stream_switch

    return _npu_stream_switch(*args, **kwargs)


def olora_tp_enable() -> bool:
    enabled = get_ascend_config().finegrained_tp_config.olora_tensor_parallel_size > 1
    if enabled:
        raise NotImplementedError("FL Ascend OLoRA tensor parallelism is not migrated")
    return False


def oproj_tp_enable() -> bool:
    enabled = get_ascend_config().finegrained_tp_config.oproj_tensor_parallel_size > 0
    if enabled:
        raise NotImplementedError("FL Ascend OProj tensor parallelism is not migrated")
    return False


def enable_sp() -> bool:
    from .impl.moe.compat import enable_sp as _enable_sp

    return _enable_sp()


def is_310p() -> bool:
    return get_ascend_device_type() is AscendDeviceType._310P


def maybe_trans_nz(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.float32 or weight.is_meta:
        return weight
    if is_310p():
        return torch_npu.npu_format_cast(weight, 29)
    nz_mode = get_ascend_config().weight_nz_mode
    if not nz_mode:
        return weight
    if weight.dtype in {torch.bfloat16, torch.float16} and nz_mode != 2:
        return weight
    return torch_npu.npu_format_cast(weight, 29)


def singleton(cls):
    instances = {}

    @wraps(cls)
    def factory(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return factory
