# Copyright (c) 2026 BAAI. All rights reserved.

"""Select the complete public Ascend runtime for MiniMax BF16.

This module deliberately has no torch, FlagGems or vendor imports at module scope.
Platform selection precedes ModelConfig construction, so the runtime choice is
explicit and inherited by engine/worker subprocesses, rather than inferred from
a checkpoint path or changed during a forward pass.
"""

import importlib
import importlib.metadata
import inspect
import json
import os
import threading

ENV_NAME = "VLLM_FL_MINIMAX_ASCEND_RUNTIME"
VENDOR_COMMIT = "412cda26814ff70c326f6eb6510f1b610f67bbc0"
PLATFORM_PATH = "vllm_fl.platform_minimax_ascend.MiniMaxAscendPlatform"
WORKER_PATH = "vllm_ascend.worker.worker.NPUWorker"
ARCHITECTURES = frozenset(
    ("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration")
)
GENERAL_HOOKS = (
    "register_connector",
    "register_model_loader",
    "register_service_profiling",
    "register_model",
)
_general_lock = threading.RLock()
_general_state = "new"


def enabled() -> bool:
    value = os.environ.get(ENV_NAME, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENV_NAME} must be '0' or '1', got {value!r}")
    return value == "1"


def require_plugin_filter() -> None:
    # Both distributions expose an OOT platform entry point. Select FL only;
    # its dedicated entry points then load the vendor runtime exactly once.
    if os.environ.get("VLLM_PLUGINS", "").strip() != "fl":
        raise RuntimeError(f"{ENV_NAME}=1 requires VLLM_PLUGINS=fl")


def check_dependencies() -> None:
    """Reject incompatible installations before importing optional vendor code."""
    if _general_state == "failed":
        raise RuntimeError(
            "MiniMax Ascend startup is incomplete; start a fresh process"
        )
    require_plugin_filter()
    try:
        vendor = importlib.metadata.distribution("vllm-ascend")
        vllm_version = importlib.metadata.version("vllm")
        npu_version = importlib.metadata.version("torch-npu")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "MiniMax Ascend runtime requires requirements/minimax-ascend.txt "
            "and the matching Ascend native build; no FL/CUDA fallback is used"
        ) from error
    direct_url = json.loads(vendor.read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit is not None and commit != VENDOR_COMMIT:
        raise RuntimeError(
            f"MiniMax Ascend requires vllm-ascend commit {VENDOR_COMMIT}"
        )
    # Source installs record their exact commit above. Published wheels must
    # identify the corresponding release; a development version is insufficient.
    if commit is None and vendor.version.split("+", 1)[0] != "0.24.0rc1":
        raise RuntimeError("MiniMax Ascend requires vllm-ascend 0.24.0rc1")
    if vllm_version.split("+", 1)[0] != "0.24.0":
        raise RuntimeError("MiniMax Ascend requires vLLM 0.24.0")
    if npu_version.split("+", 1)[0] != "2.10.0.post2":
        raise RuntimeError("MiniMax Ascend requires torch-npu 2.10.0.post2")


def register_general_plugins() -> None:
    """Reuse vendor public startup hooks, including its global patch lifecycle."""
    global _general_state
    with _general_lock:
        if _general_state == "ready":
            return
        if _general_state != "new":
            raise RuntimeError(
                "MiniMax Ascend startup is incomplete; start a fresh process "
                "instead of retrying a partially initialized vendor runtime"
            )
        try:
            check_dependencies()
            vendor = importlib.import_module("vllm_ascend")
            # Preflight all hooks before invoking any of them. A failure still
            # poisons this selected runtime, even if vLLM logs and swallows it.
            hooks = []
            for name in GENERAL_HOOKS:
                hook = getattr(vendor, name, None)
                if not callable(hook):
                    raise RuntimeError(
                        f"vllm-ascend is missing public startup hook {name}"
                    )
                inspect.signature(hook).bind()
                hooks.append(hook)
            _general_state = "starting"
            for hook in hooks:
                hook()
        except BaseException:
            # Vendor registrations/patches are process-wide. Do not claim to
            # undo that external lifecycle or re-enter it after partial failure.
            _general_state = "failed"
            raise
        _general_state = "ready"


def validate_config(config) -> None:
    """Validate support boundaries before the vendor updates the configuration."""
    model = config.model_config
    if model is None:
        raise ValueError("MiniMax Ascend runtime requires a model configuration")
    hf_config = model.hf_config
    architectures = set(getattr(hf_config, "architectures", None) or ())
    if not architectures or not architectures <= ARCHITECTURES:
        raise ValueError("MiniMax Ascend runtime accepts only MiniMax-M3 architectures")
    if str(model.dtype) not in {"torch.bfloat16", "bfloat16"}:
        raise ValueError("MiniMax Ascend runtime requires BF16 model dtype")
    text_config = getattr(hf_config, "text_config", None)
    if model.quantization is not None or any(
        getattr(part, "quantization_config", None)
        for part in (hf_config, text_config)
        if part is not None
    ):
        raise ValueError(
            "MiniMax Ascend BF16 runtime does not accept quantized checkpoints"
        )
    if config.cache_config.cache_dtype != "auto":
        raise ValueError("MiniMax Ascend BF16 runtime requires kv_cache_dtype='auto'")
    device = config.device_config
    if device is not None and device.device_type != "npu":
        raise ValueError("MiniMax Ascend runtime requires an NPU device")
    if config.parallel_config.worker_cls not in {"auto", WORKER_PATH}:
        raise ValueError("MiniMax Ascend runtime requires the vendor NPUWorker")
    parallel = config.parallel_config
    if (
        parallel.tensor_parallel_size != 8
        or parallel.data_parallel_size != 2
        or not parallel.enable_expert_parallel
    ):
        raise ValueError(
            "MiniMax Ascend BF16 runtime currently requires the vendor A3 recipe: "
            "tensor_parallel_size=8, data_parallel_size=2, enable_expert_parallel=True"
        )
