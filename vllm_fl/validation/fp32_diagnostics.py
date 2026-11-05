# Copyright (c) 2026 BAAI. All rights reserved.

"""Optional worker diagnostics; enabled only by an explicit worker extension.

Counters and replay instrumentation live here rather than in numerical kernels.
The replay count records successful NPUGraph.replay calls, not capture attempts.
"""

from collections import Counter
from functools import wraps

import torch

from vllm_fl.dispatch.backends.vendor.ascend.impl import fp32_graph, fp32_reference

_STATS = Counter()
_DTYPES = set()


def _observe(tensors, metadata, *, full=False):
    _STATS["calls"] += 1
    lengths = metadata.seq_lens_list
    _STATS["max_sequences"] = max(_STATS["max_sequences"], len(lengths))
    _STATS["batched_calls"] += int(len(lengths) > 1)
    _STATS["full_attention_captures"] += int(full)
    _DTYPES.update(str(t.dtype) for t in tensors)


fp32_reference._observer = _observe
fp32_graph._observer = _observe


def _install_replay_observer():
    import torch_npu  # noqa: F401

    original = torch.npu.NPUGraph.replay
    if getattr(original, "_fl_validation_observer", False):
        return

    @wraps(original)
    def replay(graph, *args, **kwargs):
        result = original(graph, *args, **kwargs)
        _STATS["full_decode_replays"] += 1
        return result

    replay._fl_validation_observer = True
    torch.npu.NPUGraph.replay = replay


_install_replay_observer()


class FP32Diagnostics:
    """Explicit validation RPC extension, retained for CosyVoice clients."""

    def fp32_dtype_report(self):
        model = self.get_model()
        parameters = Counter(
            str(p.dtype) for p in model.parameters() if p.is_floating_point()
        )
        devices = Counter(str(p.device) for p in model.parameters())
        caches = Counter()
        for module in model.modules():
            impl = getattr(module, "impl", None)
            if impl is not None:
                for name in ("key_cache", "value_cache"):
                    tensor = getattr(impl, name, None)
                    if isinstance(tensor, torch.Tensor):
                        caches[str(tensor.dtype)] += 1
        return {
            "floating_parameter_dtypes": dict(parameters),
            "parameter_devices": dict(devices),
            "kv_cache_dtypes": dict(caches),
            "attention": {**dict(_STATS), "dtypes": sorted(_DTYPES)},
            "graph_replay_calls": _STATS["full_decode_replays"],
            "full_decode": {
                name: _STATS[name]
                for name in ("full_decode_replays", "full_attention_captures")
            },
            "backend": "NPU FP32 matmul/softmax with vLLM paged KV cache",
            "matmul_hf32": torch.npu.matmul.allow_hf32,
        }
