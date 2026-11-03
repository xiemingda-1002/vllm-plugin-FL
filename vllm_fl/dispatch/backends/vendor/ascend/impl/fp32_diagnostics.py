from collections import Counter
import torch
from .fp32_reference import _STATS
from .fp32_graph import COUNTS
class FP32Diagnostics:
    def fp32_dtype_report(self):
        model = self.get_model()
        parameters = Counter(str(p.dtype) for p in model.parameters() if p.is_floating_point())
        devices = Counter(str(p.device) for p in model.parameters())
        caches = Counter()
        for module in model.modules():
            impl = getattr(module, "impl", None)
            if impl is not None:
                for name in ["key_cache", "value_cache"]:
                    tensor = getattr(impl, name, None)
                    if isinstance(tensor, torch.Tensor):
                        caches[str(tensor.dtype)] += 1
        return {"floating_parameter_dtypes": dict(parameters), "parameter_devices": dict(devices),
                "kv_cache_dtypes": dict(caches), "attention": {
                    **{k: v for k, v in _STATS.items() if k != "dtypes"},
                    "dtypes": sorted(_STATS["dtypes"])},
                "graph_replay_calls": COUNTS['full_decode_replays'],
                "full_decode": COUNTS,
                "backend": "NPU FP32 matmul/softmax with vLLM paged KV cache",
                "matmul_hf32": torch.npu.matmul.allow_hf32}
