# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _minimal_vllm_config(*, dtype=torch.bfloat16, quant_config=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=dtype,
            hf_text_config=SimpleNamespace(routed_scaling_factor=1.0),
            use_mla=False,
            get_head_size=lambda: 128,
        ),
        device_config=SimpleNamespace(device="npu"),
        compilation_config=SimpleNamespace(
            splitting_ops=[],
            use_inductor_graph_partition=False,
            pass_config=SimpleNamespace(),
        ),
        quant_config=quant_config,
    )


def test_base_pattern_registers_each_local_pass_and_dedupes_npugraph(
    monkeypatch,
) -> None:
    import torch._inductor.pattern_matcher as pm
    import vllm_fl.compilation.passes.base_pattern as base_module

    local_events = []
    nge_events = []
    monkeypatch.setattr(
        pm,
        "register_replacement",
        lambda *args, **kwargs: local_events.append((args, kwargs)),
    )
    nge = ModuleType("npugraph_ex")
    nge.register_replacement = lambda **kwargs: nge_events.append(kwargs)
    monkeypatch.setitem(sys.modules, "npugraph_ex", nge)
    base_module._registered_npugraph_patterns.clear()

    class Pattern(base_module.BasePattern):
        def __init__(self, config, scale, rope_dim=128):
            super().__init__(config)
            self.scale = scale
            self.rope_dim = rope_dim

        def get_inputs(self):
            return [torch.zeros(1)]

        def get_pattern(self):
            return lambda value: value * self.scale

        def get_replacement(self):
            return lambda value: value * self.scale

    first_pm = object()
    second_pm = object()
    Pattern(_minimal_vllm_config(), 1.0).register(first_pm)
    Pattern(_minimal_vllm_config(), 1.0).register(second_pm)
    Pattern(_minimal_vllm_config(), 2.0).register(second_pm)
    Pattern(_minimal_vllm_config(), 2.0, rope_dim=64).register(second_pm)
    assert len(local_events) == 4
    assert len(nge_events) == 3


def test_stream_guard_rejects_cross_stream_matches() -> None:
    from vllm_fl.compilation.passes.utils.npugraph_ex_utils_check import (
        extra_stream_scope_check,
    )

    node = lambda stream: SimpleNamespace(  # noqa: E731
        op="call_function", meta={"stream_label": stream}
    )
    assert extra_stream_scope_check(SimpleNamespace(nodes=[node(None), node(None)]))
    assert not extra_stream_scope_check(
        SimpleNamespace(nodes=[node(None), node("decode")])
    )
    assert not extra_stream_scope_check(
        SimpleNamespace(nodes=[node("a"), node("b")])
    )


def test_norm_quant_pass_registers_dynamic_patterns_without_fl_custom_op(
    monkeypatch,
) -> None:
    import vllm_fl.compilation.passes.norm_quant_fusion_pass as norm_module

    events = []
    registered = []

    class PatternMatcher:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def apply(self, graph):
            events.append(("apply", graph))
            return 0

    monkeypatch.setattr(norm_module, "PatternMatcherPass", PatternMatcher)
    monkeypatch.setattr(
        norm_module.BasePattern,
        "register",
        lambda self, _pm_pass: registered.append(self.__class__.__name__),
    )
    monkeypatch.setattr(
        norm_module,
        "is_add_rms_norm_dynamic_mx_quant_fusion_available",
        lambda: False,
    )
    monkeypatch.setattr(
        norm_module,
        "is_rms_norm_dynamic_mx_quant_fusion_available",
        lambda: False,
    )
    monkeypatch.setattr(norm_module, "_static_norm_quant_patterns_available", lambda: False)
    norm_module.AddRMSNormQuantFusionPass(_minimal_vllm_config())
    assert events == [("init", {"pass_name": "rmsnorm_quant_fusion_pass"})]
    assert registered == [
        "AddRMSNormDynamicQuantPattern",
        "AddRMSNormDynamicQuantSPPattern",
        "AddRMSNormDynamicQuantPattern",
        "AddRMSNormDynamicQuantSPPattern",
    ]


def test_norm_quant_pass_keeps_mx_and_static_branches_capability_gated(
    monkeypatch,
) -> None:
    import vllm_fl.compilation.passes.norm_quant_fusion_pass as norm_module

    registered = []

    class PatternMatcher:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(norm_module, "PatternMatcherPass", PatternMatcher)
    monkeypatch.setattr(
        norm_module.BasePattern,
        "register",
        lambda self, _pm_pass: registered.append(self.__class__.__name__),
    )
    monkeypatch.setattr(
        norm_module,
        "is_add_rms_norm_dynamic_mx_quant_fusion_available",
        lambda: True,
    )
    monkeypatch.setattr(
        norm_module,
        "is_rms_norm_dynamic_mx_quant_fusion_available",
        lambda: True,
    )
    monkeypatch.setattr(norm_module, "_static_norm_quant_patterns_available", lambda: True)
    norm_module.AddRMSNormQuantFusionPass(_minimal_vllm_config())

    assert registered.count("AddRMSNormDynamicQuantPattern") == 2
    assert registered.count("AddRMSNormDynamicMXQuantPattern") == 2
    assert registered.count("RMSNormDynamicMXQuantPattern") == 2
    # Static producer-gated patterns include static quant and dynamic+bias.
    assert registered.count("AddRMSNormQuantPattern") == 2
    assert registered.count("AddRMSNormDynamicQuantPatternWithBias") == 2


def test_norm_quant_pass_rejects_w4a4_and_unsupported_dtype(monkeypatch) -> None:
    import vllm_fl.compilation.passes.norm_quant_fusion_pass as norm_module

    registered = []

    class PatternMatcher:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(norm_module, "PatternMatcherPass", PatternMatcher)
    monkeypatch.setattr(
        norm_module.BasePattern,
        "register",
        lambda self, _pm_pass: registered.append(self.__class__.__name__),
    )
    w4a4 = SimpleNamespace(quant_description={"layer": "W4A4_DYNAMIC"})
    norm_module.AddRMSNormQuantFusionPass(
        _minimal_vllm_config(quant_config=w4a4)
    )
    norm_module.AddRMSNormQuantFusionPass(_minimal_vllm_config(dtype=torch.float32))
    assert registered == []


def test_static_norm_quant_gate_requires_the_actual_fl_producer(monkeypatch) -> None:
    import vllm_fl.compilation.passes.norm_quant_fusion_pass as norm_module

    monkeypatch.setattr(norm_module, "enable_custom_op", lambda: False)
    assert not norm_module._static_norm_quant_patterns_available()

    monkeypatch.setattr(norm_module, "enable_custom_op", lambda: True)
    monkeypatch.setattr(
        norm_module.torch,
        "ops",
        SimpleNamespace(
            _C_ascend=SimpleNamespace(npu_add_rms_norm_bias=object()),
            vllm=SimpleNamespace(),
        ),
    )
    assert not norm_module._static_norm_quant_patterns_available()

    monkeypatch.setattr(
        norm_module.torch,
        "ops",
        SimpleNamespace(
            _C_ascend=SimpleNamespace(npu_add_rms_norm_bias=object()),
            vllm=SimpleNamespace(quantize=object()),
        ),
    )
    assert norm_module._static_norm_quant_patterns_available()


def test_qknorm_registers_bias_and_no_bias_patterns_for_bf16_head128(
    monkeypatch,
) -> None:
    import vllm_fl.compilation.passes.base_pattern as base_module
    import vllm_fl.compilation.passes.qknorm_rope_fusion_pass as qk_module
    import vllm_fl.dispatch.backends.vendor.ascend.impl.graph_fusion_ops as ops

    records = []
    monkeypatch.setattr(ops, "ensure_graph_fusion_ops_registered", lambda: None)
    monkeypatch.setattr(
        qk_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {
            "layer": SimpleNamespace(
                head_size=128, num_heads=8, num_kv_heads=2
            )
        },
    )
    monkeypatch.setattr(
        base_module.BasePattern,
        "register",
        lambda self, pm_pass: records.append(
            (self.__class__.__name__, self.eps, self.head_dim)
        ),
    )
    qk_module.QKNormRopeFusionPass(_minimal_vllm_config())
    assert records == [
        ("QKNormRopeFusionPattern", 1e-6, 128),
        ("QKNormRopeFusionPatternWithBias", 1e-6, 128),
        ("QKNormRopeFusionPattern", 1e-5, 128),
        ("QKNormRopeFusionPatternWithBias", 1e-5, 128),
    ]

    records.clear()
    qk_module.QKNormRopeFusionPass(
        _minimal_vllm_config(dtype=torch.float16)
    )
    assert records == []


def test_qkv_fake_shapes_and_rotary_fake_do_not_alias() -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.impl.canonical_rotary import (
        _npu_rotary_embedding_fake,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.linearnorm.split_qkv_rmsnorm_rope import (
        split_qkv_rmsnorm_rope_impl_fake,
    )

    qkv = torch.empty(3, 1536, dtype=torch.bfloat16)
    q, k, v = split_qkv_rmsnorm_rope_impl_fake(
        qkv,
        torch.empty(16, 128),
        torch.arange(3),
        torch.empty(128),
        torch.empty(128),
        1024,
        256,
        128,
        1e-6,
    )
    assert (q.shape, k.shape, v.shape) == ((3, 1024), (3, 256), (3, 256))
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16

    query = torch.empty(3, 1024)
    key = torch.empty(3, 256)
    query_out, key_out = _npu_rotary_embedding_fake(
        torch.arange(3), query, key, torch.empty(16, 128), 128, 128, True
    )
    assert query_out.shape == query.shape and key_out.shape == key.shape
    assert query_out.data_ptr() != query.data_ptr()
    assert key_out.data_ptr() != key.data_ptr()
