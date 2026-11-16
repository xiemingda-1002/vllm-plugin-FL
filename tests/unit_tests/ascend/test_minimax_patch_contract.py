# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU wiring tests: these do not establish Ascend numerical correctness."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3] / "vllm_fl"
VENDOR = "dispatch/backends/vendor/ascend"


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def entry(monkeypatch):
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        vendor_name="ascend", device_type="npu"
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    module = load_file(f"{VENDOR}/patches/patch_minimax_m3.py", "m3_patch_test")
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    return module, config, platforms.current_platform


@pytest.mark.parametrize(
    "vendor,architectures",
    [
        ("cuda", ["MiniMaxM3SparseForCausalLM"]),
        ("other_npu", ["MiniMaxM3SparseForCausalLM"]),
        ("ascend", ["Qwen3ForCausalLM"]),
        ("ascend", None),
    ],
)
def test_non_target_does_not_load_dependencies(
    entry, monkeypatch, vendor, architectures
):
    module, config, platform = entry
    platform.vendor_name = vendor
    config.model_config.hf_config.architectures = architectures

    def unexpected():
        raise AssertionError("loaded MiniMax dependencies for a non-target model")

    monkeypatch.setattr(module, "_prepare_patches", unexpected)
    before = dict(sys.modules)
    assert module.apply_minimax_m3_patches(config) is False
    assert dict(sys.modules) == before


def installers(module, monkeypatch, failure=None):
    owner = ModuleType("upstream_m3")
    original = owner.layer = object()
    calls = []
    modules = {}
    names = [
        (
            "vllm_fl.models.minimax_m3_ascend",
            ["_install_official_gemma_rmsnorm", "install_ascend_minimax_m3_model"],
        ),
        (f"{module._VENDOR}.minimax_m3_injections", ["install_minimax_m3_injections"]),
        (
            f"{module._VENDOR}.indexer.index_topk_ascend",
            ["install_index_topk_replacement"],
        ),
        (
            f"{module._VENDOR}.attention.sparse_attn_m3",
            ["install_sparse_attn_replacement"],
        ),
        (
            f"{module._VENDOR}.minimax_m3_graph_replay",
            ["install_minimax_m3_graph_conformance"],
        ),
    ]
    for path, methods in names:
        target = modules[path] = ModuleType(path)
        target._installed = False
        target._register_ops = lambda: None
        for method in methods:

            def install(method=method, target=target):
                calls.append(method)
                owner.layer = object()
                owner.new_wrapper = object()
                target._installed = True
                if method == failure:
                    raise ValueError("original installer error")
                return True

            install.__name__ = method
            setattr(target, method, install)
    monkeypatch.setattr(
        module, "_prepare_patches", lambda: (modules, [owner, *modules.values()])
    )
    return modules, owner, original, calls


@pytest.mark.parametrize(
    "failure",
    [
        "_install_official_gemma_rmsnorm",
        "install_minimax_m3_injections",
        "install_index_topk_replacement",
        "install_sparse_attn_replacement",
        "install_minimax_m3_graph_conformance",
        "install_ascend_minimax_m3_model",
    ],
)
def test_real_entry_rolls_back_each_failure_and_can_retry(entry, monkeypatch, failure):
    module, config, _ = entry
    modules, owner, original, calls = installers(module, monkeypatch, failure)
    with pytest.raises(ValueError, match="original installer error"):
        module.apply_minimax_m3_patches(config)
    assert owner.layer is original
    assert not hasattr(owner, "new_wrapper")
    assert not module._installed
    assert all(not target._installed for target in modules.values())
    assert calls.count(failure) == 1
    # A later successful attempt must not be suppressed by a cached failure.
    _, _, _, retry_calls = installers(module, monkeypatch)
    assert module.apply_minimax_m3_patches(config)
    count = len(retry_calls)
    assert module.apply_minimax_m3_patches(config)
    assert len(retry_calls) == count


def test_false_required_adapter_is_not_cached(entry, monkeypatch):
    module, config, _ = entry
    modules, owner, original, _ = installers(module, monkeypatch)
    model = modules["vllm_fl.models.minimax_m3_ascend"]
    model._install_official_gemma_rmsnorm = lambda: False
    with pytest.raises(RuntimeError, match="required adapter failed"):
        module.apply_minimax_m3_patches(config)
    assert owner.layer is original
    assert not module._installed


@pytest.mark.parametrize("value", ["fp8", "fp8_e4m3", "int8"])
def test_cache_dtype_rejected_before_dependency_loading(entry, monkeypatch, value):
    module, config, _ = entry
    config.cache_config.cache_dtype = value
    monkeypatch.setattr(
        module, "_prepare_patches", lambda: pytest.fail("loaded dependencies")
    )
    with pytest.raises(ValueError, match="cache_dtype='auto'"):
        module.apply_minimax_m3_patches(config)


def test_qkv_dtype_rejected_before_any_write():
    module = load_file(
        f"{VENDOR}/impl/linearnorm/minimax_m3_qknorm_rope.py", "m3_qkv_test"
    )
    qkv = torch.randn(2, 12)
    cache = torch.randn(2, 2, 4, 1, 4)
    original_qkv, original_cache = qkv.clone(), cache.clone()
    with pytest.raises(ValueError, match="kv_cache_dtype='auto'"):
        module.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            torch.ones(4),
            torch.ones(4),
            torch.zeros(4, 4),
            torch.arange(2),
            1,
            1,
            4,
            1e-6,
            kv_cache=cache,
            kv_cache_dtype="fp8",
        )
    torch.testing.assert_close(qkv, original_qkv)
    torch.testing.assert_close(cache, original_cache)


def test_spawn_worker_installs_using_received_config(monkeypatch):
    path = ROOT / "worker/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constructor = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    block = next(
        node
        for node in constructor.body
        if isinstance(node, ast.If) and "apply_minimax_m3_patches" in ast.unparse(node)
    )
    patch_module = ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.patches.patch_minimax_m3"
    )
    calls = []
    patch_module.apply_minimax_m3_patches = calls.append
    monkeypatch.setitem(sys.modules, patch_module.__name__, patch_module)
    received_config = object()
    namespace = {
        "current_platform": SimpleNamespace(vendor_name="ascend"),
        "vllm_config": received_config,
    }
    exec(
        compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), namespace
    )
    assert calls == [received_config]
    namespace["current_platform"].vendor_name = "cuda"
    exec(
        compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), namespace
    )
    assert calls == [received_config]


def test_local_adapters_preserve_global_layers_and_dependencies(monkeypatch):
    upstream = ModuleType("vllm.models.minimax_m3.nvidia.model")
    vision = ModuleType("vllm.models.minimax_m3.common.vision_tower")
    activation = ModuleType("vllm.model_executor.layers.activation")
    rotary = ModuleType("vllm.model_executor.layers.rotary_embedding.common")

    class Activation:
        pass

    class Rotary:
        pass

    activation.SiluAndMulWithClamp = upstream.SiluAndMulWithClamp = Activation
    rotary.ApplyRotaryEmb = vision.ApplyRotaryEmb = Rotary
    nvidia = ModuleType("vllm.models.minimax_m3.nvidia")
    nvidia.model = upstream
    common = ModuleType("vllm.models.minimax_m3.common")
    common.vision_tower = vision
    for module in (nvidia, common, upstream, vision, activation, rotary):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    # A real dependency must remain untouched, rather than being shadowed.
    flashinfer = ModuleType("flashinfer.norm")
    monkeypatch.setitem(sys.modules, "flashinfer.norm", flashinfer)
    injections = load_file(
        f"{VENDOR}/impl/minimax_m3_injections.py", "m3_injections_test"
    )
    rotary_adapter = load_file(f"{VENDOR}/impl/minimax_m3_rotary.py", "m3_rotary_test")
    before = dict(sys.modules)
    assert injections._install_activation_injection()
    assert rotary_adapter.install_apply_rotary_emb_override()
    assert upstream.SiluAndMulWithClamp is not Activation
    assert issubclass(upstream.SiluAndMulWithClamp, Activation)
    assert vision.ApplyRotaryEmb is rotary_adapter.AscendApplyRotaryEmb
    assert activation.SiluAndMulWithClamp is Activation
    assert rotary.ApplyRotaryEmb is Rotary
    assert dict(sys.modules) == before


def test_actual_preparation_captures_classes_and_copied_consumers(entry, monkeypatch):
    module, config, _ = entry
    implementations, _, _, _ = installers(module, monkeypatch)
    # Restore the actual preparation code rather than testing only a fake list.
    real_entry = load_file(
        f"{VENDOR}/patches/patch_minimax_m3.py", "m3_real_prepare_test"
    )
    monkeypatch.setattr(module, "_prepare_patches", real_entry._prepare_patches)

    def norm_init(self, hidden_size, eps=1e-6):
        pass

    def activation_init(self, swiglu_limit, alpha=1.0, beta=0.0):
        pass

    def rotary_init(
        self, enforce_enable=False, is_neox_style=True, enable_fp32_compute=False
    ):
        pass

    def original_forward(self, layer, query, kv_cache, output):
        pass

    classes = {
        "MiniMaxM3SparseBackend": type(
            "Backend", (), {"get_kv_cache_shape": staticmethod(lambda: "original")}
        ),
        "MiniMaxM3SparseImpl": type("SparseImpl", (), {}),
        "MiniMaxM3SparseTritonImpl": type(
            "TritonImpl", (), {"forward": original_forward}
        ),
        "MiniMaxM3IndexerImpl": type("IndexerImpl", (), {}),
    }
    qkv = load_file(
        f"{VENDOR}/impl/linearnorm/minimax_m3_qknorm_rope.py", "m3_prepare_qkv_test"
    )
    upstream = ModuleType("vllm.models.minimax_m3.nvidia.model")
    upstream.__dict__.update(
        {
            "MiniMAXGemmaRMSNorm": type("Norm", (), {}),
            "SiluAndMulWithClamp": type(
                "Activation", (), {"__init__": activation_init}
            ),
            "fused_allreduce_gemma_rms_norm": lambda *args: None,
            "MiniMaxM3Attention": type("Attention", (), {}),
            "MiniMaxM3SparseAttention": type("SparseAttention", (), {}),
            "MiniMaxM3Model": type("Model", (), {}),
        }
    )

    def get_module(name):
        if name in implementations:
            return implementations[name]
        target = ModuleType(name)
        target.__dict__.update(classes)
        target.fused_minimax_m3_qknorm_rope_kv_insert = (
            qkv.fused_minimax_m3_qknorm_rope_kv_insert
        )
        target.ApplyRotaryEmb = type("Rotary", (), {"__init__": rotary_init})
        target.GemmaRMSNorm = type("Norm", (), {"__init__": norm_init})
        target.get_minimax_sparse_kernels = lambda: SimpleNamespace(
            qknorm_rope_cache=qkv.fused_minimax_m3_qknorm_rope_kv_insert,
            rotary=lambda *args: None,
            gemma_weight=lambda *args: None,
            attend=lambda *args, **kwargs: None,
        )
        target.minimax_m3_index_topk = target.minimax_m3_index_decode = (
            target.minimax_m3_index_score
        ) = lambda: None
        return upstream if name == upstream.__name__ else target

    monkeypatch.setattr(real_entry.importlib, "import_module", get_module)
    consumer = ModuleType("vllm.models.minimax_m3.copied_consumer")
    original = consumer.reference = object()
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    backend = classes["MiniMaxM3SparseBackend"]
    original_shape = backend.__dict__["get_kv_cache_shape"]

    def fail_after_mutating_targets():
        backend.get_kv_cache_shape = staticmethod(lambda: "changed")
        classes["MiniMaxM3SparseTritonImpl"].forward = lambda: None
        classes["MiniMaxM3IndexerImpl"].update_graph_params = staticmethod(lambda: None)
        consumer.reference = object()
        raise LookupError("failure after class patches")

    implementations[
        f"{module._VENDOR}.minimax_m3_graph_replay"
    ].install_minimax_m3_graph_conformance = fail_after_mutating_targets
    with pytest.raises(LookupError, match="failure after class patches"):
        module.apply_minimax_m3_patches(config)
    assert backend.__dict__["get_kv_cache_shape"] is original_shape
    assert classes["MiniMaxM3SparseTritonImpl"].forward is original_forward
    assert "update_graph_params" not in vars(classes["MiniMaxM3IndexerImpl"])
    assert consumer.reference is original
    assert not module._installed


@pytest.fixture
def operator_bundle(monkeypatch):
    library = ModuleType("flaggems_vllm")
    calls = []
    names = (
        "apply_minimax_m3_rotary_ascend",
        "attend_minimax_m3_sparse_ascend",
        "fused_minimax_m3_qknorm_rope_kv_insert_ascend",
        "insert_minimax_m3_index_cache_ascend",
        "insert_minimax_m3_kv_cache_ascend",
        "load_minimax_m3_ascend_native",
        "minimax_m3_index_decode_ascend",
        "minimax_m3_index_score_ascend",
        "minimax_m3_index_topk_ascend",
        "prepare_minimax_m3_gemma_weight_ascend",
    )
    for name in names:

        def operation(*args, name=name, **kwargs):
            calls.append((name, args, kwargs))
            return "operator-result"

        setattr(library, name, operation)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", library)
    bundle = load_file(f"{VENDOR}/impl/minimax_m3_sparse.py", "m3_bundle_test")
    return bundle.MiniMaxM3SparseKernels, library, calls


@pytest.mark.parametrize("setting,loads", [(None, 1), ("1", 1), ("0", 0)])
def test_prepare_honors_existing_native_disable(
    operator_bundle, monkeypatch, setting, loads
):
    kernels, _, calls = operator_bundle
    if setting is None:
        monkeypatch.delenv("VLLM_FL_M3_NPU_SPARSE_ATTN", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_M3_NPU_SPARSE_ATTN", setting)

    def activation(*args, **kwargs):
        pass

    activation._schemas = {
        "": SimpleNamespace(
            arguments=[
                SimpleNamespace(name=name)
                for name in ("alpha", "limit", "bias", "interleaved")
            ]
        )
    }
    monkeypatch.setattr(torch.ops.npu, "npu_clipped_swiglu", activation, raising=False)
    monkeypatch.setattr(
        torch.ops._flaggems_m3_ascend,
        "npu_sparse_attention_score",
        lambda: None,
        raising=False,
    )
    kernels.prepare()
    assert len(calls) == loads
    assert all(call[0] == "load_minimax_m3_ascend_native" for call in calls)


def test_prepare_rejects_missing_activation_before_native_load(
    operator_bundle, monkeypatch
):
    kernels, _, calls = operator_bundle
    monkeypatch.setattr(torch.ops.npu, "npu_clipped_swiglu", None, raising=False)
    with pytest.raises(RuntimeError, match="clipped-SwiGLU"):
        kernels.prepare()
    assert not calls


@pytest.mark.parametrize(
    "file,function,method,count",
    [
        ("indexer/index_vendor.py", "minimax_m3_index_score", "index_score", 10),
        ("indexer/index_vendor.py", "minimax_m3_index_topk", "index_topk", 8),
        (
            "indexer/index_decode_vendor.py",
            "minimax_m3_index_decode",
            "index_decode",
            13,
        ),
    ],
)
def test_index_bridge_forwards_and_does_not_retry(
    monkeypatch, file, function, method, count
):
    selector = ModuleType("vllm_fl.dispatch.backends.vendor.ascend.attention_selector")
    calls = []

    def selected(*args, **kwargs):
        calls.append((args, kwargs))
        raise ArithmeticError("error after operator launch")

    selector.get_minimax_sparse_kernels = lambda: SimpleNamespace(**{method: selected})
    monkeypatch.setitem(sys.modules, selector.__name__, selector)
    bridge = load_file(f"{VENDOR}/impl/{file}", "m3_bridge_" + function)
    args = tuple(object() for _ in range(count))
    with pytest.raises(ArithmeticError, match="after operator launch"):
        getattr(bridge, function)(*args)
    assert calls == [(args, {})]


def test_derived_weights_publish_only_after_all_preparations(monkeypatch):
    path = ROOT / "models/minimax_m3_ascend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "cache_gemma_plus_one_weights"
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_GEMMA_PLUS_ONE_ATTRS"
                for target in node.targets
            )
        )
    ]
    selector = ModuleType("vllm_fl.dispatch.backends.vendor.ascend.attention_selector")
    calls = []

    def prepare(weight):
        calls.append(weight)
        if len(calls) == 2:
            raise ValueError("second weight preparation failed")
        return "new-cache"

    selector.get_minimax_sparse_kernels = lambda: SimpleNamespace(gemma_weight=prepare)
    monkeypatch.setitem(sys.modules, selector.__name__, selector)

    class Layer:
        pass

    layer = Layer()
    layer.q_norm = SimpleNamespace(weight=object())
    layer.k_norm = SimpleNamespace(weight=object())
    layer._fl_q_norm_plus_one = old = object()
    model = SimpleNamespace(modules=lambda: [layer])
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(ValueError, match="second weight"):
        namespace["cache_gemma_plus_one_weights"](model)
    assert layer._fl_q_norm_plus_one is old
    assert not hasattr(layer, "_fl_k_norm_plus_one")
    assert len(calls) == 2
