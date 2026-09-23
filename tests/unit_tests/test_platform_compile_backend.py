# Copyright (c) 2026 BAAI. All rights reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm.config import CompilationMode, CUDAGraphMode


def _platform_config(*, mode=CompilationMode.VLLM_COMPILE):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls="auto", all2all_backend="", data_parallel_size=1
        ),
        model_config=None,
        cache_config=None,
        attention_config=None,
        compilation_config=SimpleNamespace(
            mode=mode,
            compile_sizes=[],
            pass_config=SimpleNamespace(enable_sp=False),
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_num_of_warmups=0,
            use_inductor_graph_partition=True,
            splitting_ops=["vllm::unified_attention_with_output"],
        ),
        additional_config={},
    )


def _capture_default_config(
    *,
    max_num_seqs=48,
    max_capture_size=None,
    capture_sizes=None,
    num_speculative_tokens=None,
):
    speculative_config = (
        None
        if num_speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    )
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=max_capture_size,
            cudagraph_capture_sizes=capture_sizes,
        ),
        speculative_config=speculative_config,
    )


def _stub_refresh_block_size(monkeypatch):
    cache_module = importlib.import_module("vllm_fl.configs.ascend_cache")
    monkeypatch.setattr(cache_module, "refresh_block_size", lambda config: None)


def test_platform_selects_fl_compiler_only_for_npu(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    assert (
        PlatformFL.get_compile_backend()
        == "vllm_fl.compilation.compiler_interface.AscendCompiler"
    )

    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    monkeypatch.setattr(PlatformFL, "simple_compile_backend", "inductor")
    assert PlatformFL.get_compile_backend() == "inductor"


def test_platform_selects_fl_pass_manager_only_for_npu(monkeypatch) -> None:
    from vllm_fl.compilation.config import COMPILATION_PASS_KEY
    from vllm_fl.platform import PlatformFL

    platform = PlatformFL()
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    assert platform.pass_key == COMPILATION_PASS_KEY
    assert PlatformFL.get_pass_manager_cls() == (
        "vllm_fl.compilation.graph_fusion_pass_manager.GraphFusionPassManager"
    )

    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    assert platform.pass_key == "post_grad_custom_post_pass"
    assert (
        PlatformFL.get_pass_manager_cls()
        == "vllm.compilation.passes.pass_manager.PostGradPassManager"
    )


def test_active_platform_uses_eager_for_simple_compile_helpers() -> None:
    from vllm_fl.platform import PlatformFL

    expected = "eager" if PlatformFL.device_type == "npu" else "inductor"
    assert PlatformFL.simple_compile_backend == expected


@pytest.mark.parametrize(
    ("max_num_seqs", "num_speculative_tokens", "expected"),
    [
        (48, None, 48),
        (48, 5, 288),
        (200, 5, 512),
    ],
)
def test_ascend_capture_default_matches_rc1(
    monkeypatch, max_num_seqs, num_speculative_tokens, expected
) -> None:
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", "ascend")
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = _capture_default_config(
        max_num_seqs=max_num_seqs,
        num_speculative_tokens=num_speculative_tokens,
    )

    PlatformFL.apply_config_platform_defaults(config)

    assert config.compilation_config.max_cudagraph_capture_size == expected


def test_ascend_capture_default_preserves_explicit_max_and_sizes(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", "ascend")
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    explicit_max = _capture_default_config(max_capture_size=96)
    explicit_sizes = _capture_default_config(capture_sizes=[1, 8, 96])

    PlatformFL.apply_config_platform_defaults(explicit_max)
    PlatformFL.apply_config_platform_defaults(explicit_sizes)

    assert explicit_max.compilation_config.max_cudagraph_capture_size == 96
    assert explicit_sizes.compilation_config.max_cudagraph_capture_size is None


def test_ascend_capture_default_requires_scheduler_limit(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", "ascend")
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = _capture_default_config()
    config.scheduler_config = SimpleNamespace()

    PlatformFL.apply_config_platform_defaults(config)

    assert config.compilation_config.max_cudagraph_capture_size is None


def test_capture_default_is_ascend_vendor_scoped(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "vendor_name", "cuda")
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    config = _capture_default_config()

    PlatformFL.apply_config_platform_defaults(config)

    assert config.compilation_config.max_cudagraph_capture_size is None


def test_ascend_backend_preserves_model_derived_hybrid_block_size(
    monkeypatch,
) -> None:
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=1536,
            enable_prefix_caching=False,
            mamba_cache_mode="align",
            mamba_block_size=4096,
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False
        ),
        kv_transfer_config=None,
    )

    PlatformFL.update_block_size_for_backend(config)

    assert config.cache_config.block_size == 1536
    assert config.cache_config.mamba_block_size == 4096


def test_ascend_backend_aligns_mamba_block_for_kv_transfer(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=1536,
            enable_prefix_caching=False,
            mamba_cache_mode="align",
            mamba_block_size=4096,
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False
        ),
        kv_transfer_config=SimpleNamespace(),
    )

    PlatformFL.update_block_size_for_backend(config)

    assert config.cache_config.block_size == 1536
    assert config.cache_config.mamba_block_size == 1536


def test_full_decode_only_uses_supported_ascend_config(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    _stub_refresh_block_size(monkeypatch)
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = _platform_config()
    PlatformFL.check_and_update_config(config)
    compilation_config = config.compilation_config

    assert config.additional_config["ascend_compilation_config"] == {
        "enable_npugraph_ex": True,
        "enable_static_kernel": False,
        "fuse_norm_quant": True,
        "fuse_qknorm_rope": True,
        "fuse_allreduce_rms": False,
        "fuse_muls_add": True,
    }
    assert compilation_config.cudagraph_num_of_warmups == 1
    assert compilation_config.use_inductor_graph_partition is False
    assert compilation_config.splitting_ops == []
    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


def test_ascend_sp_pass_preserves_all2all_backend(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    _stub_refresh_block_size(monkeypatch)
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    config = _platform_config()
    config.parallel_config.all2all_backend = "allgather_reducescatter"
    config.compilation_config.pass_config.enable_sp = True

    PlatformFL.check_and_update_config(config)

    assert config.parallel_config.all2all_backend == "allgather_reducescatter"


def test_non_ascend_does_not_install_moe_backend_sentinel(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    config = _platform_config()
    config.parallel_config.all2all_backend = "allgather_reducescatter"

    PlatformFL.check_and_update_config(config)

    assert config.parallel_config.all2all_backend == "allgather_reducescatter"


def test_forward_context_hook_is_ascend_only(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "cuda")

    assert PlatformFL.set_additional_forward_context(
        attn_metadata={},
        vllm_config=None,
        dp_metadata=None,
    ) == {}


def test_ascend_forward_context_hook_delegates_lazily(monkeypatch) -> None:
    from vllm_fl import ascend_forward_context
    from vllm_fl.platform import PlatformFL

    expected = {"moe_comm_type": "allgather"}
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    monkeypatch.setattr(
        ascend_forward_context,
        "build_additional_forward_context",
        lambda **kwargs: expected,
    )

    assert PlatformFL.set_additional_forward_context(
        attn_metadata={},
        vllm_config=None,
        dp_metadata=None,
    ) is expected


def test_ascend_attention_selector_uses_compress_for_dsa(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.ascend import AscendBackend
    from vllm_fl.platform import PlatformFL

    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    monkeypatch.setattr(
        "vllm_fl.dispatch.call_op",
        lambda _name, **kwargs: AscendBackend().attention_backend(**kwargs),
    )

    dsa_config = SimpleNamespace(
        use_mla=True, use_sparse=False, use_compress=True
    )
    assert PlatformFL.get_attn_backend_cls(None, dsa_config) == (
        "vllm_fl.attention.ascend.dsa_v1.AscendDSABackend"
    )

    # Existing MHA and MLA selector branches retain their original paths.
    assert PlatformFL.get_attn_backend_cls(
        None, SimpleNamespace(use_mla=False, use_sparse=False)
    ) == "vllm_fl.attention.ascend.attention.AscendAttentionBackend"
    assert PlatformFL.get_attn_backend_cls(
        None, SimpleNamespace(use_mla=True, use_sparse=False, use_compress=False)
    ) == "vllm_fl.attention.ascend.attention.AscendMLABackend"


def test_non_npu_attention_selector_keeps_existing_dispatch_contract(
    monkeypatch,
) -> None:
    from vllm_fl.platform import PlatformFL

    captured = {}
    monkeypatch.setattr(PlatformFL, "device_type", "cuda")

    def capture_call(name, **kwargs):
        captured.update(name=name, **kwargs)
        return "fallback.backend"

    monkeypatch.setattr("vllm_fl.dispatch.call_op", capture_call)

    assert PlatformFL.get_attn_backend_cls(
        None, SimpleNamespace(use_mla=False, use_sparse=False, use_compress=True)
    ) == "fallback.backend"
    assert captured == {
        "name": "attention_backend",
        "use_mla": False,
        "use_sparse": False,
    }


def test_ascend_kv_cache_spec_hook_is_vendor_scoped_and_idempotent(monkeypatch) -> None:
    from vllm_fl.kv_cache.ascend import (
        deepseek_v4_kv_cache,
        kv_cache_interface,
    )
    from vllm_fl.platform import PlatformFL

    calls = []
    monkeypatch.setattr(
        kv_cache_interface,
        "register_ascend_kv_cache_specs",
        lambda: calls.append("ascend"),
    )
    monkeypatch.setattr(
        deepseek_v4_kv_cache,
        "apply_deepseek_v4_kv_cache_patches",
        lambda: calls.append("patch"),
    )
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    PlatformFL.register_custom_kv_cache_specs(None)
    PlatformFL.register_custom_kv_cache_specs(None)
    assert calls == ["ascend", "patch", "ascend", "patch"]

    monkeypatch.setattr(PlatformFL, "device_type", "cuda")
    PlatformFL.register_custom_kv_cache_specs(None)
    assert calls == ["ascend", "patch", "ascend", "patch"]


def test_ascend_kv_cache_specs_register_the_expected_managers() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        SlidingWindowManager,
    )
    from vllm.v1.kv_cache_spec_registry import _REGISTRY_KVCACHESPEC_LIST

    from vllm_fl.kv_cache.ascend.deepseek_v4_kv_cache import (
        CompressAttentionManager,
    )
    from vllm_fl.kv_cache.ascend.kv_cache_interface import (
        AscendMLAAttentionSpec,
        AscendSFAIndexerCacheSpec,
        AscendSlidingWindowMLASpec,
        register_ascend_kv_cache_specs,
    )

    register_ascend_kv_cache_specs()
    register_ascend_kv_cache_specs()
    assert (
        _REGISTRY_KVCACHESPEC_LIST[AscendMLAAttentionSpec].manager_class
        is CompressAttentionManager
    )
    assert (
        _REGISTRY_KVCACHESPEC_LIST[AscendSFAIndexerCacheSpec].manager_class
        is FullAttentionManager
    )
    assert (
        _REGISTRY_KVCACHESPEC_LIST[AscendSlidingWindowMLASpec].manager_class
        is SlidingWindowManager
    )


def test_full_decode_only_rejects_launch_blocking(monkeypatch) -> None:
    from vllm_fl.platform import PlatformFL

    _stub_refresh_block_size(monkeypatch)
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    with pytest.raises(ValueError, match="ASCEND_LAUNCH_BLOCKING=1"):
        PlatformFL.check_and_update_config(_platform_config())


@pytest.mark.parametrize(
    "mode",
    [CompilationMode.STOCK_TORCH_COMPILE, CompilationMode.DYNAMO_TRACE_ONCE],
)
def test_platform_rejects_unsupported_ascend_compile_modes(monkeypatch, mode) -> None:
    from vllm_fl.platform import PlatformFL

    _stub_refresh_block_size(monkeypatch)
    monkeypatch.setattr(PlatformFL, "device_type", "npu")
    with pytest.raises(NotImplementedError, match="only VLLM_COMPILE"):
        PlatformFL.check_and_update_config(_platform_config(mode=mode))
