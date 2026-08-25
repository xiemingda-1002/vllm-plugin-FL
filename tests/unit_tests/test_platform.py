import os
from types import SimpleNamespace

import pytest

from vllm_fl.platform import PlatformFL


def test_npu_simple_compile_backend_is_eager():
    expected_backend = "eager" if PlatformFL.device_type == "npu" else "inductor"

    assert PlatformFL.simple_compile_backend == expected_backend


def test_ascend_entrypoint_stays_inside_fl(monkeypatch):
    if PlatformFL.device_type != "npu":
        pytest.skip("Ascend-only entrypoint behavior")

    import vllm_fl

    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)

    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert os.environ["TRITON_CACHE_DIR"].endswith(
        "/.triton/vllm-fl-ascend-v1"
    )


def test_ascend_triton_cache_preserves_explicit_override(monkeypatch):
    import vllm_fl

    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/user-triton-cache")

    vllm_fl._configure_ascend_triton_cache()

    assert os.environ["TRITON_CACHE_DIR"] == "/tmp/user-triton-cache"


def test_ascend_hccl_preserves_explicit_override(monkeypatch):
    import vllm_fl

    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AICPU")

    vllm_fl._configure_ascend_hccl()

    assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AICPU"


def test_ascend_hccl_defaults_to_aiv(monkeypatch):
    import vllm_fl

    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)

    vllm_fl._configure_ascend_hccl()

    assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AIV"


def test_ascend_platform_hook_installs_hybrid_cache_patch_early(monkeypatch):
    if PlatformFL.device_type != "npu":
        pytest.skip("Ascend-only entrypoint behavior")

    from vllm_fl.dispatch.backends.vendor.ascend import patch as ascend_patch

    called = []
    monkeypatch.setattr(ascend_patch, "patch_mamba_config", lambda: called.append(True))

    PlatformFL.pre_register_and_update()

    assert called == [True]


def test_ascend_refresh_block_size_preserves_hybrid_page_alignment():
    from vllm_fl.dispatch.backends.vendor.ascend.patch import refresh_block_size

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2048,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        model_config=SimpleNamespace(is_hybrid=True),
    )

    refresh_block_size(config)

    assert config.cache_config.block_size == 2048


def test_ascend_refresh_block_size_keeps_generic_chunked_prefill_default():
    from vllm_fl.dispatch.backends.vendor.ascend.patch import refresh_block_size

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=2048,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        model_config=SimpleNamespace(is_hybrid=False),
    )

    refresh_block_size(config)

    assert config.cache_config.block_size == 128
