"""Runtime contracts for the GLM-5.2 SFA config and cache backend API."""

from types import SimpleNamespace
import importlib

import pytest


def _install_config(monkeypatch, additional_config: dict) -> None:
    from vllm_fl.configs import ascend as ascend_config

    monkeypatch.setattr(
        ascend_config,
        "_WORKER_VLLM_CONFIG",
        SimpleNamespace(
            additional_config=additional_config,
            model_config=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        ascend_config, "shared_expert_dp_enabled_for_config", lambda _: False
    )


def _import_sfa(monkeypatch):
    """Import the backend with the same initialized device-type precondition."""
    from vllm_fl.dispatch.backends.vendor.ascend import dsa_compat
    from vllm_fl.platforms.ascend import hardware

    monkeypatch.setattr(
        hardware,
        "get_ascend_device_type",
        lambda: hardware.AscendDeviceType.A2,
    )
    monkeypatch.setattr(
        dsa_compat,
        "get_ascend_device_type",
        lambda: hardware.AscendDeviceType.A2,
    )
    return importlib.import_module(
        "vllm_fl.attention.ascend.sfa_v1"
    )


def test_sfa_sparse_c8_defaults_to_a_safe_disabled_config(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.compat import (
        get_ascend_config,
    )

    _install_config(monkeypatch, {})
    config = get_ascend_config()

    assert config.enable_sparse_c8 is False
    assert config.c8_enable_reshape_optim is False
    assert config.is_sparse_c8_layer("model.layers.0.self_attn.indexer.k_cache") is False


def test_sfa_sparse_c8_opt_in_fails_closed_before_cache_selection(monkeypatch) -> None:
    from vllm_fl.dispatch.backends.vendor.ascend import dsa_compat
    from vllm_fl.dispatch.backends.vendor.ascend.impl.moe.compat import (
        get_ascend_config,
    )

    _install_config(monkeypatch, {"enable_sparse_c8": True})
    monkeypatch.setattr(dsa_compat, "model_uses_sfa_sparse", lambda _: True)

    with pytest.raises(NotImplementedError, match="sparse-C8 cache is not migrated"):
        get_ascend_config()


@pytest.mark.parametrize(
    "backend_path",
    [
        "vllm_fl.attention.ascend.sfa_v1.AscendSFABackend",
        "vllm_fl.attention.ascend.indexer.AscendSFAIndexerBackend",
    ],
)
def test_sfa_cache_backends_accept_upstream_cache_dtype_keyword(
    monkeypatch, backend_path: str
) -> None:
    module_path, class_name = backend_path.rsplit(".", 1)
    module = _import_sfa(monkeypatch) if module_path.endswith("sfa_v1") else __import__(
        module_path, fromlist=[class_name]
    )
    backend = getattr(module, class_name)

    # vLLM's KV-connector runner calls this API with cache_dtype_str as a
    # keyword. Exercise that invocation, not merely the source signature.
    assert backend.get_kv_cache_shape(
        3, 128, 1, 576, cache_dtype_str="bfloat16"
    ) == (3, 128, 1, 576)


def test_sfa_cp_off_uses_the_local_builder_and_impl(monkeypatch) -> None:
    sfa_v1 = _import_sfa(monkeypatch)

    monkeypatch.setattr(sfa_v1, "enable_sfa_dcp_replicated_indexer", lambda: False)
    monkeypatch.setattr(sfa_v1, "enable_cp", lambda: False)

    assert sfa_v1.AscendSFABackend.get_builder_cls() is sfa_v1.AscendSFAMetadataBuilder
    assert sfa_v1.AscendSFABackend.get_impl_cls() is sfa_v1.AscendSFAImpl


@pytest.mark.parametrize("replicated_indexer", [False, True])
def test_sfa_cp_opt_in_fails_closed_without_unmigrated_import(
    monkeypatch, replicated_indexer: bool
) -> None:
    sfa_v1 = _import_sfa(monkeypatch)

    monkeypatch.setattr(
        sfa_v1,
        "enable_sfa_dcp_replicated_indexer",
        lambda: replicated_indexer,
    )
    monkeypatch.setattr(sfa_v1, "enable_cp", lambda: not replicated_indexer)

    with pytest.raises(NotImplementedError, match="context-parallel cache and metadata are not migrated"):
        sfa_v1.AscendSFABackend.get_builder_cls()
    with pytest.raises(NotImplementedError, match="context-parallel cache and metadata are not migrated"):
        sfa_v1.AscendSFABackend.get_impl_cls()
