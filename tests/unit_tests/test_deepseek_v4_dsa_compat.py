from types import SimpleNamespace

import pytest
import torch

vllm_config_module = pytest.importorskip("vllm.config")

from vllm_fl.dispatch.backends.vendor.ascend import dsa_compat


def _config(additional_config=None, *, has_indexer=True, kv_transfer_config=None):
    text_config = SimpleNamespace()
    if has_indexer:
        text_config.index_topk = 2048
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_config),
        additional_config=additional_config,
        kv_transfer_config=kv_transfer_config,
    )


def test_dsa_cp_defaults_off_and_requested_mode_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config",
        lambda: _config(),
    )
    assert dsa_compat.enable_dsa_cp() is False

    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config",
        lambda: _config({"enable_dsa_cp": True}),
    )
    with pytest.raises(NotImplementedError, match="DSA-CP"):
        dsa_compat.enable_dsa_cp()


def test_unmigrated_finegrained_tp_fails_closed(monkeypatch) -> None:
    config = SimpleNamespace(
        finegrained_tp_config=SimpleNamespace(
            oproj_tensor_parallel_size=2,
            olora_tensor_parallel_size=2,
        )
    )
    monkeypatch.setattr(dsa_compat, "get_ascend_config", lambda: config)
    with pytest.raises(NotImplementedError, match="OProj"):
        dsa_compat.oproj_tp_enable()
    with pytest.raises(NotImplementedError, match="OLoRA"):
        dsa_compat.olora_tp_enable()


def test_meta_weights_do_not_request_npu_format_conversion(monkeypatch) -> None:
    monkeypatch.setattr(
        dsa_compat,
        "get_ascend_config",
        lambda: SimpleNamespace(weight_nz_mode=2),
    )
    weight = torch.empty((16, 16), device="meta", dtype=torch.bfloat16)
    assert dsa_compat.maybe_trans_nz(weight) is weight
