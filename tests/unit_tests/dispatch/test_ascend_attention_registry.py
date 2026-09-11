# Copyright (c) 2026 BAAI. All rights reserved.

import importlib
import sys

from vllm.v1.attention.backends.registry import AttentionBackendEnum


_BACKEND_MODULE = (
    "vllm_fl.dispatch.backends.vendor.ascend.impl.attention"
)
_BACKEND_PATH = f"{_BACKEND_MODULE}.AscendAttentionBackend"


def _assert_vllm_ascend_not_imported() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name == "vllm_ascend" or name.startswith("vllm_ascend.")
    )
    assert imported == []


def test_platform_registration_keeps_ascend_attention_lazy() -> None:
    import vllm_fl

    sys.modules.pop(_BACKEND_MODULE, None)

    assert vllm_fl.register() == "vllm_fl.platform.PlatformFL"
    assert _BACKEND_MODULE not in sys.modules
    _assert_vllm_ascend_not_imported()


def test_ascend_attention_uses_registered_custom_slot() -> None:
    backend_module = importlib.import_module(_BACKEND_MODULE)
    backend_cls = backend_module.AscendAttentionBackend

    assert backend_cls.get_name() == "CUSTOM"
    assert AttentionBackendEnum[backend_cls.get_name()] is AttentionBackendEnum.CUSTOM
    assert AttentionBackendEnum.CUSTOM.is_overridden()
    assert AttentionBackendEnum.CUSTOM.get_path() == _BACKEND_PATH
    assert AttentionBackendEnum.CUSTOM.get_class() is backend_cls
    _assert_vllm_ascend_not_imported()
