# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from torch import fx
from vllm.compilation.passes.inductor_pass import InductorPass, pass_context
from vllm.config.compilation import Range


class _Pass(InductorPass):
    def __init__(self, name: str, events: list[str] | None = None):
        self.name = name
        self.events = events

    def __call__(self, graph: fx.Graph) -> None:
        if self.events is not None:
            self.events.append(self.name)

    def uuid(self) -> str:
        return self.name


def _config(**options):
    defaults = {
        "fuse_norm_quant": True,
        "fuse_qknorm_rope": True,
        "fuse_allreduce_rms": False,
        "fuse_muls_add": True,
    }
    defaults.update(options)
    return SimpleNamespace(
        additional_config={"ascend_compilation_config": defaults},
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
    )


def _stub_default_passes(monkeypatch):
    modules = {
        "vllm_fl.compilation.passes.norm_quant_fusion_pass": (
            "AddRMSNormQuantFusionPass",
            "norm",
        ),
        "vllm_fl.compilation.passes.qknorm_rope_fusion_pass": (
            "QKNormRopeFusionPass",
            "qknorm",
        ),
        "vllm_fl.compilation.passes.muls_add_pass": (
            "MulsAddFusionPass",
            "muls_add",
        ),
    }
    for module_name, (class_name, pass_name) in modules.items():
        module = ModuleType(module_name)
        setattr(module, class_name, lambda config, name=pass_name: _Pass(name))
        monkeypatch.setitem(sys.modules, module_name, module)


def test_manager_order_and_idempotency(monkeypatch) -> None:
    import vllm_fl.compilation.graph_fusion_pass_manager as manager_module

    _stub_default_passes(monkeypatch)
    monkeypatch.setattr(
        manager_module,
        "set_current_vllm_config",
        lambda *args, **kwargs: nullcontext(),
    )
    manager = manager_module.GraphFusionPassManager()
    manager.configure(_config())
    assert [item.name for item in manager.passes] == [
        "norm",
        "qknorm",
        "muls_add",
    ]

    manager.configure(_config())
    assert [item.name for item in manager.passes] == [
        "norm",
        "qknorm",
        "muls_add",
    ]


def test_manager_rejects_external_inductor_passes() -> None:
    from vllm_fl.compilation.graph_fusion_pass_manager import (
        GraphFusionPassManager,
    )

    manager = GraphFusionPassManager()
    with pytest.raises(NotImplementedError, match="external InductorPass"):
        manager.add(_Pass("generic"))


def test_manager_executes_configured_default_passes(monkeypatch) -> None:
    from vllm_fl.compilation.graph_fusion_pass_manager import (
        GraphFusionPassManager,
    )

    events: list[str] = []
    manager = GraphFusionPassManager()
    manager.passes = [_Pass("default", events)]
    graph_module = fx.symbolic_trace(lambda value: value + 1)
    with pass_context(Range(1, 1)):
        assert manager(graph_module.graph) is None
        first_uuid = manager.uuid()
        second_uuid = manager.uuid()
    assert events == ["default"]
    assert first_uuid == second_uuid


def test_manager_optional_unsupported_features_fail_closed(monkeypatch) -> None:
    import vllm_fl.compilation.graph_fusion_pass_manager as manager_module

    monkeypatch.setattr(
        manager_module,
        "set_current_vllm_config",
        lambda *args, **kwargs: nullcontext(),
    )
    with pytest.raises(NotImplementedError, match="fuse_allreduce_rms=True"):
        manager_module.GraphFusionPassManager().configure(
            _config(fuse_allreduce_rms=True)
        )

    config = _config()
    config.compilation_config.pass_config.enable_sp = True
    with pytest.raises(NotImplementedError, match="sequence-parallel"):
        manager_module.GraphFusionPassManager().configure(config)


def test_default_configuration_does_not_import_optional_pass_modules(
    monkeypatch,
) -> None:
    import vllm_fl.compilation.graph_fusion_pass_manager as manager_module

    _stub_default_passes(monkeypatch)
    monkeypatch.setattr(
        manager_module,
        "set_current_vllm_config",
        lambda *args, **kwargs: nullcontext(),
    )
    manager_module.GraphFusionPassManager().configure(_config())
    assert "vllm_fl.compilation.passes.allreduce_rmsnorm_fusion_pass" not in sys.modules
    assert "vllm_fl.compilation.passes.sequence_parallelism" not in sys.modules


def test_manager_never_delegates_to_upstream_post_grad(monkeypatch) -> None:
    import vllm.compilation.passes.pass_manager as upstream
    import vllm_fl.compilation.graph_fusion_pass_manager as manager_module

    _stub_default_passes(monkeypatch)
    monkeypatch.setattr(
        manager_module,
        "set_current_vllm_config",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        upstream.PostGradPassManager,
        "configure",
        lambda *args, **kwargs: pytest.fail("upstream manager must not run"),
    )
    manager_module.GraphFusionPassManager().configure(_config())
