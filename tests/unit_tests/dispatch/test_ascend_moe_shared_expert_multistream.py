# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import ast
import builtins
import importlib.util
import logging
import sys
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[3]
MOE = ROOT / "vllm_fl" / "dispatch" / "backends" / "vendor" / "ascend" / "impl" / "moe"


def _source(name: str) -> str:
    return (MOE / name).read_text(encoding="utf-8")


def _load_compat():
    # Load this narrow facade without importing the plugin package initializer:
    # this CPU unit module intentionally has no FlagGems/NPU dependency.
    package_names = (
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.ascend",
        "vllm_fl.configs",
    )
    modules = {name: _package(name) for name in package_names}
    vllm_logger = type(sys)("vllm.logger")

    def init_logger(name):
        logger = logging.getLogger(name)
        emitted = set()

        def warning_once(message, *args, **kwargs):
            if message not in emitted:
                emitted.add(message)
                logger.warning(message, *args, **kwargs)

        logger.warning_once = warning_once
        return logger

    vllm_logger.init_logger = init_logger
    vllm_config = type(sys)("vllm.config")
    vllm_config.get_current_vllm_config = lambda: (_ for _ in ()).throw(
        AssertionError("No current VllmConfig")
    )
    modules.update(
        {
            "vllm": _package("vllm"),
            "vllm.logger": vllm_logger,
            "vllm.config": vllm_config,
        }
    )

    def shared_expert_dp_enabled_for_config(vllm_config):
        additional_config = getattr(vllm_config, "additional_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        return bool(
            additional_config
            and additional_config.get("enable_shared_expert_dp", False)
            and parallel_config is not None
            and getattr(parallel_config, "enable_expert_parallel", False)
            and getattr(parallel_config, "tensor_parallel_size", 1) > 1
        )

    modules["vllm_fl.ascend_flashcomm"] = SimpleNamespace(
        enable_flashcomm1=lambda *_args, **_kwargs: False,
        shared_expert_dp_enabled_for_config=shared_expert_dp_enabled_for_config,
    )
    modules["vllm_fl.platforms.ascend.hardware"] = SimpleNamespace(
        AscendDeviceType=Enum("AscendDeviceType", "A2 A3 A5 _310P"),
        get_ascend_device_type=lambda: None,
    )
    config_spec = importlib.util.spec_from_file_location(
        "vllm_fl.configs.ascend", ROOT / "vllm_fl" / "configs" / "ascend.py"
    )
    assert config_spec is not None and config_spec.loader is not None
    config_module = importlib.util.module_from_spec(config_spec)
    modules["vllm_fl.configs.ascend"] = config_module
    path = MOE / "compat.py"
    spec = importlib.util.spec_from_file_location("_fl_ascend_moe_compat_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_modules = {name: sys.modules.get(name) for name in modules}
    try:
        sys.modules.update(modules)
        config_spec.loader.exec_module(config_module)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    original_import = __import__

    def config_import(name, globals=None, locals=None, fromlist=(), level=0):
        # The isolated config's deferred vLLM lookup must continue to work
        # after its temporary sys.modules registration is restored. A test may
        # still install its own vllm.config module to exercise a custom owner.
        if name == "vllm.config" and name not in sys.modules:
            return vllm_config
        return original_import(name, globals, locals, fromlist, level)

    config_module.__dict__["__builtins__"] = dict(
        vars(builtins), __import__=config_import
    )
    # Keep the exact owner alive after its temporary sys.modules registration
    # is restored. Config function globals must never resolve through module
    # state leaked from an earlier test.
    module._config_owner = config_module
    return module


def _config_owner(compat):
    """Return the module whose globals execute compat's config re-export."""
    return compat._config_owner


def _package(name: str):
    module = type(sys)(name)
    module.__path__ = []
    return module


@pytest.fixture(autouse=True)
def _isolate_fused_mc2_environment(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_FUSED_MC2", raising=False)


def test_shared_expert_stream_is_created_lazily_once(monkeypatch) -> None:
    compat = _load_compat()
    stream = object()
    stream_factory = Mock(return_value=stream)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(Stream=stream_factory)),
    )

    assert compat._SHARED_EXPERTS_CALCULATION_STREAM is None
    assert compat.shared_experts_calculation_stream() is stream
    assert compat.shared_experts_calculation_stream() is stream
    stream_factory.assert_called_once_with()


def test_stream_switch_is_noop_when_disabled() -> None:
    compat = _load_compat()
    with compat.npu_stream_switch(None, enabled=False) as result:
        assert result is None


def test_stream_switch_uses_npu_context_when_enabled(monkeypatch) -> None:
    compat = _load_compat()
    target_stream = object()
    marker = object()
    stream_context = Mock(return_value=nullcontext(marker))
    monkeypatch.setattr(
        compat,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(stream=stream_context)),
    )

    with compat.npu_stream_switch(target_stream, enabled=True) as result:
        assert result is marker
    stream_context.assert_called_once_with(target_stream)


def test_multistream_activation_matches_current_rc1_gate() -> None:
    runner = _source("fused_moe.py")
    assert (
        "self.multistream_overlap_shared_expert = (\n"
        "            ascend_config.multistream_overlap_shared_expert\n"
        "            and shared_experts is not None\n"
        "        )"
    ) in runner
    assert (
        "if ascend_config.multistream_overlap_shared_expert:\n"
        "            raise NotImplementedError"
    ) not in runner


def test_post_load_validation_wrap_matches_current_rc1_order() -> None:
    runner = _source("fused_moe.py")
    setup = runner.index("setup_moe_comm_method(self.moe_config)")
    capture = runner.index("original_process_weights = (", setup)
    original = runner.index(
        "result = original_process_weights(*args, **kwargs)", capture
    )
    validation = runner.index("self._validate_shared_expert_consistency()", original)
    result_return = runner.index("return result", validation)
    assert setup < capture < original < validation < result_return
    assert "@wraps(original_process_weights)" in runner[capture:result_return]


def test_multistream_forward_retains_stream_and_event_lifecycle() -> None:
    runner = _source("fused_moe.py")
    start = runner.index("def _forward_shared_experts(")
    end = runner.index("def shared_forward_impl(", start)
    forward = runner[start:end]

    assert (
        "with npu_stream_switch(shared_experts_calculation_stream(), "
        "enabled=self.multistream_overlap_shared_expert):"
    ) in forward
    assert "wait_event(fused_moe_evts.before_routed_experts)" in forward
    assert "maybe_wait_event(fused_moe_evts.before_dispatch)" in forward
    assert "maybe_wait_event(fused_moe_evts.before_combine)" in forward
    assert "wait_stream(shared_experts_calculation_stream())" in forward
    assert "self._shared_experts_part1(hidden_states)" in forward
    assert "self._shared_experts_part2(hidden_states, part1_out)" in forward


def test_unmigrated_neighbor_modes_still_fail_closed() -> None:
    runner = _source("fused_moe.py")
    for rejection in ("EPLB is not migrated",):
        assert rejection in runner


def test_compat_configuration_exports_retain_config_owner_identity() -> None:
    compat = _load_compat()
    config_owner = _config_owner(compat)

    for name in (
        "init_ascend_config",
        "clear_ascend_config",
        "_current_vllm_config_or_none",
        "get_ascend_additional_config",
        "balance_scheduling_enabled",
        "_nested_config",
        "_reject_unsupported",
        "get_ascend_config",
        "enable_sp",
        "enable_sp_by_pass",
    ):
        assert getattr(compat, name) is getattr(config_owner, name)


def test_moe_compat_uses_rc1_nested_defaults(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(_config_owner(compat), "_additional_config", lambda: {})

    config = compat.get_ascend_config()

    assert config.eplb_config.dynamic_eplb is False
    assert config.eplb_config.expert_heat_collection_interval == 600
    assert config.eplb_config.algorithm_execution_interval == 50
    assert config.eplb_config.eplb_policy_type == 2
    assert config.eplb_config.expert_map_record_path is None
    assert config.ascend_compilation_config.enable_static_kernel is False
    assert config.ascend_fusion_config.fusion_ops_gmmswigluquant is False


@pytest.mark.parametrize(
    ("additional_config", "message"),
    [
        ({"eplb_config": {"dynamic_eplb": True}}, "EPLB"),
        ({"eplb_config": {"expert_map_record_path": "record.json"}}, "EPLB"),
        ({"eplb_config": {"expert_map_record_path": ""}}, "EPLB"),
        ({"eplb_config": {"expert_map_path": "map.json"}}, "EPLB"),
        ({"eplb_config": {"num_redundant_experts": 1}}, "EPLB"),
        ({"mix_placement": True}, "mixed shared-expert placement"),
        ({"enable_mc2_hierarchy_comm": True}, "MC2 hierarchy communication"),
        (
            {"ascend_compilation_config": {"enable_static_kernel": True}},
            "static kernel generation",
        ),
        (
            {"ascend_fusion_config": {"fusion_ops_gmmswigluquant": True}},
            "gmmswigluquant fusion",
        ),
    ],
)
def test_moe_compat_rejects_known_nested_unsupported_opt_ins(
    monkeypatch, additional_config, message
) -> None:
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat), "_additional_config", lambda: additional_config
    )

    with pytest.raises(NotImplementedError, match=message):
        compat.get_ascend_config()


def test_moe_compat_preserves_dormant_nested_eplb_tuning(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat),
        "_additional_config",
        lambda: {
            "mega_moe_max_tokens": 8192,
            "eplb_config": {
                "eplb_policy_type": 3,
                "expert_heat_collection_interval": 10,
            },
        },
    )

    config = compat.get_ascend_config()

    assert config.mega_moe_max_tokens == 8192
    assert config.eplb_config.eplb_policy_type == 3
    assert config.eplb_config.expert_heat_collection_interval == 10


def test_moe_compat_rejects_legacy_flat_eplb_key(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat), "_additional_config", lambda: {"dynamic_eplb": False}
    )

    with pytest.raises(
        ValueError, match=r"additional_config\.eplb_config\.dynamic_eplb"
    ):
        compat.get_ascend_config()


def test_moe_compat_accepts_fused_mc2_environment_opt_in(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(_config_owner(compat), "_additional_config", lambda: {})
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "1")

    assert compat.get_ascend_config().enable_fused_mc2 == 1


def test_fused_mc2_disables_shared_expert_overlap_once_without_caching_config(
    monkeypatch, caplog
) -> None:
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat),
        "_additional_config",
        lambda: {
            "enable_fused_mc2": 1,
            "multistream_overlap_shared_expert": True,
        },
    )

    with caplog.at_level(logging.WARNING):
        first = compat.get_ascend_config()
        second = compat.get_ascend_config()

    assert first is not second
    assert first.enable_fused_mc2 == second.enable_fused_mc2 == 1
    assert first.multistream_overlap_shared_expert is False
    assert second.multistream_overlap_shared_expert is False
    warning = (
        "enable_fused_mc2 and multistream_overlap_shared_expert cannot "
        "be enabled together; disabling shared-expert overlap."
    )
    assert caplog.text.count(warning) == 1


def test_moe_compat_config_overrides_fused_mc2_environment(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat), "_additional_config", lambda: {"enable_fused_mc2": 0}
    )
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "1")

    assert compat.get_ascend_config().enable_fused_mc2 == 0


def test_moe_compat_preserves_shared_expert_overlap_opt_in(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat),
        "_additional_config",
        lambda: {"multistream_overlap_shared_expert": True},
    )

    assert compat.get_ascend_config().multistream_overlap_shared_expert is True


@pytest.mark.parametrize(
    ("enable_ep", "tp_size", "expected"),
    [
        (True, 2, True),
        (True, 4, True),
        (True, 1, False),
        (False, 2, False),
    ],
)
def test_shared_expert_dp_requires_requested_ep_and_tp_gt_one(
    monkeypatch, enable_ep, tp_size, expected
) -> None:
    compat = _load_compat()
    vllm_config = SimpleNamespace(
        additional_config={"enable_shared_expert_dp": True},
        parallel_config=SimpleNamespace(
            enable_expert_parallel=enable_ep,
            tensor_parallel_size=tp_size,
        ),
    )
    monkeypatch.setattr(
        _config_owner(compat),
        "_current_vllm_config_or_none",
        lambda: vllm_config,
    )
    monkeypatch.setattr(
        _config_owner(compat),
        "_additional_config",
        lambda: vllm_config.additional_config,
    )
    enable_sp = Mock(return_value=True)
    monkeypatch.setattr(_config_owner(compat), "enable_sp", enable_sp)

    config = compat.get_ascend_config()

    assert config.enable_shared_expert_dp is expected
    if expected:
        enable_sp.assert_called_once_with(
            vllm_config=vllm_config,
            enable_shared_expert_dp=True,
        )
    else:
        enable_sp.assert_not_called()


def test_shared_expert_dp_defaults_false_without_current_config(monkeypatch) -> None:
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat), "_current_vllm_config_or_none", lambda: None
    )
    monkeypatch.setattr(
        _config_owner(compat),
        "_additional_config",
        lambda: {"enable_shared_expert_dp": True},
    )

    assert compat.get_ascend_config().enable_shared_expert_dp is False


def test_all2all_prepare_finalize_honors_shared_expert_dp_skip(monkeypatch) -> None:
    """Exercise the existing prepare/finalize flag, without a device group."""
    import torch
    from vllm_fl.dispatch.backends.vendor.ascend.impl.moe import prepare_finalize

    method = object.__new__(prepare_finalize.PrepareAndFinalizeWithAll2All)
    method.tp_size = 2
    method.tp_rank = 0
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    router_logits = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    input_ids = torch.tensor([1, 2, 3, 4])

    skipped = method.prepare(
        hidden_states,
        router_logits,
        enable_shared_expert_dp=True,
        replace_allreduce=False,
    )
    assert skipped.hidden_states.shape[0] == 4
    assert skipped.router_logits.shape[0] == 4
    assert method.pad_and_split_input_ids(input_ids).tolist() == [1, 2, 3, 4]

    all_gather = Mock()
    monkeypatch.setattr(prepare_finalize.dist, "all_gather", all_gather)
    finalized = method.finalize(skipped.hidden_states, reduce_results=False)
    assert finalized.shape[0] == 4
    all_gather.assert_not_called()

    partitioned = method.prepare(
        hidden_states,
        router_logits,
        enable_shared_expert_dp=False,
        replace_allreduce=False,
    )
    assert partitioned.hidden_states.shape[0] == 2
    assert partitioned.router_logits.shape[0] == 2
    assert method.pad_and_split_input_ids(input_ids).tolist() == [1, 2]


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"expert_heat_collection_interval": "10"}, TypeError),
        ({"algorithm_execution_interval": -1}, ValueError),
        ({"num_redundant_experts": -1}, ValueError),
        ({"eplb_policy_type": 4}, ValueError),
        ({"eplb_heat_collection_stage": "invalid"}, ValueError),
        ({"unknown_key": True}, ValueError),
    ],
)
def test_invalid_eplb_values_follow_rc1_validation(monkeypatch, settings, error):
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat),
        "_additional_config",
        lambda: {"eplb_config": settings},
    )
    with pytest.raises(error):
        compat.get_ascend_config()


@pytest.mark.parametrize(
    "name", ["eplb_config", "ascend_compilation_config", "ascend_fusion_config"]
)
def test_malformed_nested_configuration_is_not_ignored(monkeypatch, name):
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat), "_additional_config", lambda: {name: []}
    )
    with pytest.raises(TypeError, match="must be a mapping"):
        compat.get_ascend_config()


@pytest.mark.parametrize("value", [0, -1, "8192", 1.5])
def test_invalid_mega_capacity_is_rejected(monkeypatch, value):
    compat = _load_compat()
    monkeypatch.setattr(
        _config_owner(compat),
        "_additional_config",
        lambda: {"mega_moe_max_tokens": value},
    )
    with pytest.raises(ValueError, match="positive integer"):
        compat.get_ascend_config()


def test_malformed_current_config_does_not_fall_back_to_defaults(monkeypatch):
    compat = _load_compat()
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(
            get_current_vllm_config=lambda: SimpleNamespace(additional_config=[])
        ),
    )
    with pytest.raises(TypeError, match="additional_config must be a mapping"):
        compat.get_ascend_config()


def test_no_current_context_keeps_supported_defaults(monkeypatch):
    compat = _load_compat()

    def no_context():
        raise AssertionError("No current VllmConfig")

    monkeypatch.setitem(
        sys.modules, "vllm.config", SimpleNamespace(get_current_vllm_config=no_context)
    )
    assert compat.get_ascend_config().enable_fused_mc2 == 0


def _runtime_vllm_config(*, additional_config=None, enable_ep=False, tp_size=1):
    """Build a real vLLM config while keeping this module CPU-only."""
    from vllm.config import DeviceConfig, VllmConfig

    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_config.additional_config = (
        {} if additional_config is None else additional_config
    )
    vllm_config.parallel_config.enable_expert_parallel = enable_ep
    vllm_config.parallel_config.tensor_parallel_size = tp_size
    return vllm_config


def test_worker_owned_config_survives_real_vllm_context_exit():
    """The worker reference, not an expired context, drives fused settings."""
    from vllm.config import set_current_vllm_config

    compat = _load_compat()
    worker_config = _runtime_vllm_config(additional_config={"enable_fused_mc2": 1})
    try:
        with set_current_vllm_config(worker_config):
            compat.init_ascend_config(worker_config)
            assert compat.get_ascend_config().enable_fused_mc2 == 1

        assert compat.get_ascend_config().enable_fused_mc2 == 1
    finally:
        compat.clear_ascend_config()


def test_active_real_context_overrides_then_restores_worker_reference():
    from vllm.config import set_current_vllm_config

    compat = _load_compat()
    worker_config = _runtime_vllm_config(additional_config={"enable_fused_mc2": 1})
    active_config = _runtime_vllm_config(additional_config={"enable_fused_mc2": 0})
    try:
        compat.init_ascend_config(worker_config)
        assert compat.get_ascend_config().enable_fused_mc2 == 1

        with set_current_vllm_config(active_config):
            assert compat.get_ascend_config().enable_fused_mc2 == 0

        assert compat.get_ascend_config().enable_fused_mc2 == 1
    finally:
        compat.clear_ascend_config()


def test_worker_owned_reference_refreshes_and_config_overrides_environment(monkeypatch):
    compat = _load_compat()
    worker_config = _runtime_vllm_config(additional_config={})
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "1")
    try:
        compat.init_ascend_config(worker_config)
        assert compat.get_ascend_config().enable_fused_mc2 == 1

        # The stored owner is deliberately a reference, so a worker refresh
        # takes effect without retaining a stale derived compatibility view.
        worker_config.additional_config["enable_fused_mc2"] = 0
        assert compat.get_ascend_config().enable_fused_mc2 == 0
        worker_config.additional_config["enable_fused_mc2"] = 1
        assert compat.get_ascend_config().enable_fused_mc2 == 1
    finally:
        compat.clear_ascend_config()


def test_clear_and_reinit_change_the_outside_context_owner():
    compat = _load_compat()
    enabled = _runtime_vllm_config(additional_config={"enable_fused_mc2": 1})
    disabled = _runtime_vllm_config(additional_config={"enable_fused_mc2": 0})
    try:
        compat.init_ascend_config(enabled)
        assert compat.get_ascend_config().enable_fused_mc2 == 1

        compat.clear_ascend_config()
        assert compat.get_ascend_config().enable_fused_mc2 == 0

        compat.init_ascend_config(disabled)
        assert compat.get_ascend_config().enable_fused_mc2 == 0
    finally:
        compat.clear_ascend_config()


def test_shared_dp_sp_gate_receives_worker_fallback_config_after_context_exit(
    monkeypatch,
):
    from vllm.config import set_current_vllm_config

    compat = _load_compat()
    worker_config = _runtime_vllm_config(
        additional_config={"enable_shared_expert_dp": True},
        enable_ep=True,
        tp_size=2,
    )
    enable_sp = Mock(return_value=True)
    monkeypatch.setattr(_config_owner(compat), "enable_sp", enable_sp)
    try:
        with set_current_vllm_config(worker_config):
            compat.init_ascend_config(worker_config)
        config = compat.get_ascend_config()
    finally:
        compat.clear_ascend_config()

    assert config.enable_shared_expert_dp is True
    enable_sp.assert_called_once_with(
        vllm_config=worker_config,
        enable_shared_expert_dp=True,
    )


@pytest.mark.parametrize(
    ("additional_config", "error"),
    [
        ({"eplb_config": {"dynamic_eplb": True}}, NotImplementedError),
        ([], TypeError),
    ],
)
def test_outside_context_fallback_remains_fail_closed_for_bad_or_unsupported_config(
    additional_config, error
):
    compat = _load_compat()
    worker_config = _runtime_vllm_config(additional_config=additional_config)
    try:
        compat.init_ascend_config(worker_config)
        with pytest.raises(error):
            compat.get_ascend_config()
    finally:
        compat.clear_ascend_config()


def test_worker_ascend_config_import_is_vendor_guarded_in_constructor_ast():
    """Non-Ascend workers cannot enter the Ascend import branch at init."""
    worker_path = ROOT / "vllm_fl" / "worker" / "worker.py"
    tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    guarded_imports = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        has_ascend_guard = (
            isinstance(node.test, ast.Compare)
            and any(
                isinstance(comparator, ast.Constant)
                and comparator.value == "ascend"
                for comparator in node.test.comparators
            )
        )
        imports_config = any(
            isinstance(child, ast.ImportFrom)
            and child.module
            == "vllm_fl.configs.ascend"
            and any(alias.name == "init_ascend_config" for alias in child.names)
            for child in ast.walk(node)
        )
        calls_init = any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "init_ascend_config"
            for child in ast.walk(node)
        )
        if has_ascend_guard and imports_config and calls_init:
            guarded_imports.append(node)

    assert len(guarded_imports) == 1

    # Execute the actual guard, not a rewritten predicate, without constructing
    # devices or the rest of WorkerFL. Non-Ascend must not even import config.
    import builtins

    branch = compile(ast.Module(body=guarded_imports, type_ignores=[]),
                     str(worker_path), "exec")
    for vendor in ("cuda", "musa", None, "ascend"):
        calls = []
        owner = object()

        def guarded_import(name, *args, **kwargs):
            calls.append(("import", name))
            return SimpleNamespace(
                init_ascend_config=lambda config: calls.append(("init", config))
            )

        namespace = {
            "__builtins__": dict(vars(builtins), __import__=guarded_import),
            "current_platform": SimpleNamespace(vendor_name=vendor),
            "vllm_config": owner,
        }
        exec(branch, namespace)
        if vendor == "ascend":
            assert calls == [
                ("import", "vllm_fl.configs.ascend"),
                ("init", owner),
            ]
        else:
            assert calls == []


def test_sp_helper_resolves_worker_owner_without_explicit_argument(monkeypatch):
    compat = _load_compat()
    owner = _runtime_vllm_config(additional_config={"enable_flashcomm1": True})
    flashcomm = Mock(return_value=True)
    monkeypatch.setattr(_config_owner(compat), "enable_flashcomm1", flashcomm)
    try:
        compat.init_ascend_config(owner)
        assert compat.enable_sp() is True
        flashcomm.assert_called_once_with(owner, enable_shared_expert_dp=False)
    finally:
        compat.clear_ascend_config()
