# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for the dual decode-graph dispatch state.

``vllm_fl.compilation.graph_kind`` keeps every dispatch decision in plain
Python: the start-up switch, the short/long step state machine, the
admission-time class of a request, the replay guard and the FX fingerprints
that prove the two captured graph sets really differ. None of that needs a
device, so the module is executed straight from its source file here.
(Importing ``vllm_fl.compilation`` instead would pull in torch, vLLM and the
vendor plugins, which this logic deliberately does not depend on.)

vLLM and torch are imported lazily inside two helpers; when they are not
installed, minimal stand-ins are registered for exactly the symbols those
helpers use, and the real modules are used whenever they are importable.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "vllm_fl" / "compilation" / "graph_kind.py"
_SWITCHES = (
    "VLLM_FL_M3_DUAL_GRAPH",
    "VLLM_FL_M3_DUAL_GRAPH_SIZES",
    "VLLM_FL_M3_DUAL_GRAPH_RETRACE",
)


def _install_missing_dependencies() -> None:
    """Register stand-ins only for the modules that cannot be imported."""
    try:
        importlib.import_module("torch._dynamo")
    except ImportError:
        torch = types.ModuleType("torch")
        dynamo = types.ModuleType("torch._dynamo")
        dynamo.reset = lambda: None
        torch._dynamo = dynamo
        sys.modules.setdefault("torch", torch)
        sys.modules["torch._dynamo"] = dynamo
    try:
        importlib.import_module("vllm.compilation.decorators")
        importlib.import_module("vllm.compilation.wrapper")
    except Exception:
        # Import can fail with more than ImportError, e.g. two platform plugins.
        vllm = types.ModuleType("vllm")
        compilation = types.ModuleType("vllm.compilation")
        decorators = types.ModuleType("vllm.compilation.decorators")
        wrapper = types.ModuleType("vllm.compilation.wrapper")
        wrapper.reset_compile_wrapper = lambda model: None
        vllm.compilation = compilation
        compilation.decorators = decorators
        compilation.wrapper = wrapper
        sys.modules.setdefault("vllm", vllm)
        sys.modules["vllm.compilation"] = compilation
        sys.modules["vllm.compilation.decorators"] = decorators
        sys.modules["vllm.compilation.wrapper"] = wrapper


_install_missing_dependencies()


def _exec_module():
    """Execute ``graph_kind.py`` in a fresh module under the current environment."""
    spec = importlib.util.spec_from_file_location("graph_kind_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_graph_kind(monkeypatch):
    """Return a loader that executes graph_kind with exactly the given switches."""

    def _load(**env):
        for switch in _SWITCHES:
            monkeypatch.delenv(switch, raising=False)
        for switch, value in env.items():
            assert switch in _SWITCHES, f"unexpected switch {switch}"
            monkeypatch.setenv(switch, value)
        return _exec_module()

    return _load


@pytest.fixture
def gk(load_graph_kind):
    """graph_kind loaded with the dual-graph switch on."""
    return load_graph_kind(VLLM_FL_M3_DUAL_GRAPH="1")


@pytest.fixture
def gk_off(load_graph_kind):
    """graph_kind loaded with every switch at its default (feature off)."""
    return load_graph_kind()


def _decide(module, **overrides):
    """Call ``decide_step_kind`` with an all-short decode batch by default."""
    kwargs = {
        "full_graph": True,
        "uniform": True,
        "num_tokens": 32,
        "all_short": True,
    }
    kwargs.update(overrides)
    return module.decide_step_kind(**kwargs)


def _fake_fx_graph(*targets):
    """A ``torch.fx.GraphModule`` look-alike; only ``.graph.nodes`` is read."""
    nodes = [types.SimpleNamespace(op="placeholder", target="inputs")]
    nodes.extend(
        types.SimpleNamespace(op="call_function", target=target) for target in targets
    )
    nodes.append(types.SimpleNamespace(op="output", target="output"))
    return types.SimpleNamespace(graph=types.SimpleNamespace(nodes=nodes))


#: Two bodies for the same batch shape: the long one also runs the indexer and
#: writes the index-key cache, the short one does neither.
_LONG_BODY = ("rms_norm", "indexer", "index_key_insert", "sparse_attention")
_SHORT_BODY = ("rms_norm", "sparse_attention")


class TestSwitch:
    """The dual-graph switch and the two auxiliary switches."""

    def test_switch_names_are_stable(self, gk_off):
        assert gk_off.DUAL_GRAPH_SWITCH == "VLLM_FL_M3_DUAL_GRAPH"
        assert gk_off.SHORT_SIZES_SWITCH == "VLLM_FL_M3_DUAL_GRAPH_SIZES"
        assert gk_off.RETRACE_SWITCH == "VLLM_FL_M3_DUAL_GRAPH_RETRACE"

    def test_kind_constants_are_distinct(self, gk_off):
        assert gk_off.KIND_LONG != gk_off.KIND_SHORT

    def test_unset_switch_is_off(self, gk_off):
        assert gk_off.dual_graph_enabled() is False
        assert gk_off.short_active() is False

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", True),
            ("0", False),
            ("", False),
            ("2", False),
            ("true", False),
            ("yes", False),
        ],
    )
    def test_switch_truth_table(self, load_graph_kind, raw, expected):
        module = load_graph_kind(**{"VLLM_FL_M3_DUAL_GRAPH": raw})
        assert module.dual_graph_enabled() is expected

    def test_switch_tolerates_surrounding_whitespace(self, load_graph_kind):
        module = load_graph_kind(**{"VLLM_FL_M3_DUAL_GRAPH": " 1 "})
        assert module.dual_graph_enabled() is True

    def test_retrace_defaults_to_on(self, gk_off):
        assert gk_off.retrace_enabled() is True

    @pytest.mark.parametrize(
        ("raw", "expected"), [("1", True), ("0", False), ("2", True)]
    )
    def test_retrace_truth_table(self, load_graph_kind, raw, expected):
        module = load_graph_kind(**{"VLLM_FL_M3_DUAL_GRAPH_RETRACE": raw})
        assert module.retrace_enabled() is expected


class TestKindStateMachine:
    """``graph_kind`` / ``short_active`` / ``begin_capture`` / ``end_capture``."""

    def test_default_kind_is_long(self, gk):
        assert gk.graph_kind() == gk.KIND_LONG
        assert gk.capture_kind() is None

    def test_set_graph_kind_round_trip(self, gk):
        gk.set_graph_kind(gk.KIND_SHORT)
        assert gk.graph_kind() == gk.KIND_SHORT
        gk.set_graph_kind(gk.KIND_LONG)
        assert gk.graph_kind() == gk.KIND_LONG

    @pytest.mark.parametrize("kind", [-1, 2, 99, None])
    def test_set_graph_kind_rejects_unknown_values(self, gk, kind):
        with pytest.raises(ValueError, match="invalid graph kind"):
            gk.set_graph_kind(kind)

    def test_short_active_needs_kind_and_switch(self, gk):
        assert gk.short_active() is False
        gk.set_graph_kind(gk.KIND_SHORT)
        assert gk.short_active() is True

    def test_short_kind_is_invisible_when_switch_is_off(self, gk_off):
        gk_off.set_graph_kind(gk_off.KIND_SHORT)
        assert gk_off.graph_kind() == gk_off.KIND_SHORT
        assert gk_off.short_active() is False

    def test_begin_capture_forces_the_kind(self, gk):
        gk.begin_capture(gk.KIND_SHORT)
        assert gk.capture_kind() == gk.KIND_SHORT
        assert gk.graph_kind() == gk.KIND_SHORT
        assert gk.short_active() is True

    @pytest.mark.parametrize("kind", [-1, 2, None])
    def test_begin_capture_rejects_unknown_values(self, gk, kind):
        with pytest.raises(ValueError, match="invalid graph kind"):
            gk.begin_capture(kind)

    def test_end_capture_restores_the_long_kind(self, gk):
        gk.begin_capture(gk.KIND_SHORT)
        gk.end_capture()
        assert gk.capture_kind() is None
        assert gk.graph_kind() == gk.KIND_LONG
        assert gk.short_active() is False

    def test_end_capture_is_idempotent(self, gk):
        gk.end_capture()
        gk.end_capture()
        assert gk.capture_kind() is None
        assert gk.graph_kind() == gk.KIND_LONG


class TestRequestLifetime:
    """``request_is_short_lifetime`` -- admission-time, conservative."""

    def test_bound_must_be_registered(self, gk):
        assert gk.selection_bound() is None
        assert gk.request_is_short_lifetime(1024, 1024) is False

    def test_registered_bound_is_reported(self, gk):
        gk.set_selection_bound(2048)
        assert gk.selection_bound() == 2048

    @pytest.mark.parametrize("bound", [0, -1, -2048])
    def test_set_selection_bound_rejects_non_positive_values(self, gk, bound):
        with pytest.raises(ValueError, match="invalid short-context selection bound"):
            gk.set_selection_bound(bound)

    def test_total_equal_to_the_bound_is_short(self, gk):
        gk.set_selection_bound(2048)
        assert gk.request_is_short_lifetime(1024, 1024) is True

    def test_total_below_the_bound_is_short(self, gk):
        gk.set_selection_bound(2048)
        assert gk.request_is_short_lifetime(1, 1) is True

    def test_one_token_past_the_bound_is_long(self, gk):
        gk.set_selection_bound(2048)
        assert gk.request_is_short_lifetime(1024, 1025) is False

    def test_unbounded_max_tokens_is_long(self, gk):
        gk.set_selection_bound(2048)
        assert gk.request_is_short_lifetime(16, None) is False

    @pytest.mark.parametrize(
        ("prompt_tokens", "max_tokens"), [(1024, "many"), (None, 5), ("x", 1)]
    )
    def test_non_numeric_input_is_long(self, gk, prompt_tokens, max_tokens):
        gk.set_selection_bound(2048)
        assert gk.request_is_short_lifetime(prompt_tokens, max_tokens) is False

    def test_numeric_strings_are_accepted(self, gk):
        gk.set_selection_bound(2048)
        assert gk.request_is_short_lifetime("1024", "1024") is True

    def test_class_is_long_when_switch_is_off(self, gk_off):
        gk_off.set_selection_bound(2048)
        assert gk_off.request_is_short_lifetime(1024, 1024) is False


class TestDecideStepKind:
    """The dispatch table: short only for a covered all-short full-graph step."""

    def test_all_conditions_hold_gives_short(self, gk):
        gk.set_short_covered_sizes([32, 64])
        assert _decide(gk) == gk.KIND_SHORT

    def test_prefill_gives_long(self, gk):
        gk.set_short_covered_sizes([32, 64])
        assert _decide(gk, full_graph=False) == gk.KIND_LONG

    def test_non_uniform_batch_gives_long(self, gk):
        gk.set_short_covered_sizes([32, 64])
        assert _decide(gk, uniform=False) == gk.KIND_LONG

    def test_mixed_batch_gives_long(self, gk):
        gk.set_short_covered_sizes([32, 64])
        assert _decide(gk, all_short=False) == gk.KIND_LONG

    def test_uncovered_shape_gives_long(self, gk):
        gk.set_short_covered_sizes([32, 64])
        assert _decide(gk, num_tokens=128) == gk.KIND_LONG

    def test_no_captured_short_set_gives_long(self, gk):
        assert gk.short_covered_sizes() == frozenset()
        assert _decide(gk) == gk.KIND_LONG

    def test_switch_off_gives_long_for_every_input(self, gk_off):
        gk_off.set_short_covered_sizes([32, 64])
        gk_off.set_selection_bound(2048)
        assert _decide(gk_off) == gk_off.KIND_LONG
        assert _decide(gk_off, all_short=False) == gk_off.KIND_LONG

    def test_decision_has_no_side_effects(self, gk):
        gk.set_short_covered_sizes([32])
        before = dict(gk.STATS)
        assert _decide(gk) == gk.KIND_SHORT
        assert _decide(gk, num_tokens=100) == gk.KIND_LONG
        assert dict(gk.STATS) == before
        assert gk.graph_kind() == gk.KIND_LONG


class TestCheckShortReplay:
    """``check_short_replay`` -- the fail-fast guard around a short replay."""

    def test_equal_to_the_bound_is_allowed(self, gk):
        gk.set_selection_bound(2048)
        assert gk.check_short_replay(True, 2048) is None

    def test_below_the_bound_is_allowed(self, gk):
        gk.set_selection_bound(2048)
        assert gk.check_short_replay(True, 1) is None

    def test_above_the_bound_raises(self, gk):
        gk.set_selection_bound(2048)
        with pytest.raises(RuntimeError, match="would reach seq_len"):
            gk.check_short_replay(True, 2049)

    def test_mixed_batch_raises(self, gk):
        gk.set_selection_bound(2048)
        with pytest.raises(RuntimeError, match="not entirely short-lived"):
            gk.check_short_replay(False, 0)

    def test_unregistered_bound_raises(self, gk):
        with pytest.raises(RuntimeError, match="never registered"):
            gk.check_short_replay(True, 0)

    def test_guard_does_not_change_the_kind(self, gk):
        gk.set_selection_bound(2048)
        with pytest.raises(RuntimeError):
            gk.check_short_replay(True, 4096)
        assert gk.graph_kind() == gk.KIND_LONG


class TestShortSizesForCapture:
    """``short_sizes_for_capture`` -- which shapes get a short graph."""

    def test_every_size_by_default(self, load_graph_kind):
        module = load_graph_kind()
        assert module.short_sizes_for_capture([16, 32, 64]) == [16, 32, 64]

    def test_filter_keeps_the_intersection(self, load_graph_kind):
        module = load_graph_kind(VLLM_FL_M3_DUAL_GRAPH_SIZES="32,64")
        assert module.short_sizes_for_capture([16, 32, 64]) == [32, 64]

    def test_filter_with_a_single_size(self, load_graph_kind):
        module = load_graph_kind(VLLM_FL_M3_DUAL_GRAPH_SIZES="64")
        assert module.short_sizes_for_capture([16, 32, 64]) == [64]

    def test_filter_with_spaces(self, load_graph_kind):
        module = load_graph_kind(VLLM_FL_M3_DUAL_GRAPH_SIZES=" 32 , 64 ")
        assert module.short_sizes_for_capture([16, 32, 64]) == [32, 64]

    def test_unrelated_filter_captures_nothing(self, load_graph_kind):
        module = load_graph_kind(VLLM_FL_M3_DUAL_GRAPH_SIZES="1000")
        assert module.short_sizes_for_capture([16, 32, 64]) == []

    def test_empty_filter_keeps_everything(self, load_graph_kind):
        module = load_graph_kind(VLLM_FL_M3_DUAL_GRAPH_SIZES="  ")
        assert module.short_sizes_for_capture([16, 32]) == [16, 32]

    def test_filter_does_not_register_the_sizes(self, load_graph_kind):
        module = load_graph_kind(VLLM_FL_M3_DUAL_GRAPH_SIZES="32,64")
        module.short_sizes_for_capture([16, 32, 64])
        assert module.short_covered_sizes() == frozenset()


class TestGraphFingerprints:
    """``note_traced_graph`` / ``traced_graphs`` -- the two sets must differ."""

    def test_switch_off_records_nothing(self, gk_off):
        assert gk_off.note_traced_graph(_fake_fx_graph(*_LONG_BODY)) is None
        assert gk_off.traced_graphs(gk_off.KIND_LONG) == []
        assert gk_off.traced_graphs(gk_off.KIND_SHORT) == []

    def test_fingerprint_is_recorded_under_the_capture_kind(self, gk):
        gk.begin_capture(gk.KIND_SHORT)
        fingerprint = gk.note_traced_graph(_fake_fx_graph(*_SHORT_BODY))
        gk.end_capture()
        assert fingerprint is not None
        assert gk.traced_graphs(gk.KIND_SHORT) == [fingerprint]
        assert gk.traced_graphs(gk.KIND_LONG) == []

    def test_fingerprint_describes_the_emitted_graph(self, gk):
        gk.begin_capture(gk.KIND_LONG)
        fingerprint = gk.note_traced_graph(
            _fake_fx_graph(*_LONG_BODY), compile_range="[0, 1)"
        )
        assert set(fingerprint) == {
            "fx_md5",
            "nodes",
            "kernels",
            "distinct_kernels",
            "top_kernels",
            "compile_range",
        }
        # placeholder + 4 call_function + output
        assert fingerprint["nodes"] == 6
        assert fingerprint["kernels"] == 4
        assert fingerprint["distinct_kernels"] == 4
        assert dict(fingerprint["top_kernels"]) == {name: 1 for name in _LONG_BODY}
        assert fingerprint["compile_range"] == "[0, 1)"
        assert len(fingerprint["fx_md5"]) == 32

    def test_same_body_gives_the_same_fingerprint(self, gk):
        gk.begin_capture(gk.KIND_SHORT)
        first = gk.note_traced_graph(_fake_fx_graph(*_SHORT_BODY))
        second = gk.note_traced_graph(_fake_fx_graph(*_SHORT_BODY))
        assert first["fx_md5"] == second["fx_md5"]
        assert len(gk.traced_graphs(gk.KIND_SHORT)) == 2

    def test_the_two_bodies_are_disjoint(self, gk):
        gk.begin_capture(gk.KIND_SHORT)
        gk.note_traced_graph(_fake_fx_graph(*_SHORT_BODY))
        gk.end_capture()
        gk.begin_capture(gk.KIND_LONG)
        gk.note_traced_graph(_fake_fx_graph(*_LONG_BODY))
        gk.end_capture()
        short = {item["fx_md5"] for item in gk.traced_graphs(gk.KIND_SHORT)}
        long = {item["fx_md5"] for item in gk.traced_graphs(gk.KIND_LONG)}
        assert short and long
        assert short.isdisjoint(long)

    def test_reused_artifact_is_detected_as_a_shared_fingerprint(self, gk):
        gk.begin_capture(gk.KIND_SHORT)
        gk.note_traced_graph(_fake_fx_graph(*_LONG_BODY))
        gk.end_capture()
        gk.begin_capture(gk.KIND_LONG)
        gk.note_traced_graph(_fake_fx_graph(*_LONG_BODY))
        gk.end_capture()
        short = {item["fx_md5"] for item in gk.traced_graphs(gk.KIND_SHORT)}
        long = {item["fx_md5"] for item in gk.traced_graphs(gk.KIND_LONG)}
        assert short & long

    def test_non_fx_object_is_ignored(self, gk):
        gk.begin_capture(gk.KIND_SHORT)
        assert gk.note_traced_graph(object()) is None
        assert gk.traced_graphs(gk.KIND_SHORT) == []

    def test_reset_traced_graphs_clears_every_pass(self, gk):
        gk.begin_capture(gk.KIND_SHORT)
        gk.note_traced_graph(_fake_fx_graph(*_SHORT_BODY))
        gk.reset_traced_graphs()
        assert gk.traced_graphs(gk.KIND_SHORT) == []

    def test_real_torch_fx_graph_is_fingerprinted(self, gk):
        torch = pytest.importorskip("torch")
        fx = pytest.importorskip("torch.fx")
        graph = fx.Graph()
        placeholder = graph.placeholder("x")
        graph.output(graph.call_function(torch.sin, (placeholder,)))
        graph_module = fx.GraphModule(torch.nn.Module(), graph)

        gk.begin_capture(gk.KIND_SHORT)
        fingerprint = gk.note_traced_graph(graph_module)

        assert fingerprint["nodes"] == 3
        assert fingerprint["kernels"] == 1
        assert [name for name, _ in fingerprint["top_kernels"]] == ["sin"]


class TestResetForTests:
    """``reset_for_tests`` must restore every piece of module state."""

    def test_reset_clears_state_and_counters(self, gk):
        gk.set_selection_bound(2048)
        gk.set_short_covered_sizes([32])
        gk.begin_capture(gk.KIND_SHORT)
        gk.note_traced_graph(_fake_fx_graph(*_SHORT_BODY))
        gk.STATS["short_steps"] = 5
        gk.STATS["short_fallbacks"] = 2

        gk.reset_for_tests()

        assert gk.capture_kind() is None
        assert gk.graph_kind() == gk.KIND_LONG
        assert gk.selection_bound() is None
        assert gk.short_covered_sizes() == frozenset()
        assert gk.traced_graphs(gk.KIND_SHORT) == []
        assert gk.traced_graphs(gk.KIND_LONG) == []
        assert gk.STATS == {
            "short_steps": 0,
            "long_steps": 0,
            "short_fallbacks": 0,
            "short_trace_hits": 0,
        }


class TestDisabledSemantics:
    """With the switch off the module is inert: everything takes the long path."""

    def test_dispatch_always_long(self, gk_off):
        gk_off.set_short_covered_sizes([32])
        gk_off.set_selection_bound(2048)
        assert _decide(gk_off) == gk_off.KIND_LONG

    def test_short_replay_is_never_active(self, gk_off):
        gk_off.set_graph_kind(gk_off.KIND_SHORT)
        assert gk_off.short_active() is False

    def test_requests_are_never_short_lived(self, gk_off):
        gk_off.set_selection_bound(2048)
        assert gk_off.request_is_short_lifetime(1, 1) is False

    def test_no_fingerprints_are_recorded(self, gk_off):
        gk_off.begin_capture(gk_off.KIND_SHORT)
        assert gk_off.note_traced_graph(_fake_fx_graph(*_SHORT_BODY)) is None
        assert gk_off.traced_graphs(gk_off.KIND_SHORT) == []

    def test_force_fresh_trace_is_a_no_op(self, gk_off, tmp_path):
        artifact = tmp_path / "model"
        artifact.write_text("stale", encoding="utf-8")
        model = types.SimpleNamespace(_aot_compilation_path=str(artifact))

        status = gk_off.force_fresh_trace(model)

        assert status == {
            "reset_wrapper": False,
            "dynamo_reset": False,
            "removed_artifact": "",
            "error": "",
        }
        assert artifact.exists()


class TestFreshTraceHelpers:
    """The two helpers that do reach into vLLM / torch (lazily imported)."""

    def test_install_no_aot_reload_patches_once(self, gk_off, monkeypatch):
        decorators = importlib.import_module("vllm.compilation.decorators")
        monkeypatch.setattr(
            decorators, "_try_load_aot_compiled_fn", lambda *args: "original"
        )
        monkeypatch.setattr(
            decorators, "_fl_dual_graph_no_aot_reload", False, raising=False
        )

        assert gk_off.install_no_aot_reload() is True
        patched = decorators._try_load_aot_compiled_fn
        assert patched(object(), "/tmp/model") is None
        assert decorators._fl_dual_graph_no_aot_reload is True
        # idempotent: the second call keeps the same patch object
        assert gk_off.install_no_aot_reload() is True
        assert decorators._try_load_aot_compiled_fn is patched

    def test_force_fresh_trace_resets_and_drops_the_artifact(
        self, gk, monkeypatch, tmp_path
    ):
        calls = []
        wrapper = importlib.import_module("vllm.compilation.wrapper")
        dynamo = importlib.import_module("torch._dynamo")
        monkeypatch.setattr(
            wrapper, "reset_compile_wrapper", lambda model: calls.append("wrapper")
        )
        monkeypatch.setattr(dynamo, "reset", lambda: calls.append("dynamo"))
        artifact = tmp_path / "model"
        artifact.write_text("stale", encoding="utf-8")
        model = types.SimpleNamespace(_aot_compilation_path=str(artifact))

        status = gk.force_fresh_trace(model)

        assert calls == ["wrapper", "dynamo"]
        assert status["reset_wrapper"] is True
        assert status["dynamo_reset"] is True
        assert status["removed_artifact"] == str(artifact)
        assert status["error"] == ""
        assert not artifact.exists()

    def test_force_fresh_trace_reports_a_failing_helper(self, gk, monkeypatch):
        calls = []
        wrapper = importlib.import_module("vllm.compilation.wrapper")
        dynamo = importlib.import_module("torch._dynamo")

        def _boom(model):
            raise RuntimeError("no wrapper")

        monkeypatch.setattr(wrapper, "reset_compile_wrapper", _boom)
        monkeypatch.setattr(dynamo, "reset", lambda: calls.append("dynamo"))

        status = gk.force_fresh_trace(types.SimpleNamespace())

        assert status["reset_wrapper"] is False
        assert status["dynamo_reset"] is True
        assert "reset_compile_wrapper failed" in status["error"]
