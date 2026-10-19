# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual decode-graph dispatch state.

Why this exists
---------------
With the general configuration (``--max-model-len 69632`` + block-sparse
attention) only the 1k-1k scenario misses the 80 % line, and the whole gap is
the block indexer plus the index-key chain that the short-context configuration
skips (measured: 57.83 ms -> 47.16 ms per decode step). The indexer cannot
be skipped per operator (it is worth <= 2.3 ms, and the ``n <= topk`` predicate
silently computes the wrong thing: the index-key cache is cross-step state).
What *can* be switched safely is the **graph**, a start-up-time artifact:

* ``LONG``  -- the shipping block-sparse path (indexer + index-K writes);
* ``SHORT`` -- the identity selection (no indexer, no index-K write).

A step replays ``SHORT`` only when **every** request in the batch has a
lifetime upper bound ``prompt_tokens + max_tokens <= bound``, which the
scheduler fixes at admission. Such a request can never pass ``bound``, so it
never changes class, so the graph never has to change mid-request -- which is
what makes the skipped index-key state unobservable (no mid-request switch).

The switch is a start-up constant (``VLLM_FL_M3_DUAL_GRAPH=1``; default off).
When it is off every function here is a no-op and the runner/wrapper code takes
the exact pre-existing path.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter

#: Graph kinds. ``KIND_LONG`` is the shipping path and the default everywhere.
KIND_LONG = 0
KIND_SHORT = 1
_KINDS = (KIND_LONG, KIND_SHORT)

DUAL_GRAPH_SWITCH = "VLLM_FL_M3_DUAL_GRAPH"
SHORT_SIZES_SWITCH = "VLLM_FL_M3_DUAL_GRAPH_SIZES"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() == "1"


#: Resolved once at import: the switch must not change inside a process, because
#: a second graph set is captured at start-up and cannot appear later.
_DUAL_GRAPH = _env_flag(DUAL_GRAPH_SWITCH)

#: Current step's kind. Read by ``GraphWrapper`` (which graph set to use) and by
#: the model (which traced path to emit at capture time / which eager body to
#: run). Written by the runner once per step, before the model is called.
_kind = KIND_LONG

#: Non-``None`` only inside ``begin_capture``/``end_capture``: forces the kind so
#: the two capture passes cannot be re-decided from the dummy batch.
_capture_kind: int | None = None

#: Capture sizes actually covered by the short set (empty => dual is inert).
_short_sizes: frozenset[int] = frozenset()

#: ``sparse_topk_blocks * sparse_block_size``, registered by the model once the
#: layers are built (single source of truth: the model's
#: ``_shortctx_selection_bound``).
_bound: int | None = None

#: Per-step counters for operational visibility (never used for control flow).
#: ``short_trace_hits`` counts executions of the *traced* short branch during the
#: short capture pass; zero means the compiler reused the long artifact.
STATS = {
    "short_steps": 0,
    "long_steps": 0,
    "short_fallbacks": 0,
    "short_trace_hits": 0,
}


def dual_graph_enabled() -> bool:
    """True when the two-graph mode was requested at start-up."""
    return _DUAL_GRAPH


def graph_kind() -> int:
    """Kind of the step currently being dispatched."""
    return _kind


def short_active() -> bool:
    """True only when dual mode is on *and* this step replays the short graph."""
    return _DUAL_GRAPH and _kind == KIND_SHORT


def set_graph_kind(kind: int) -> None:
    global _kind
    if kind not in _KINDS:
        raise ValueError(f"invalid graph kind {kind!r}")
    _kind = kind


def capture_kind() -> int | None:
    """Kind being captured, or ``None`` outside the capture loop."""
    return _capture_kind


def begin_capture(kind: int) -> None:
    global _capture_kind
    if kind not in _KINDS:
        raise ValueError(f"invalid graph kind {kind!r}")
    _capture_kind = kind
    set_graph_kind(kind)


def end_capture() -> None:
    global _capture_kind
    _capture_kind = None
    set_graph_kind(KIND_LONG)


#: ``VLLM_FL_M3_DUAL_GRAPH_RETRACE=0`` turns the per-pass compile reset off.
#: Kept only as a bisecting escape hatch: with the reset off the second capture
#: pass reuses the artifact of the first one, which silently produces a "short"
#: set identical to the long one.
RETRACE_SWITCH = "VLLM_FL_M3_DUAL_GRAPH_RETRACE"

#: FX-graph fingerprints of every graph the custom backend compiled, keyed by
#: the pass that compiled it (``None`` = the pre-capture profiling run).
#: Fingerprints are what proves "the two graph sets really differ": with the
#: reset installed the short trace must emit a *different* FX graph (fewer
#: index norm/RoPE nodes), so identical fingerprints mean the compiler reused
#: the artifact and the runner must refuse to start.
_TRACED: dict[int | None, list[dict]] = {}


def retrace_enabled() -> bool:
    """Whether the per-pass compile reset is armed (default: yes)."""
    return os.environ.get(RETRACE_SWITCH, "1").strip() != "0"


#: ``VLLM_FL_M3_DUAL_GRAPH_KV_RESERVE=1`` arms the KV-budget reservation for the
#: duplicated graph set. Off by default, and the switch *is* connected to the
#: real budget point (``vllm_fl/worker/worker.py`` ``determine_available_memory``)
#: -- unlike T58's doubling, which sat behind a CUDA-only call gate and therefore
#: never ran on this stack (T60: KV pool 416,640 tokens, the single-set value).
#: T60 also showed 22 graphs fit without the reservation, so arming it is a
#: margin decision, not a correctness one: it costs two extra capture shapes at
#: start-up and shrinks the pool by the graph memory.
KV_RESERVE_SWITCH = "VLLM_FL_M3_DUAL_GRAPH_KV_RESERVE"


def kv_reserve_enabled() -> bool:
    """Whether the NPU KV budget reserves memory for both graph sets."""
    return _env_flag(KV_RESERVE_SWITCH)


def short_covered_sizes() -> frozenset[int]:
    """Padded batch sizes for which a short graph was captured."""
    return _short_sizes


def set_short_covered_sizes(sizes) -> None:
    global _short_sizes
    _short_sizes = frozenset(int(x) for x in sizes)


def short_sizes_for_capture(all_sizes) -> list[int]:
    """Capture sizes for the short set (``VLLM_FL_M3_DUAL_GRAPH_SIZES`` filter).

    Default: every size, i.e. a short graph for any batch shape. The filter is a
    memory lever (graph memory scales with the captured token counts); a size
    that is not captured simply falls back to the long graph, which is always
    correct.
    """
    sizes = [int(x) for x in all_sizes]
    raw = os.environ.get(SHORT_SIZES_SWITCH, "").strip()
    if not raw:
        return sizes
    wanted = set()
    for item in raw.replace(" ", ",").split(","):
        if item.strip():
            wanted.add(int(item))
    return [x for x in sizes if x in wanted]


def selection_bound() -> int | None:
    """``sparse_topk_blocks * sparse_block_size``, or ``None`` before the model."""
    return _bound


def set_selection_bound(bound: int) -> None:
    global _bound
    bound = int(bound)
    if bound <= 0:
        raise ValueError(f"invalid short-context selection bound {bound}")
    _bound = bound


def request_is_short_lifetime(num_prompt_tokens: int, max_tokens) -> bool:
    """Whether a request can never exceed the identity-selection bound.

    Conservative by construction: an unspecified ``max_tokens`` (generate until
    EOS) is *not* short, and neither is a request whose bound is unknown yet.
    The class is fixed at admission and never recomputed for a running request,
    which is what guarantees "never switch graphs mid-request".
    """
    bound = _bound
    if not _DUAL_GRAPH or bound is None:
        return False
    if max_tokens is None:
        return False
    try:
        total = int(num_prompt_tokens) + int(max_tokens)
    except (TypeError, ValueError):
        return False
    return total <= bound


def decide_step_kind(
    *,
    full_graph: bool,
    uniform: bool,
    num_tokens: int,
    all_short: bool,
) -> int:
    """Pure dispatch decision for one step: which graph set to replay.

    Short only for a **uniform full-graph decode** whose **every** request is
    short-lived and whose padded shape was captured for the short set. Anything
    else -- prefill, chunked prefill, mixed batch, eager step, uncovered shape,
    or a single long-lived request -- replays the long graph, which is always
    correct. Keeping the table here (instead of in the runner) makes it testable
    without torch and gives the fallback a single definition.
    """
    if not _DUAL_GRAPH:
        return KIND_LONG
    if not (full_graph and uniform):
        return KIND_LONG
    if int(num_tokens) not in _short_sizes:
        return KIND_LONG
    return KIND_SHORT if all_short else KIND_LONG


def check_short_replay(all_short: bool, worst_seq: int) -> None:
    """Fail fast on anything that would make a short replay unsafe.

    ``worst_seq`` is the sequence length this step reaches
    (``num_computed_tokens`` + this step's scheduled tokens, max over the
    batch), computed by the caller from host arrays only.

    Raising is the point: past the bound the identity selection attends a
    different key set than the indexer would, which changes the output without
    any error. A misclassified request must abort the engine, not degrade.
    """
    bound = _bound
    if bound is None:
        raise RuntimeError(
            "Dual-graph: the short-context selection bound was never registered "
            "(the MiniMax-M3 sparse layers were not built through "
            "vllm_fl.models.minimax_m3_ascend)."
        )
    if not all_short:
        raise RuntimeError(
            "Dual-graph: refusing to replay the SHORT graph for a batch that is "
            "not entirely short-lived."
        )
    if int(worst_seq) > bound:
        raise RuntimeError(
            f"Dual-graph: SHORT graph replay would reach seq_len {worst_seq} > "
            f"{bound} (sparse_topk_blocks * sparse_block_size). The identity "
            "selection equals the indexer's key set only up to that bound; "
            "aborting instead of silently attending a different key set. This "
            "means a request was misclassified at admission."
        )
    return None


def traced_graphs(kind: int | None) -> list[dict]:
    """FX fingerprints the backend recorded while tracing ``kind``."""
    return list(_TRACED.get(kind, ()))


def reset_traced_graphs() -> None:
    """Forget every fingerprint (start of a fresh capture, and for tests)."""
    _TRACED.clear()


def _target_name(target) -> str:
    name = getattr(target, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return str(target)


def note_traced_graph(graph, compile_range=None) -> dict | None:
    """Fingerprint one FX graph handed to the custom backend.

    Called from ``vllm_fl.compilation.compiler_interface.AscendCompiler.compile``
    while a capture pass is open, so :func:`capture_kind` says which set this
    graph belongs to. The fingerprint is the *emitted kernel sequence* (node op
    + target name, in FX order) plus its length and md5 -- no tensor values and
    no node ids, so two traces of the same Python body give the same value and
    two different bodies (short: no indexer norm/RoPE, no ``cat``) do not.
    """
    if not _DUAL_GRAPH:
        return None
    try:
        nodes = list(graph.graph.nodes)
    except Exception:  # pragma: no cover - a non-FX object means nothing to do
        return None
    calls = Counter()
    sequence = []
    for node in nodes:
        name = _target_name(node.target)
        sequence.append(f"{node.op}:{name}")
        if node.op == "call_function":
            calls[name] += 1
    text = "\n".join(sequence)
    fingerprint = {
        "fx_md5": hashlib.md5(text.encode("utf-8")).hexdigest(),
        "nodes": len(nodes),
        "kernels": sum(calls.values()),
        "distinct_kernels": len(calls),
        "top_kernels": calls.most_common(4),
        "compile_range": str(compile_range),
    }
    _TRACED.setdefault(capture_kind(), []).append(fingerprint)
    return fingerprint


def install_no_aot_reload() -> bool:
    """Never load a pre-compiled AOT artifact while dual mode is on.

    ``vllm/compilation/decorators.py:284-330`` loads
    ``$VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile/<hash>/rank_R_D/model``
    at start-up and calls ``disable_guard_check()`` on it. That artifact was
    compiled with *one* body (whichever kind happened to be active), and the
    cache key does not carry the kind, so loading it would silently pin both
    graph sets to one body -- the same failure one process earlier. The patch is
    process-wide and idempotent; ``__call__`` resolves the name at call time.
    """
    try:
        from vllm.compilation import decorators as core_decorators
    except Exception:
        return False
    if getattr(core_decorators, "_fl_dual_graph_no_aot_reload", False):
        return True

    def _never_load(model, aot_compilation_path):
        print(
            "[DUAL_GRAPH] not loading the AOT artifact "
            f"{aot_compilation_path}: it was compiled for a single graph kind "
            "(a fresh trace is needed on every capture pass)",
            flush=True,
        )
        return None

    core_decorators._try_load_aot_compiled_fn = _never_load
    core_decorators._fl_dual_graph_no_aot_reload = True
    return True


def _reset_model_state(model) -> None:
    """Minimal fallback for :func:`force_fresh_trace` (no vLLM helper)."""
    if hasattr(model, "aot_compiled_fn"):
        model.aot_compiled_fn = None
    if hasattr(model, "was_aot_compile_fn_loaded_from_disk"):
        model.was_aot_compile_fn_loaded_from_disk = False
    try:
        model.__class__.forward.__code__ = model.original_code_object()
    except Exception:
        pass


def force_fresh_trace(model) -> dict:
    """Make the next model call re-trace the body under the current kind.

    Three caches would otherwise serve the artifact compiled for the *other*
    kind, and none of them can see a Python flag:

    * ``vllm/compilation/decorators.py:513-520`` stores ``aot_compiled_fn`` on
      the model instance and calls it for every later forward, so Dynamo is not
      in the call chain at all;
    * ``vllm/compilation/wrapper.py:47-53/106-125`` drops every Dynamo guard
      (``dynamic=False``, ``fullgraph=True``) -- "Dynamo should never be traced
      again after that" -- so even a fresh Dynamo call would hit the cached
      frame without re-running the Python body;
    * ``vllm/compilation/decorators.py:284-330`` reloads the AOT artifact from
      disk across restarts (see :func:`install_no_aot_reload`).

    ``reset_compile_wrapper`` is vLLM's own re-compile primitive (it clears
    ``aot_compiled_fn``, restores ``forward.__code__`` from the bytecode hook
    and rebuilds the ``torch.compile`` callable; elastic EP uses it at
    ``vllm/distributed/elastic_ep/elastic_execute.py:344``). ``torch._dynamo.reset``
    covers the frame cache that it leaves alone. The artifact the *previous*
    pass saved is removed so a short-body artifact can never outlive the
    process (the last pass is the long one, which then saves a long artifact).
    """
    status = {
        "reset_wrapper": False,
        "dynamo_reset": False,
        "removed_artifact": "",
        "error": "",
    }
    if not (_DUAL_GRAPH and retrace_enabled()):
        return status
    install_no_aot_reload()
    try:
        from vllm.compilation.wrapper import reset_compile_wrapper

        reset_compile_wrapper(model)
        status["reset_wrapper"] = True
    except Exception as exc:  # pragma: no cover - needs a live vLLM install
        _reset_model_state(model)
        status["error"] = f"reset_compile_wrapper failed: {exc!r}"
    try:
        import torch

        torch._dynamo.reset()
        status["dynamo_reset"] = True
    except Exception as exc:  # pragma: no cover - needs torch
        status["error"] = (status["error"] + " | " if status["error"] else "") + (
            f"torch._dynamo.reset failed: {exc!r}"
        )
    path = getattr(model, "_aot_compilation_path", None)
    if isinstance(path, str) and os.path.exists(path):
        try:
            os.remove(path)
            status["removed_artifact"] = path
        except OSError as exc:  # pragma: no cover - cache dir permissions
            status["error"] = (status["error"] + " | " if status["error"] else "") + (
                f"could not remove {path}: {exc!r}"
            )
    return status


def reset_for_tests() -> None:  # pragma: no cover - exercised by local checks
    global _capture_kind, _short_sizes, _bound
    _capture_kind = None
    _short_sizes = frozenset()
    _bound = None
    _TRACED.clear()
    STATS.update(
        short_steps=0, long_steps=0, short_fallbacks=0, short_trace_hits=0
    )
    set_graph_kind(KIND_LONG)


# Install the AOT-reload block as early as possible: the *profiling* compile
# happens before the capture loop, and it must not load a stale artifact either
# (otherwise the first pass would have no verified-fresh trace and no
# ``_aot_compilation_path`` to drop). No-op when dual mode is off.
if _DUAL_GRAPH:  # pragma: no cover - needs a live vLLM install
    install_no_aot_reload()
