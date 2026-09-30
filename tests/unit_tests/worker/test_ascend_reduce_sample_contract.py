"""CPU contracts for the Ascend reduce-sample execution boundary."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path(__file__).parents[3]
SAMPLER = REPO / "vllm_fl/sample/sampler.py"
RUNNER = REPO / "vllm_fl/worker/model_runner.py"
CONFIG = REPO / "vllm_fl/configs/ascend.py"


def _function(path: Path, name: str):
    source = path.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "Any": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _runner_method(name: str, namespace: dict):
    tree = ast.parse(RUNNER.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunnerFL")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace[name]


class _GatherGroup:
    """Two-rank CPU collective mock; rank zero invokes the tested helper."""

    rank_in_group = 0
    world_size = 2

    def all_gather(self, tensor, dim=-1):
        # Candidate rank one wins row 0; ties must choose rank zero because
        # torch.argmax returns the first gathered position.
        if tensor.dtype.is_floating_point:
            peer = torch.tensor([[4.0], [3.0]])
        else:
            peer = torch.tensor([[3], [2]], dtype=tensor.dtype)
        return torch.cat((tensor, peer), dim=dim)


def test_reduce_greedy_uses_active_group_offsets_and_stable_ties() -> None:
    greedy = _function(SAMPLER, "reduce_sample_greedy")
    logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    # Row 0 peer's value 4 maps to global token 3. Row 1 ties with peer's
    # value 3, so first gathered rank maps to local token 0.
    assert greedy(logits, _GatherGroup()).tolist() == [3, 0]


def test_reduce_contract_has_finite_topk_and_safe_full_vocab_fallback() -> None:
    runner = RUNNER.read_text()
    assert "eligible.size == 0" in runner
    assert "eligible.size != top_k_cpu.size" in runner
    assert "tp_group.all_gather(logits, dim=-1)" in runner
    assert "sampling_metadata.max_num_logprobs is not None" in runner
    assert "sampling_metadata.logprob_token_ids" in runner
    # Empty processor lists do not accidentally disable ordinary requests.
    assert "reduce_sample_processors_are_inactive" in runner
    assert "allowed_token_ids_mask is not None" in runner
    assert "bad_words_token_ids" in runner


class _RunnerGroup:
    rank_in_group = 0
    world_size = 2

    def __init__(self):
        self.calls = []

    def all_gather(self, tensor, dim=-1):
        self.calls.append(tensor.clone())
        return torch.cat((tensor, tensor + 100), dim=dim)


class _RunnerSampler:
    def __init__(self, enabled=True):
        self.enable_reduce_sample = enabled
        self.prepared = []

    def prepare_sampling(self, max_top_k, enabled):
        self.prepared.append((max_top_k, enabled))


def _metadata(**overrides):
    values = dict(
        max_num_logprobs=None, logprob_token_ids={}, allowed_token_ids_mask=None,
        bad_words_token_ids={}, no_penalties=True,
        thinking_budget_state_holder=None, logitsprocs=SimpleNamespace(all=()),
        all_greedy=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _prepared_runner(monkeypatch, enabled=True, device_type="npu"):
    group = _RunnerGroup()
    namespace = {
        "current_platform": SimpleNamespace(device_type=device_type),
        "get_tp_group": lambda: group,
        "envs": SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        "HAS_TRITON": True,
        "reduce_sample_processors_are_inactive": lambda _: True,
    }
    prepare = _runner_method("_prepare_ascend_reduce_sample", namespace)
    restore = _runner_method("_restore_full_ascend_logits", namespace)
    runner = SimpleNamespace(
        sampler=_RunnerSampler(enabled),
        input_batch=SimpleNamespace(vocab_size=10, num_reqs=2,
                                    top_k_cpu=torch.tensor([3, 4, 999]).numpy()),
    )
    runner._restore_full_ascend_logits = types.MethodType(restore, runner)
    return prepare, runner, group, namespace


def test_prepare_executes_finite_topk_and_current_request_slice(monkeypatch) -> None:
    prepare, runner, group, _ = _prepared_runner(monkeypatch)
    logits = torch.zeros(2, 5)
    assert prepare(runner, logits, _metadata()) is logits
    assert runner.sampler.prepared == [(4, True)]
    assert group.calls == []


def test_prepare_empty_topk_logprobs_mask_and_no_triton_penalty_fallback(monkeypatch) -> None:
    prepare, runner, group, namespace = _prepared_runner(monkeypatch)
    logits = torch.zeros(2, 5)
    runner.input_batch.top_k_cpu[:2] = 5
    full = prepare(runner, logits, _metadata())
    assert full.shape == (2, 10) and runner.sampler.prepared[-1] == (None, False)
    assert len(group.calls) == 1
    for metadata in (_metadata(max_num_logprobs=1),
                     _metadata(allowed_token_ids_mask=torch.zeros(2, 5, dtype=torch.bool))):
        prepare(runner, logits, metadata)
        assert runner.sampler.prepared[-1][1] is False
    namespace["HAS_TRITON"] = False
    prepare(runner, logits, _metadata(no_penalties=False))
    assert runner.sampler.prepared[-1][1] is False


def test_prepare_nonascend_or_default_off_has_no_collective(monkeypatch) -> None:
    for enabled, device in ((False, "npu"), (True, "cuda")):
        prepare, runner, group, _ = _prepared_runner(monkeypatch, enabled, device)
        logits = torch.zeros(2, 5)
        assert prepare(runner, logits, _metadata()) is logits
        assert group.calls == [] and runner.sampler.prepared == []


def test_prompt_logprobs_restores_full_vocab_before_global_id_gather() -> None:
    class LogprobsTensors:
        @staticmethod
        def empty_cpu(rows, cols):
            return SimpleNamespace(
                logprob_token_ids=torch.empty(rows, cols, dtype=torch.int32),
                logprobs=torch.empty(rows, cols),
                selected_token_ranks=torch.empty(rows, cols, dtype=torch.int32),
            )

    namespace = {
        "torch": torch,
        "LogprobsTensors": LogprobsTensors,
        "current_platform": SimpleNamespace(device_type="npu"),
    }
    prompt = _runner_method("_get_prompt_logprobs_dict", namespace)
    sampler_calls = []
    sampler = SimpleNamespace(
        enable_reduce_sample=True,
        compute_logprobs=lambda logits: sampler_calls.append(logits) or logits,
        gather_logprobs=lambda logits, n, ids: (
            torch.zeros(1, n + 1, dtype=torch.int32),
            torch.zeros(1, n + 1), torch.zeros(1, n + 1, dtype=torch.int32), None),
    )
    request = SimpleNamespace(prompt_token_ids=[1, 7], num_computed_tokens=0,
                              in_progress_prompt_logprobs_cpu=None)
    restored = []
    runner = SimpleNamespace(
        num_prompt_logprobs={"r": 1}, requests={"r": request}, device="cpu",
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}),
        query_start_loc=SimpleNamespace(np=torch.tensor([0]).numpy()),
        model=SimpleNamespace(compute_logits=lambda _: torch.zeros(1, 5)),
        sampler=sampler, _sync_device=lambda: None,
    )
    runner._restore_full_ascend_logits = lambda logits: restored.append(logits) or torch.zeros(1, 10)
    result = prompt(runner, torch.zeros(2, 2), {"r": 2})
    assert "r" in result and len(restored) == 1
    assert sampler_calls[0].shape == (1, 10)


def test_builtin_processor_gate_is_exact_and_custom_types_fail_closed() -> None:
    source = RUNNER.read_text()
    assert "type(processor) is MinPLogitsProcessor" in source
    assert "hasattr(processor" not in source
    assert "Do not use structural/``hasattr`` matching" in source


def test_reduce_is_vendor_opt_in_and_async_is_separate() -> None:
    config = CONFIG.read_text()
    sampler = SAMPLER.read_text()
    assert 'enable_reduce_sample = bool(extra.get("enable_reduce_sample", False))' in config
    assert "reduce-sample does not support speculative decoding" in config
    assert "reduce-sample does not support LoRA" in config
    assert "if self.enable_reduce_sample:\n            return" in sampler
    assert "self.enable_reduce_sample and enabled" in sampler


def test_local_penalty_padding_is_not_a_valid_next_shard_token() -> None:
    penalties = (REPO / "vllm_fl/sample/penalties.py").read_text()
    assert "padding_token_id = (tp_rank + 1) * vocab_size" in penalties
    assert "reduce_sample=self.topk_topp_sampler.enable_reduce_sample" in SAMPLER.read_text()


def test_candidate_collective_uses_global_ids_and_local_topk(monkeypatch) -> None:
    candidates = _function(SAMPLER, "reduce_sample_candidates")
    calls = []
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(
        npu_top_k_top_p=lambda values, k, p: calls.append((values.clone(), k, p)) or values
    ))

    class Group:
        rank_in_group = 0
        world_size = 2
        def all_gather(self, value, dim=-1):
            if value.dtype.is_floating_point:
                return torch.cat((value, torch.tensor([[4.0, 0.0]])), dim=dim)
            return torch.cat((value, torch.tensor([[4, 5]], dtype=value.dtype)), dim=dim)

    values, ids = candidates(torch.tensor([[1.0, 5.0, 3.0]]),
                             torch.tensor([2]), torch.tensor([0.9]), 2, Group())
    assert values.shape[-1] == 4 and ids.tolist() == [[1, 2, 4, 5]]
    assert len(calls) == 1 and calls[0][1].tolist() == [2]
