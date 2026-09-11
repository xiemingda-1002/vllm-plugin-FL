# Copyright (c) 2026 BAAI. All rights reserved.

"""Static contracts for the current rc1 Ascend sampling completion path."""

import ast
from pathlib import Path


REPO = Path(__file__).parents[3]
RUNNER_PATH = REPO / "vllm_fl" / "worker" / "model_runner.py"
SAMPLER_PATH = REPO / "vllm_fl" / "sample" / "sampler.py"
RUNNER_SOURCE = RUNNER_PATH.read_text()
SAMPLER_SOURCE = SAMPLER_PATH.read_text()
RUNNER_TREE = ast.parse(RUNNER_SOURCE)
SAMPLER_TREE = ast.parse(SAMPLER_SOURCE)


def _method_source(tree: ast.Module, source: str, cls: str, name: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    segment = ast.get_source_segment(source, item)
                    assert segment is not None
                    return segment
    raise AssertionError(f"{cls}.{name} was not found")


def test_sampler_is_injected_only_for_ascend() -> None:
    init = _method_source(RUNNER_TREE, RUNNER_SOURCE, "ModelRunnerFL", "__init__")

    upstream_default = init.index("sampler_cls = Sampler")
    ascend_guard = init.index('current_platform.device_type == "npu"')
    ascend_import = init.index(
        "from vllm_fl.sample.sampler import AscendSampler"
    )
    constructor = init.index("self.sampler = sampler_cls")
    assert upstream_default < ascend_guard < ascend_import < constructor
    assert "use_fp64_gumbel=self.model_config.use_fp64_gumbel" in init


def test_sampler_module_keeps_torch_npu_import_lazy() -> None:
    top_level_imports = [
        node
        for node in SAMPLER_TREE.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not (
            isinstance(node, ast.Import)
            and any(alias.name == "torch_npu" for alias in node.names)
        )
        for node in top_level_imports
    )
    apply_filter = next(
        node
        for node in SAMPLER_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_top_k_top_p"
    )
    assert "import torch_npu" in ast.unparse(apply_filter)


def test_random_sampling_uses_global_stream_and_return_dependency() -> None:
    random_sample = next(
        node
        for node in SAMPLER_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "random_sample"
    )
    source = ast.get_source_segment(SAMPLER_SOURCE, random_sample)
    assert source is not None
    assert "with torch.npu.stream(stream):" in source
    assert "noise.exponential_" in source
    assert "torch.npu.current_stream().wait_stream(stream)" in source
    assert "use_fp64_gumbel" in source


def test_async_exponential_is_launched_before_model_forward() -> None:
    execute = _method_source(
        RUNNER_TREE, RUNNER_SOURCE, "ModelRunnerFL", "execute_model"
    )
    launch = execute.index("self.sampler.do_async_exponential")
    forward_marker = execute.index("# Run the model.")
    assert launch < forward_marker
    assert "async_exponential_enabled()" in execute
    assert "self.input_batch.sampling_metadata.generators" in execute


def test_hybrid_state_update_follows_sampling_event_before_return() -> None:
    sample_tokens = _method_source(
        RUNNER_TREE, RUNNER_SOURCE, "ModelRunnerFL", "sample_tokens"
    )
    sampled = sample_tokens.index("sampler_output = self._sample")
    assert "self.sampling_done_event = torch.npu.Event()" in sample_tokens
    event_record = sample_tokens.index("self.sampling_done_event.record()")
    bookkeeping = sample_tokens.index("self._bookkeeping_sync")
    state_stream = sample_tokens.index(
        'record_function_or_nullcontext("async_state_update")'
    )
    event_wait = sample_tokens.index("stream.wait_event(self.sampling_done_event)")
    state_update = sample_tokens.rindex("self._update_states_after_model_execute")
    sync_return = sample_tokens.index("if not self.use_async_scheduling:")

    assert sampled < event_record < bookkeeping < state_stream
    assert state_stream < event_wait < state_update < sync_return
    assert "else:\n            self._update_states_after_model_execute" in sample_tokens


def test_mamba_groups_select_the_hybrid_completion_path_only_on_ascend() -> None:
    initialize = _method_source(
        RUNNER_TREE, RUNNER_SOURCE, "ModelRunnerFL", "initialize_kv_cache"
    )
    assert 'current_platform.device_type == "npu"' in initialize
    assert "self.need_accepted_tokens = any(" in initialize
    assert "isinstance(attn_group[0].kv_cache_spec, MambaSpec)" in initialize


def test_batch_invariant_mode_retains_upstream_sampling() -> None:
    forward = _method_source(
        SAMPLER_TREE,
        SAMPLER_SOURCE,
        "AscendTopKTopPSampler",
        "forward_native",
    )
    assert "if envs.VLLM_BATCH_INVARIANT:" in forward
    assert "return super().forward_native(logits, generators, k, p)" in forward
