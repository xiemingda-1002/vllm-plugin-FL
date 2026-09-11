# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
RUNNER_PATH = ROOT / "vllm_fl" / "worker" / "model_runner.py"
WORKER_PATH = ROOT / "vllm_fl" / "worker" / "worker.py"
RUNNER_SOURCE = RUNNER_PATH.read_text(encoding="utf-8")
WORKER_SOURCE = WORKER_PATH.read_text(encoding="utf-8")
SAFE_GATE_PATH = ROOT / "vllm_fl" / "ascend_flashcomm.py"
SAFE_GATE_SOURCE = SAFE_GATE_PATH.read_text(encoding="utf-8")


def _class_method_source(source: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    segment = ast.get_source_segment(source, method)
    assert segment is not None
    return segment


def test_runner_and_worker_reuse_the_unified_ascend_gate() -> None:
    runner_gate_import = (
        "from vllm_fl.ascend_flashcomm import (\n"
        "    enable_flashcomm1,\n"
        "    is_vl_model,\n"
        ")"
    )
    worker_gate_import = (
        "from vllm_fl.ascend_flashcomm import (\n"
        "    enable_flashcomm1,\n"
        ")"
    )
    assert runner_gate_import in RUNNER_SOURCE
    assert worker_gate_import in WORKER_SOURCE
    assert "def enable_flashcomm1(" not in RUNNER_SOURCE
    assert "def enable_flashcomm1(" not in WORKER_SOURCE


def test_runner_primes_shared_vl_cache_only_for_npu() -> None:
    init = _class_method_source(RUNNER_SOURCE, "ModelRunnerFL", "__init__")
    uniform_decode = init.index("self.uniform_decode_query_len =")
    guard = init.index(
        'if current_platform.device_type == "npu":', uniform_decode
    )
    warmup = init.index("is_vl_model(self.vllm_config)")
    mc2_init = init.index("set_mc2_tokens_capacity(")

    assert uniform_decode < guard < warmup < mc2_init
    assert init.count("is_vl_model(self.vllm_config)") == 1


def test_common_runner_worker_gate_does_not_import_ascend_impl_package() -> None:
    forbidden = "vllm_fl.dispatch.backends.vendor.ascend.impl"
    for source in (RUNNER_SOURCE, WORKER_SOURCE):
        top_level_imports = [
            node
            for node in ast.parse(source).body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all(
            not str(getattr(node, "module", "")).startswith(forbidden)
            and all(
                not alias.name.startswith(forbidden)
                for alias in getattr(node, "names", ())
            )
            for node in top_level_imports
        )
    assert forbidden not in SAFE_GATE_SOURCE
    assert "torch_npu" not in SAFE_GATE_SOURCE


def test_query_start_buffer_reserves_only_an_ascend_fia_slot() -> None:
    init = _class_method_source(RUNNER_SOURCE, "ModelRunnerFL", "__init__")
    assert "query_start_loc_size = self.max_num_reqs + 1" in init
    assert 'if current_platform.device_type == "npu":' in init
    assert "query_start_loc_size += 1" in init


def test_flashcomm_and_upstream_pass_sp_have_distinct_padding_gates() -> None:
    pad = _class_method_source(
        RUNNER_SOURCE, "ModelRunnerFL", "_pad_for_sequence_parallelism"
    )
    assert "self.compilation_config.pass_config.enable_sp" in pad
    assert "enable_flashcomm1(self.vllm_config)" in pad
    assert "round_up(num_scheduled_tokens, tp_size)" in pad


def test_fia_padding_preserves_current_rc1_mixed_and_uniform_rules() -> None:
    fia = _class_method_source(
        RUNNER_SOURCE, "ModelRunnerFL", "_pad_query_start_loc_for_fia"
    )
    assert "cudagraph_runtime_mode == CUDAGraphMode.FULL" in fia
    assert "self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL" in fia
    assert "num_reqs_padded * self.uniform_decode_query_len" in fia
    assert "self.arange_np" in fia
    assert "query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded" in fia
    assert "query_start_loc.copy_to_gpu()" in fia


def test_real_forward_uses_fia_padding_and_flashcomm_dp_force_padding() -> None:
    execute = _class_method_source(RUNNER_SOURCE, "ModelRunnerFL", "execute_model")
    determine = _class_method_source(
        RUNNER_SOURCE,
        "ModelRunnerFL",
        "_determine_batch_execution_and_padding",
    )
    assert "self._pad_query_start_loc_for_fia(" in execute
    assert "cudagraph_mode == CUDAGraphMode.FULL" in execute
    assert "flashcomm1_enabled and not self.model_config.use_mla" in execute
    assert "self.compilation_config.pass_config.enable_sp" in determine
    assert "or enable_flashcomm1(self.vllm_config)" in determine


def test_final_hidden_and_aux_states_are_gathered_then_unpadded() -> None:
    model_forward = _class_method_source(
        RUNNER_SOURCE, "ModelRunnerFL", "_model_forward"
    )
    gather = _class_method_source(
        RUNNER_SOURCE,
        "ModelRunnerFL",
        "_all_gather_flashcomm_hidden_states",
    )
    gather_aux = _class_method_source(
        RUNNER_SOURCE,
        "ModelRunnerFL",
        "_all_gather_flashcomm_hidden_states_and_aux",
    )
    assert 'current_platform.device_type != "npu"' in model_forward
    assert '"flash_comm_v1_enabled"' in model_forward
    assert "not isinstance(model_output, IntermediateTensors)" in model_forward
    assert "get_tp_group().all_gather(hidden_states, dim=0)" in gather
    assert 'additional_kwargs.get("pad_size", 0)' in gather
    assert "hidden_states[:-pad_size, :]" in gather
    assert "isinstance(hidden_states, tuple)" in gather_aux
    assert "for aux_hidden_state in aux_hidden_states" in gather_aux


def test_flashcomm_pp_intermediates_stay_local_without_duplicate_gather() -> None:
    sync = _class_method_source(
        RUNNER_SOURCE,
        "ModelRunnerFL",
        "sync_and_gather_intermediate_tensors",
    )
    flash_branch = sync[: sync.index("is_rs =")]
    upstream_branch = sync[sync.index("is_rs =") :]
    assert "local_len = cdiv(num_tokens, tp)" in flash_branch
    assert "value[:local_len]" in flash_branch
    assert ".all_gather(" not in flash_branch
    assert "get_tp_group().all_gather" in upstream_branch


def test_dummy_pp_shape_uses_local_flashcomm_shard() -> None:
    dummy = _class_method_source(RUNNER_SOURCE, "ModelRunnerFL", "_dummy_run")
    assert "self._pad_query_start_loc_for_fia(" in dummy
    assert "max_intermediate_tokens = self.max_num_tokens" in dummy
    assert "max_intermediate_tokens = cdiv(" in dummy
    assert "batch_size=max_intermediate_tokens" in dummy


def test_graph_shape_resolution_keeps_tp_padding_before_dispatch() -> None:
    determine = _class_method_source(
        RUNNER_SOURCE,
        "ModelRunnerFL",
        "_determine_batch_execution_and_padding",
    )
    resolve_graph = _class_method_source(
        RUNNER_SOURCE, "ModelRunnerFL", "_check_and_update_cudagraph_mode"
    )
    assert determine.index("self._pad_for_sequence_parallelism(num_tokens)") < (
        determine.index("dispatch_cudagraph(")
    )
    assert "self.parallel_config.tensor_parallel_size" in resolve_graph
    assert "self.cudagraph_dispatcher.initialize_cudagraph_keys(" in resolve_graph


def test_worker_uses_current_vllm_async_pp_api_and_flashcomm_bypass() -> None:
    init = _class_method_source(WORKER_SOURCE, "WorkerFL", "__init__")
    execute = _class_method_source(WORKER_SOURCE, "WorkerFL", "execute_model")
    assert "self._pp_send_work: list[Handle] = []" in init
    assert "for handle in self._pp_send_work" in execute
    assert "get_pp_group().irecv_tensor_dict(" in execute
    assert "AsyncIntermediateTensors(" in execute
    assert "get_pp_group().isend_tensor_dict(" in execute
    assert "None if flashcomm1_enabled else get_tp_group()" in execute
    assert "and not flashcomm1_enabled" in execute
    assert "compilation_config.pass_config.enable_sp" in execute
