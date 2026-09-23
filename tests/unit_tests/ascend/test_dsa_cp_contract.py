"""Static contracts for the Ascend-only DeepSeek-V4 DSA-CP closure."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


_ROOT = Path(__file__).resolve().parents[3]
_ASCEND = _ROOT / "vllm_fl" / "dispatch" / "backends" / "vendor" / "ascend"
_ATTENTION = _ROOT / "vllm_fl" / "attention" / "ascend"


def _build_local_token_metadata_method(rank: int):
    """Compile the production method body with CPU-only TP/RoPE stubs."""
    source = (_ATTENTION / "context_parallel" / "dsa_cp.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_local_token_metadata"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "get_tp_group": lambda: SimpleNamespace(world_size=4, rank_in_group=rank),
        "get_cos_and_sin_dsa": lambda *_args, **_kwargs: (None, None),
    }
    exec(compile(module, "dsa_cp_cpu_method", "exec"), namespace)
    return namespace["_build_local_token_metadata"]


def test_cpu_tp4_uneven_partition_keeps_real_lengths_and_dummy_zero() -> None:
    # Three real requests plus a zero-token dummy request; 10 tokens must pad
    # to 12 for TP4, while request metadata must never claim dummy tokens.
    starts = torch.tensor([0, 3, 7, 10, 10], dtype=torch.int32)
    lens = torch.tensor([3, 4, 3, 0], dtype=torch.int32)
    results = []
    for rank in range(4):
        query_buffer = torch.empty(5, dtype=torch.int32)
        seq_buffer = torch.empty(4, dtype=torch.int32)
        result = _build_local_token_metadata_method(rank)(
            SimpleNamespace(),
            num_reqs=4,
            num_input_tokens=10,
            input_positions=None,
            query_start_loc=starts,
            seq_lens=lens,
            use_cache=False,
            local_query_start_loc=query_buffer,
            local_seq_lens=seq_buffer,
        )
        assert result[4].data_ptr() == query_buffer.data_ptr()
        assert result[5].data_ptr() == seq_buffer.data_ptr()
        results.append(result)
    assert [(item[0], item[1], item[2], item[3]) for item in results] == [
        (0, 3, 3, 12), (3, 6, 3, 12), (6, 9, 3, 12), (9, 12, 3, 12)
    ]
    assert [item[4].tolist() for item in results] == [
        [0, 3, 3, 3, 3], [0, 0, 3, 3, 3], [0, 0, 1, 3, 3], [0, 0, 0, 1, 1]
    ]
    assert [item[5].tolist() for item in results] == [
        [3, 0, 0, 0], [0, 3, 0, 0], [0, 4, 2, 0], [0, 0, 3, 0]
    ]


def test_dsa_cp_module_has_no_vllm_ascend_runtime_import() -> None:
    source = (_ATTENTION / "context_parallel" / "dsa_cp.py").read_text()
    assert "vllm_ascend" not in source
    assert "all_gather_async" in source
    assert "get_tp_group" in source


def test_dsa_cp_uses_existing_dsa_ops_without_new_kernel_stub() -> None:
    source = (_ATTENTION / "context_parallel" / "dsa_cp.py").read_text()
    device_ops = (_ASCEND / "device" / "device_op.py").read_text()
    for symbol in (
        "get_dsa_sparse_attn_op",
        "dsa_kv_compress_scatter",
        "unpack_dsa_forward_kv_cache",
    ):
        assert symbol in source
        assert symbol in device_ops


def test_dsa_cp_is_tp_not_dcp_or_pcp_closure() -> None:
    source = (_ATTENTION / "context_parallel" / "dsa_cp.py").read_text()
    assert "get_tp_group" in source
    assert "get_dcp_group" not in source
    assert "get_pcp_group" not in source


def test_runner_distinguishes_regular_and_cp_dsa_ratio_maps() -> None:
    source = (_ROOT / "vllm_fl" / "worker" / "model_runner.py").read_text()
    assert "AscendDSACPMetadataBuilder" in source
    assert "if isinstance(builder, AscendDSAMetadataBuilder):" in source


def test_sequence_row_parallel_skips_flashcomm_padding_for_dsa_cp_wo_b() -> None:
    source = (_ASCEND / "impl" / "linear_op.py").read_text()
    assert "dsa_cp_attention_output" in source
    assert '"wo_b" in self.layer.prefix' in source
    assert "and not dsa_cp_attention_output" in source
