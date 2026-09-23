from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_fl/worker/model_runner.py"


def test_dsv4_compressed_runner_lifecycle_is_vendor_scoped() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'current_platform.device_type == "npu"' in source
    assert 'getattr(hf_config, "model_type", None) == "deepseek_v4"' in source
    assert 'hasattr(hf_config, "compress_ratios")' in source
    assert "_dsa_positions_cpu_buf" in source
    assert "np.copyto(" in source
    assert "AscendCommonAttentionMetadata" in source
    assert "AscendDSAMetadataBuilder" in source


def test_dsv4_compressed_graph_and_cache_paths_are_explicit() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "self.positions.fill_(127)" in source
    assert "self._dsa_positions_cpu_buf.fill_(127)" in source
    assert "self.positions.zero_()" in source
    assert "self.use_compress and force_attention" in source
    assert "DeepSeek-V4 compressed DSA sparse-C8 is not" in source
    assert "DeepSeek-V4 compressed DSA does not support" in source
    assert "extract_dsv4_layer_index" in source
    assert "DeepSeek-V4 compressed DSA does not support a" in source
    assert "compressed/SWA/state/indexer attention caches" in source


def test_dummy_slot_sentinel_precedes_ascend_metadata_builder() -> None:
    """Keep the pre-#442 Ascend dummy ordering without changing CUDA's path."""
    source = RUNNER.read_text(encoding="utf-8")
    dummy_start = source.index("    def _dummy_run(")
    next_method = source.index("\n    def ", dummy_start + 1)
    dummy_source = source[dummy_start:next_method]
    ascend_fill = dummy_source.index(
        'current_platform.device_type == "npu"'
    )
    builder = dummy_source.index("attn_metadata, _ = self._build_attention_metadata(")
    common_fill = dummy_source.rindex(
        "self.common_attention_metadata_graph is not None"
    )

    assert ascend_fill < builder < common_fill
    assert "slot_mapping.fill_(-1)" in dummy_source[ascend_fill:builder]
    assert "slot_mapping.fill_(-1)" in dummy_source[common_fill:]


def test_dsv4_indexer_page_strides_keep_k_and_scale_in_each_page() -> None:
    """Exercise the rc1 per-page K/scale layout used by the runner reshape."""
    blocks, block_size, head_dim, scale_dim = 2, 4, 8, 1
    k_bytes = block_size * head_dim
    scale_bytes = block_size * scale_dim * torch.empty((), dtype=torch.float16).element_size()
    page_bytes = k_bytes + scale_bytes
    raw = torch.arange(blocks * page_bytes, dtype=torch.int8)

    k_cache = torch.as_strided(
        raw,
        size=(blocks, block_size, 1, head_dim),
        stride=(page_bytes, head_dim, head_dim, 1),
    )
    scale_cache = torch.as_strided(
        raw.view(torch.float16),
        size=(blocks, block_size, 1, scale_dim),
        stride=(page_bytes // 2, scale_dim, scale_dim, 1),
        storage_offset=k_bytes // 2,
    )

    assert k_cache.stride(0) == page_bytes
    assert scale_cache.stride(0) == page_bytes // 2
    assert k_cache[1, 0, 0, 0].item() == raw[page_bytes].item()
    assert scale_cache.data_ptr() - raw.data_ptr() == k_bytes


def _keyword_expression(function_name: str, keyword_name: str) -> ast.expr:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == function_name)
            or (isinstance(node.func, ast.Name) and node.func.id == function_name)
        )
    )
    keyword = next(kw for kw in call.keywords if kw.arg == keyword_name)
    assert keyword.value is not None
    return keyword.value


def _dummy_keyword_expression(function_name: str, keyword_name: str) -> ast.expr:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    dummy_run = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_dummy_run"
    )
    call = next(
        node
        for node in ast.walk(dummy_run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == function_name
    )
    keyword = next(kw for kw in call.keywords if kw.arg == keyword_name)
    assert keyword.value is not None
    return keyword.value


def _eval_expr(expr: ast.expr, **locals_: object) -> object:
    expression = ast.Expression(expr)
    ast.fix_missing_locations(expression)
    return eval(compile(expression, str(RUNNER), "eval"), {}, locals_)


def test_dsv4_compressed_metadata_retains_dp_padding_after_sync_mode_none() -> None:
    """CPU contract for local decode-3 captured as 4 then DP-synced to NONE."""
    compressed = SimpleNamespace(use_compress=True)
    generic = SimpleNamespace(use_compress=False)
    # The rank initially dispatches FULL at 4, then synchronized mode becomes
    # NONE because another DP rank prefills. Its padded extent must survive.
    values = {
        "self": compressed,
        "pad_attn": False,
        "has_separate_kv_update": False,
        "num_tokens_unpadded": 3,
        "num_tokens_padded": 4,
        "num_reqs_padded": 2,
    }

    slot_expr = _keyword_expression("_get_slot_mappings", "num_tokens_padded")
    assert _eval_expr(slot_expr, **values) == 4

    metadata_expr = _keyword_expression(
        "_build_attention_metadata", "num_tokens_padded"
    )
    assert _eval_expr(metadata_expr, **values) == 4
    assert _eval_expr(metadata_expr, **{**values, "self": generic}) is None
    request_expr = _keyword_expression(
        "_build_attention_metadata", "num_reqs_padded"
    )
    assert _eval_expr(request_expr, **values) is None

    full_values = {**values, "self": generic, "pad_attn": True}
    assert _eval_expr(slot_expr, **full_values) == 4
    assert _eval_expr(metadata_expr, **full_values) == 4
    assert _eval_expr(request_expr, **full_values) == 2

    common_expr = _keyword_expression("common_metadata_cls", "num_actual_tokens")
    assert (
        _eval_expr(
            common_expr, self=compressed, num_tokens=3, num_tokens_padded=4
        )
        == 3
    )
    assert (
        _eval_expr(
            common_expr, self=generic, num_tokens=3, num_tokens_padded=4
        )
        == 4
    )

    source = RUNNER.read_text(encoding="utf-8")
    assert "cm_base.num_input_tokens = num_tokens_padded" in source
    assert "slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)" in source


def test_normal_runner_bridges_actual_tokens_but_dummy_keeps_padded_rc1_contract() -> None:
    """Only normal execution scopes scheduler actual tokens for the MC2 mask."""
    source = RUNNER.read_text(encoding="utf-8")
    assert "from vllm_fl.ascend_forward_context import override_actual_num_tokens" in source
    assert "scheduler_output.total_num_scheduled_tokens" in source
    assert "ascend_actual_tokens_scope" in source

    tree = ast.parse(source)
    set_context_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set_forward_context"
    ]
    assert len(set_context_calls) == 2  # normal execution and dummy/profile
    assert all(
        any(
            keyword.arg == "num_tokens"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "num_tokens_padded"
            for keyword in call.keywords
        )
        for call in set_context_calls
    )


def test_dsv4_dummy_keeps_actual_and_dp_padded_request_counts_distinct() -> None:
    """Idle DP DSA metadata must clear rows added by graph padding."""
    actual_expr = _dummy_keyword_expression("_build_attention_metadata", "num_reqs")
    padded_expr = _dummy_keyword_expression(
        "_build_attention_metadata", "num_reqs_padded"
    )
    values = {"num_reqs": 1, "num_reqs_padded": 4}

    compressed = SimpleNamespace(use_compress=True)
    assert _eval_expr(actual_expr, self=compressed, **values) == 1
    assert _eval_expr(padded_expr, self=compressed, **values) == 4

    generic = SimpleNamespace(use_compress=False)
    assert _eval_expr(actual_expr, self=generic, **values) == 4
    assert _eval_expr(padded_expr, self=generic, **values) is None
