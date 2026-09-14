import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASCEND = ROOT / "vllm_fl/dispatch/backends/vendor/ascend"
RUNNER = ROOT / "vllm_fl/worker/model_runner.py"


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_deepseek_runtime_is_fl_owned_and_vendor_scoped() -> None:
    checked = [
        *ASCEND.joinpath("attention").rglob("*.py"),
        *ASCEND.joinpath("core").rglob("*.py"),
        *ASCEND.joinpath("device").rglob("*.py"),
        *ASCEND.joinpath("distributed").rglob("*.py"),
        *ASCEND.joinpath("models").rglob("*.py"),
        *ASCEND.joinpath("ops").rglob("*.py"),
        ASCEND / "dsa_compat.py",
        ASCEND / "memcache_comm_fence.py",
        ROOT / "vllm_fl/models/deepseek_v4.py",
        ROOT / "vllm_fl/patches/deepseek_v4.py",
    ]
    for path in checked:
        source = path.read_text(encoding="utf-8")
        assert "from vllm_ascend" not in source, path
        assert "import vllm_ascend" not in source, path


def test_deepseek_registry_and_dsa_registration_are_explicit() -> None:
    registry = _source("vllm_fl/patches/deepseek_v4.py")
    assert 'register_model("DeepseekV4ForCausalLM"' in registry
    assert "vllm_fl.models.deepseek_v4:AscendDeepseekV4ForCausalLM" in registry

    patch = _source("vllm_fl/dispatch/backends/vendor/ascend/patch.py")
    assert "ensure_dsa_forward_registered()" in patch
    assert "apply_deepseek_v4_patches()" in patch

    dsa = _source("vllm_fl/dispatch/backends/vendor/ascend/ops/dsa.py")
    assert "def ensure_dsa_forward_registered()" in dsa
    assert 'op_name="dsa_forward"' in dsa
    assert 'mutates_args=["output"]' in dsa
    assert 'dispatch_key="PrivateUse1"' in dsa


def test_a2_a3_dsa_cache_contract_has_exact_six_slots() -> None:
    dsa = _source("vllm_fl/dispatch/backends/vendor/ascend/ops/dsa.py")
    expected_order = """(
                    compress_kv_cache,
                    swa_kv_cache,
                    state_cache,
                    indexer_state_cache,
                    indexer_k_cache,
                    indexer_scale_cache,
                )"""
    assert expected_order in dsa


def _unfold_kvcache_from_source():
    """Execute only the dependency-free boundary helper, never device imports."""
    source = _source("vllm_fl/dispatch/backends/vendor/ascend/ops/dsa.py")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "unfold_kvcache"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(isolated), "<unfold_kvcache>", "exec"), namespace)
    return namespace["unfold_kvcache"]


def test_dsa_cache_normalization_unwraps_only_singleton_wrappers() -> None:
    unfold_kvcache = _unfold_kvcache_from_source()
    tensor_sentinel = object()
    k_cache, scale_cache = object(), object()
    k_scale_pair = (k_cache, scale_cache)
    multi_list = [k_cache, scale_cache]

    assert unfold_kvcache([(tensor_sentinel,)]) is tensor_sentinel
    assert unfold_kvcache((tensor_sentinel,)) is tensor_sentinel
    assert unfold_kvcache(k_scale_pair) is k_scale_pair
    assert unfold_kvcache([k_scale_pair]) is k_scale_pair
    assert unfold_kvcache(multi_list) is multi_list
    assert unfold_kvcache(()) == ()
    assert unfold_kvcache([]) == []
    assert unfold_kvcache(None) is None


def test_worker_initializes_ascend_parallel_groups_only_for_hccl() -> None:
    worker = _source("vllm_fl/worker/worker.py")
    assert 'if backend == "hccl":' in worker
    assert "init_ascend_model_parallel(parallel_config)" in worker


def test_active_ascend_linear_preserves_deepseek_wo_a_grouped_layout() -> None:
    patch = _source("vllm_fl/dispatch/backends/vendor/ascend/patch.py")
    assert "from .impl.linear import (" in patch
    assert '"ColumnParallelLinear": AscendColumnParallelLinear' in patch

    linear = _source(
        "vllm_fl/dispatch/backends/vendor/ascend/impl/linear.py"
    )
    assert 'if "wo_a" in prefix:' in linear
    assert 'getattr(hf_config, "o_groups", 0) // self.tp_size' in linear
    assert 'getattr(hf_config, "o_lora_rank", 0)' in linear
    assert 'if "wo_a" in self.prefix' in linear
    assert "get_ascend_device_type() != AscendDeviceType.A5" in linear
    assert "self.weight.ndim == 2" in linear
    assert ".view(\n                        self.n_local_groups," in linear
    assert ".transpose(2, 1)" in linear
    assert "loaded_weight.narrow(" in linear
    assert "Unexpected wo_a weight shape" in linear


def _runner_call_keyword_expressions(
    function_name: str, keyword_name: str
) -> list[ast.expr]:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    calls = sorted(
        (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == function_name)
            or (isinstance(node.func, ast.Name) and node.func.id == function_name)
        )
        ),
        key=lambda node: node.lineno,
    )
    expressions = []
    for call in calls:
        keyword = next((kw for kw in call.keywords if kw.arg == keyword_name), None)
        if keyword is not None:
            assert keyword.value is not None
            expressions.append(keyword.value)
    return expressions


def _eval_runner_expression(expr: ast.expr, **locals_: object) -> object:
    expression = ast.Expression(expr)
    ast.fix_missing_locations(expression)
    return eval(compile(expression, str(RUNNER), "eval"), {}, locals_)


def test_dsa_sync_none_keeps_padded_slot_mapping_extent() -> None:
    """A local decode 3 can retain capture padding 4 after DP sync selects NONE."""
    expr = _runner_call_keyword_expressions(
        "_get_slot_mappings", "num_tokens_padded"
    )[0]
    value = _eval_runner_expression(
        expr,
        self=type("Runner", (), {"use_compress": True})(),
        pad_attn=False,
        has_separate_kv_update=False,
        num_tokens_unpadded=3,
        num_tokens_padded=4,
    )
    assert value == 4


def test_dsa_metadata_calls_keep_padded_extent_for_mixed_and_dummy_none() -> None:
    expressions = _runner_call_keyword_expressions(
        "_build_attention_metadata", "num_tokens_padded"
    )
    assert len(expressions) == 2  # normal execute and dummy/capture only
    locals_ = {
        "self": type("Runner", (), {"use_compress": True})(),
        "pad_attn": False,
        "num_tokens_padded": 4,
    }
    assert [_eval_runner_expression(expr, **locals_) for expr in expressions] == [4, 4]


def test_dsa_metadata_calls_keep_padded_extent_for_idle_and_full_capture() -> None:
    expressions = _runner_call_keyword_expressions(
        "_build_attention_metadata", "num_tokens_padded"
    )
    dsa_idle = {
        "self": type("Runner", (), {"use_compress": True})(),
        "pad_attn": False,
        "num_tokens_padded": 4,
    }
    assert [_eval_runner_expression(expr, **dsa_idle) for expr in expressions] == [4, 4]
    generic_full = {**dsa_idle, "self": type("Runner", (), {"use_compress": False})(), "pad_attn": True}
    assert [_eval_runner_expression(expr, **generic_full) for expr in expressions] == [4, 4]


def test_dsa_metadata_keeps_local_actual_and_padded_input_contract() -> None:
    expr = _runner_call_keyword_expressions(
        "common_metadata_cls", "num_actual_tokens"
    )[0]
    assert (
        _eval_runner_expression(
            expr,
            self=type("Runner", (), {"use_compress": True})(),
            num_tokens=3,
            num_tokens_padded=4,
        )
        == 3
    )
    runner = _source("vllm_fl/worker/model_runner.py")
    assert "cm_base.num_input_tokens = num_tokens_padded" in runner
    assert "slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)" in runner


def test_non_dsa_none_and_full_metadata_rules_remain_unchanged() -> None:
    tokens_exprs = _runner_call_keyword_expressions(
        "_build_attention_metadata", "num_tokens_padded"
    )
    reqs_expr = _runner_call_keyword_expressions(
        "_build_attention_metadata", "num_reqs_padded"
    )[0]
    generic = type("Runner", (), {"use_compress": False})()
    none_locals = {
        "self": generic,
        "pad_attn": False,
        "num_tokens_padded": 4,
        "num_reqs_padded": 2,
    }
    assert [_eval_runner_expression(expr, **none_locals) for expr in tokens_exprs] == [None, None]
    assert _eval_runner_expression(reqs_expr, **none_locals) is None
    full_locals = {**none_locals, "pad_attn": True}
    assert [_eval_runner_expression(expr, **full_locals) for expr in tokens_exprs] == [4, 4]
    assert _eval_runner_expression(reqs_expr, **full_locals) == 2

    dsa_single_dp = {
        "self": type("Runner", (), {"use_compress": True})(),
        "pad_attn": False,
        "num_tokens_padded": 3,
    }
    assert [_eval_runner_expression(expr, **dsa_single_dp) for expr in tokens_exprs] == [3, 3]
