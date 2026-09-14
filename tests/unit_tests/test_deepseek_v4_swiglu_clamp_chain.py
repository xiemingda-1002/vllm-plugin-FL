"""DeepSeek-V4 routed/shared SwiGLU clamp-chain regressions."""

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/models/deepseek_v4.py"
MOE_MLP = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl/moe/moe_mlp.py"


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _method(path: Path, class_name: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _branch(function: ast.FunctionDef, name: str) -> ast.If:
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == name
    )


def _positive_limit_branch(function: ast.FunctionDef) -> ast.If:
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "swiglu_limit"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Gt)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == 0
    )


def _run_branch(branch: ast.If, namespace: dict) -> dict:
    condition = compile(ast.Expression(branch.test), "<clamp-condition>", "eval")
    assert eval(condition, namespace)
    # The Triton fallback imports its implementation locally. The extracted
    # branch test supplies that callable directly so it can execute without an
    # Ascend/FlagGems import environment.
    body = ast.Module(
        body=[statement for statement in branch.body if not isinstance(statement, (ast.Import, ast.ImportFrom))],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(body), "<clamp-branch>", "exec"), namespace)
    return namespace


def test_deepseek_v4_passes_checkpoint_limit_to_routed_fused_moe() -> None:
    init = _method(MODEL, "DeepseekV4MoE", "__init__")
    calls = [node for node in ast.walk(init) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    shared_call = next(node for node in calls if node.func.id == "DeepseekV2MLP")
    routed_call = next(node for node in calls if node.func.id == "FusedMoE")

    def swiglu_keyword(call: ast.Call) -> ast.expr:
        return next(keyword.value for keyword in call.keywords if keyword.arg == "swiglu_limit")

    # Evaluate the extracted expressions against the same model instance: both
    # construction paths must receive the checkpoint limit, not a default.
    model = SimpleNamespace(swiglu_limit=10.0)
    namespace = {"self": model}
    assert eval(compile(ast.Expression(swiglu_keyword(shared_call)), "<shared-limit>", "eval"), namespace) == 10.0
    assert eval(compile(ast.Expression(swiglu_keyword(routed_call)), "<routed-limit>", "eval"), namespace) == 10.0


def test_routed_quant_positive_limit_bypasses_limit_unaware_triton_fallback() -> None:
    function = _function(MOE_MLP, "quant_apply_mlp")
    branch = _positive_limit_branch(function)
    calls: list[tuple[str, object]] = []

    def clipped(hidden_states, **kwargs):
        calls.append(("clipped", kwargs))
        return "clipped-output"

    def dynamic_quant(hidden_states, **kwargs):
        calls.append(("dynamic_quant", {"hidden_states": hidden_states, **kwargs}))
        return "quantized-output", "quant-scale"

    namespace = {
        "swiglu_limit": 10.0,
        "swiglu_alpha": 1.0,
        "swiglu_beta": 0.0,
        "hidden_states": "gate-up",
        "act_quant_type": "int8",
        "use_mxfp_quant": False,
        "torch_npu": SimpleNamespace(npu_clipped_swiglu=clipped),
        "DeviceOperator": SimpleNamespace(npu_dynamic_quant=dynamic_quant),
    }
    result = _run_branch(branch, namespace)

    assert calls == [
        ("clipped", {"interleaved": False, "alpha": 1.0, "limit": 10.0, "bias": 0.0}),
        ("dynamic_quant", {"hidden_states": "clipped-output", "act_quant_type": "int8", "use_mxfp_quant": False}),
    ]
    assert result["hidden_states"] == "quantized-output"
    assert result["swiglu_out_scale"] == "quant-scale"


def test_zero_limit_retains_existing_triton_fallback_and_special_activation_branches() -> None:
    function = _function(MOE_MLP, "quant_apply_mlp")
    positive = _positive_limit_branch(function)
    triton = _branch(function, "HAS_TRITON")
    assert not eval(compile(ast.Expression(positive.test), "<positive-limit>", "eval"), {"swiglu_limit": 0.0})

    calls: list[tuple] = []

    def swiglu_quant(hidden_states, group_list, group_list_type):
        calls.append((hidden_states, group_list, group_list_type))
        return "triton-output", "triton-scale"

    namespace = {
        "HAS_TRITON": True,
        "hidden_states": "gate-up",
        "group_list": "groups",
        "group_list_type": 1,
        "swiglu_quant": swiglu_quant,
    }
    result = _run_branch(triton, namespace)

    assert calls == [("gate-up", "groups", 1)]
    assert result["hidden_states"] == "triton-output"
    assert result["swiglu_out_scale"] == "triton-scale"
