import ast
from pathlib import Path

DSA_SOURCE = (
    Path(__file__).parents[2]
    / "vllm_fl/dispatch/backends/vendor/ascend/attention/dsa_v1.py"
)


def _isolated_dsa_backend() -> type:
    """Execute only the backend class so this contract needs no NPU runtime."""
    tree = ast.parse(DSA_SOURCE.read_text())
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendDSABackend"
    )
    namespace = {"AttentionBackend": type("AttentionBackend", (), {})}
    exec(compile(ast.Module(body=[backend], type_ignores=[]), str(DSA_SOURCE), "exec"), namespace)
    return namespace["AscendDSABackend"]


def test_dsa_kv_cache_shape_accepts_current_vllm_cache_dtype_contract() -> None:
    backend = _isolated_dsa_backend()

    assert backend.get_kv_cache_shape(11, 128, 2, 576) == (11, 128, 2, 576)
    assert backend.get_kv_cache_shape(
        11, 128, 2, 576, cache_dtype_str="bfloat16"
    ) == (11, 128, 2, 576)


def test_dsa_kv_cache_shape_signature_has_optional_cache_dtype() -> None:
    tree = ast.parse(DSA_SOURCE.read_text())
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendDSABackend"
    )
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_kv_cache_shape"
    )

    assert [arg.arg for arg in method.args.args] == [
        "num_blocks",
        "block_size",
        "num_kv_heads",
        "head_size",
        "cache_dtype_str",
    ]
    assert len(method.args.defaults) == 1
    assert isinstance(method.args.defaults[0], ast.Constant)
    assert method.args.defaults[0].value == "auto"
