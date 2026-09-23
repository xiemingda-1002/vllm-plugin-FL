"""CPU-only contracts for the GLM-5.2 non-MTP Ascend SFA closure."""

import ast
from functools import wraps
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
ASCEND = ROOT / "vllm_fl/dispatch/backends/vendor/ascend"
ATTENTION = ROOT / "vllm_fl/attention/ascend"
PATCHES = ASCEND / "patches"


def test_glm52_constructor_patch_preserves_other_models_and_is_idempotent():
    tree = ast.parse((PATCHES / "patch_glm52.py").read_text())
    patch_fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_glm52_shared_indexer_patch"
    )
    calls = []

    class Attention:
        def __init__(self, *args, **kwargs):
            calls.append(("original", args, kwargs))

    def migrated(self, *args, **kwargs):
        calls.append(("glm", args, kwargs))

    namespace = {
        "wraps": wraps,
        "DeepseekV2MLAAttention": Attention,
        "_deepseek_v2_mla_attention_init": migrated,
    }
    exec(compile(ast.Module(body=[patch_fn], type_ignores=[]), "patch_glm52.py", "exec"), namespace)
    apply = namespace["apply_glm52_shared_indexer_patch"]
    apply()
    patched = Attention.__init__
    apply()
    assert Attention.__init__ is patched
    Attention(None, SimpleNamespace(model_type="deepseek_v2"), 3, prefix="dense")
    Attention(None, SimpleNamespace(model_type="glm_moe_dsa"), 4, prefix="sfa")
    assert [call[0] for call in calls] == ["original", "glm"]
    assert calls[0][1][-1] == 3
    assert calls[1][2] == {"prefix": "sfa"}


def test_glm52_selector_uses_vendor_scoped_sfa_backend() -> None:
    source = (ASCEND / "ascend.py").read_text()
    assert "vllm_fl.attention.ascend.sfa_v1." in source
    assert "AscendSFABackend" in source
    assert "MLA with sparse attention is not implemented" not in source


def test_glm52_sfa_backend_is_non_mtp_and_vendor_scoped() -> None:
    source = (ATTENTION / "sfa_v1.py").read_text()
    rope_source = (ASCEND / "impl/rope.py").read_text()
    assert "class AscendSFABackend" in source
    assert "class AscendSFAImpl" in source
    assert "vllm_ascend" not in source
    assert "patch_deepseek_mtp" not in source
    assert "_EXTRA_CTX, \"cos\"" not in source
    assert "get_cos_and_sin_mla" in source
    assert "record_cos_and_sin_cache_interleaved" in rope_source
    assert "if _cos_cache is not None or _sin_cache is not None" in rope_source
    assert "attr_value.set_" in source
    assert "Placeholder type" not in source


def test_glm52_runner_routes_split_indexer_cache_to_ascend_backend() -> None:
    source = (ROOT / "vllm_fl/worker/model_runner.py").read_text()
    assert "AscendSFAIndexerCacheSpec" in source
    assert "AscendSFAIndexerBackend" in source
    assert "GLM5.2 SFA non-C8 cache" in source
    assert "kv_caches[layer_name] = (k_cache, v_cache)" in source
    assert "kv_caches[layer_name] = (indexer_k_cache,)" in source
    assert "GLM5.2 SFA C8/DCP cache is not migrated" in source
    assert "set_cos_and_sin(" in source


def test_glm52_shared_indexer_init_requires_checkpoint_metadata() -> None:
    source = (PATCHES / "patch_glm52.py").read_text()
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_should_skip_indexer_init"
    )
    helper_text = ast.get_source_segment(source, helper)
    assert "if not skip_topk" in helper_text
    assert "indexer_types" in helper_text
    assert 'indexer_type.lower() == "shared"' in helper_text
    assert "record_cos_and_sin_cache_interleaved" in source


def test_glm52_modelslim_attention_and_indexer_kv_guards_remain_explicit() -> None:
    source = (ASCEND / "impl/quantization/config.py").read_text()
    # The audited GLM-5.2 W8A8 checkpoint has no FA/indexer/C8 metadata.
    # Do not silently invent a quantized attention implementation for it.
    assert "FL ModelSlim quantized attention is not migrated" in source
    assert "FL ModelSlim C8 KV cache is not migrated" in source
