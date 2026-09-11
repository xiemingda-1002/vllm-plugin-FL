from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = (
    ROOT
    / "vllm_fl/dispatch/backends/vendor/ascend/patches/patch_mamba_config.py"
)
ASCEND_PATCH_PATH = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/patch.py"


class _Logger:
    def debug(self, *args) -> None:
        pass

    def info(self, *args) -> None:
        pass

    def warning(self, *args) -> None:
        pass


class _MambaModelConfig:
    calls = 0

    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        cls.calls += 1


class _ModelRegistry:
    model_cls = None

    @classmethod
    def resolve_model_cls(cls, architecture, *, model_config):
        assert architecture == model_config.architecture
        return cls.model_cls, None


def _load_functions():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_using_kv_store", "verify_and_update_config"}
    ]
    namespace = {
        "__name__": "test_ascend_mamba_config",
        "math": math,
        "init_logger": lambda name: _Logger(),
        "MambaModelConfig": _MambaModelConfig,
        "ModelRegistry": _ModelRegistry,
        "STR_DTYPE_TO_TORCH_DTYPE": {"bf16": "bf16"},
        "get_dtype_size": lambda dtype: {"bf16": 2, "fp32": 4}[dtype],
        "cdiv": lambda a, b: (a + b - 1) // b,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), SOURCE_PATH, "exec"),
        namespace,
    )
    return (
        namespace["_using_kv_store"],
        namespace["verify_and_update_config"].__func__,
    )


def _config(
    *,
    shapes=((64, 128), (64, 4)),
    dtypes=("bf16", "bf16"),
    block_size=None,
    cache_dtype="auto",
    model_dtype="bf16",
    use_mla=False,
    prefix=False,
    cache_mode="none",
    connector=None,
    connector_extra=None,
    disable_hybrid=False,
    speculative_method=None,
):
    class Model:
        @staticmethod
        def get_mamba_state_shape_from_config(vllm_config):
            return shapes

        @staticmethod
        def get_mamba_state_dtype_from_config(vllm_config):
            return dtypes

    _ModelRegistry.model_cls = Model
    model_config = SimpleNamespace(
        architecture="Qwen3_5ForConditionalGeneration",
        dtype=model_dtype,
        use_mla=use_mla,
        max_model_len=4096,
        hf_text_config=SimpleNamespace(kv_lora_rank=48, qk_rope_head_dim=16),
        get_num_kv_heads=lambda parallel_config: 1,
        get_head_size=lambda: 64,
    )
    cache_config = SimpleNamespace(
        cache_dtype=cache_dtype,
        block_size=block_size,
        mamba_page_size_padded=None,
        mamba_cache_mode=cache_mode,
        enable_prefix_caching=prefix,
        mamba_block_size=None,
    )
    transfer = (
        None
        if connector is None
        else SimpleNamespace(
            kv_connector=connector,
            kv_connector_extra_config=connector_extra,
        )
    )
    return SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hybrid
        ),
        kv_transfer_config=transfer,
        speculative_config=(
            None
            if speculative_method is None
            else SimpleNamespace(method=speculative_method)
        ),
    )


def _run(config) -> None:
    _, verify = _load_functions()
    _MambaModelConfig.calls = 0
    verify(object, config)
    assert _MambaModelConfig.calls == 1


def _load_refresh_block_size():
    tree = ast.parse(ASCEND_PATCH_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "refresh_block_size"
    )
    namespace = {"logger": _Logger()}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            ASCEND_PATCH_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace["refresh_block_size"]


def test_refresh_block_size_preserves_hybrid_geometry() -> None:
    refresh_block_size = _load_refresh_block_size()
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=1536,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        model_config=SimpleNamespace(
            is_hybrid=True,
            hf_config=SimpleNamespace(model_type="qwen3_5"),
        ),
    )

    refresh_block_size(config)

    assert config.cache_config.block_size == 1536


def test_refresh_block_size_uses_deepseek_v4_supported_size() -> None:
    refresh_block_size = _load_refresh_block_size()
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=1536,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        model_config=SimpleNamespace(
            is_hybrid=False,
            hf_config=SimpleNamespace(model_type="deepseek_v4"),
        ),
    )

    refresh_block_size(config)

    assert config.cache_config.block_size == 32


def test_hybrid_page_layout_uses_ssm_conv_and_attention_bytes() -> None:
    config = _config()

    _run(config)

    # SSM=64*128*2=16384 bytes; conv=64*4*2=512 bytes.
    # One K token is 64*1*2=128 bytes, so 128-token alignment is exact.
    assert config.cache_config.block_size == 128
    # K+V attention page plus the separately stored conv page.
    assert config.cache_config.mamba_page_size_padded == 128 * 256 + 512
    assert config.cache_config.mamba_block_size == 4096


def test_existing_larger_block_size_is_preserved_in_page_layout() -> None:
    config = _config(block_size=256)

    _run(config)

    assert config.cache_config.block_size == 256
    assert config.cache_config.mamba_page_size_padded == 256 * 256 + 512


def test_explicit_cache_dtype_controls_attention_page_bytes() -> None:
    config = _config(cache_dtype="bf16", model_dtype="fp32")

    _run(config)

    assert config.cache_config.block_size == 128
    assert config.cache_config.mamba_page_size_padded == 128 * 256 + 512


def test_pure_linear_attention_has_no_conv_page() -> None:
    config = _config(shapes=((1, 64, 128),), dtypes=("bf16",))

    _run(config)

    assert config.cache_config.block_size == 128
    assert config.cache_config.mamba_page_size_padded == 128 * 256


@pytest.mark.parametrize(
    ("connector", "extra", "expected"),
    [
        ("AscendStoreConnector", None, True),
        (
            "MultiConnector",
            {"connectors": [{"kv_connector": "AscendStoreConnector"}]},
            True,
        ),
        ("MultiConnector", None, False),
        ("MultiConnector", {"connectors": [{"kv_connector": "Other"}]}, False),
        ("Other", None, False),
    ],
)
def test_kv_store_detection(connector, extra, expected) -> None:
    using_kv_store, _ = _load_functions()
    config = _config(connector=connector, connector_extra=extra)

    assert using_kv_store(config) is expected


def test_kv_store_aligns_hybrid_prefix_cache_blocks() -> None:
    config = _config(
        connector="AscendStoreConnector",
        prefix=True,
        cache_mode="none",
    )

    _run(config)

    assert config.cache_config.mamba_cache_mode == "align"
    assert config.cache_config.mamba_block_size == config.cache_config.block_size


def test_align_mode_without_prefix_uses_full_model_mamba_block() -> None:
    config = _config(
        connector="AscendStoreConnector",
        prefix=False,
        cache_mode="none",
    )

    _run(config)

    assert config.cache_config.mamba_cache_mode == "align"
    assert config.cache_config.mamba_block_size == 4096


def test_existing_align_mode_with_prefix_uses_attention_block() -> None:
    config = _config(prefix=True, cache_mode="align")

    _run(config)

    assert config.cache_config.mamba_cache_mode == "align"
    assert config.cache_config.mamba_block_size == 128


@pytest.mark.parametrize(
    ("disable_hybrid", "speculative_method"),
    [(True, None), (False, "extract_hidden_states")],
)
def test_kv_store_alignment_exclusions(disable_hybrid, speculative_method) -> None:
    config = _config(
        connector="AscendStoreConnector",
        prefix=True,
        cache_mode="none",
        disable_hybrid=disable_hybrid,
        speculative_method=speculative_method,
    )

    _run(config)

    assert config.cache_config.mamba_cache_mode == "none"
    assert config.cache_config.mamba_block_size == 4096


def test_kv_store_rejects_non_align_explicit_mode() -> None:
    config = _config(
        connector="AscendStoreConnector",
        cache_mode="all",
    )

    with pytest.raises(AssertionError, match="only support 'align'"):
        _run(config)


def test_mla_attention_page_uses_kv_and_rope_components() -> None:
    config = _config(
        shapes=((24, 256), (24, 4)),
        use_mla=True,
    )

    _run(config)

    # K token=48*1*2=96; rope token=16*1*2=32; SSM=12288.
    assert config.cache_config.block_size == 128
    assert config.cache_config.mamba_page_size_padded == 128 * 128 + 192


def test_patch_is_fl_owned_and_has_no_vllm_ascend_dependency() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(
        name == "vllm_ascend" or name.startswith("vllm_ascend.")
        for name in imports
    )
