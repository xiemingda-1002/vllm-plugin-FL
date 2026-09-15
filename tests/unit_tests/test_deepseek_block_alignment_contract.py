"""Exercise the production MLA alignment guard without loading an NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "device,vendor,model_type,block_size,expected",
    [
        ("npu", "ascend", "deepseek_v4", 32, 32),
        ("npu", "ascend", "deepseek_v4", 64, 64),
        ("npu", "ascend", "deepseek_v4", 128, 128),
        ("npu", "ascend", "deepseek_v3", 32, 64),
        ("npu", "other", "deepseek_v4", 32, 64),
        ("cuda", "nvidia", "deepseek_v4", 32, 64),
    ],
)
def test_mla_alignment_preserves_only_ascend_dsv4_geometry(
    device, vendor, model_type, block_size, expected
):
    source = Path(__file__).resolve().parents[2] / "vllm_fl/platform.py"
    tree = ast.parse(source.read_text())
    platform = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlatformFL"
    )
    method = next(
        n for n in platform.body
        if isinstance(n, ast.FunctionDef) and n.name == "check_and_update_config"
    )
    alignment = next(
        n for n in method.body
        if isinstance(n, ast.If)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "use_mla"
            for child in ast.walk(n.test)
        )
    )
    cache = SimpleNamespace(block_size=block_size)
    namespace = {
        "cls": SimpleNamespace(device_type=device, vendor_name=vendor),
        "model_config": SimpleNamespace(
            use_mla=True, hf_config=SimpleNamespace(model_type=model_type)
        ),
        "cache_config": cache,
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(ast.Module(body=[alignment], type_ignores=[]), str(source), "exec"), namespace)
    assert cache.block_size == expected
