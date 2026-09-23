"""Dependency-free checks for Ascend functional-layer boundaries."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "vllm_fl"


def test_functional_packages_do_not_install_vendor_runtime_on_import():
    for relative in (
        "attention/__init__.py",
        "attention/ascend/__init__.py",
        "kv_cache/__init__.py",
        "kv_cache/ascend/__init__.py",
        "dispatch/backends/vendor/ascend/patches/__init__.py",
        "platforms/__init__.py",
        "platforms/ascend/__init__.py",
        "profiler/__init__.py",
        "scheduling/__init__.py",
    ):
        tree = ast.parse((PACKAGE / relative).read_text())
        # Package discovery must not load NPU models, install scheduler patches,
        # register custom ops, or construct communication groups.
        assert all(
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for node in tree.body
        ), relative


def test_layout_has_one_owner_and_no_obsolete_runtime_path():
    expected = (
        "attention/ascend/attention.py",
        "attention/ascend/standard_mask.py",
        "attention/ascend/attention_connector.py",
        "attention/ascend/gdn_attn_builder.py",
        "profiler/ascend/torch_npu_profiler.py",
        "platforms/ascend/hardware.py",
        "platforms/ascend/runtime.py",
        "compilation/ascend_tensor_utils.py",
        "configs/ascend_cache.py",
        "configs/ascend_mamba.py",
        "dispatch/backends/vendor/ascend/patch.py",
    )
    for relative in expected:
        assert (PACKAGE / relative).is_file(), relative

    obsolete = (
        "dispatch/backends/vendor/ascend/attention",
        "dispatch/backends/vendor/ascend/profiler",
        "dispatch/backends/vendor/ascend/hardware.py",
        "dispatch/backends/vendor/ascend/runtime.py",
        "dispatch/backends/vendor/ascend/tensor_utils.py",
        "patches/ascend",
        "dispatch/backends/vendor/ascend/worker",
        "dispatch/backends/vendor/ascend/distributed",
    )
    for relative in obsolete:
        assert not any((PACKAGE / relative).rglob("*.py")), relative


def test_cache_spec_and_planner_share_one_class_owner():
    source = (PACKAGE / "kv_cache/ascend/kv_cache_interface.py").read_text()
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "deepseek_v4_kv_cache"
        and any(alias.name == "CompressAttentionManager" for alias in node.names)
        for node in tree.body
    )
    owners = []
    for path in PACKAGE.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and node.name == "CompressAttentionManager":
                owners.append(path.relative_to(PACKAGE).as_posix())
    assert owners == ["kv_cache/ascend/deepseek_v4_kv_cache.py"]


def test_old_cache_and_scheduler_import_paths_are_absent():
    obsolete = (
        "vllm_fl.dispatch.backends.vendor.ascend.core.",
        "vllm_fl.dispatch.backends.vendor.ascend.patches.patch_balance_schedule",
        "vllm_fl.dispatch.backends.vendor.ascend.models.dsa_layer",
    )
    for path in PACKAGE.rglob("*.py"):
        source = path.read_text()
        assert not any(old in source for old in obsolete), path


def test_unconsumed_memcache_fence_is_not_in_attention_path():
    vendor = PACKAGE / "dispatch/backends/vendor/ascend"
    attention = PACKAGE / "attention/ascend"
    assert not (vendor / "memcache_comm_fence.py").exists()
    assert not (vendor / "patches/patch_multimodal_merge.py").exists()
    for path in (*vendor.rglob("*.py"), *attention.rglob("*.py")):
        source = path.read_text()
        assert "record_attention_compute_start" not in source, path
    # Connector completion notification is independent and must survive cleanup.
    for name in ("dsa_v1.py", "sfa_v1.py", "context_parallel/dsa_cp.py"):
        assert "notify_kv_cache_written(" in (attention / name).read_text()
