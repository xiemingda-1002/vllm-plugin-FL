from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_runtime_has_no_vllm_ascend_imports() -> None:
    offenders = []
    for path in (ROOT / "vllm_fl").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            if any(name == "vllm_ascend" or name.startswith("vllm_ascend.") for name in names):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_ascend_build_contract_is_soc_specific_and_python_optional() -> None:
    setup = _text("setup.py")
    cmake = _text("csrc/CMakeLists.txt")
    manifest = _text("csrc/ascend/build_opp.sh")
    assert 'SUPPORTED_VENDORS = ("cuda", "ascend")' in setup
    assert 'os.environ.get("SOC_VERSION", "ascend910_93")' in setup
    assert 'extension_name = "vllm_fl._C_ascend"' in setup
    assert "set(SUPPORTED_VENDORS cuda ascend)" in cmake
    assert "${CMAKE_COMMAND} -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0" in cmake
    assert "TORCH_CMAKE_PREFIX_RESULT" in cmake
    for op in (
        "causal_conv1d",
        "recurrent_gated_delta_rule",
        "fused_gdn_gating",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_fwd_o",
    ):
        assert op in manifest
    assert "export FL_BUILD_CANN_OPP=1" in manifest
    assert "--vendor_name=custom" in manifest
    assert "if VLLM_VENDOR:" in setup


def test_current_rc1_opp_build_file_closure_is_present() -> None:
    wrapper = _text("csrc/ascend/CMakeLists.txt")
    opp_branch = wrapper.index("if(FL_BUILD_CANN_OPP")
    assert wrapper.index("cmake_minimum_required(VERSION 3.26)") < opp_branch
    assert wrapper.index("project(vllm_fl_ascend LANGUAGES CXX)") < opp_branch
    assert "include(${CMAKE_CURRENT_LIST_DIR}/opp_project.cmake)" in wrapper
    assert "add_library(_C_ascend SHARED" in wrapper

    # These hashes are the current-vllm-ascend-0.24.0rc1 build files.  The
    # repository applies a final-newline normalization when adding them.
    expected = {
        "csrc/ascend/attention/CMakeLists.txt": (
            "5b0210b6b70daa3d63801b75bd927d88373a8f90fc90d363c1b09e47a4bf7953"
        ),
        "csrc/ascend/moe/CMakeLists.txt": (
            "1887be7afd1aa3097d2d753ba90b8a773fdd6f668edf9e73119af0ae20d9f855"
        ),
        "csrc/ascend/version.info": (
            "dfe472a2a6306520c0c306927ddcfbfbd60a1fead8dde4800da9fe2c0567020f"
        ),
    }
    for relative, digest in expected.items():
        data = (ROOT / relative).read_bytes().rstrip(b"\n")
        assert hashlib.sha256(data).hexdigest() == digest

    common_header = (
        ROOT
        / "csrc/ascend/moe/common/kernel_utils/block/"
        "block_mmad_pingpong_tla_multi.hpp"
    )
    assert hashlib.sha256(common_header.read_bytes()).hexdigest() == (
        "7c6f61cd4aefce5d334cb10671f9541e6e6ea82592d2d304dd4560c2e1e8187f"
    )

    selected_ops = {
        "moe/causal_conv1d",
        "moe/chunk_gated_delta_rule_fwd_h",
        "moe/chunk_fwd_o",
        "attention/recurrent_gated_delta_rule",
        "attention/fused_gdn_gating",
    }
    for relative in selected_ops:
        assert (ROOT / "csrc/ascend" / relative).is_dir()

    manifest = _text("MANIFEST.in")
    assert "recursive-include csrc *" in manifest


def test_current_rc1_protobuf_patches_are_deliverable() -> None:
    expected = {
        "protobuf-hide_absl_symbols.patch": (
            "ff1419dfcb83e9c9478f63e2ece972c0362b8ecda34535b8d2628cdaac81be92"
        ),
        "protobuf_25.1_change_version.patch": (
            "210033396c1a3551f4e24dd5d49b5b4588c53b450ab05fef27d31ae389d94c4f"
        ),
    }
    patch_root = ROOT / "csrc/ascend/cmake/third_party/build/modules/patch"
    paths = []
    for filename, digest in expected.items():
        path = patch_root / filename
        paths.append(path)
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    if (ROOT / ".git").exists():
        for path in paths:
            result = subprocess.run(
                ["git", "check-ignore", "--quiet", str(path)],
                cwd=ROOT,
                check=False,
            )
            assert result.returncode == 1, (
                f"required protobuf source patch is git-ignored: {path}"
            )


def test_current_rc1_dispatcher_contract_has_privateuse1_and_meta() -> None:
    binding = _text("csrc/ascend/torch_binding.cpp")
    for op in (
        "npu_causal_conv1d_custom",
        "npu_recurrent_gated_delta_rule",
        "npu_fused_gdn_gating",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_fwd_o",
    ):
        assert binding.count(f'ops.impl("{op}"') == 2
    assert "TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops)" in binding
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops)" in binding
    assert "Tensor? query_start_loc_opt" in binding
    assert "Tensor? cache_indices_opt" in binding


def test_qwen_gdn_backend_and_flaggems_off_provider_are_owned_by_fl() -> None:
    gdn = _text("vllm_fl/dispatch/backends/vendor/ascend/impl/gdn.py")
    builder = _text(
        "vllm_fl/dispatch/backends/vendor/ascend/impl/gdn_attn_builder.py"
    )
    patch = _text("vllm_fl/dispatch/backends/vendor/ascend/patch.py")
    assert "class AscendGatedDeltaNetAttention" in gdn
    assert "class AscendGDNAttentionMetadataBuilder" in builder
    assert "class AscendGDNAttentionBackend" in builder
    assert "flag_gems.runtime.backend._ascend.fla" not in patch
    assert "from .patches import patch_qwen3_5" in patch


def test_vendor_patch_is_outside_oot_per_op_loop() -> None:
    tree = ast.parse(_text("vllm_fl/ops/custom_ops.py"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "register_oot_ops"
    )
    apply_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_ascend_patches"
    ]
    assert len(apply_calls) == 1
    assert not any(
        isinstance(parent, (ast.For, ast.While))
        and call in ast.walk(parent)
        for call in apply_calls
        for parent in ast.walk(function)
    )


def test_qwen_moe_factory_is_patched_before_qwen_module_import() -> None:
    tree = ast.parse(_text("vllm_fl/ops/custom_ops.py"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "register_oot_ops"
    )
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    factory_line = next(
        node.lineno for node in calls if node.func.id == "_patch_fused_moe_factory"
    )
    ascend_line = next(
        node.lineno for node in calls if node.func.id == "apply_ascend_patches"
    )
    assert factory_line < ascend_line

    source = _text("vllm_fl/ops/custom_ops.py")
    assert 'sys.modules.get("vllm.model_executor.models.qwen3_next")' in source
    assert "qwen_module.FusedMoE = FusedMoEFL" in source


def test_tp2_hccl_and_bf16_moe_do_not_route_to_cuda_or_flaggems() -> None:
    platform = _text("vllm_fl/platform.py")
    moe = _text(
        "vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe.py"
    )
    selector = _text("vllm_fl/ops/fused_moe/fused_moe_utils.py")
    assert "npu_communicator.NPUCommunicator" in platform
    assert "torch_npu.npu_grouped_matmul" in moe
    assert "torch_npu.npu_swiglu" in moe
    assert ".transpose(1, 2).contiguous()" not in moe
    assert "flag_gems" not in moe
    assert 'current_platform.device_type == "npu"' in selector

    layer = _text("vllm_fl/ops/fused_moe/layer.py")
    assert "def process_weights_after_loading(self, layer)" in layer
    assert layer.count(".transpose(1, 2).contiguous()") == 2
