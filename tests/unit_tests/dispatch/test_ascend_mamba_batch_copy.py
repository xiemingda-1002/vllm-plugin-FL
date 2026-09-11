from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[3]
IMPL_PATH = (
    ROOT
    / "vllm_fl/dispatch/backends/vendor/ascend/impl/batch_memcpy.py"
)
PATCH_PATH = (
    ROOT
    / "vllm_fl/dispatch/backends/vendor/ascend/patches/patch_mamba_utils.py"
)
ASCEND_PATCH_PATH = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/patch.py"
CUSTOM_OPS_PATH = ROOT / "vllm_fl/ops/custom_ops.py"


def _load_patch_module(
    monkeypatch,
    mamba_utils,
    batch_memcpy,
    kernel,
    *,
    vendor_name="ascend",
    device_type="npu",
):
    worker = ModuleType("vllm.v1.worker")
    worker.mamba_utils = mamba_utils
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker)

    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        vendor_name=vendor_name,
        device_type=device_type,
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    utils = ModuleType("vllm.utils")
    utils.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    math_utils = ModuleType("vllm.utils.math_utils")
    math_utils.cdiv = lambda numerator, denominator: (
        numerator + denominator - 1
    ) // denominator
    monkeypatch.setitem(sys.modules, "vllm.utils.math_utils", math_utils)

    impl_name = (
        "vllm_fl.dispatch.backends.vendor.ascend.impl.batch_memcpy"
    )
    impl = ModuleType(impl_name)
    impl.batch_memcpy = batch_memcpy
    impl.batch_memcpy_kernel = kernel
    monkeypatch.setitem(sys.modules, impl_name, impl)

    module_name = (
        "vllm_fl.dispatch.backends.vendor.ascend.patches."
        "_test_patch_mamba_utils"
    )
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_installer_patches_kernel_and_only_pointer_dtypes(monkeypatch) -> None:
    allocations = []

    class MambaCopyBuffers:
        @classmethod
        def create(cls, max_num_reqs, kv_cache_config, copy_funcs, make_buffer):
            del cls, max_num_reqs, kv_cache_config, copy_funcs
            return (
                make_buffer(2, dtype=torch.uint64),
                make_buffer(2, dtype=torch.uint64),
                make_buffer(2, dtype=torch.int32),
            )

    def make_buffer(n, dtype):
        allocations.append((n, dtype))
        return dtype

    sentinel_copy = object()
    sentinel_kernel = object()
    mamba_utils = SimpleNamespace(
        MambaCopyBuffers=MambaCopyBuffers,
        batch_memcpy=object(),
        batch_memcpy_kernel=object(),
    )
    patch = _load_patch_module(
        monkeypatch, mamba_utils, sentinel_copy, sentinel_kernel
    )

    assert patch.patch_mamba_batch_copy() is True
    result = MambaCopyBuffers.create(None, None, None, make_buffer)

    assert result == (torch.int64, torch.int64, torch.int32)
    assert allocations == [
        (2, torch.int64),
        (2, torch.int64),
        (2, torch.int32),
    ]
    assert mamba_utils.batch_memcpy is sentinel_copy
    assert mamba_utils.batch_memcpy_kernel is sentinel_kernel
    assert patch.patch_mamba_batch_copy() is False


def test_installer_preserves_upstream_symbols_off_ascend(monkeypatch) -> None:
    class MambaCopyBuffers:
        @classmethod
        def create(cls, *args, **kwargs):
            return cls, args, kwargs

    original_create = MambaCopyBuffers.create
    original_copy = object()
    original_kernel = object()
    original_preprocess = object()
    mamba_utils = SimpleNamespace(
        MambaCopyBuffers=MambaCopyBuffers,
        batch_memcpy=original_copy,
        batch_memcpy_kernel=original_kernel,
        preprocess_mamba=original_preprocess,
    )

    for vendor_name, device_type in (
        ("nvidia", "cuda"),
        ("other_npu", "npu"),
        ("ascend", "cpu"),
    ):
        patch = _load_patch_module(
            monkeypatch,
            mamba_utils,
            object(),
            object(),
            vendor_name=vendor_name,
            device_type=device_type,
        )
        assert patch.patch_mamba_batch_copy() is False
        assert MambaCopyBuffers.create == original_create
        assert mamba_utils.batch_memcpy is original_copy
        assert mamba_utils.batch_memcpy_kernel is original_kernel
        assert mamba_utils.preprocess_mamba is original_preprocess
        assert not hasattr(
            mamba_utils, "_vllm_fl_ascend_mamba_batch_copy_patched"
        )


def test_ascend_preprocess_collects_without_copy_and_patch_is_idempotent(
    monkeypatch,
) -> None:
    class MambaCopyBuffers:
        @classmethod
        def create(cls, *args, **kwargs):
            return cls, args, kwargs

    copied = []
    collected = []

    def original_preprocess(*args, **kwargs):
        del args, kwargs

    def collect_mamba_copy_meta(*args):
        collected.append(args)

    def do_mamba_copy_block(*args):
        copied.append(args)

    mamba_utils = SimpleNamespace(
        MambaCopyBuffers=MambaCopyBuffers,
        batch_memcpy=object(),
        batch_memcpy_kernel=object(),
        preprocess_mamba=original_preprocess,
        collect_mamba_copy_meta=collect_mamba_copy_meta,
        do_mamba_copy_block=do_mamba_copy_block,
    )
    patch = _load_patch_module(
        monkeypatch,
        mamba_utils,
        object(),
        object(),
    )
    assert patch.patch_mamba_batch_copy() is True
    installed_preprocess = mamba_utils.preprocess_mamba
    assert installed_preprocess is not original_preprocess

    scheduler_output = SimpleNamespace(
        finished_req_ids={"finished"},
        preempted_req_ids={"preempted"},
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids={"resumed"}
        ),
        num_scheduled_tokens={"req": 5},
    )
    copy_bufs = SimpleNamespace(
        mamba_group_ids=[7],
        mamba_spec=SimpleNamespace(
            num_speculative_blocks=0,
            block_size=4,
        ),
        offset=99,
    )
    input_batch = SimpleNamespace(
        req_ids=["req"],
        num_accepted_tokens_cpu=[3],
    )
    req_state = SimpleNamespace(num_computed_tokens=4)
    state_indices = {
        "req": 0,
        "finished": 1,
        "preempted": 1,
        "resumed": 1,
    }
    installed_preprocess(
        scheduler_output,
        object(),
        object(),
        state_indices,
        input_batch,
        {"req": req_state},
        {"layer": object()},
        (object(),),
        copy_bufs,
    )

    assert copied == []
    assert copy_bufs.offset == 0
    assert state_indices == {"req": 2}
    assert input_batch.num_accepted_tokens_cpu == [1]
    assert len(collected) == 1
    assert collected[0][3] == [7]
    assert collected[0][4:7] == (0, 2, 2)
    assert collected[0][7] is req_state
    assert set(collected[0][8]) == {"layer"}
    assert patch.patch_mamba_batch_copy() is False
    assert mamba_utils.preprocess_mamba is installed_preprocess


def test_kernel_keeps_pointer_cast_outside_loop_and_uses_rc1_launch() -> None:
    tree = ast.parse(IMPL_PATH.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "batch_memcpy_kernel"
    )
    loop = next(node for node in ast.walk(kernel) if isinstance(node, ast.For))
    loop_lines = ast.get_source_segment(
        IMPL_PATH.read_text(encoding="utf-8"), loop
    )
    assert loop_lines is not None
    assert "pointer_type" not in loop_lines

    source = IMPL_PATH.read_text(encoding="utf-8")
    assert source.count(".to(tl.pointer_type(tl.uint8))") == 2
    assert 'cache_modifier=".cg"' in source
    assert "block_size = 8192" in source
    assert "batch_memcpy_kernel[(batch,)]" in source


def test_installer_is_called_by_ascend_lifecycle_before_model_patches() -> None:
    tree = ast.parse(ASCEND_PATCH_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_ascend_patches"
    )
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls["patch_mamba_batch_copy"] < calls["patch_causal_conv1d"]
    assert calls["patch_mamba_batch_copy"] < calls["patch_op_cls"]


def test_outer_npu_lifecycle_and_installer_dual_platform_gate() -> None:
    tree = ast.parse(CUSTOM_OPS_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "register_oot_ops"
    )
    ascend_assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "is_ascend"
            for target in node.targets
        )
    )
    ascend_if = next(
        node
        for node in function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "is_ascend"
    )
    call_names = {
        node.func.id
        for statement in ascend_if.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "apply_ascend_patches" in call_names

    condition = compile(
        ast.Expression(body=ascend_assignment.value),
        CUSTOM_OPS_PATH,
        "eval",
    )
    assert not eval(
        condition,
        {
            "current_platform": SimpleNamespace(
                vendor_name="other_npu",
                device_type="npu",
            )
        },
    )

    non_ascend_nodes = [
        node
        for statement in ascend_if.orelse
        for node in ast.walk(statement)
    ]
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_ascend_patches"
        for node in non_ascend_nodes
    )

    installer_source = PATCH_PATH.read_text(encoding="utf-8")
    assert 'current_platform.vendor_name == "ascend"' in installer_source
    assert 'current_platform.device_type == "npu"' in installer_source
