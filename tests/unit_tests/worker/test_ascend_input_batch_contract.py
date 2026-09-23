import ast
from pathlib import Path
from types import SimpleNamespace

import torch


WORKER_DIR = Path(__file__).resolve().parents[3] / "vllm_fl/worker"
RUNNER = WORKER_DIR / "model_runner.py"


def _tree(name: str) -> ast.Module:
    return ast.parse((WORKER_DIR / name).read_text(encoding="utf-8"))


def _is_torch_int32(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr == "int32"
    )


def test_ascend_block_table_uses_int32_device_inputs() -> None:
    tree = _tree("block_table.py")
    block_table = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BlockTable"
    )
    init = next(
        node
        for node in block_table.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    buffer_dtypes: dict[str, ast.expr] = {}
    for node in ast.walk(init):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "_make_buffer"
        ):
            dtype = next(
                (kw.value for kw in value.keywords if kw.arg == "dtype"), None
            )
            if dtype is not None:
                buffer_dtypes[target.attr] = dtype

    assert _is_torch_int32(buffer_dtypes["block_table"])
    assert _is_torch_int32(buffer_dtypes["slot_mapping"])


def test_npu_input_batch_owns_the_ascend_block_table() -> None:
    tree = _tree("npu_input_batch.py")
    imports = {
        (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (
        "vllm_fl.worker.block_table",
        "MultiGroupBlockTable",
    ) in imports

    npu_input_batch = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUInputBatch"
    )
    init = next(
        node
        for node in npu_input_batch.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "block_table"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Call)
    assert isinstance(assignments[0].value.func, ast.Name)
    assert assignments[0].value.func.id == "MultiGroupBlockTable"


def test_runner_selects_the_vendor_input_batch_in_one_factory() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerFL"
    )
    factory = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_create_input_batch"
    )
    returned_classes = {
        node.value.func.id
        for node in ast.walk(factory)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    }
    assert returned_classes == {"InputBatch", "NPUInputBatch"}

    factory_calls = [
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_create_input_batch"
    ]
    assert len(factory_calls) == 2


def test_runner_keeps_rc1_discard_and_contiguous_mrope_inputs_npu_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "self.discard_request_indices: CpuGpuBuffer | None = None" in source
    assert "discard_request_indices = np.nonzero(discard_requests_mask)[0]" in source
    assert "self.mrope_positions.gpu.copy_(" in source
    assert 'if current_platform.device_type == "npu":' in source


def test_ascend_block_table_clear_uses_canonical_cpu_gpu_buffer_views(
    monkeypatch,
) -> None:
    import vllm_fl.worker.block_table as block_table_module

    singleton_group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(block_table_module, "get_pcp_group", lambda: singleton_group)
    monkeypatch.setattr(block_table_module, "get_dcp_group", lambda: singleton_group)

    table = block_table_module.BlockTable(
        block_size=128,
        max_num_reqs=2,
        max_num_blocks_per_req=2,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_sizes=[64],
    )
    table.block_table.cpu.fill_(3)
    table.block_table.gpu.fill_(4)

    table.clear()

    assert not table.block_table.cpu.any()
    assert not table.block_table.gpu.any()
