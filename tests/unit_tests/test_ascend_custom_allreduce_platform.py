"""Ascend collective selection must not change other FL vendors."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def _isolated_custom_allreduce_method():
    source = Path(__file__).parents[2] / "vllm_fl/platform.py"
    tree = ast.parse(source.read_text())
    platform = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlatformFL"
    )
    method = next(
        node for node in platform.body
        if isinstance(node, ast.FunctionDef) and node.name == "use_custom_allreduce"
    )
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(source), "exec"), namespace)
    return namespace["use_custom_allreduce"]


@pytest.mark.parametrize(
    ("vendor", "backend", "expected"),
    [
        ("ascend", "hccl", False),
        ("hygon", "nccl", False),
        ("nvidia", "flagcx", False),
        ("nvidia", "nccl", True),
        ("metax", "nccl", True),
    ],
)
def test_collective_selection_is_vendor_scoped(vendor, backend, expected):
    use_custom_allreduce = _isolated_custom_allreduce_method()
    platform = SimpleNamespace(vendor_name=vendor, dist_backend=backend)
    assert use_custom_allreduce(platform) is expected
