"""CPU-only packaging/generator contracts without importing setup.py."""
import ast
import glob
from pathlib import Path
from types import SimpleNamespace

import pytest

SETUP = Path(__file__).resolve().parents[3] / "setup.py"


@pytest.mark.parametrize("vendor,generator", [("ascend", "Unix Makefiles"), ("cuda", "Ninja")])
def test_vendor_build_generator(vendor, generator):
    tree = ast.parse(SETUP.read_text())
    configure = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "configure")
    branch = next(
        n for n in configure.body
        if isinstance(n, ast.If)
        and any(isinstance(t, ast.Name) and t.id == "build_tool" for t in ast.walk(n))
    )
    scope = dict(_is_ascend=lambda: vendor == "ascend", _which=lambda _: True,
                 build_tool=[], cmake_args=[], num_jobs=4)
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(SETUP), "exec"), scope)
    assert scope["build_tool"] == ["-G", generator]


def test_packaging_finds_real_cmake_library_name(tmp_path):
    library = tmp_path / "ascend" / "libvllm_fl_ascend_kernels.so"
    library.parent.mkdir()
    library.touch()
    tree = ast.parse(SETUP.read_text())
    assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "kernel_patterns" for t in n.targets))
    patterns = eval(compile(ast.Expression(assign.value), str(SETUP), "eval"),
                    dict(self=SimpleNamespace(build_temp=str(tmp_path)), VLLM_VENDOR="ascend"))
    assert [Path(p) for pattern in patterns for p in glob.glob(pattern)] == [library]
