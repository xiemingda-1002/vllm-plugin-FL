"""Regression for incremental Ascend OPP index generation."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FUNC_CMAKE = ROOT / "csrc/ascend/cmake/func.cmake"


def test_ops_config_target_refreshes_an_existing_index_after_new_binary(
    tmp_path: Path,
) -> None:
    """A metadata-only second build must rerun the index generator.

    The miniature target mirrors the dependency shape in ``func.cmake``: an
    index custom target depends on completed kernel targets but owns no output
    timestamp. Adding a binary between builds therefore refreshes both index
    files without a kernel compilation command.
    """
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for the OPP index regression")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kernel = bin_dir / "kernel_a.bin"
    kernel.write_bytes(b"compiled-once")
    generator = tmp_path / "generate_index.py"
    generator.write_text(
        """import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=Path, required=True)
args = parser.parse_args()
names = sorted(path.name for path in args.p.glob('*.bin'))
(args.p / 'binary_info_config.json').write_text(json.dumps(names))
(args.p / 'relocatable_kernel_info_config.json').write_text(json.dumps(names))
count = args.p / 'refresh-count'
count.write_text(str(int(count.read_text() if count.exists() else '0') + 1))
""",
        encoding="utf-8",
    )
    (tmp_path / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.18)
project(opp_index_refresh NONE)
add_custom_target(kernel_target DEPENDS \"{kernel}\")
add_custom_target(ops_config
  COMMAND \"{sys.executable}\" \"{generator}\" -p \"{bin_dir}\"
  BYPRODUCTS \"{bin_dir / 'binary_info_config.json'}\" \"{bin_dir / 'relocatable_kernel_info_config.json'}\"
  VERBATIM)
add_dependencies(ops_config kernel_target)
""",
        encoding="utf-8",
    )
    build_dir = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(build_dir)], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "ops_config"], check=True)
    assert json.loads((bin_dir / "binary_info_config.json").read_text()) == ["kernel_a.bin"]

    # Simulate an already-built newly selected kernel: no compile target or
    # source changes occur between these two metadata builds.
    (bin_dir / "kernel_b.bin").write_bytes(b"already-compiled")
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "ops_config"], check=True)
    assert json.loads((bin_dir / "binary_info_config.json").read_text()) == [
        "kernel_a.bin",
        "kernel_b.bin",
    ]
    assert json.loads((bin_dir / "relocatable_kernel_info_config.json").read_text()) == [
        "kernel_a.bin",
        "kernel_b.bin",
    ]
    assert (bin_dir / "refresh-count").read_text() == "2"
    assert kernel.read_bytes() == b"compiled-once"


def test_vendor_function_uses_the_refresh_target_pattern() -> None:
    source = FUNC_CMAKE.read_text(encoding="utf-8")
    assert "add_custom_target(${OPS_CONFIG_TARGET}" in source
    assert "BYPRODUCTS ${BINARY_INFO_CONFIG_FILE} ${RELOCATABLE_KERNEL_INFO_CONFIG_FILE}" in source
    assert "add_custom_command(OUTPUT ${BINARY_INFO_CONFIG_FILE}" not in source
