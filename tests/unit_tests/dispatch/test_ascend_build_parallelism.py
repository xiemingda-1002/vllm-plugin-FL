"""Exercise the real CMake command fragment without compiling kernels."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("jobs", [None, "", "1", "8", "0", "-2", "invalid", "8;invalid"])
def test_explicit_inner_jobs_preserve_default_and_validate_input(tmp_path, jobs):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to validate the build command")
    root = Path(__file__).resolve().parents[3]
    source = (root / "csrc/ascend/cmake/func.cmake").read_text()
    start = source.index("            # CANN's value 1")
    end = source.index("            list(APPEND _BUILD_COMMAND export BIN_FILENAME_HASHED=1", start)
    script = tmp_path / "parallel.cmake"
    script.write_text(source[start:end] + '\nmessage("COMMAND=${_BUILD_COMMAND}")\n')
    env = os.environ.copy()
    env.pop("TILINGKEY_PARALLEL_JOB", None)
    if jobs is not None:
        env["TILINGKEY_PARALLEL_JOB"] = jobs
    result = subprocess.run([cmake, "-P", str(script)], env=env,
                            capture_output=True, text=True)
    if jobs in (None, ""):
        assert result.returncode == 0, result.stderr
        assert "COMMAND=export;TILINGKEY_PAR_COMPILE=1;&&" in result.stderr
        assert "TILINGKEY_PARALLEL_JOB=" not in result.stderr
    elif jobs in ("1", "8"):
        assert result.returncode == 0, result.stderr
        assert "TILINGKEY_PAR_COMPILE=0;&&" in result.stderr
        assert f"TILINGKEY_PARALLEL_JOB={jobs};&&" in result.stderr
    else:
        assert result.returncode != 0
        assert "TILINGKEY_PARALLEL_JOB must be a positive integer" in result.stderr
