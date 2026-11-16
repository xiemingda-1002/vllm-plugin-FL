# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU checks for the MiniMax example's real command-line entry point."""

import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

EXAMPLE = Path(__file__).parents[2] / "examples/minimax_m31_offline_inference.py"


@pytest.mark.parametrize("arguments,code", [(["--help"], 0), ([], 2)])
def test_cli_is_validated_before_runtime_import(arguments, code):
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), *arguments], capture_output=True, text=True
    )
    assert result.returncode == code
    assert "--model" in result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_model_path_is_forwarded_and_generation_stays_on_stdout(monkeypatch, capsys):
    calls = []
    runtime = ModuleType("vllm")

    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def chat(self, messages, sampling):
            assert messages[0]["role"] == "user"
            assert sampling.max_tokens == 64
            return [SimpleNamespace(outputs=[SimpleNamespace(text="4")])]

    runtime.LLM = LLM
    runtime.SamplingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", runtime)
    checkpoint = "/models/team checkpoints/MiniMax-M3.1-Flash"
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), "--model", checkpoint])
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["model"] == checkpoint
    assert calls[0]["dtype"] == "bfloat16"
    assert capsys.readouterr().out == "Generated: '4'\n"


@pytest.mark.parametrize("stage", ["load", "generate"])
def test_runtime_failures_are_not_swallowed_or_retried(monkeypatch, stage):
    runtime = ModuleType("vllm")
    calls = []
    error = RuntimeError("MiniMax runtime failed")

    class LLM:
        def __init__(self, **kwargs):
            calls.append("load")
            if stage == "load":
                raise error

        def chat(self, *args):
            calls.append("generate")
            raise error

    runtime.LLM = LLM
    runtime.SamplingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", runtime)
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), "--model", "/models/checkpoint"])
    with pytest.raises(RuntimeError) as failure:
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert failure.value is error
    assert calls == (["load"] if stage == "load" else ["load", "generate"])
