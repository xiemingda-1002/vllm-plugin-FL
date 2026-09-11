# Copyright (c) 2026 BAAI. All rights reserved.

from pathlib import Path


ROOT = Path(__file__).parents[3]
MOE = (
    ROOT
    / "vllm_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "ascend"
    / "impl"
    / "moe"
)


def _source(name: str) -> str:
    return (MOE / name).read_text(encoding="utf-8")


def test_rc1_moe_package_has_no_vllm_ascend_runtime_dependency() -> None:
    for path in MOE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from vllm_ascend" not in source
        assert "import vllm_ascend" not in source


def test_a2_allgather_owns_complete_runner_lifecycle() -> None:
    runner = _source("fused_moe.py")
    assert "moe_comm_method.prepare(" in runner
    assert "self._quant_method.apply(" in runner
    assert "moe_comm_method.finalize(" in runner
    assert "setup_moe_comm_method(self.moe_config)" in runner

    comm = _source("moe_comm_method.py")
    assert "TokenDispatcherWithAllGather" in comm
    assert "PrepareAndFinalizeWithAllGather" in comm
    assert "only A2 ALLGATHER is supported" in comm


def test_rc1_weight_conversion_is_idempotent_and_loader_safe() -> None:
    runner = _source("fused_moe.py")
    assert 'getattr(layer, _WEIGHTS_PROCESSED_ATTR, False)' in runner
    assert 'replace_parameter(layer, "w13_weight", w13_data)' in runner
    assert 'replace_parameter(layer, "w2_weight", w2_data)' in runner
    assert "setattr(layer, _WEIGHTS_PROCESSED_ATTR, True)" in runner


def test_unmigrated_execution_modes_fail_closed() -> None:
    runner = _source("fused_moe.py")
    for feature in (
        "unquantized BF16/FP16",
        "fused MC2",
        "shared-expert DP",
        "shared-expert multistream",
        "EPLB",
    ):
        assert feature in runner

    device_operator = (
        ROOT
        / "vllm_fl"
        / "dispatch"
        / "backends"
        / "vendor"
        / "ascend"
        / "impl"
        / "device_operator.py"
    ).read_text(encoding="utf-8")
    assert "does not support MXFP MoE quantization" in device_operator
