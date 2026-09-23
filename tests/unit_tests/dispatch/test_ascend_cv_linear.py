import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


CV_LINEAR_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/ascend/ops/cv_linear.py"
)
LINEAR_OP_MODULE = "vllm_fl.dispatch.backends.vendor.ascend.impl.linear_op"
QUANTIZATION_MODULE = (
    "vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.linear"
)


@pytest.fixture
def cv_linear_module(monkeypatch):
    class CustomReplicatedOp:
        pass

    class AscendW8A8DynamicLinearMethod:
        pass

    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    linear_op = ModuleType(LINEAR_OP_MODULE)
    linear_op.CustomReplicatedOp = CustomReplicatedOp
    monkeypatch.setitem(sys.modules, LINEAR_OP_MODULE, linear_op)
    quantization = ModuleType(QUANTIZATION_MODULE)
    quantization.AscendW8A8DynamicLinearMethod = AscendW8A8DynamicLinearMethod
    monkeypatch.setitem(sys.modules, QUANTIZATION_MODULE, quantization)

    spec = importlib.util.spec_from_file_location("test_cv_linear", CV_LINEAR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, CustomReplicatedOp


def _linear(*, custom_op=None, gather_output=False):
    return SimpleNamespace(
        custom_op=custom_op,
        gather_output=gather_output,
        quant_method=object(),
    )


def test_cv_linear_treats_replicated_op_as_non_communicating(
    cv_linear_module,
) -> None:
    module, CustomReplicatedOp = cv_linear_module
    wrapper = module.CVLinearWrapper(_linear(custom_op=CustomReplicatedOp()))

    assert not wrapper._has_communication


def test_cv_linear_treats_non_replicated_custom_op_as_communicating(
    cv_linear_module,
) -> None:
    module, _ = cv_linear_module

    assert module.CVLinearWrapper._detect_communication(_linear(custom_op=object()))


def test_cv_linear_treats_gather_output_as_communicating(cv_linear_module) -> None:
    module, _ = cv_linear_module

    assert module.CVLinearWrapper._detect_communication(_linear(gather_output=True))
