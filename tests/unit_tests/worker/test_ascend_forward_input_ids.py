"""Hash routing must see the current model input, including padded IDs."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.config")
from vllm.forward_context import ForwardContext, override_forward_context
from vllm_fl.worker import model_runner


def test_model_forward_refreshes_ascend_input_ids(monkeypatch):
    monkeypatch.setattr(
        model_runner, "current_platform",
        SimpleNamespace(vendor_name="ascend", device_type="npu"),
    )
    runner = object.__new__(model_runner.ModelRunnerFL)
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={},
        additional_kwargs={"flash_comm_v1_enabled": False},
    )
    seen = []

    def model(**kwargs):
        assert context.additional_kwargs["input_ids"] is kwargs["input_ids"]
        seen.append(kwargs["input_ids"])
        return "output"

    runner.model = model
    first = torch.tensor([2, 5, -1])
    second = torch.tensor([7, 1])
    with override_forward_context(context):
        for ids in (first, second, None):
            assert runner._model_forward(input_ids=ids) == "output"
    assert seen[0] is first and seen[1] is second and seen[2] is None
    assert not hasattr(context, "input_ids")


@pytest.mark.parametrize("vendor,device", [("cuda", "cuda"), ("other", "npu")])
def test_input_ids_binding_preserves_other_vendors(monkeypatch, vendor, device):
    monkeypatch.setattr(
        model_runner, "current_platform",
        SimpleNamespace(vendor_name=vendor, device_type=device),
    )

    def unexpected_context_access():
        raise AssertionError("Non-Ascend binding must not access context")

    monkeypatch.setattr(model_runner, "get_forward_context", unexpected_context_access)
    model_runner.ModelRunnerFL._set_ascend_forward_input_ids(torch.tensor([1]))
