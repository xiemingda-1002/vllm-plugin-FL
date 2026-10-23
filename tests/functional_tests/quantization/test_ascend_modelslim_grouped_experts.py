# SPDX-License-Identifier: Apache-2.0
"""Pre-dispatched numerical experts; no routing, groups or model loading."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        current_platform.device_type != "npu", reason="Requires Ascend NPU"
    ),
]


def _case(dtype, list_type, counts=(2, 0, 1), capacity=6):
    from vllm_fl.dispatch.backends.reference.impl import w8a8_moe as reference
    from vllm_fl.dispatch.backends.vendor.ascend.impl import w8a8_moe as native

    torch.manual_seed(37)
    w1 = torch.randint(-24, 24, (3, 128, 128), dtype=torch.int8, device="npu")
    w2 = torch.randint(-24, 24, (3, 128, 64), dtype=torch.int8, device="npu")
    s1 = torch.full((3, 128, 1), 0.015625, dtype=dtype, device="npu")
    s2 = torch.full((3, 128, 1), 0.015625, dtype=dtype, device="npu")
    x = torch.randint(-100, 100, (capacity, 128), dtype=torch.int8, device="npu")
    scale = torch.full((capacity,), 0.03125, dtype=torch.float32, device="npu")
    scale[sum(counts) :] = float("nan")
    groups = torch.tensor(counts, dtype=torch.int64, device="npu")
    if list_type == 0:
        groups = groups.cumsum(0)
    kwargs = dict(group_list=groups, group_list_type=list_type, output_dtype=dtype)
    return (
        native,
        reference,
        x,
        scale,
        native.modelslim_w8a8_prepare_experts(w1, w2, s1, s2),
        reference.modelslim_w8a8_prepare_experts(w1, w2, s1, s2),
        kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("list_type", [0, 1])
@pytest.mark.parametrize("counts", [(2, 0, 1), (0, 0, 0)])
@pytest.mark.parametrize(
    "activation,limit", [("silu", None), ("silu", 0.5), ("swiglustep", 0.5)]
)
def test_grouped_oracle(dtype, list_type, counts, activation, limit):
    native, ref, x, scale, w, rw, kwargs = _case(dtype, list_type, counts)
    kwargs.update(activation=activation, limit=limit)
    actual = native.modelslim_w8a8_grouped_experts(x, scale, *w, **kwargs)
    expected = ref.modelslim_w8a8_grouped_experts(x, scale, *rw, **kwargs)
    assert actual.shape == x.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.06)
    assert torch.count_nonzero(actual[sum(counts) :]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("list_type", [0, 1])
def test_zero_capacity(dtype, list_type):
    native, ref, x, scale, w, rw, kwargs = _case(
        dtype, list_type, counts=(0, 0, 0), capacity=0
    )
    for provider, weights in ((native, w), (ref, rw)):
        output = provider.modelslim_w8a8_grouped_experts(x, scale, *weights, **kwargs)
        assert output.shape == (0, 128)
        assert output.dtype == dtype


def test_grouped_cached_dispatch():
    from vllm_fl.dispatch import CachedOp, get_default_manager, with_strict_mode

    _, ref, x, scale, weights, rw, kwargs = _case(torch.bfloat16, 0)
    operation = CachedOp("modelslim_w8a8_grouped_experts")
    expected = ref.modelslim_w8a8_grouped_experts(x, scale, *rw, **kwargs)
    with with_strict_mode():
        assert (
            get_default_manager().get_selected_impl_id("modelslim_w8a8_grouped_experts")
            == "vendor.ascend"
        )
        output = operation(x, scale, *weights, **kwargs)
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.06)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_grouped_quantization_half_integer(dtype):
    import torch_npu

    from vllm_fl.dispatch.backends.reference.impl.w8a8_moe import (
        _quantize_grouped_activation,
    )

    # A real GMM/SiLU fixture produced 25.25 with row absmax 50.5.
    # Check both signs: do not hide a one-code rounding mismatch by
    # increasing the final expert-output tolerance.
    x = torch.tensor([[25.25, 50.5], [-25.25, -50.5]], dtype=dtype, device="npu")
    quantized, scale = _quantize_grouped_activation(x)
    native_quantized, native_scale = torch_npu.npu_dynamic_quant(x)
    torch.testing.assert_close(quantized, native_quantized, rtol=0, atol=0)
    torch.testing.assert_close(scale.flatten(), native_scale.flatten(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("list_type", [0, 1])
@pytest.mark.parametrize("idle", [False, True])
def test_grouped_changed_input_graph(dtype, list_type, idle):
    import torch_npu  # noqa: F401

    counts = (0, 0, 0) if idle else (2, 0, 1)
    native, ref, x, scale, w, rw, kwargs = _case(dtype, list_type, counts)
    # Both accepted scale layouts exercise the same row ordering.
    scale = scale[:, None]
    for _ in range(3):
        native.modelslim_w8a8_grouped_experts(x, scale, *w, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = native.modelslim_w8a8_grouped_experts(x, scale, *w, **kwargs)
    torch.npu.synchronize()
    previous = output.clone()
    for _ in range(2):
        x.copy_(torch.randint_like(x, -100, 100))
        scale[: sum(counts)].mul_(1.125)
        expected = ref.modelslim_w8a8_grouped_experts(x, scale, *rw, **kwargs)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.03, atol=0.06)
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output[sum(counts) :]) == 0
        if not idle:
            assert not torch.equal(previous, output)
        previous = output.clone()


@pytest.mark.parametrize(
    "invalid",
    [
        "input_dtype",
        "input_shape",
        "scale_dtype",
        "scale_shape",
        "list_type",
        "list_dtype",
        "list_shape",
        "weight_dtype",
        "weight_shape",
        "channel_scale",
        "output_dtype",
        "limit",
        "activation",
    ],
)
def test_structural_rejection(invalid):
    native, ref, x, scale, w, rw, kwargs = _case(torch.bfloat16, 1)
    for provider, weights in ((native, w), (ref, rw)):
        xx, ss, ww, args = x, scale, list(weights), dict(kwargs)
        if invalid == "input_dtype":
            xx = x.float()
        elif invalid == "input_shape":
            xx = x.unsqueeze(0)
        elif invalid == "scale_dtype":
            ss = scale.half()
        elif invalid == "scale_shape":
            ss = scale[:-1]
        elif invalid == "list_type":
            args["group_list_type"] = 2
        elif invalid == "list_dtype":
            args["group_list"] = args["group_list"].int()
        elif invalid == "list_shape":
            args["group_list"] = args["group_list"][:-1]
        elif invalid == "weight_dtype":
            ww[0] = ww[0].float()
        elif invalid == "weight_shape":
            ww[1] = ww[1][:, :, :-1]
        elif invalid == "channel_scale":
            ww[2] = ww[2].squeeze(-1)
        elif invalid == "output_dtype":
            args["output_dtype"] = torch.float32
        elif invalid == "limit":
            args["limit"] = float("nan")
        else:
            args["activation"] = "gelu"
        with pytest.raises((ValueError, NotImplementedError)):
            provider.modelslim_w8a8_grouped_experts(xx, ss, *ww, **args)


@pytest.mark.parametrize("list_type", [0, 1])
def test_grouped_changed_counts_graph(list_type):
    import torch_npu  # noqa: F401

    native, ref, x, scale, w, rw, kwargs = _case(torch.bfloat16, list_type)
    for _ in range(3):
        native.modelslim_w8a8_grouped_experts(x, scale, *w, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = native.modelslim_w8a8_grouped_experts(x, scale, *w, **kwargs)
    for counts in ((0, 2, 1), (0, 0, 0), (1, 1, 2)):
        values = torch.tensor(counts, dtype=torch.int64, device="npu")
        kwargs["group_list"].copy_(values.cumsum(0) if list_type == 0 else values)
        scale.fill_(0.03125)
        scale[sum(counts) :] = float("nan")
        x.copy_(torch.randint_like(x, -100, 100))
        expected = ref.modelslim_w8a8_grouped_experts(x, scale, *rw, **kwargs)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.03, atol=0.06)
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output[sum(counts) :]) == 0


@pytest.mark.parametrize(
    "list_type,counts", [(0, [2, 1, 3]), (1, [2, -1, 2]), (1, [9, 0, 0])]
)
def test_eager_oracle_rejects_invalid_count_values(list_type, counts):
    _, ref, x, scale, _, rw, kwargs = _case(torch.bfloat16, list_type)
    kwargs["group_list"] = torch.tensor(counts, dtype=torch.int64, device="npu")
    with pytest.raises(ValueError, match="counts"):
        ref.modelslim_w8a8_grouped_experts(x, scale, *rw, **kwargs)
