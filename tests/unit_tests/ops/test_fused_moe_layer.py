# Copyright (c) 2026 BAAI. All rights reserved.

from unittest.mock import MagicMock, patch

from vllm_fl.ops.fused_moe import layer


def _runner_with_quant_method(quant_method):
    runner = MagicMock()
    runner._quant_method = quant_method
    runner.moe_config = MagicMock()
    return runner


def test_fused_moe_fl_replaces_unquantized_method():
    quant_method = MagicMock(spec=layer.UnquantizedFusedMoEMethod)
    runner = _runner_with_quant_method(quant_method)
    replacement = MagicMock()

    with (
        patch.object(layer, "_OrigFusedMoE", return_value=runner),
        patch.object(
            layer,
            "UnquantizedFusedMoEMethodFL",
            return_value=replacement,
        ) as replacement_cls,
        patch.object(layer, "replace_router_with_fl") as replace_router,
        patch(
            "vllm.platforms.current_platform",
            vendor_name="generic",
            device_type="cuda",
        ),
    ):
        result = layer.FusedMoEFL(test_arg=True)

    assert result is runner
    replacement_cls.assert_called_once_with(runner.moe_config)
    runner._replace_quant_method.assert_called_once_with(replacement)
    replace_router.assert_called_once_with()


def test_non_ascend_method_keeps_upstream_weight_layout_lifecycle():
    assert "process_weights_after_loading" not in (
        layer.UnquantizedFusedMoEMethodFL.__dict__
    )
    assert (
        layer.UnquantizedFusedMoEMethodFL.process_weights_after_loading
        is layer.UnquantizedFusedMoEMethod.process_weights_after_loading
    )


def test_fused_moe_fl_preserves_quantized_method():
    quant_method = object()
    runner = _runner_with_quant_method(quant_method)

    with (
        patch.object(layer, "_OrigFusedMoE", return_value=runner),
        patch.object(layer, "UnquantizedFusedMoEMethodFL") as replacement_cls,
        patch.object(layer, "replace_router_with_fl") as replace_router,
        patch.object(layer.logger, "info_once") as info_once,
        patch(
            "vllm.platforms.current_platform",
            vendor_name="generic",
            device_type="cuda",
        ),
    ):
        result = layer.FusedMoEFL()

    assert result is runner
    replacement_cls.assert_not_called()
    runner._replace_quant_method.assert_not_called()
    replace_router.assert_called_once_with()
    info_once.assert_called_once_with(
        "Preserving upstream quantized MoE method %s in FusedMoEFL.",
        "object",
    )


def test_fused_moe_fl_preserves_explicit_ascend_runner_cls():
    runner = _runner_with_quant_method(MagicMock())
    runner_cls = type("ExplicitRunner", (), {})

    with (
        patch.object(layer, "_OrigFusedMoE", return_value=runner) as original,
        patch(
            "vllm.platforms.current_platform",
            vendor_name="ascend",
            device_type="npu",
        ),
        patch.object(layer, "replace_router_with_fl") as replace_router,
    ):
        result = layer.FusedMoEFL(
            runner_cls=runner_cls,
            runner_args={"caller_owned": True},
        )

    assert result is runner
    original.assert_called_once_with(
        runner_cls=runner_cls,
        runner_args={"caller_owned": True},
    )
    runner._replace_quant_method.assert_not_called()
    replace_router.assert_called_once_with()


def test_fused_moe_fl_injects_rc1_ascend_runner_at_factory_construction():
    runner = _runner_with_quant_method(MagicMock())
    ascend_runner_cls = MagicMock(name="AscendMoERunner")
    with (
        patch.object(layer, "_OrigFusedMoE", return_value=runner) as original,
        patch(
            "vllm.platforms.current_platform",
            vendor_name="ascend",
            device_type="npu",
        ),
        patch(
            "vllm_fl.dispatch.backends.vendor.ascend.impl.moe.fused_moe."
            "AscendMoERunner",
            ascend_runner_cls,
        ),
        patch.object(layer, "replace_router_with_fl"),
    ):
        result = layer.FusedMoEFL(runner_args={"tid2eid": None})

    assert result is runner
    original.assert_called_once_with(
        runner_cls=ascend_runner_cls,
        runner_args={"tid2eid": None},
    )
    runner._replace_quant_method.assert_not_called()
