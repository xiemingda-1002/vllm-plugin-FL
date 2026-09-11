from __future__ import annotations

import ast
import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest
import torch

import vllm_fl.ascend_flashcomm as flashcomm


MODULE_PREFIX = "vllm_fl.dispatch.backends.vendor.ascend.impl"


@pytest.mark.parametrize(
    ("prefix", "is_vl", "expected"),
    [
        ("model.layers.0.linear_attn.in_proj_qkvz", True, False),
        ("model.layers.1.linear_attn.in_proj_qkvz", True, True),
        ("model.layers.0.linear_attn.in_proj_qkvz", False, True),
        ("model.layers.0.mlp.gate_up_proj", True, True),
    ],
    ids=(
        "vl-layer0-attn",
        "vl-later-attn",
        "non-vl-layer0-attn",
        "unrelated",
    ),
)
def test_sequence_column_all_gather_matches_current_rc1(
    prefix: str,
    is_vl: bool,
    expected: bool,
) -> None:
    linear_op = importlib.import_module(f"{MODULE_PREFIX}.linear_op")

    assert linear_op._sequence_column_needs_all_gather(prefix, is_vl) is expected


def test_vl_detection_uses_nested_config_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flashcomm, "_IS_VL_MODEL", None)
    text_config = SimpleNamespace(to_dict=lambda: {})
    config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: {}),
                hf_text_config=text_config,
            )
    )
    assert flashcomm.is_vl_model(config) is True


def test_vl_detection_uses_vision_config_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flashcomm, "_IS_VL_MODEL", None)
    vision_config = SimpleNamespace(
        to_dict=lambda: {"vision_config": {"hidden_size": 1024}}
    )
    config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=vision_config,
                hf_text_config=vision_config,
            )
    )
    assert flashcomm.is_vl_model(config) is True


def test_vl_detection_is_false_for_a_non_vl_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flashcomm, "_IS_VL_MODEL", None)
    text_config = SimpleNamespace(to_dict=lambda: {})
    config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=text_config,
                hf_text_config=text_config,
            )
    )
    assert flashcomm.is_vl_model(config) is False


def test_vl_detection_keeps_cached_value_without_current_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flashcomm, "_IS_VL_MODEL", None)
    text_config = SimpleNamespace(to_dict=lambda: {})
    config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: {}),
                hf_text_config=text_config,
            )
    )
    assert flashcomm.is_vl_model(config) is True

    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: None)
    assert flashcomm.is_vl_model() is True


def test_vl_detection_no_context_does_not_freeze_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flashcomm, "_IS_VL_MODEL", None)
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: None)
    assert flashcomm.is_vl_model() is None

    text_config = SimpleNamespace(to_dict=lambda: {})
    config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: {}),
                hf_text_config=text_config,
            )
    )
    assert flashcomm.is_vl_model(config) is True


def test_linear_op_consumes_shared_vl_cache() -> None:
    linear_op = importlib.import_module(f"{MODULE_PREFIX}.linear_op")

    assert linear_op.is_vl_model is flashcomm.is_vl_model


def test_sequence_column_passes_false_label_for_first_vl_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear_op = importlib.import_module(f"{MODULE_PREFIX}.linear_op")
    labels = []
    monkeypatch.setattr(linear_op, "is_vl_model", lambda: True)
    monkeypatch.setattr(
        linear_op.torch.ops,
        "vllm",
        SimpleNamespace(
            maybe_all_gather_and_maybe_unpad=lambda x, *, label: (
                labels.append(label) or x
            )
        ),
    )
    layer = SimpleNamespace(
        prefix="model.layers.0.linear_attn.in_proj_qkvz",
        skip_bias_add=False,
        return_bias=True,
        quant_method=SimpleNamespace(apply=lambda layer, x, bias: x),
        gather_output=False,
        bias=None,
    )
    op = linear_op.SequenceColumnParallelOp(layer)
    op.update_attrs()

    output, output_bias = op.apply_impl(torch.ones(2, 3))

    assert labels == [False]
    torch.testing.assert_close(output, torch.ones(2, 3))
    assert output_bias is None


def test_flashcomm_modules_have_no_import_time_torch_npu_dependency() -> None:
    names = (
        f"{MODULE_PREFIX}.flashcomm_custom_ops",
        f"{MODULE_PREFIX}.linear_op",
        f"{MODULE_PREFIX}.linear",
    )
    for name in names:
        sys.modules.pop(name, None)
    for name in names:
        module = importlib.import_module(name)
        tree = ast.parse(inspect.getsource(module))
        imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all(
            getattr(node, "module", None) != "torch_npu"
            and all(alias.name != "torch_npu" for alias in getattr(node, "names", ()))
            for node in imports
        )


def test_flashcomm_custom_op_registration_is_explicit_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = importlib.import_module(f"{MODULE_PREFIX}.flashcomm_custom_ops")
    registrations = []
    monkeypatch.setattr(ops, "_REGISTERED", False)
    monkeypatch.setattr(
        ops,
        "direct_register_custom_op",
        lambda **kwargs: registrations.append(kwargs),
    )
    monkeypatch.setattr(
        ops.torch.ops,
        "vllm",
        SimpleNamespace(),
    )

    ops.ensure_ascend_flashcomm_custom_ops_registered()
    ops.ensure_ascend_flashcomm_custom_ops_registered()

    assert [entry["op_name"] for entry in registrations] == [
        "maybe_chunk_residual",
        "maybe_all_gather_and_maybe_unpad",
        "maybe_pad_and_reduce",
        "matmul_and_reduce",
    ]
    assert all(entry["dispatch_key"] == "PrivateUse1" for entry in registrations)


def test_unquantized_gemm_registration_is_explicit_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear = importlib.import_module(f"{MODULE_PREFIX}.linear")
    registrations = []
    monkeypatch.setattr(linear, "_UNQUANTIZED_GEMM_REGISTERED", False)
    monkeypatch.setattr(
        linear,
        "direct_register_custom_op",
        lambda **kwargs: registrations.append(kwargs),
    )
    monkeypatch.setattr(linear.torch.ops, "vllm", SimpleNamespace())

    linear.ensure_ascend_linear_custom_ops_registered()
    linear.ensure_ascend_linear_custom_ops_registered()

    assert [entry["op_name"] for entry in registrations] == [
        "unquantized_gemm"
    ]
    assert registrations[0]["dispatch_key"] == "PrivateUse1"


def test_maybe_chunk_residual_pads_then_selects_tp_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = importlib.import_module(f"{MODULE_PREFIX}.flashcomm_custom_ops")
    monkeypatch.setattr(ops, "get_forward_context", lambda: object())
    monkeypatch.setattr(
        ops, "_EXTRA_CTX", SimpleNamespace(pad_size=1)
    )
    monkeypatch.setattr(ops, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ops, "get_tensor_model_parallel_rank", lambda: 1)
    x = torch.empty(2, 2)
    residual = torch.arange(6).reshape(3, 2)

    result = ops._maybe_chunk_residual_impl(x, residual)

    torch.testing.assert_close(result, torch.tensor([[4, 5], [0, 0]]))


def test_maybe_gather_unpads_and_reduce_scatter_pads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = importlib.import_module(f"{MODULE_PREFIX}.flashcomm_custom_ops")
    context = SimpleNamespace(dp_metadata=None)
    monkeypatch.setattr(ops, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        ops,
        "_EXTRA_CTX",
        SimpleNamespace(flash_comm_v1_enabled=True, pad_size=1),
    )
    monkeypatch.setattr(
        ops,
        "tensor_model_parallel_all_gather",
        lambda x, dim: torch.cat((x, x + 10), dim=dim),
    )
    reduced = []
    monkeypatch.setattr(
        ops,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim: reduced.append((x, dim)) or x[:2],
    )
    x = torch.tensor([[1.0], [2.0]])

    gathered = ops._maybe_all_gather_and_maybe_unpad_impl(x, True)
    scattered = ops._maybe_pad_and_reduce_impl(x)

    torch.testing.assert_close(gathered, torch.tensor([[1.0], [2.0], [11.0]]))
    torch.testing.assert_close(reduced[0][0], torch.tensor([[1.0], [2.0], [0.0]]))
    assert reduced[0][1] == 0
    torch.testing.assert_close(scattered, torch.tensor([[1.0], [2.0]]))


def test_sequence_row_mmrs_is_bf16_unquantized_and_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear_op = importlib.import_module(f"{MODULE_PREFIX}.linear_op")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class Backend:
        def get_hccl_comm_name(self, rank):
            assert rank == 0
            return "hcom"

    group = SimpleNamespace(
        rank_in_group=0,
        world_size=2,
        device_group=SimpleNamespace(_get_backend=lambda device: Backend()),
    )
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        linear_op,
        "_EXTRA_CTX",
        SimpleNamespace(
            flash_comm_v1_enabled=True,
            mmrs_fusion=True,
            pad_size=0,
        ),
    )
    calls = []
    monkeypatch.setattr(
        linear_op.DeviceOperator,
        "npu_mm_reduce_scatter_base",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or torch.ones(1, 3),
    )
    layer = SimpleNamespace(
        quant_method=UnquantizedLinearMethod(),
        weight=torch.ones(3, 4),
        tp_rank=0,
        tp_size=2,
    )
    op = linear_op.SequenceRowParallelOp(layer)
    op.quant_method = layer.quant_method

    result = op.matmul_and_reduce(torch.ones(2, 4), None)

    assert result.shape == (1, 3)
    assert calls[0][0][2:4] == ("hcom", 2)


def test_linear_selector_covers_qwen_projections_and_skips_shared_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear_op = importlib.import_module(f"{MODULE_PREFIX}.linear_op")
    group = SimpleNamespace(rank_in_group=1, world_size=2)
    monkeypatch.setattr(linear_op, "flashcomm1_configured", lambda: True)
    monkeypatch.setattr(linear_op, "shared_expert_dp_enabled", lambda: True)
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: group)
    layer = object()

    column, rank, size = linear_op.get_parallel_op(
        False, "model.layers.0.in_proj_qkvz", layer, "column"
    )
    row, _, _ = linear_op.get_parallel_op(
        False, "model.layers.0.out_proj", layer, "row"
    )
    shared, _, _ = linear_op.get_parallel_op(
        False, "model.layers.0.shared_expert.down_proj", layer, "row"
    )

    assert isinstance(column, linear_op.SequenceColumnParallelOp)
    assert isinstance(row, linear_op.SequenceRowParallelOp)
    assert shared is None
    assert (rank, size) == (1, 2)
    assert linear_op.get_parallel_op(
        False, "model.layers.0.shared_expert.down_proj", layer, "row"
    )[1:] == (0, 1)


@pytest.mark.parametrize("fake_rank", [0, 1])
def test_flashcomm_shared_expert_weights_are_full_and_unsharded_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
    fake_rank: int,
) -> None:
    linear_op = importlib.import_module(f"{MODULE_PREFIX}.linear_op")
    linear = importlib.import_module(f"{MODULE_PREFIX}.linear")
    parameter = importlib.import_module("vllm.model_executor.parameter")
    group = SimpleNamespace(rank_in_group=fake_rank, world_size=2)
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: group)
    monkeypatch.setattr(linear_op, "shared_expert_dp_enabled", lambda: True)
    monkeypatch.setattr(
        parameter, "get_tensor_model_parallel_rank", lambda: fake_rank
    )
    monkeypatch.setattr(
        parameter, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(linear, "flashcomm1_configured", lambda: True)
    monkeypatch.setattr(
        linear,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
    )

    gate_up = linear.AscendMergedColumnParallelLinear(
        input_size=4,
        output_sizes=[6, 6],
        bias=False,
        prefix="model.layers.0.mlp.shared_expert.gate_up_proj",
    )
    down = linear.AscendRowParallelLinear(
        input_size=6,
        output_size=4,
        bias=False,
        reduce_results=False,
        prefix="model.layers.0.mlp.shared_expert.down_proj",
    )

    assert (gate_up.tp_rank, gate_up.tp_size) == (0, 1)
    assert tuple(gate_up.weight.shape) == (12, 4)
    assert (down.tp_rank, down.tp_size) == (0, 1)
    assert tuple(down.weight.shape) == (4, 6)


@pytest.mark.parametrize("fake_rank", [0, 1])
def test_flashcomm_regular_qwen_linears_keep_tp2_sequence_ops_and_shards(
    monkeypatch: pytest.MonkeyPatch,
    fake_rank: int,
) -> None:
    linear_op = importlib.import_module(f"{MODULE_PREFIX}.linear_op")
    linear = importlib.import_module(f"{MODULE_PREFIX}.linear")
    parameter = importlib.import_module("vllm.model_executor.parameter")
    group = SimpleNamespace(rank_in_group=fake_rank, world_size=2)
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: group)
    monkeypatch.setattr(linear_op, "flashcomm1_configured", lambda: True)
    monkeypatch.setattr(linear_op, "shared_expert_dp_enabled", lambda: True)
    monkeypatch.setattr(
        parameter, "get_tensor_model_parallel_rank", lambda: fake_rank
    )
    monkeypatch.setattr(
        parameter, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(linear, "flashcomm1_configured", lambda: True)
    monkeypatch.setattr(
        linear,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
    )

    gate_up = linear.AscendMergedColumnParallelLinear(
        input_size=4,
        output_sizes=[6, 6],
        bias=False,
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    down = linear.AscendRowParallelLinear(
        input_size=6,
        output_size=4,
        bias=False,
        prefix="model.layers.0.mlp.down_proj",
    )

    assert isinstance(gate_up.custom_op, linear_op.SequenceColumnParallelOp)
    assert (gate_up.tp_rank, gate_up.tp_size) == (fake_rank, 2)
    assert tuple(gate_up.weight.shape) == (6, 4)
    assert isinstance(down.custom_op, linear_op.SequenceRowParallelOp)
    assert (down.tp_rank, down.tp_size) == (fake_rank, 2)
    assert tuple(down.weight.shape) == (4, 3)


def test_flashcomm_constructor_selects_effective_tp_before_weight_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear_op = importlib.import_module(f"{MODULE_PREFIX}.linear_op")
    linear = importlib.import_module(f"{MODULE_PREFIX}.linear")
    parameter = importlib.import_module("vllm.model_executor.parameter")
    group = SimpleNamespace(rank_in_group=1, world_size=2)
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: group)
    monkeypatch.setattr(linear_op, "flashcomm1_configured", lambda: True)
    monkeypatch.setattr(linear_op, "shared_expert_dp_enabled", lambda: True)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        parameter, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(linear, "flashcomm1_configured", lambda: True)
    monkeypatch.setattr(
        linear,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
    )

    original_create_weights = linear.AscendUnquantizedLinearMethod.create_weights
    observed = []

    def record_create_weights(method, *args, **kwargs):
        layer = kwargs.get("layer", args[0] if args else None)
        observed.append((layer.prefix, layer.tp_rank, layer.tp_size))
        return original_create_weights(method, *args, **kwargs)

    monkeypatch.setattr(
        linear.AscendUnquantizedLinearMethod,
        "create_weights",
        record_create_weights,
    )

    linear.AscendMergedColumnParallelLinear(
        input_size=4,
        output_sizes=[6, 6],
        bias=False,
        prefix="model.layers.0.mlp.shared_expert.gate_up_proj",
    )
    linear.AscendRowParallelLinear(
        input_size=6,
        output_size=4,
        bias=False,
        reduce_results=False,
        prefix="model.layers.0.mlp.shared_expert.down_proj",
    )
    linear.AscendMergedColumnParallelLinear(
        input_size=4,
        output_sizes=[6, 6],
        bias=False,
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    linear.AscendRowParallelLinear(
        input_size=6,
        output_size=4,
        bias=False,
        prefix="model.layers.0.mlp.down_proj",
    )

    assert observed == [
        ("model.layers.0.mlp.shared_expert.gate_up_proj", 0, 1),
        ("model.layers.0.mlp.shared_expert.down_proj", 0, 1),
        ("model.layers.0.mlp.gate_up_proj", 1, 2),
        ("model.layers.0.mlp.down_proj", 1, 2),
    ]


def test_device_mmrs_delegates_all_rc1_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = importlib.import_module(f"{MODULE_PREFIX}.device_operator")
    calls = []
    fake_torch_npu = SimpleNamespace(
        npu_mm_reduce_scatter_base=lambda *args, **kwargs: calls.append(
            (args, kwargs)
        )
        or "result"
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    x1 = torch.empty(2, 4)
    x2 = torch.empty(4, 3)

    result = device.DeviceOperator.npu_mm_reduce_scatter_base(
        x1, x2, "hcom", 2, comm_mode="aiv"
    )

    assert result == "result"
    assert calls[0][0] == (x1, x2, "hcom", 2)
    assert calls[0][1]["reduce_op"] == "sum"
    assert calls[0][1]["comm_mode"] == "aiv"


def test_layernorm_flashcomm_residual_boundary_is_vendor_local() -> None:
    layernorm = importlib.import_module(f"{MODULE_PREFIX}.layernorm")

    assert issubclass(
        layernorm.AscendRMSNorm,
        importlib.import_module(
            "vllm.model_executor.layers.layernorm"
        ).RMSNorm,
    )
    assert "maybe_chunk_residual" in inspect.getsource(
        layernorm.AscendRMSNorm.forward_oot
    )
    assert "maybe_chunk_residual" in inspect.getsource(
        layernorm.AscendGemmaRMSNorm.forward_oot
    )
