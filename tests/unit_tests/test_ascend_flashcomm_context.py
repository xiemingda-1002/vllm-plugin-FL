# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from vllm_fl import ascend_forward_context as afc
import vllm_fl.ascend_flashcomm as flashcomm


def _config(
    *,
    is_moe: bool = True,
    enable_ep: bool = True,
    tp_size: int = 2,
    enabled: bool = True,
    refresh: bool = True,
    enforce_eager: bool = False,
    capture_sizes: list[int] | None = None,
):
    additional_config = {
        "enable_flashcomm1": enabled,
        "refresh": refresh,
    }
    compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
        cudagraph_capture_sizes=capture_sizes or [1, 2, 4, 5, 8],
        max_cudagraph_capture_size=8,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            is_moe_model=is_moe,
            enable_expert_parallel=enable_ep,
            world_size_across_dp=4,
            pipeline_parallel_size=1,
            tensor_parallel_size=tp_size,
        ),
        model_config=SimpleNamespace(
            enforce_eager=enforce_eager,
            get_num_experts=lambda: 128 if is_moe else 0,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
        compilation_config=compilation_config,
        additional_config=additional_config,
    )
    config.update_sizes_for_sequence_parallelism = lambda sizes: [
        size for size in sizes if size % tp_size == 0
    ]
    return config


@pytest.fixture(autouse=True)
def _reset_flashcomm_cache(monkeypatch):
    monkeypatch.setattr(flashcomm, "_ENABLE_FLASHCOMM1", None)
    monkeypatch.setattr(flashcomm, "_IS_VL_MODEL", None)


def _context(config, *, num_tokens: int, dp_tokens=None):
    dp_metadata = None
    if dp_tokens is not None:
        dp_metadata = SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor(dp_tokens)
        )
    return afc.build_additional_forward_context(
        attn_metadata=None,
        vllm_config=config,
        dp_metadata=dp_metadata,
        num_tokens=num_tokens,
    )


def test_additional_config_gate_uses_rc1_refresh_cache() -> None:
    assert flashcomm.enable_flashcomm1(_config(enabled=True))

    cached_false = _config(enabled=False, refresh=False)
    assert flashcomm.enable_flashcomm1(cached_false)

    cached_false.additional_config["refresh"] = True
    assert not flashcomm.enable_flashcomm1(cached_false)


def test_moe_enables_flashcomm1_for_every_token_count(monkeypatch) -> None:
    monkeypatch.setattr(afc, "_get_moe_comm_method", lambda _: object())
    for num_tokens in (0, 1, 1000):
        context = _context(_config(is_moe=True), num_tokens=num_tokens)
        assert context["flash_comm_v1_enabled"] is True
        assert context["mmrs_fusion"] is False


def test_missing_token_count_does_not_enable_flashcomm1(monkeypatch) -> None:
    monkeypatch.setattr(afc, "_get_moe_comm_method", lambda _: object())
    context = afc.build_additional_forward_context(
        attn_metadata=None,
        vllm_config=_config(is_moe=True),
        dp_metadata=None,
        num_tokens=None,
    )

    assert context["flash_comm_v1_enabled"] is False


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [(1000, False), (1001, True)],
)
def test_dense_flashcomm1_threshold(num_tokens: int, expected: bool) -> None:
    context = _context(
        _config(is_moe=False),
        num_tokens=num_tokens,
    )
    assert context["flash_comm_v1_enabled"] is expected
    assert context["mmrs_fusion"] is True


def test_tp_padding_without_dp_keeps_padded_length_unset(monkeypatch) -> None:
    monkeypatch.setattr(afc, "_get_moe_comm_method", lambda _: object())
    context = _context(_config(tp_size=4), num_tokens=5)

    assert context["pad_size"] == 3
    assert context["padded_length"] is None
    assert context["padded_num_tokens"] == 8


def test_dp_padding_uses_largest_replica_then_tp_rounding(monkeypatch) -> None:
    monkeypatch.setattr(afc, "_get_moe_comm_method", lambda _: object())
    context = _context(
        _config(tp_size=4),
        num_tokens=5,
        dp_tokens=[5, 10],
    )

    assert context["max_tokens_across_dp"] == 10
    assert context["padded_length"] == 12
    assert context["pad_size"] == 7
    assert context["padded_num_tokens"] == 12


def test_flashcomm1_validation_requires_tp_and_moe_ep() -> None:
    with pytest.raises(AssertionError, match="tp_size > 1"):
        flashcomm.validate_and_update_flashcomm1_config(_config(tp_size=1))

    with pytest.raises(AssertionError, match="enable_expert_parallel=True"):
        flashcomm.validate_and_update_flashcomm1_config(
            _config(is_moe=True, enable_ep=False)
        )


def test_flashcomm1_graph_capture_sizes_are_tp_divisible() -> None:
    config = _config(tp_size=4, capture_sizes=[1, 2, 4, 5, 8])

    flashcomm.validate_and_update_flashcomm1_config(config)

    assert config.compilation_config.cudagraph_capture_sizes == [4, 8]
    assert config.compilation_config.max_cudagraph_capture_size == 8


def test_disabled_gate_does_not_rewrite_graph_shapes() -> None:
    config = _config(enabled=False, tp_size=1, capture_sizes=[1, 2, 3])

    flashcomm.validate_and_update_flashcomm1_config(config)

    assert config.compilation_config.cudagraph_capture_sizes == [1, 2, 3]
