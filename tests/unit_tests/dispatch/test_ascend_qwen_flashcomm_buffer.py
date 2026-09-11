# Copyright (c) 2026 BAAI. All rights reserved.

import pytest

from vllm_fl.ascend_flashcomm import flashcomm1_attention_output_tokens


def test_first_qwen_layer_uses_ceil_divided_flashcomm_output() -> None:
    assert (
        flashcomm1_attention_output_tokens(
            layer_idx=0,
            num_tokens=5,
            tp_size=2,
            enabled=True,
        )
        == 3
    )


def test_non_first_qwen_layer_keeps_full_attention_output() -> None:
    assert (
        flashcomm1_attention_output_tokens(
            layer_idx=1,
            num_tokens=5,
            tp_size=2,
            enabled=True,
        )
        == 5
    )


def test_qwen_output_is_not_sharded_when_flashcomm_is_disabled() -> None:
    assert (
        flashcomm1_attention_output_tokens(
            layer_idx=0,
            num_tokens=5,
            tp_size=2,
            enabled=False,
        )
        == 5
    )


def test_qwen_output_rejects_invalid_enabled_tp_size() -> None:
    with pytest.raises(ValueError, match="tp_size must be positive"):
        flashcomm1_attention_output_tokens(
            layer_idx=0,
            num_tokens=5,
            tp_size=0,
            enabled=True,
        )
