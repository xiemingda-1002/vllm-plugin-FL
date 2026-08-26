import pytest


def _import_attention():
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.impl import attention
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"Ascend attention dependencies are unavailable: {exc}")
    return attention


@pytest.mark.parametrize(
    "runner_state",
    [
        "PrefillNoCache",
        "PrefillCacheHit",
        "DecodeOnly",
        "ChunkedPrefill",
    ],
)
def test_attention_builder_preserves_runner_state(runner_state):
    attention = _import_attention()
    builder = object.__new__(attention.AscendAttentionMetadataBuilder)
    expected = getattr(attention.AscendAttentionState, runner_state)

    resolved = builder._resolve_attn_state(
        expected,
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
        num_prefill_tokens=4096,
    )

    assert resolved is expected


def test_attention_builder_infers_state_when_runner_state_is_missing():
    attention = _import_attention()
    builder = object.__new__(attention.AscendAttentionMetadataBuilder)

    resolved = builder._resolve_attn_state(
        None,
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
        num_prefill_tokens=4095,
    )

    assert resolved is attention.AscendAttentionState.ChunkedPrefill
