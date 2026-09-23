import pytest
import torch


def _load_native_dispatcher() -> None:
    from vllm_fl.ascend_custom_ops import enable_custom_op

    if not enable_custom_op():
        pytest.skip("FL Ascend native dispatcher is not installed")


def test_sparse_flash_attention_dispatcher_schema_and_meta_outputs():
    """Exercise the registered graph-time contract, not source text only."""
    _load_native_dispatcher()

    schema = torch._C._dispatch_find_schema_or_throw(
        "_C_ascend::npu_sparse_flash_attention", "").schema()
    assert "Tensor sparse_indices" in str(schema)
    assert "Tensor? query_rope=None" in str(schema)
    assert "Tensor? key_rope=None" in str(schema)

    query = torch.empty((7, 4, 16), device="meta", dtype=torch.bfloat16)
    kv = torch.empty((2, 8, 1, 16), device="meta", dtype=torch.bfloat16)
    topk = torch.empty((7, 1, 4), device="meta", dtype=torch.int32)
    rope = torch.empty((7, 1, 16), device="meta", dtype=torch.bfloat16)
    output, softmax_max, softmax_sum = (
        torch.ops._C_ascend.npu_sparse_flash_attention(
            query, kv, kv, topk, 0.25,
            query_rope=rope,
            key_rope=kv,
            layout_query="TND",
            layout_kv="PA_BSND",
            return_softmax_lse=True,
        )
    )
    assert output.device.type == "meta"
    assert tuple(output.shape) == (7, 4, 16)
    assert tuple(softmax_max.shape) == (1, 7, 4)
    assert tuple(softmax_sum.shape) == (1, 7, 4)
    assert softmax_max.dtype is torch.float32
    assert softmax_sum.dtype is torch.float32
