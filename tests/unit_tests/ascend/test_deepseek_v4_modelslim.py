# Copyright 2026 FlagOS Contributors

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map

from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa_cache import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    reshape_dsa_kv_cache,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa import (
    AscendDSAMetadataBuilder,
    _build_hadamard,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa_cache_manager import (
    AscendCompressedAttentionManager,
    install_dsa_cache_manager,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa_cache_utils import (
    _get_dsa_kv_cache_groups,
    _group_and_unify_dsa_kv_cache_specs,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa_utils import (
    compute_dsa_slot_mappings,
    get_compressed_pos_and_indices,
)
from vllm_fl.models.deepseek_v4_ascend import (
    _deepseek_v4_modelslim_prefix,
)
from vllm_fl.quantization.modelslim import AscendModelSlimConfig


def _config() -> AscendModelSlimConfig:
    config = AscendModelSlimConfig(
        {
            "layers.0.attn.wq_a.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.shared_experts.w1.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.shared_experts.w3.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.shared_experts.w2.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
            "embed.weight": "FLOAT",
            "head.weight": "FLOAT",
        }
    )
    config.model_type = "deepseek_v4"
    config.packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "experts": [
            "experts.0.w1",
            "experts.0.w2",
            "experts.0.w3",
        ],
    }
    return config


def test_dsa_hadamard_has_no_scipy_runtime_dependency():
    matrix = _build_hadamard(
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(matrix, expected)
    torch.testing.assert_close(matrix @ matrix.T, 4 * torch.eye(4))


def test_deepseek_dsa_supports_uniform_decode_graph_batches():
    assert (
        AscendDSAMetadataBuilder.aclgraph_support
        is AttentionCGSupport.UNIFORM_BATCH
    )
    assert (
        AscendDSAMetadataBuilder.get_cudagraph_support(None, None)
        is AttentionCGSupport.UNIFORM_BATCH
    )


def test_dsa_metadata_keeps_padded_forward_shape_after_dp_graph_downgrade(
    monkeypatch,
):
    """C15 real decode metadata must provide C16 RoPE for a C16 forward."""
    import vllm_fl.dispatch.backends.vendor.ascend.impl.dsa as dsa_module

    builder = object.__new__(AscendDSAMetadataBuilder)
    builder.decode_threshold = 1
    builder.model_config = SimpleNamespace(get_head_size=lambda: 128)
    builder.metadata_cls = lambda **kwargs: SimpleNamespace(**kwargs)
    builder.slot_mapping = torch.zeros((32, 2), dtype=torch.int32)
    builder.get_block_table_size = MagicMock(return_value=15)
    builder.build_decode_metadata = MagicMock(return_value=SimpleNamespace())

    def fake_get_cos_and_sin(positions, is_decode=False):
        assert positions.shape == (16,)
        return (
            torch.zeros((positions.shape[0], 64)),
            torch.ones((positions.shape[0], 64)),
        )

    monkeypatch.setattr(
        dsa_module,
        "get_cos_and_sin_dsa",
        fake_get_cos_and_sin,
    )
    common = SimpleNamespace(
        num_reqs=15,
        num_actual_tokens=15,
        num_input_tokens=16,
        max_query_len=1,
        query_start_loc=torch.arange(16, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(16, dtype=torch.int32),
        positions=torch.arange(16, dtype=torch.int64),
        seq_lens=torch.arange(15, dtype=torch.int32),
        slot_mapping=torch.cat(
            [
                torch.arange(15, dtype=torch.int32),
                torch.tensor([-1], dtype=torch.int32),
            ]
        ),
        graph_pad_size=-1,
        block_table_tensor=torch.zeros((15, 1), dtype=torch.int32),
        attn_state=SimpleNamespace(),
        prefill_context_parallel_metadata=None,
    )

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_reqs_actual=15,
        block_size=128,
    )

    assert metadata.num_input_tokens == 16
    assert metadata.num_actual_tokens == 15
    assert metadata.num_decodes == 15
    assert metadata.num_decode_tokens == 15
    assert metadata.cos.shape[0] == 16
    assert metadata.sin.shape[0] == 16
    torch.testing.assert_close(
        builder.slot_mapping[:16, 0],
        torch.tensor([0] * 15 + [-1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        builder.slot_mapping[:16, 1],
        torch.tensor(list(range(15)) + [127], dtype=torch.int32),
    )


def test_deepseek_v4_modelslim_prefix_mapping():
    assert _deepseek_v4_modelslim_prefix(
        "model.layers.0.self_attn.wq_a"
    ) == "layers.0.attn.wq_a"
    assert _deepseek_v4_modelslim_prefix(
        "model.layers.0.mlp.shared_experts.down_proj"
    ) == "layers.0.ffn.shared_experts.w2"
    assert _deepseek_v4_modelslim_prefix("model.embed_tokens") == "embed"
    assert _deepseek_v4_modelslim_prefix("lm_head") == "head"


def test_deepseek_v4_modelslim_quant_contracts():
    config = _config()
    assert config._get_linear_quant_type(
        "model.layers.0.self_attn.wq_a"
    ) == "W8A8_DYNAMIC"
    assert config._get_linear_quant_type(
        "model.layers.0.mlp.shared_experts.gate_up_proj"
    ) == "W8A8_DYNAMIC"
    assert config._get_linear_quant_type(
        "model.layers.0.mlp.experts"
    ) == "W8A8_DYNAMIC"
    assert config._get_linear_quant_type("model.embed_tokens") == "FLOAT"
    assert config._get_linear_quant_type("lm_head") == "FLOAT"


def test_quantized_indexer_cache_page_layout():
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        compress_ratio=4,
        scale_dim=1,
        scale_dtype=torch.float16,
    )
    assert spec.page_size_bytes == 16640

    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.int8)
    views = reshape_dsa_kv_cache(raw, spec, num_blocks=2)
    assert views is not None
    key_cache, scale_cache = views
    assert key_cache.shape == (2, 128, 1, 128)
    assert scale_cache.shape == (2, 128, 1, 1)
    assert key_cache.stride(0) == spec.page_size_bytes
    assert scale_cache.stride(0) == spec.page_size_bytes // 2

    key_cache[0].fill_(3)
    scale_cache[0].fill_(5)
    assert torch.count_nonzero(key_cache[1]) == 0
    assert torch.count_nonzero(scale_cache[1]) == 0


def test_compressor_state_cache_uses_padded_page_stride():
    spec = AscendSlidingWindowMLASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
        sliding_window=128,
        page_size_padded=131072,
    )
    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.int8)
    views = reshape_dsa_kv_cache(raw, spec, num_blocks=2)
    assert views is not None
    assert len(views) == 1
    state_cache = views[0]
    assert state_cache.shape == (2, 32, 1, 512)
    assert state_cache.stride(0) == spec.page_size_bytes // 4

    state_cache[0].fill_(3)
    assert torch.count_nonzero(state_cache[1]) == 0


def test_compressed_indexer_cache_uses_logical_token_ratio():
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        compress_ratio=4,
        scale_dim=1,
        scale_dtype=torch.float16,
    )
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=False,
        hash_block_size=128,
    )
    manager = AscendCompressedAttentionManager(
        spec,
        pool,
        enable_caching=False,
        kv_cache_group_id=0,
    )
    assert manager.get_num_blocks_to_allocate(
        "request",
        num_tokens=512,
        new_computed_blocks=[],
        total_computed_tokens=0,
        num_tokens_main_model=512,
    ) == 1
    assert len(manager.allocate_new_blocks("request", 512, 512)) == 1

    install_dsa_cache_manager()
    assert spec_manager_map[AscendMLAAttentionSpec] is (
        AscendCompressedAttentionManager
    )

    import vllm.v1.core.single_type_kv_cache_manager as manager_module

    capped_manager = manager_module.get_manager_for_kv_cache_spec(
        spec,
        max_num_batched_tokens=4096,
        max_model_len=4096,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=1,
    )
    assert isinstance(capped_manager, AscendCompressedAttentionManager)
    assert capped_manager._max_admission_blocks_per_request == 9


def test_dsa_cache_groups_keep_compression_ratios_independent():
    specs = {
        "c4.attn": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=4,
            model_version="deepseek_v4",
        ),
        "c4.indexer": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            compress_ratio=4,
            model_version="deepseek_v4",
            scale_dim=1,
            scale_dtype=torch.float16,
        ),
        "c128.attn": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=128,
            model_version="deepseek_v4",
        ),
        "c4.indexer_state": AscendSlidingWindowMLASpec(
            block_size=8,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.float32,
            sliding_window=8,
            page_size_padded=16640,
        ),
        "c4.attn_state": AscendSlidingWindowMLASpec(
            block_size=8,
            num_kv_heads=1,
            head_size=2048,
            dtype=torch.float32,
            sliding_window=8,
            page_size_padded=131072,
        ),
        "c128.attn_state": AscendSlidingWindowMLASpec(
            block_size=32,
            num_kv_heads=1,
            head_size=1024,
            dtype=torch.float32,
            sliding_window=128,
            page_size_padded=131072,
        ),
    }

    grouped = _group_and_unify_dsa_kv_cache_specs(specs)
    assert grouped is not None
    groups = _get_dsa_kv_cache_groups(grouped)

    c4_ratios = {
        spec.compress_ratio
        for spec in groups[0].kv_cache_spec.kv_cache_specs.values()
    }
    c128_ratios = {
        spec.compress_ratio
        for spec in groups[1].kv_cache_spec.kv_cache_specs.values()
    }
    assert c4_ratios == {4}
    assert c128_ratios == {128}
    assert all(
        page_size in groups[0].kv_cache_spec.get_page_sizes()
        for group in groups[2:]
        for page_size in group.kv_cache_spec.get_page_sizes()
    )


def test_dsa_compressed_positions_follow_each_cache_group_ratio():
    specs = {
        "c4": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            compress_ratio=4,
            model_version="deepseek_v4",
        ),
        "c128": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=128,
            model_version="deepseek_v4",
        ),
        "state": AscendSlidingWindowMLASpec(
            block_size=128,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=8,
        ),
    }
    grouped = _group_and_unify_dsa_kv_cache_specs(specs)
    assert grouped is not None
    groups = _get_dsa_kv_cache_groups(grouped)

    positions, request_indices, scheduled_counts = (
        get_compressed_pos_and_indices(
            num_computed_tokens=np.array([3, 127], dtype=np.int32),
            num_scheduled_tokens=np.array([5, 3], dtype=np.int32),
            arrange_np=np.array([0, 1], dtype=np.int32),
            kv_cache_groups=groups,
        )
    )

    np.testing.assert_array_equal(positions[0], np.array([0, 1, 31]))
    np.testing.assert_array_equal(request_indices[0], np.array([0, 0, 1]))
    np.testing.assert_array_equal(scheduled_counts[0], np.array([2, 1]))
    np.testing.assert_array_equal(positions[1], np.array([0]))
    np.testing.assert_array_equal(request_indices[1], np.array([1]))
    np.testing.assert_array_equal(scheduled_counts[1], np.array([0, 1]))


def test_dsa_slot_mapping_uses_compressed_position_when_lengths_match():
    class _HostDeviceBuffer:
        def __init__(self, value: np.ndarray):
            self.np = value
            self.copied = 0

        def copy_to_gpu(self, count: int) -> None:
            self.copied = count

    class _BlockTable:
        block_size = 128

        def __init__(self):
            self.block_table = SimpleNamespace(
                np=np.array([[7, 11]], dtype=np.int32)
            )
            self.slot_mapping = _HostDeviceBuffer(
                np.zeros(1, dtype=np.int32)
            )

        def compute_slot_mapping(self, *args, **kwargs):
            raise AssertionError(
                "compressed DSA groups must not use full token positions"
            )

    block_table = _BlockTable()
    compute_dsa_slot_mappings(
        block_tables=SimpleNamespace(block_tables=[block_table]),
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        positions=torch.tensor([3], dtype=torch.int64),
        positions_by_group=[np.array([0], dtype=np.int64)],
        request_indices_by_group=[np.array([0], dtype=np.int64)],
    )

    assert block_table.slot_mapping.np[0] == 7 * 128
    assert block_table.slot_mapping.copied == 1
