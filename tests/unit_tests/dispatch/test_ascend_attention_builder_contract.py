# Copyright (c) 2026 BAAI. All rights reserved.

import inspect
from types import SimpleNamespace

import torch
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes

from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
    AscendAttentionBackend,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)


def test_ascend_builder_matches_vllm_024_base_contract() -> None:
    assert issubclass(AscendAttentionMetadataBuilder, AttentionMetadataBuilder)
    assert (
        AttentionMetadataBuilder[AscendMetadata]
        in AscendAttentionMetadataBuilder.__orig_bases__
    )
    assert AscendAttentionMetadataBuilder.supports_update_block_table is False
    assert (
        AscendAttentionMetadataBuilder.get_cudagraph_support(None, None)
        is AttentionCGSupport.ALWAYS
    )

    build_signature = inspect.signature(AscendAttentionMetadataBuilder.build)
    assert build_signature.parameters["fast_build"].default is False
    assert "model" not in build_signature.parameters


def test_ascend_builder_initializes_inherited_state() -> None:
    kv_cache_spec = object()
    layer_names = ["model.layers.0.self_attn"]
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=256,
            runner_type="generate",
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    builder = AscendAttentionMetadataBuilder(
        kv_cache_spec,
        layer_names,
        vllm_config,
        torch.device("cpu"),
    )

    assert builder.kv_cache_spec is kv_cache_spec
    assert builder.layer_names is layer_names
    assert builder.vllm_config is vllm_config
    assert builder.device == torch.device("cpu")


def test_decode_metadata_keeps_real_splitfuse_mask_for_default_fia() -> None:
    builder = object.__new__(AscendAttentionMetadataBuilder)
    builder.model_config = SimpleNamespace(runner_type="generate", use_mla=False)
    builder.device = torch.device("cpu")

    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
        AscendAttentionState,
    )

    mask = builder._make_attention_mask(AscendAttentionState.DecodeOnly)
    assert mask is not None
    assert mask.shape == (2048, 2048)
    assert mask.dtype is torch.int8


def test_builder_prefers_live_private_lengths_and_pads_fia_dummy_request(
    monkeypatch,
) -> None:
    builder = object.__new__(AscendAttentionMetadataBuilder)
    builder.model_config = SimpleNamespace(runner_type="generate", use_mla=False)
    builder.device = torch.device("cpu")
    builder.decode_threshold = 1
    monkeypatch.setattr(builder, "_make_attention_mask", lambda state: object())
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=2,
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_query_len=1,
        block_table_tensor=torch.tensor([[7]], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([9], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([99], dtype=torch.int32),
        seq_lens=torch.tensor([999], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 4], dtype=torch.int64),
        causal=True,
    )

    metadata = builder.build(0, common)

    assert metadata.seq_lens.tolist() == [9, 1]
    assert metadata.seq_lens_list == [9, 1]
    assert metadata.block_tables.tolist() == [[7], [0]]
    assert metadata.actual_seq_lengths_q == [1, 2]


def _kernel_block_size_for(backend: type[AttentionBackend]) -> int:
    spec = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
    )
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=spec)
        ],
    )
    groups = [[AttentionGroup(backend, ["layer"], spec, 0)]]
    return prepare_kernel_block_sizes(config, groups)[0]


def test_ascend_backend_splits_1536_logical_blocks_for_fia() -> None:
    assert AscendAttentionBackend.get_supported_kernel_block_sizes() == [128]
    assert _kernel_block_size_for(AscendAttentionBackend) == 128

    table = BlockTable(
        block_size=1536,
        kernel_block_size=128,
        max_num_reqs=1,
        max_num_blocks_per_req=1,
        max_num_batched_tokens=1,
        pin_memory=False,
        device=torch.device("cpu"),
        cp_kv_cache_interleave_size=1,
    )
    table.add_row([3], 0)

    assert table.blocks_per_kv_block == 12
    assert table.block_size == 128
    assert table.get_numpy_array()[0, :12].tolist() == list(range(36, 48))


def test_default_non_ascend_backend_keeps_1536_kernel_block() -> None:
    assert _kernel_block_size_for(AttentionBackend) == 1536
