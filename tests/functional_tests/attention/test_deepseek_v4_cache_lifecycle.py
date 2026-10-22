# SPDX-License-Identifier: Apache-2.0
"""Real scheduler allocations with installed NPU cache-layer components."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.fixture(scope="module")
def runtime():
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        CompilationMode,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.network_utils import get_open_port

    from vllm_fl import _patch_ascend_torch_accelerator

    torch.npu.set_device(0)
    _patch_ascend_torch_accelerator()
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, custom_ops=["all"]
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=2,
        ),
        cache_config=CacheConfig(block_size=128, enable_prefix_caching=False),
    )
    with set_current_vllm_config(config), torch.inference_mode():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="hccl",
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        initialize_model_parallel(1, backend="hccl")
        try:
            yield config
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _specs():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

    # Match the planner's page-size coverage contract even with toy head widths.
    # Padded pages also exercise the actual strided-view reshape/write path.
    common = dict(
        block_size=128,
        num_kv_heads=1,
        model_version="deepseek_v4",
        page_size_padded=8192,
    )
    return {
        "state4": SlidingWindowMLASpec(
            **common, head_size=16, dtype=torch.float32, sliding_window=8
        ),
        "state128": SlidingWindowMLASpec(
            **common, head_size=8, dtype=torch.float32, sliding_window=128
        ),
        "swa": SlidingWindowMLASpec(
            **common, head_size=4, dtype=torch.bfloat16, sliding_window=8
        ),
        "compact4": MLAAttentionSpec(
            **common, head_size=4, dtype=torch.bfloat16, compress_ratio=4
        ),
        "compact128": MLAAttentionSpec(
            **common, head_size=4, dtype=torch.bfloat16, compress_ratio=128
        ),
        "key4": MLAAttentionSpec(
            **common, head_size=4, dtype=torch.int8, compress_ratio=4
        ),
        "scale4": MLAAttentionSpec(
            **common, head_size=1, dtype=torch.float16, compress_ratio=4
        ),
    }


@pytest.fixture
def pools(runtime):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_groups,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache

    from vllm_fl.kv_cache.deepseek_v4 import DeepseekV4CacheLayer
    from vllm_fl.worker.kv_cache_utils import bind_kv_cache
    from vllm_fl.worker.model_runner import ModelRunnerFL

    specs = {f"model.layers.0.{key}": spec for key, spec in _specs().items()}
    groups = get_kv_cache_groups(runtime, specs)
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(size=64 * spec.page_size_bytes, shared_by=[name])
            for name, spec in specs.items()
        ],
    )
    owner = SimpleNamespace(
        device=torch.device("npu:0"),
        runner_only_attn_layers=set(),
        use_ascend_hybrid_cache=False,
    )
    raw = ModelRunnerFL._allocate_kv_cache_tensors(owner, config)
    views = {
        name: _reshape_attention_kv_cache(
            raw[name],
            spec,
            (64, spec.storage_block_size, spec.head_size),
            (0, 1, 2),
            64,
            None,
        )
        for name, spec in specs.items()
    }
    manager = KVCacheManager(
        generate_scheduler_kv_cache_config([config]),
        max_model_len=512,
        scheduler_block_size=128,
        hash_block_size=128,
        max_num_batched_tokens=512,
        enable_caching=False,
    )
    leaves = {
        name: DeepseekV4CacheLayer(name, spec, vllm_config=runtime)
        for name, spec in specs.items()
    }
    runner_caches = []
    bind_kv_cache(views, leaves, runner_caches)
    initial = manager.block_pool.get_num_free_blocks()
    env = SimpleNamespace(
        config=runtime,
        specs=specs,
        groups=groups,
        views=views,
        raw=raw,
        leaves=leaves,
        runner_caches=runner_caches,
        manager=manager,
        requests=[],
        initial=initial,
    )
    try:
        yield env
    finally:
        for request in env.requests:
            manager.free(request)
        assert manager.block_pool.get_num_free_blocks() == initial
        context = runtime.compilation_config.static_forward_context
        for name in leaves:
            context.pop(name, None)


def _request(env, name="request-0"):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    request = Request(name, list(range(256)), SamplingParams(max_tokens=8), None)
    env.requests.append(request)
    return request


def _allocate(env, request, count):
    assert env.manager.allocate_slots(request, num_new_tokens=count) is not None


def _common(env, request, leaf_name, count, *, padded=None):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    name = "model.layers.0." + leaf_name
    index = next(i for i, group in enumerate(env.groups) if name in group.layer_names)
    physical = env.manager.get_blocks(request.request_id).get_block_ids()[index]
    # IDs come exclusively from the real scheduler manager. Zero is its null page.
    pages = list(physical) + [0] * max(0, 4 - len(physical))
    end = request.num_computed_tokens + count
    qpad = count if padded is None else padded
    slots = []
    for position in range(request.num_computed_tokens, end):
        page = pages[position // 128]
        slots.append(page * 128 + position % 128 if page else -1)
    slots.extend([-1] * (qpad - count))
    starts = torch.tensor([0, count], dtype=torch.int32)
    seq = torch.tensor([end], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=starts.to("npu"),
        query_start_loc_cpu=starts,
        seq_lens=seq.to("npu"),
        num_reqs=1,
        num_actual_tokens=qpad,
        max_query_len=count,
        max_seq_len=end,
        block_table_tensor=torch.tensor([pages], device="npu", dtype=torch.int32),
        slot_mapping=torch.tensor(slots, device="npu", dtype=torch.int64),
        _seq_lens_cpu=seq,
    ), pages


def _builder(env, leaf_name):
    from vllm_fl.attention.deepseek_v4 import DeepseekV4MetadataBuilder

    name = "model.layers.0." + leaf_name
    return DeepseekV4MetadataBuilder(
        env.specs[name], [name], env.config, torch.device("npu:0")
    )


def test_scheduler_groups_typed_views_and_binding(pools):
    key_group = next(
        group for group in pools.groups if "model.layers.0.key4" in group.layer_names
    )
    assert "model.layers.0.scale4" in key_group.layer_names
    assert len(pools.runner_caches) == len(pools.leaves)
    for name, leaf in pools.leaves.items():
        spec = pools.specs[name]
        assert leaf.kv_cache is pools.views[name]
        assert leaf.get_kv_cache_spec(pools.config) is spec
        assert pools.views[name].shape == (
            64,
            128 // spec.compress_ratio,
            spec.head_size,
        )
        assert pools.views[name].dtype == spec.dtype
        assert (
            pools.views[name].untyped_storage().data_ptr()
            == pools.raw[name].untyped_storage().data_ptr()
        )
        assert any(view is pools.views[name] for view in pools.runner_caches)


@pytest.mark.parametrize("ratio", [4, 128])
def test_scheduler_completed_row_arithmetic_and_private_metadata(pools, ratio):
    request = _request(pools)
    builder = _builder(pools, f"compact{ratio}")
    for count in (ratio - 1, 1, ratio):
        _allocate(pools, request, count)
        common, pages = _common(pools, request, f"compact{ratio}", count)
        public_table, public_slots = (
            common.block_table_tensor.clone(),
            common.slot_mapping.clone(),
        )
        meta = builder.build(0, common)
        start, end = request.num_computed_tokens, request.num_computed_tokens + count
        complete = list(range(start // ratio, end // ratio))
        assert meta.num_completed == len(complete)
        expected_ends = [(index + 1) * ratio - 1 for index in complete]
        expected_slots = [
            pages[(index * ratio) // 128] * (128 // ratio) + index % (128 // ratio)
            for index in complete
        ]
        torch.testing.assert_close(
            meta.output_positions[: len(complete)],
            torch.tensor(
                expected_ends, device="npu", dtype=meta.output_positions.dtype
            ),
        )
        torch.testing.assert_close(
            meta.output_slots[: len(complete)],
            torch.tensor(expected_slots, device="npu", dtype=meta.output_slots.dtype),
        )
        assert torch.all(meta.output_positions[len(complete) :] == -1)
        assert torch.all(meta.output_slots[len(complete) :] == -1)
        torch.testing.assert_close(
            meta.key_lengths[0],
            torch.tensor(end // ratio, device="npu"),
            check_dtype=False,
        )
        torch.testing.assert_close(common.block_table_tensor, public_table)
        torch.testing.assert_close(common.slot_mapping, public_slots)
        expected_table = torch.tensor(
            [page if page else -1 for page in pages],
            device="npu",
            dtype=meta.block_table.dtype,
        )
        torch.testing.assert_close(meta.block_table[0, : len(pages)], expected_table)
        assert meta.block_table.data_ptr() != common.block_table_tensor.data_ptr()
        request.num_computed_tokens = end


def test_two_scheduler_requests_keep_completed_slots_isolated(pools):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    first, second = _request(pools, "first"), _request(pools, "second")
    _allocate(pools, first, 4)
    _allocate(pools, second, 8)
    ca, pa = _common(pools, first, "compact4", 4)
    cb, pb = _common(pools, second, "compact4", 8)
    starts = torch.tensor([0, 4, 12], dtype=torch.int32)
    lengths = torch.tensor([4, 8], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts.to("npu"),
        query_start_loc_cpu=starts,
        seq_lens=lengths.to("npu"),
        num_reqs=2,
        num_actual_tokens=12,
        max_query_len=8,
        max_seq_len=8,
        block_table_tensor=torch.cat([ca.block_table_tensor, cb.block_table_tensor]),
        slot_mapping=torch.cat([ca.slot_mapping, cb.slot_mapping]),
        _seq_lens_cpu=lengths,
    )
    metadata = _builder(pools, "compact4").build(0, common)
    assert metadata.num_completed == 3
    torch.testing.assert_close(
        metadata.output_request_ids[:3],
        torch.tensor([0, 1, 1], device="npu", dtype=metadata.output_request_ids.dtype),
    )
    torch.testing.assert_close(
        metadata.output_positions[:3],
        torch.tensor([3, 3, 7], device="npu", dtype=metadata.output_positions.dtype),
    )
    torch.testing.assert_close(
        metadata.output_slots[:3],
        torch.tensor(
            [pa[0] * 32, pb[0] * 32, pb[0] * 32 + 1],
            device="npu",
            dtype=metadata.output_slots.dtype,
        ),
    )
    assert pa[0] != pb[0]


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int8]
)
def test_cache_write_typed_padding_and_null_page(pools, dtype):
    from vllm_fl.dispatch import call_op, get_default_manager

    assert (
        get_default_manager().get_selected_impl_id("deepseek_v4_cache_write")
        == "reference.torch"
    )
    cache = torch.zeros(3, 4, 4, device="npu", dtype=dtype)
    values = torch.arange(12, device="npu").reshape(3, 4).to(dtype)
    if dtype.is_floating_point:
        values[-1].fill_(float("nan"))
    call_op(
        "deepseek_v4_cache_write", values, torch.tensor([4, 9, -1], device="npu"), cache
    )
    torch.testing.assert_close(cache[1, 0], values[0])
    torch.testing.assert_close(cache[2, 1], values[1])
    assert torch.count_nonzero(cache[0]) == 0


@pytest.mark.parametrize("missing", ["seq_lens", "query_start"])
def test_missing_exact_cpu_metadata_rejected(pools, missing):
    request = _request(pools)
    _allocate(pools, request, 3)
    common, _ = _common(pools, request, "compact4", 3)
    if missing == "seq_lens":
        common._seq_lens_cpu = None
    else:
        common.query_start_loc_cpu = None
    with pytest.raises((ValueError, RuntimeError), match="CPU|cpu"):
        _builder(pools, "compact4").build(0, common)


def test_duplicate_prefix_and_invalid_spec_rejected(pools):
    from vllm_fl.kv_cache.deepseek_v4 import DeepseekV4CacheLayer

    name = "model.layers.0.compact4"
    with pytest.raises((ValueError, RuntimeError)):
        DeepseekV4CacheLayer(name, pools.specs[name], vllm_config=pools.config)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        DeepseekV4CacheLayer(
            "model.layers.0.invalid", object(), vllm_config=pools.config
        )


def _compressor(ratio):
    from vllm_fl.models.deepseek_v4 import DeepseekV4Compressor

    module = DeepseekV4Compressor(8, 4, ratio).to("npu")
    for name, param in module.named_parameters():
        values = (
            torch.arange(param.numel(), device="npu").reshape(param.shape) % 23 - 11
        ).float() * 0.017
        param.copy_(
            values.to(param.dtype) if name != "norm.weight" else values.abs() + 0.7
        )
    return module


def _compression_oracle(module, history, end, ratio, rope):
    coff = 2 if ratio == 4 else 1
    projected = F.linear(history.float(), module.wkv.weight.float())
    gates = F.linear(history.float(), module.wgate.weight.float())
    values, logits = [], []
    for time in range(max(0, end - ratio * coff + 1), end + 1):
        half = int(coff == 2 and time >= end - ratio + 1)
        values.append(projected[time, half * 4 : (half + 1) * 4])
        logits.append(
            (gates[time] + module.ape[time % ratio])[half * 4 : (half + 1) * 4]
        )
    result = (torch.stack(values) * torch.stack(logits).softmax(0)).sum(0)
    result = (
        result * (result.square().mean() + 1e-6).rsqrt() * module.norm.weight.float()
    )
    c, s = rope[(end // ratio) * ratio]
    return torch.stack(
        [
            result[0],
            result[1],
            result[2] * c - result[3] * s,
            result[3] * c + result[2] * s,
        ]
    ).to(history.dtype)


@pytest.mark.parametrize("ratio", [4, 128])
def test_scheduler_compressor_chunk_decode_and_store(pools, ratio):
    request = _request(pools)
    module = _compressor(ratio)
    state_leaf = pools.leaves[f"model.layers.0.state{ratio}"]
    compact_leaf = pools.leaves[f"model.layers.0.compact{ratio}"]
    state_builder, compact_builder = (
        _builder(pools, f"state{ratio}"),
        _builder(pools, f"compact{ratio}"),
    )
    history = (
        (torch.arange(2 * ratio * 8, device="npu").reshape(2 * ratio, 8) % 29 - 14)
        * 0.03
    ).bfloat16()
    angle = torch.arange(2 * ratio, device="npu").float() * 0.013
    rope = torch.stack([angle.cos(), angle.sin()], -1)
    for count in (ratio - 1, 1, ratio):
        _allocate(pools, request, count)
        state_common, _ = _common(pools, request, f"state{ratio}", count)
        compact_common, _ = _common(pools, request, f"compact{ratio}", count)
        smeta, cmeta = (
            state_builder.build(0, state_common),
            compact_builder.build(0, compact_common),
        )
        start, end = request.num_computed_tokens, request.num_computed_tokens + count
        output = module.forward_from_cache(
            history[start:end], state_leaf, compact_leaf, smeta, cmeta, rope
        )
        completed = [(j + 1) * ratio - 1 for j in range(start // ratio, end // ratio)]
        for row, boundary in enumerate(completed):
            expected = _compression_oracle(module, history, boundary, ratio, rope)
            torch.testing.assert_close(output[row], expected, rtol=0.01, atol=0.01)
            slot = int(cmeta.output_slots[row].item())
            cache = compact_leaf.kv_cache
            torch.testing.assert_close(
                cache[slot // cache.shape[1], slot % cache.shape[1]],
                expected,
                rtol=0.01,
                atol=0.01,
            )
        assert torch.count_nonzero(state_leaf.kv_cache[0]) == 0
        assert torch.count_nonzero(compact_leaf.kv_cache[0]) == 0
        request.num_computed_tokens = end


def test_stable_plans_rebuild_and_changed_input_graph(pools):
    request = _request(pools)
    module = _compressor(4)
    state_leaf, compact_leaf = (
        pools.leaves["model.layers.0." + name] for name in ("state4", "compact4")
    )
    sb, cb = _builder(pools, "state4"), _builder(pools, "compact4")
    _allocate(pools, request, 3)
    scommon, _ = _common(pools, request, "state4", 3)
    ccommon, _ = _common(pools, request, "compact4", 3)
    rope = torch.tensor([[1.0, 0.0]] * 8, device="npu")
    x0 = torch.full((3, 8), 0.1, device="npu", dtype=torch.bfloat16)
    module.forward_from_cache(
        x0, state_leaf, compact_leaf, sb.build(0, scommon), cb.build(0, ccommon), rope
    )
    request.num_computed_tokens = 3
    _allocate(pools, request, 1)
    scommon, _ = _common(pools, request, "state4", 1, padded=4)
    ccommon, _ = _common(pools, request, "compact4", 1, padded=4)
    smeta, cmeta = sb.build(0, scommon), cb.build(0, ccommon)
    pointers = (
        smeta.positions.data_ptr(),
        smeta.slot_mapping.data_ptr(),
        cmeta.output_positions.data_ptr(),
        cmeta.output_slots.data_ptr(),
        smeta.block_table.data_ptr(),
        cmeta.block_table.data_ptr(),
    )
    x = torch.full((4, 8), 0.2, device="npu", dtype=torch.bfloat16)

    def run():
        return module.forward_from_cache(
            x, state_leaf, compact_leaf, smeta, cmeta, rope
        )

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    request.num_computed_tokens = 4
    _allocate(pools, request, 4)
    scommon, _ = _common(pools, request, "state4", 4, padded=4)
    ccommon, _ = _common(pools, request, "compact4", 4, padded=4)
    updated_s, updated_c = sb.build(0, scommon), cb.build(0, ccommon)
    assert pointers == (
        updated_s.positions.data_ptr(),
        updated_s.slot_mapping.data_ptr(),
        updated_c.output_positions.data_ptr(),
        updated_c.output_slots.data_ptr(),
        updated_s.block_table.data_ptr(),
        updated_c.block_table.data_ptr(),
    )
    assert int(updated_c.output_positions[0].item()) == 7
    for step in range(2):
        x.copy_(
            (
                (torch.arange(32, device="npu").reshape(4, 8) % 13 - 6) * 0.03
                + step * 0.05
            ).bfloat16()
        )
        expected = run().clone()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
        assert torch.count_nonzero(compact_leaf.kv_cache[0]) == 0
