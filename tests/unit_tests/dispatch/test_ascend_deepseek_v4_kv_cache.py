"""CPU contracts for Ascend DeepSeek-V4's non-packed cache closure."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_ascend_mla_spec_keeps_physical_page_width(block_size: int,
                                                    compress_ratio: int) -> None:
    """Ascend allocates ``block_size`` compressed slots per physical page."""
    try:
        import torch

        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        from vllm_fl.kv_cache.ascend.kv_cache_interface import (
            AscendMLAAttentionSpec,
        )
    except ModuleNotFoundError as error:
        pytest.skip(f"requires vLLM test dependencies: {error}")

    spec = AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
    )
    upstream_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
    )

    assert spec.storage_block_size == block_size
    assert spec.page_size_bytes == block_size * 16 * torch.empty(
        (), dtype=torch.bfloat16).element_size()
    assert upstream_spec.storage_block_size == block_size // compress_ratio

    # This is the same blocks-first shape returned by AscendDSABackend; ensure
    # one planned page can be viewed at the physical, not double-compressed,
    # width without exceeding its byte budget.
    cache = torch.empty(spec.page_size_bytes, dtype=torch.uint8).view(spec.dtype)
    cache = cache.view(1, spec.storage_block_size, spec.num_kv_heads,
                       spec.head_size)
    assert cache.shape == (1, block_size, 1, 16)
    assert cache.numel() * cache.element_size() == spec.page_size_bytes


def test_cache_closure_is_vendor_scoped_and_non_packed() -> None:
    source = (ROOT / "vllm_fl/kv_cache/ascend/deepseek_v4_kv_cache.py").read_text()
    assert "_get_kv_cache_config_packed = _get_kv_cache_config_packed_for_ascend" in source
    planner = source.split("def _get_kv_cache_config_non_packed", 1)[1].split("class AscendHybrid", 1)[0]
    assert "block_stride=" not in planner
    assert "if _is_deepseek_v4_config(config):" in source
    assert "return original_factory(*args, **kwargs)" in source
    assert "def _get_kv_cache_config_packed_for_ascend" in source
    assert "return _ORIGINAL_PACKED_KV_CACHE_CONFIG(" in source
    assert "isinstance(kv_cache_spec, AscendMLAAttentionSpec)" in source
    assert 'kwargs["max_admission_blocks_per_request"] = cdiv(' in source
    assert "SlidingWindowSpec, ChunkedLocalAttentionSpec" in source
    assert "def cache_blocks(self, request: Request, num_computed_tokens: int)" in source
    assert "num_computed_tokens // self.scheduler_block_size" in source
    assert "aligned + manager.block_size" in source


@pytest.fixture
def kv_module():
    try:
        from vllm_fl.kv_cache.ascend import deepseek_v4_kv_cache
    except ModuleNotFoundError as error:
        pytest.skip(f"requires vLLM test dependencies: {error}")
    return deepseek_v4_kv_cache


def test_compressed_manager_scales_allocation_cache_and_hit(kv_module) -> None:
    spec = SimpleNamespace(block_size=32, compress_ratio=4)
    pool = MagicMock()
    pool.null_block = MagicMock()
    manager = object.__new__(kv_module.CompressAttentionManager)
    manager.kv_cache_spec = spec
    manager.block_size = 32
    manager.compress_ratio = 4
    manager.block_pool = pool
    manager.kv_cache_group_id = 3
    manager.num_cached_block = {}
    manager.req_to_blocks = {"request": [MagicMock(), MagicMock()]}
    manager.get_num_skipped_tokens = lambda _: 0

    assert manager.get_num_blocks_to_allocate("new", 256, [], 0, 256) == 2

    # Only complete logical blocks are cached: 257 // (4 * 32) == 2.
    manager.cache_blocks(SimpleNamespace(request_id="request"), 257)
    assert pool.cache_full_blocks.call_args.kwargs["num_full_blocks"] == 2
    assert pool.cache_full_blocks.call_args.kwargs["block_size"] == 128

    cached = (MagicMock(),)
    pool.get_cached_block.side_effect = [cached, cached, None]
    with patch.object(kv_module, "BlockHashListWithBlockSize", return_value=[1, 2, 3]):
        blocks = kv_module.CompressAttentionManager.find_longest_cache_hit(
            [1, 2, 3], 256, [3], pool, spec, alignment_tokens=128)
    assert len(blocks[0]) == 2


def test_non_packed_plan_has_expected_shared_layers_and_no_stride(kv_module) -> None:
    try:
        import torch

        from vllm.v1.kv_cache_interface import (
            KVCacheGroupSpec,
            UniformTypeKVCacheSpecs,
        )

        from vllm_fl.kv_cache.ascend.kv_cache_interface import (
            AscendMLAAttentionSpec,
        )
    except ModuleNotFoundError as error:
        pytest.skip(f"requires vLLM test dependencies: {error}")
    specs = {
        name: AscendMLAAttentionSpec(block_size=32, num_kv_heads=1, head_size=16,
                                     dtype=torch.bfloat16, compress_ratio=ratio,
                                     model_version="deepseek_v4")
        for name, ratio in (("layers.0", 4), ("layers.1", 4), ("layers.2", 128))
    }
    groups = [KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs.from_specs(specs))]
    config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None))
    blocks, tensors = kv_module._get_kv_cache_config_non_packed(config, groups, 32 * 32 * 2 * 3)
    assert blocks == 2
    assert [tensor.shared_by for tensor in tensors] == [["layers.0"], ["layers.1"], ["layers.2"]]
    assert all(tensor.block_stride == 0 and tensor.offset == 0 for tensor in tensors)


def test_only_deepseek_v4_selects_the_ascend_coordinator(kv_module) -> None:
    spec = SimpleNamespace(model_version="deepseek_v4")
    deepseek = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    other = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(model_version=None))])
    assert kv_module._is_deepseek_v4_config(deepseek)
    assert not kv_module._is_deepseek_v4_config(other)


def test_hybrid_hash_divisibility_only_applies_when_caching(kv_module) -> None:
    """Keep the rc1 hash invariant on active caching paths only."""
    invalid_groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=8), is_eagle_group=False),
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=32), is_eagle_group=False),
    ]
    valid_groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128), is_eagle_group=False),
    ]

    def make_config(groups):
        return SimpleNamespace(num_blocks=1, kv_cache_groups=groups)

    def set_lcm_block_size(coordinator):
        coordinator.lcm_block_size = 128

    with (
        patch.object(kv_module, "BlockPool"),
        patch.object(kv_module, "get_manager_for_kv_cache_spec", return_value=MagicMock()),
        patch.object(kv_module.AscendHybridKVCacheCoordinator,
                     "verify_and_split_kv_cache_groups",
                     new=set_lcm_block_size),
    ):
        # Recorded geometry: physical C4/C128 state blocks 8/32 and an
        # inactive hash path use the scheduler's 128-token placeholder.
        coordinator = kv_module.AscendHybridKVCacheCoordinator(
            make_config(invalid_groups), 1024, 64, False, False, False, 1, 1, 128, 128
        )
        assert coordinator.hash_block_size == 128

        with pytest.raises(AssertionError):
            kv_module.AscendHybridKVCacheCoordinator(
                make_config(invalid_groups), 1024, 64, False, True, False, 1, 1, 128, 128
            )

        coordinator = kv_module.AscendHybridKVCacheCoordinator(
            make_config(valid_groups), 1024, 64, False, True, False, 1, 1, 128, 128
        )
        assert coordinator.hash_block_size == 128


def test_hybrid_hash_guard_remains_lexically_scoped_to_caching() -> None:
    """AST regression: don't accidentally reintroduce an unconditional check."""
    import ast

    source = (ROOT / "vllm_fl/kv_cache/ascend/deepseek_v4_kv_cache.py").read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "AscendHybridKVCacheCoordinator")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == "__init__")
    guard = next(node for node in init.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Name) and node.test.id == "enable_caching")
    assert any(isinstance(node, ast.Assert) for node in ast.walk(guard))


def test_real_coordinator_keeps_incoming_write_floor_and_swa_lcm(kv_module, monkeypatch) -> None:
    """rc1 writes an 8K chunk while SWA reads/masks at the C128 LCM."""
    try:
        import torch

        from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
        from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

        from vllm_fl.kv_cache.ascend.kv_cache_interface import (
            AscendMLAAttentionSpec,
            AscendSlidingWindowMLASpec,
            register_ascend_kv_cache_specs,
        )
    except ModuleNotFoundError as error:
        pytest.skip(f"requires vLLM test dependencies: {error}")

    register_ascend_kv_cache_specs()
    specs = [
        AscendMLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=512,
            dtype=torch.bfloat16, compress_ratio=ratio,
            model_version="deepseek_v4",
        )
        for ratio in (4, 128)
    ]
    specs.extend(
        AscendSlidingWindowMLASpec(
            block_size=block, num_kv_heads=1, head_size=head, dtype=dtype,
            sliding_window=window, model_version="deepseek_v4",
        )
        for block, window, head, dtype in (
            (128, 128, 512, torch.bfloat16),
            (8, 8, 1024, torch.float32),
            (32, 128, 1024, torch.float32),
        )
    )
    config = KVCacheConfig(
        num_blocks=20000,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"group{i}"], spec)
                         for i, spec in enumerate(specs)],
    )
    vconfig = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128, enable_prefix_caching=True,
                                     hash_block_size=None),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1,
                                        prefill_context_parallel_size=1),
        kv_transfer_config=None,
    )
    scheduler, hash_size = kv_module.resolve_kv_cache_block_sizes(config, vconfig)
    assert (scheduler, hash_size) == (128, 8)
    coordinator = kv_module.AscendHybridKVCacheCoordinator(
        config, 133120, 8192, False, True, False, 1, 1, scheduler, hash_size,
    )

    assert coordinator.scheduler_block_size == 128
    assert coordinator.lcm_block_size == 16384
    assert all(manager.scheduler_block_size == 16384
               for manager in coordinator.single_type_managers
               if isinstance(manager, SlidingWindowManager))

    calls = []
    for manager in coordinator.single_type_managers:
        cache = MagicMock()
        monkeypatch.setattr(manager, "cache_blocks", cache)
        calls.append(cache)
    coordinator.cache_blocks(SimpleNamespace(request_id="floor-probe",
                                              num_prompt_tokens=8192), 8192)
    assert [call.call_args.args[1] for call in calls] == [8192] * len(calls)
    assert all(call.call_args.kwargs["retention_interval"]
               == coordinator.retention_interval for call in calls)


def test_non_deepseek_planner_uses_upstream_packed_fallback(kv_module) -> None:
    config = SimpleNamespace()
    groups = [SimpleNamespace(kv_cache_spec=SimpleNamespace(model_version=None))]
    original = MagicMock(return_value=(17, ["upstream"]))
    with patch.object(kv_module, "_ORIGINAL_PACKED_KV_CACHE_CONFIG", original):
        result = kv_module._get_kv_cache_config_packed_for_ascend(
            config, groups, 4096
        )
    assert result == (17, ["upstream"])
    original.assert_called_once_with(config, groups, 4096)


def test_manager_factory_sets_rc1_admission_caps(kv_module) -> None:
    try:
        import torch

        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        from vllm_fl.kv_cache.ascend.kv_cache_interface import (
            AscendMLAAttentionSpec,
        )
    except ModuleNotFoundError as error:
        pytest.skip(f"requires vLLM test dependencies: {error}")

    captured = {}

    class Manager:
        def __init__(self, spec, **kwargs):
            captured[type(spec).__name__] = kwargs

    mla = AscendMLAAttentionSpec(
        block_size=32, num_kv_heads=1, head_size=16, dtype=torch.bfloat16,
        compress_ratio=4, model_version="deepseek_v4",
    )
    sliding = SlidingWindowSpec(
        block_size=32, num_kv_heads=1, head_size=16, dtype=torch.bfloat16,
        sliding_window=128,
    )
    with (
        patch.object(kv_module.KVCacheSpecRegistry, "get_manager_class", return_value=Manager),
        patch.object(kv_module, "CompressAttentionManager", Manager),
    ):
        kv_module.get_manager_for_kv_cache_spec(mla, max_model_len=1024)
        kv_module.get_manager_for_kv_cache_spec(
            sliding, max_model_len=1024, max_num_batched_tokens=64,
        )
    assert captured[type(mla).__name__]["max_admission_blocks_per_request"] == 9
    assert "max_model_len" not in captured[type(mla).__name__]
    assert "max_num_batched_tokens" not in captured[type(mla).__name__]
    assert captured[type(sliding).__name__]["max_admission_blocks_per_request"] == (
        sliding.max_admission_blocks_per_request(
            max_num_batched_tokens=64, max_model_len=1024
        )
    )
