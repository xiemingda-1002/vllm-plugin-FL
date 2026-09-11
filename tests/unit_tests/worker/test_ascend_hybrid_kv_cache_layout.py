from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

if not hasattr(torch, "uint8"):
    pytest.skip("requires a complete PyTorch installation", allow_module_level=True)


def _hybrid_config():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MambaSpec,
    )

    attention_layer = "model.layers.0.self_attn"
    mamba_layer = "model.layers.1.linear_attn"
    block_size = 4096
    pool_page_bytes = 2_121_728
    attention_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        page_size_padded=pool_page_bytes,
    )
    mamba_spec = MambaSpec(
        block_size=block_size,
        shapes=((12_288,), (262_144,)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=pool_page_bytes,
    )
    num_blocks = 10
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * pool_page_bytes,
                shared_by=[attention_layer, mamba_layer],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[attention_layer],
                kv_cache_spec=attention_spec,
            ),
            KVCacheGroupSpec(
                layer_names=[mamba_layer],
                kv_cache_spec=mamba_spec,
            ),
        ],
    )
    return config, attention_layer, mamba_layer, attention_spec, mamba_spec


def _runner(config, attention_layer, mamba_layer, attention_spec, mamba_spec):
    from vllm_fl.worker.model_runner import ModelRunnerFL

    class FakeAscendBackend:
        @staticmethod
        def get_kv_cache_block_dim(*args, **kwargs):
            del args, kwargs
            return 0

        @staticmethod
        def get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str="auto",
        ):
            del cache_dtype_str
            return (2, num_blocks, block_size, num_kv_heads, head_size)

    runner = object.__new__(ModelRunnerFL)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.kv_cache_config = config
    runner.attn_groups = [
        [
            SimpleNamespace(
                kv_cache_spec=attention_spec,
                backend=FakeAscendBackend,
                kv_cache_group_id=0,
                layer_names=[attention_layer],
            )
        ],
        [
            SimpleNamespace(
                kv_cache_spec=mamba_spec,
                backend=object(),
                kv_cache_group_id=1,
                layer_names=[mamba_layer],
            )
        ],
    ]
    return runner


def test_ascend_hybrid_cache_uses_dense_shared_state_layout():
    import vllm_fl.worker.model_runner as model_runner

    config, attention_layer, mamba_layer, attention_spec, mamba_spec = (
        _hybrid_config()
    )
    runner = _runner(
        config,
        attention_layer,
        mamba_layer,
        attention_spec,
        mamba_spec,
    )
    raw_by_layer = runner._allocate_kv_cache_tensors(config)

    raw = raw_by_layer[attention_layer]
    assert raw_by_layer[mamba_layer] is raw
    assert runner.hybrid_with_attn_and_mamba

    with patch.object(
        model_runner,
        "current_platform",
        SimpleNamespace(device_type="npu"),
    ):
        caches = runner._reshape_kv_cache_tensors(
            raw_by_layer,
            kernel_block_sizes=[128, 4096],
        )

    attention_cache = caches[attention_layer]
    conv_state, ssm_state = caches[mamba_layer]
    conv_bytes = 10 * 24_576
    state_or_k_bytes = 10 * 1_048_576

    assert attention_cache.shape == (2, 320, 128, 1, 128)
    assert attention_cache.is_contiguous()
    assert conv_state.shape == (10, 12_288)
    assert ssm_state.shape == (10, 262_144)
    assert conv_state.is_contiguous()
    assert ssm_state.is_contiguous()

    raw_ptr = raw.data_ptr()
    assert conv_state.data_ptr() == raw_ptr
    assert ssm_state.data_ptr() == raw_ptr + conv_bytes
    assert attention_cache[0].data_ptr() == raw_ptr + conv_bytes
    assert attention_cache[1].data_ptr() == (
        raw_ptr + conv_bytes + state_or_k_bytes
    )
    assert attention_cache[1].data_ptr() + state_or_k_bytes == (
        raw_ptr + raw.nbytes
    )


def test_non_ascend_mamba_cache_keeps_page_interleaved_stride():
    import vllm_fl.worker.model_runner as model_runner
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor

    config, attention_layer, mamba_layer, attention_spec, mamba_spec = (
        _hybrid_config()
    )
    # A padded attention page with a kv-first backend shape is intentionally
    # rejected by upstream vLLM.  This non-Ascend contract instead uses the
    # valid Mamba-only plan that reaches the page-interleaved state views.
    config = KVCacheConfig(
        num_blocks=config.num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=config.num_blocks * mamba_spec.page_size_bytes,
                shared_by=[mamba_layer],
            )
        ],
        kv_cache_groups=[config.kv_cache_groups[1]],
    )
    runner = _runner(
        config,
        attention_layer,
        mamba_layer,
        attention_spec,
        mamba_spec,
    )
    runner.attn_groups = [runner.attn_groups[1]]
    runner.attn_groups[0][0].kv_cache_group_id = 0
    raw_by_layer = runner._allocate_kv_cache_tensors(config)

    with patch.object(
        model_runner,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    ):
        caches = runner._reshape_kv_cache_tensors(
            raw_by_layer,
            kernel_block_sizes=[4096],
        )

    conv_state, ssm_state = caches[mamba_layer]
    assert conv_state.stride(0) == mamba_spec.page_size_bytes // 2
    assert ssm_state.stride(0) == mamba_spec.page_size_bytes // 4
    assert not conv_state.is_contiguous()
    assert not ssm_state.is_contiguous()


def test_ascend_hybrid_cache_rejects_packed_block_stride():
    import vllm_fl.worker.model_runner as model_runner

    config, attention_layer, mamba_layer, attention_spec, mamba_spec = (
        _hybrid_config()
    )
    cache_tensor = config.kv_cache_tensors[0]
    cache_tensor.block_stride = cache_tensor.size // config.num_blocks
    runner = _runner(
        config,
        attention_layer,
        mamba_layer,
        attention_spec,
        mamba_spec,
    )
    raw_by_layer = runner._allocate_kv_cache_tensors(config)

    with (
        patch.object(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type="npu"),
        ),
        pytest.raises(
            AssertionError,
            match="does not support the packed block-stride plan",
        ),
    ):
        runner._reshape_kv_cache_tensors(
            raw_by_layer,
            kernel_block_sizes=[128, 4096],
        )


def test_hybrid_kv_cache_spec_pads_attention_to_mamba_page():
    import vllm_fl.worker.model_runner as model_runner
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm_fl.worker.model_runner import ModelRunnerFL

    attention_page_bytes = 3_145_728
    mamba_page_bytes = 3_176_448
    attention_spec = FullAttentionSpec(
        block_size=1536,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
    )
    assert attention_spec.page_size_bytes == attention_page_bytes
    mamba_spec = MambaSpec(
        block_size=1536,
        shapes=((mamba_page_bytes,),),
        dtypes=(torch.uint8,),
    )
    assert mamba_spec.page_size_bytes == mamba_page_bytes

    class FakeAttentionBackend:
        @staticmethod
        def indexes_kv_by_block_stride():
            return False

    attention_module = SimpleNamespace(
        kv_sharing_target_layer_name=None,
        get_kv_cache_spec=lambda config: attention_spec,
        get_attn_backend=lambda: FakeAttentionBackend,
    )
    mamba_module = SimpleNamespace(
        get_kv_cache_spec=lambda config: mamba_spec,
    )
    runner = object.__new__(ModelRunnerFL)
    runner.vllm_config = SimpleNamespace()
    runner.shared_kv_cache_layers = {}

    layers = {
        "model.layers.0.self_attn": attention_module,
        "model.layers.1.linear_attn": mamba_module,
    }
    with (
        patch.object(
            model_runner,
            "get_layers_from_vllm_config",
            return_value=layers,
        ),
        patch.object(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type="npu"),
        ),
    ):
        specs = runner.get_kv_cache_spec()

    padded_attention_spec = specs["model.layers.0.self_attn"]
    assert padded_attention_spec.page_size_bytes == mamba_page_bytes
    assert padded_attention_spec.page_size_padded == mamba_page_bytes
    assert specs["model.layers.1.linear_attn"].page_size_bytes == mamba_page_bytes

    with (
        patch.object(
            model_runner,
            "get_layers_from_vllm_config",
            return_value=layers,
        ),
        patch.object(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type="cuda"),
        ),
    ):
        non_ascend_specs = runner.get_kv_cache_spec()

    non_ascend_attention_spec = non_ascend_specs["model.layers.0.self_attn"]
    assert non_ascend_attention_spec.page_size_bytes == attention_page_bytes
    assert non_ascend_attention_spec.page_size_padded is None
