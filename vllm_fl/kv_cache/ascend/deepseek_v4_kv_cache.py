"""Ascend-only DeepSeek-V4 non-MTP KV-cache planning and coordination.

This is the current vLLM-Ascend 0.24 non-packed closure.  vLLM's default
DeepSeek-V4 planner uses a packed tensor with ``block_stride``; FL's NPU model
runner deliberately consumes independent, contiguous tensors instead.
"""

import itertools
import math
import sys
from collections import defaultdict
from collections.abc import Mapping
from math import lcm

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm import envs
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    may_override_num_blocks,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request


class CompressAttentionManager(FullAttentionManager):
    """Full-attention manager whose physical blocks represent compressed tokens."""

    def __init__(self, kv_cache_spec: KVCacheSpec, *args, **kwargs):
        super().__init__(kv_cache_spec, *args, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio

    def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                                   total_computed_tokens, num_tokens_main_model,
                                   apply_admission_cap=False):
        return super().get_num_blocks_to_allocate(
            request_id, num_tokens // self.compress_ratio, new_computed_blocks,
            total_computed_tokens // self.compress_ratio,
            num_tokens_main_model // self.compress_ratio, apply_admission_cap)

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        return super().allocate_new_blocks(
            request_id, num_tokens // self.compress_ratio,
            num_tokens_main_model // self.compress_ratio)

    def add_local_computed_blocks(self, request_id, new_computed_blocks,
                                  num_local_computed_tokens,
                                  num_external_computed_tokens):
        return super().add_local_computed_blocks(
            request_id, new_computed_blocks,
            num_local_computed_tokens // self.compress_ratio,
            num_external_computed_tokens // self.compress_ratio)

    def allocate_new_computed_blocks(self, request_id, new_computed_blocks,
                                      num_local_computed_tokens,
                                      num_external_computed_tokens):
        """rc1 compatibility entry point with compressed local/external tokens."""
        if request_id in self.num_cached_block:
            assert len(new_computed_blocks) == 0
            return
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        total = (num_local_computed_tokens + num_external_computed_tokens) // self.compress_ratio
        skipped = self.get_num_skipped_tokens(total)
        skipped_blocks = skipped // self.block_size
        if skipped_blocks:
            new_computed_blocks = new_computed_blocks[skipped_blocks:]
            num_external_computed_tokens = min(total - skipped, num_external_computed_tokens)
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks)
        req_blocks.extend([self._null_block] * skipped_blocks)
        req_blocks.extend(new_computed_blocks)
        self.num_cached_block[request_id] = len(req_blocks)
        if num_external_computed_tokens > 0:
            blocks = self.block_pool.get_new_blocks(cdiv(total, self.block_size) - len(req_blocks))
            req_blocks.extend(blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(block.block_id for block in blocks)

    def allocate_external_computed_blocks(self, request_id, num_local_computed_tokens,
                                          num_external_computed_tokens):
        return super().allocate_external_computed_blocks(
            request_id, num_local_computed_tokens // self.compress_ratio,
            num_external_computed_tokens // self.compress_ratio)

    def cache_blocks(self, request: Request, num_tokens: int,
                     retention_interval=None, *, alignment_tokens=None) -> None:
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // (self.block_size * self.compress_ratio)
        if num_cached_blocks >= num_full_blocks:
            return
        self.block_pool.cache_full_blocks(
            request=request, blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks, num_full_blocks=num_full_blocks,
            block_size=self.block_size * self.compress_ratio,
            kv_cache_group_id=self.kv_cache_group_id)
        self.num_cached_block[request.request_id] = num_full_blocks

    @classmethod
    def find_longest_cache_hit(cls, block_hashes: BlockHashList, max_length: int,
                               kv_cache_group_ids: list[int], block_pool,
                               kv_cache_spec: KVCacheSpec, alignment_tokens: int,
                               dcp_world_size: int = 1, pcp_world_size: int = 1,
                               drop_eagle_block: bool = False):
        block_size = kv_cache_spec.block_size * dcp_world_size * pcp_world_size
        logical_block_size = block_size * kv_cache_spec.compress_ratio
        logical_hashes = BlockHashListWithBlockSize(
            block_hashes, block_size, logical_block_size)
        computed = tuple([] for _ in kv_cache_group_ids)
        for block_hash in itertools.islice(logical_hashes, max_length // logical_block_size):
            cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached:
                break
            for out, block in zip(computed, cached):
                out.append(block)
        if drop_eagle_block and computed[0]:
            for out in computed:
                out.pop()
        while computed[0] and len(computed[0]) * logical_block_size % alignment_tokens:
            for out in computed:
                out.pop()
        return computed


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    **kwargs,
):
    """rc1 factory, including admission caps matching startup pool sizing."""
    # Import locally: kv_cache_interface imports this module to register the
    # manager, while this factory only runs after registration has completed.
    from .kv_cache_interface import AscendMLAAttentionSpec

    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    assert manager_class is not None, f"No manager for {type(kv_cache_spec).__name__}"
    if (isinstance(kv_cache_spec, AscendMLAAttentionSpec)
            and kv_cache_spec.compress_ratio > 1):
        manager_class = CompressAttentionManager
        if max_model_len is not None:
            kwargs["max_admission_blocks_per_request"] = cdiv(
                max_model_len // kv_cache_spec.compress_ratio,
                kv_cache_spec.block_size) + 1
    elif isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        if max_model_len is not None and max_num_batched_tokens is not None:
            kwargs["max_admission_blocks_per_request"] = (
                kv_cache_spec.max_admission_blocks_per_request(
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_model_len=max_model_len))
    return manager_class(kv_cache_spec, **kwargs)


def _is_deepseek_v4_spec(spec: KVCacheSpec) -> bool:
    if getattr(spec, "model_version", None) == "deepseek_v4":
        return True
    nested = getattr(spec, "kv_cache_specs", None)
    if isinstance(nested, Mapping):
        nested = nested.values()
    return nested is not None and any(
        getattr(item, "model_version", None) == "deepseek_v4" for item in nested)


def _is_deepseek_v4_config(config: KVCacheConfig) -> bool:
    return _is_deepseek_v4_groups(config.kv_cache_groups)


def _is_deepseek_v4_groups(groups: list[KVCacheGroupSpec]) -> bool:
    return any(_is_deepseek_v4_spec(group.kv_cache_spec) for group in groups)


_ORIGINAL_RESOLVE_BLOCK_SIZES = kv_cache_utils.resolve_kv_cache_block_sizes
_ORIGINAL_PACKED_KV_CACHE_CONFIG = kv_cache_utils._get_kv_cache_config_packed


def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
    """rc1 Ascend CP geometry; upstream's CUDA hybrid restriction is inapplicable."""
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups
    if len(groups) <= 1:
        size = vllm_config.cache_config.block_size * dcp * pcp
        return size, size
    if dcp != 1 or pcp != 1:
        sizes = [group.kv_cache_spec.block_size for group in groups]
        scheduler = math.lcm(*sizes) * dcp * pcp
        if not vllm_config.cache_config.enable_prefix_caching:
            return scheduler, scheduler
        return scheduler, math.gcd(*sizes)
    return _ORIGINAL_RESOLVE_BLOCK_SIZES(kv_cache_config, vllm_config)


def group_and_unify_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """rc1 grouping: C4 and C128 MLA layers must not be merged together."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

    if not any(isinstance(spec, SlidingWindowMLASpec)
               for spec in kv_cache_spec.values()):
        return None
    mla_by_ratio: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    swa_by_block: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            swa_by_block[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            mla_by_ratio[spec.compress_ratio][name] = spec
    return ([UniformTypeKVCacheSpecs.from_specs(mla_by_ratio[ratio])
             for ratio in sorted(mla_by_ratio, key=lambda ratio: (ratio != 4, ratio))]
            + [UniformTypeKVCacheSpecs.from_specs(specs)
               for specs in swa_by_block.values()])


def _get_kv_cache_groups_uniform_groups(grouped_specs):
    """rc1 DeepSeek-V4 group construction, excluding the out-of-scope MTP path."""
    from vllm.utils.math_utils import round_up
    from vllm.v1.core.kv_cache_utils import _approximate_gcd

    assert len(grouped_specs) >= 2
    full_groups = [KVCacheGroupSpec(list(spec.kv_cache_specs), spec)
                   for spec in grouped_specs[:2]]
    tuple_sizes = [spec.get_num_layer_tuples() for spec in grouped_specs]
    tuple_size = _approximate_gcd(tuple_sizes, lower_bound=tuple_sizes[0])
    tuple_sizes = [round_up(size, tuple_size) for size in tuple_sizes]
    swa_groups = []
    page_sizes = grouped_specs[0].get_page_sizes()
    for spec in grouped_specs[2:]:
        by_page: dict[int, list[str]] = defaultdict(list)
        for name, layer_spec in spec.kv_cache_specs.items():
            page_size = layer_spec.page_size_bytes
            padded = min(size for size in page_sizes if size >= page_size)
            if page_size < padded:
                object.__setattr__(layer_spec, "page_size_padded", padded)
            by_page[padded].append(name)
        count = len(next(iter(by_page.values())))
        assert all(len(names) == count for names in by_page.values())
        num_tuple_groups = cdiv(count, tuple_size)
        layer_tuples = list(zip(*by_page.values()))
        for index in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[index::num_tuple_groups]
            names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            sub_spec = UniformTypeKVCacheSpecs.from_specs(
                {name: spec.kv_cache_specs[name] for name in names})
            swa_groups.append(KVCacheGroupSpec(names, sub_spec))
    return [*full_groups, *swa_groups]


def _get_kv_cache_config_non_packed(vllm_config, kv_cache_groups: list[KVCacheGroupSpec],
                                    available_memory: int):
    """rc1 DeepSeek-V4 layout: independent tensors, never ``block_stride``."""
    if any("mtp" in name.lower() for group in kv_cache_groups for name in group.layer_names):
        raise NotImplementedError("Ascend DeepSeek-V4 MTP KV-cache planning is not supported")
    full = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full.get_page_sizes())
    buckets: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        bucket: dict[int, list[str]] = defaultdict(list)
        specs = group.kv_cache_spec.kv_cache_specs
        for name in group.layer_names:
            bucket[specs[name].page_size_bytes].append(name)
        buckets.append(bucket)
    tuple_count = max(len(names) for bucket in buckets for names in bucket.values())
    bytes_per_tuple = sum(page_sizes)
    num_blocks = may_override_num_blocks(
        vllm_config, available_memory // (bytes_per_tuple * tuple_count))
    tensors = []
    for index in range(tuple_count):
        for page_size in page_sizes:
            shared_by = [names[index] for bucket in buckets
                         if (names := bucket.get(page_size)) is not None and index < len(names)]
            tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))
    return num_blocks, tensors


def _get_kv_cache_config_packed_for_ascend(vllm_config, kv_cache_groups, available_memory):
    """Only DeepSeek-V4 opts out of vLLM's packed cross-layer layout."""
    if _is_deepseek_v4_groups(kv_cache_groups):
        return _get_kv_cache_config_non_packed(
            vllm_config, kv_cache_groups, available_memory
        )
    return _ORIGINAL_PACKED_KV_CACHE_CONFIG(
        vllm_config, kv_cache_groups, available_memory
    )


class AscendHybridKVCacheCoordinator(KVCacheCoordinator):
    """Current rc1 DeepSeek-V4 coordinator with compressed cache granularity."""

    def __init__(self, kv_cache_config, max_model_len, max_num_batched_tokens,
                 use_eagle, enable_caching, enable_kv_cache_events,
                 dcp_world_size, pcp_world_size, scheduler_block_size,
                 hash_block_size, metrics_collector=None):
        dcp = dcp_world_size
        pcp = pcp_world_size
        if dcp != 1 or pcp != 1:
            raise NotImplementedError("Ascend DeepSeek-V4 DSA-CP KV-cache coordination is not supported")
        # This deliberately does not call the upstream constructor: it would
        # first create non-compressed managers.  Keep rc1's initialization
        # order so pool sizing, admission and prefix-hit state share one
        # compressed manager set from the beginning.
        self.dcp_world_size = dcp
        self.pcp_world_size = pcp
        self.scheduler_block_size = scheduler_block_size
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_caching = enable_caching
        self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        validate_retention = getattr(
            sys.modules["vllm.v1.core.kv_cache_coordinator"],
            "_validate_prefix_cache_retention_interval",
            None,
        )
        if self.retention_interval is not None and validate_retention is not None:
            validate_retention(
                self.retention_interval, self.scheduler_block_size, kv_cache_config
            )
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )
        self.eagle_group_ids = {
            index for index, group in enumerate(kv_cache_config.kv_cache_groups)
            if group.is_eagle_group
        }
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec, block_pool=self.block_pool,
                enable_caching=self.enable_caching, kv_cache_group_id=index,
                dcp_world_size=dcp, pcp_world_size=pcp,
                scheduler_block_size=self.scheduler_block_size,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens)
            for index, group in enumerate(self.kv_cache_config.kv_cache_groups))
        self.hash_block_size = hash_block_size
        if enable_caching:
            assert all(self._effective_block_size(group.kv_cache_spec) % hash_block_size == 0
                       for group in kv_cache_config.kv_cache_groups)
        self.verify_and_split_kv_cache_groups()
        # Preserve the incoming scheduler write floor (for example, 128 when
        # the configured DSV4 block size is 128). The compressed LCM is
        # applied only to SlidingWindowManager inside
        # verify_and_split_kv_cache_groups, so SWA masks and hit lookup remain
        # aligned to the C128 logical block without delaying 8K cache writes.
        self.use_eagle = use_eagle

    def _effective_block_size(self, spec):
        return spec.block_size * max(getattr(spec, "compress_ratio", 1), 1)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Use the incoming scheduler floor; SWA owns compressed-LCM alignment."""
        aligned = (num_computed_tokens // self.scheduler_block_size
                   * self.scheduler_block_size)
        for manager in self.single_type_managers:
            tokens_to_cache = aligned
            # EAGLE matches a lookahead logical block and drops it at read
            # time, therefore the write path must retain that block too.
            if manager.use_eagle and aligned > 0:
                tokens_to_cache = min(
                    num_computed_tokens, aligned + manager.block_size
                )
            manager.cache_blocks(
                request, tokens_to_cache,
                retention_interval=self.retention_interval,
            )

    def verify_and_split_kv_cache_groups(self):
        groups = []
        for index, group in enumerate(self.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            manager_cls = self.single_type_managers[index].__class__
            for old_spec, group_ids, old_cls in groups:
                if old_spec == spec:
                    assert old_cls is manager_cls
                    group_ids.append(index)
                    break
            else:
                groups.append((spec, [index], manager_cls))
        assert len(groups) > 1
        self.attention_groups = sorted(groups, key=lambda group: not isinstance(group[0], FullAttentionSpec))
        self.eagle_attn_group_indices = {
            index for index, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(group_id in self.eagle_group_ids for group_id in group_ids)}
        for index in self.eagle_attn_group_indices:
            for group_id in self.attention_groups[index][1]:
                self.single_type_managers[group_id].use_eagle = True
        self.lcm_block_size = lcm(*[self._effective_block_size(spec)
                                    for spec, _, _ in self.attention_groups])
        for manager in self.single_type_managers:
            if isinstance(manager, SlidingWindowManager):
                manager.scheduler_block_size = self.lcm_block_size

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        hit_length = max_cache_hit_length
        by_group = [None] * len(self.kv_cache_config.kv_cache_groups)
        eagle_verified = set()
        while True:
            current = hit_length
            for index, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                effective = self._effective_block_size(spec)
                cached = by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached is not None:
                    current = current // effective * effective
                    continue
                use_eagle = index in self.eagle_attn_group_indices and index not in eagle_verified
                maximum = min(current + spec.block_size, max_cache_hit_length) if use_eagle else current
                source_size = spec.block_size
                source = block_hashes if source_size == self.hash_block_size else BlockHashListWithBlockSize(
                    block_hashes, self.hash_block_size, source_size)
                blocks = manager_cls.find_longest_cache_hit(
                    source, maximum, group_ids, self.block_pool, spec,
                    alignment_tokens=self.lcm_block_size, dcp_world_size=self.dcp_world_size,
                    pcp_world_size=self.pcp_world_size, drop_eagle_block=use_eagle)
                new_length = len(blocks[0]) * effective
                if use_eagle:
                    eagle_verified.add(index)
                elif new_length < current:
                    eagle_verified.clear()
                current = new_length
                for group_id, group_blocks in zip(group_ids, blocks):
                    by_group[group_id] = group_blocks
            if current >= hit_length:
                break
            hit_length = current
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            count = hit_length // self._effective_block_size(spec)
            for group_id in group_ids:
                if by_group[group_id] is not None:
                    del by_group[group_id][count:]
        return tuple(blocks if blocks is not None else [] for blocks in by_group), hit_length

    def find_longest_cache_hit_per_group(self, block_hashes, max_cache_hit_length):
        """Hybrid per-group API with the same compressed logical units."""
        hit_blocks = [[] for _ in self.kv_cache_config.kv_cache_groups]
        hit_lengths = [0] * len(self.kv_cache_config.kv_cache_groups)
        for spec, group_ids, manager_cls in self.attention_groups:
            source = (block_hashes if spec.block_size == self.hash_block_size
                      else BlockHashListWithBlockSize(
                          block_hashes, self.hash_block_size, spec.block_size))
            blocks = manager_cls.find_longest_cache_hit(
                source, max_cache_hit_length, group_ids, self.block_pool, spec,
                alignment_tokens=self.lcm_block_size,
                dcp_world_size=self.dcp_world_size,
                pcp_world_size=self.pcp_world_size,
            )
            length = len(blocks[0]) * self._effective_block_size(spec)
            for group_id, group_blocks in zip(group_ids, blocks):
                hit_blocks[group_id] = group_blocks
                hit_lengths[group_id] = length
        return tuple(hit_blocks), tuple(hit_lengths)


def apply_deepseek_v4_kv_cache_patches() -> None:
    """Install idempotent Ascend-only hooks after the Ascend platform is selected."""
    import vllm.v1.core.kv_cache_coordinator as coordinator
    import vllm.v1.core.kv_cache_utils as utils
    import vllm.v1.engine.core as engine

    if getattr(utils, "_fl_ascend_deepseek_v4_kv_cache", False):
        return
    original_factory = coordinator.get_kv_cache_coordinator

    def factory(*args, **kwargs):
        config = kwargs.get("kv_cache_config") or args[0]
        if _is_deepseek_v4_config(config):
            return AscendHybridKVCacheCoordinator(*args, **kwargs)
        return original_factory(*args, **kwargs)

    utils._get_kv_cache_config_packed = _get_kv_cache_config_packed_for_ascend
    utils.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
    utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
    utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
    coordinator.get_kv_cache_coordinator = factory
    manager = sys.modules.get("vllm.v1.core.kv_cache_manager")
    if manager is not None:
        manager.get_kv_cache_coordinator = factory
    engine.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
    utils._fl_ascend_deepseek_v4_kv_cache = True
