# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Scheduler-side KV-cache management for Ascend DSA compression."""

import itertools
import sys
from collections.abc import Sequence
from math import lcm

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.request import Request

from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa_cache import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)


class AscendCompressedAttentionManager(FullAttentionManager):
    """Manage blocks indexed in compressed-token rather than token space."""

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        block_pool: BlockPool,
        **kwargs,
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens // self.compress_ratio,
            new_computed_blocks,
            total_computed_tokens // self.compress_ratio,
            num_tokens_main_model // self.compress_ratio,
            apply_admission_cap,
        )

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        if request_id in self.num_cached_block:
            assert len(new_computed_blocks) == 0
            return

        request_blocks = self.req_to_blocks[request_id]
        assert len(request_blocks) == 0
        compressed_total = (
            num_local_computed_tokens + num_external_computed_tokens
        ) // self.compress_ratio
        skipped_tokens = self.get_num_skipped_tokens(compressed_total)
        skipped_blocks = skipped_tokens // self.block_size
        if skipped_blocks > 0:
            new_computed_blocks = new_computed_blocks[skipped_blocks:]
            num_external_computed_tokens = min(
                compressed_total - skipped_tokens,
                num_external_computed_tokens,
            )

        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks must be empty when prefix caching is disabled"
            )

        request_blocks.extend([self._null_block] * skipped_blocks)
        request_blocks.extend(new_computed_blocks)
        self.num_cached_block[request_id] = len(request_blocks)

        if num_external_computed_tokens > 0:
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(compressed_total, self.block_size)
                - len(request_blocks)
            )
            request_blocks.extend(allocated_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(
                    block.block_id for block in allocated_blocks
                )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        del num_tokens_main_model
        compressed_tokens = num_tokens // self.compress_ratio
        request_blocks = self.req_to_blocks[request_id]
        required_blocks = cdiv(compressed_tokens, self.block_size)
        num_new_blocks = required_blocks - len(request_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        request_blocks.extend(new_blocks)
        return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        super().cache_blocks(request, num_tokens // self.compress_ratio)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, AscendMLAAttentionSpec)
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in kv_cache_group_ids
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            cached = block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            )
            if not cached:
                break
            for computed, cached_for_group in zip(
                computed_blocks, cached
            ):
                computed.append(cached_for_group)
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()

        compressed_alignment = cdiv(
            alignment_tokens, kv_cache_spec.compress_ratio
        )
        while (
            block_size != compressed_alignment
            and len(computed_blocks[0]) * block_size
            % compressed_alignment
            != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks


class AscendDSAHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """Hybrid coordinator aware of compressed DSA cache block lengths."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector=None,
    ) -> None:
        # The generic hybrid constructor requires every physical block size
        # to be divisible by the token-space hash size. Compressed DSA groups
        # only need that restriction when prefix caching actually hashes
        # blocks, so initialize the common coordinator state directly.
        KVCacheCoordinator.__init__(
            self,
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.hash_block_size = hash_block_size
        if enable_caching:
            assert all(
                group.kv_cache_spec.block_size % hash_block_size == 0
                for group in kv_cache_config.kv_cache_groups
            ), "block_size must be divisible by hash_block_size"
        assert dcp_world_size == 1, "DCP is not supported by DSA hybrid cache"
        assert pcp_world_size == 1, "PCP is not supported by DSA hybrid cache"
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        super().verify_and_split_kv_cache_groups()
        self.lcm_block_size = lcm(
            *(
                spec.block_size * getattr(spec, "compress_ratio", 1)
                for spec, _, _ in self.attention_groups
            )
        )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        def group_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes,
                self.hash_block_size,
                kv_cache_spec.block_size,
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hits_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )
        eagle_verified: set[int] = set()

        while True:
            current_hit_length = hit_length
            for index, (spec, group_ids, manager_cls) in enumerate(
                self.attention_groups
            ):
                cached_blocks = hits_by_group[group_ids[0]]
                if (
                    isinstance(spec, FullAttentionSpec)
                    and cached_blocks is not None
                ):
                    current_hit_length = (
                        current_hit_length
                        // spec.block_size
                        * spec.block_size
                    )
                    continue

                use_eagle = (
                    index in self.eagle_attn_group_indices
                    and index not in eagle_verified
                )
                max_length = current_hit_length
                if use_eagle:
                    max_length = min(
                        current_hit_length + spec.block_size,
                        max_cache_hit_length,
                    )
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=group_hashes(spec),
                    max_length=max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    use_eagle=use_eagle,
                    alignment_tokens=self.lcm_block_size,
                )
                logical_block_size = spec.block_size * getattr(
                    spec, "compress_ratio", 1
                )
                new_hit_length = len(hit_blocks[0]) * logical_block_size
                if use_eagle:
                    eagle_verified.add(index)
                elif new_hit_length < current_hit_length:
                    eagle_verified.clear()
                current_hit_length = new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hits_by_group[group_id] = blocks

            if current_hit_length >= hit_length:
                break
            hit_length = current_hit_length
            if simple_hybrid:
                break

        first_spec, first_group_ids, _ = self.attention_groups[0]
        if isinstance(first_spec, FullAttentionSpec):
            num_blocks = hit_length // first_spec.block_size
            for group_id in first_group_ids:
                if (blocks := hits_by_group[group_id]) is not None:
                    del blocks[num_blocks:]
        return (
            tuple(blocks or [] for blocks in hits_by_group),
            hit_length,
        )


def install_dsa_cache_manager() -> None:
    """Register DSA cache scheduling without changing non-DSA models."""
    from vllm_fl.dispatch.backends.vendor.ascend.impl.dsa_cache_utils import (
        install_dsa_cache_utils,
    )

    install_dsa_cache_utils()
    spec_manager_map[AscendMLAAttentionSpec] = (
        AscendCompressedAttentionManager
    )
    spec_manager_map[AscendSlidingWindowMLASpec] = SlidingWindowManager

    import vllm.v1.core.kv_cache_coordinator as coordinator_module
    import vllm.v1.core.single_type_kv_cache_manager as manager_module

    original_manager_factory = manager_module.get_manager_for_kv_cache_spec
    if not getattr(original_manager_factory, "_fl_dsa_manager_factory", False):

        def get_manager_for_kv_cache_spec(
            kv_cache_spec: KVCacheSpec,
            max_num_batched_tokens: int,
            max_model_len: int,
            **kwargs,
        ) -> SingleTypeKVCacheManager:
            if (
                isinstance(kv_cache_spec, AscendMLAAttentionSpec)
                and kv_cache_spec.compress_ratio > 1
            ):
                max_compressed_tokens = (
                    max_model_len // kv_cache_spec.compress_ratio
                )
                kwargs["max_admission_blocks_per_request"] = (
                    cdiv(max_compressed_tokens, kv_cache_spec.block_size) + 1
                )
                return AscendCompressedAttentionManager(
                    kv_cache_spec,
                    **kwargs,
                )
            return original_manager_factory(
                kv_cache_spec,
                max_num_batched_tokens,
                max_model_len,
                **kwargs,
            )

        get_manager_for_kv_cache_spec._fl_dsa_manager_factory = True
        manager_module.get_manager_for_kv_cache_spec = (
            get_manager_for_kv_cache_spec
        )
        # kv_cache_coordinator imports the factory by value, so update its
        # cached binding as well as the defining module.
        coordinator_module.get_manager_for_kv_cache_spec = (
            get_manager_for_kv_cache_spec
        )

    original_factory = coordinator_module.get_kv_cache_coordinator
    if getattr(original_factory, "_fl_dsa_cache_factory", False):
        return

    def get_kv_cache_coordinator(
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector=None,
    ):
        uses_dsa_compression = any(
            isinstance(group.kv_cache_spec, AscendMLAAttentionSpec)
            and group.kv_cache_spec.compress_ratio > 1
            for group in kv_cache_config.kv_cache_groups
        )
        if not uses_dsa_compression:
            return original_factory(
                kv_cache_config,
                max_model_len,
                max_num_batched_tokens,
                use_eagle,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                pcp_world_size,
                hash_block_size,
                metrics_collector,
            )
        return AscendDSAHybridKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )

    get_kv_cache_coordinator._fl_dsa_cache_factory = True
    coordinator_module.get_kv_cache_coordinator = get_kv_cache_coordinator
    cached_module = sys.modules.get("vllm.v1.core.kv_cache_manager")
    if cached_module is not None:
        cached_module.get_kv_cache_coordinator = get_kv_cache_coordinator
