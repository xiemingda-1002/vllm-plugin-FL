# Copyright (c) 2026 BAAI. All rights reserved.

"""Lazy registration entry point for the first Ascend fusion targets."""

from __future__ import annotations


def ensure_graph_fusion_ops_registered() -> None:
    from .canonical_rotary import ensure_npu_rotary_embedding_registered
    from .linearnorm.split_qkv_rmsnorm_rope import (
        ensure_qkv_rmsnorm_rope_registered,
    )

    ensure_npu_rotary_embedding_registered()
    ensure_qkv_rmsnorm_rope_registered()
