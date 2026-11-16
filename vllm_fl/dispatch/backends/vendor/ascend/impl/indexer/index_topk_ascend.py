# Copyright (c) 2026 BAAI. All rights reserved.
"""Install MiniMax indexer bridges without embedding numerical kernels."""

import logging

from .index_decode_vendor import minimax_m3_index_decode
from .index_vendor import minimax_m3_index_topk

logger = logging.getLogger(__name__)


def install_index_topk_replacement() -> bool:
    """Patch upstream ``index_topk`` to use the Ascend top-k. Idempotent."""
    try:
        from vllm.models.minimax_m3.common.ops import index_topk as _up
    except ImportError:
        logger.warning(
            "MiniMax-M3: upstream index_topk not importable; skipping top-k patch"
        )
        return False

    if getattr(_up, "_fl_ascend_topk", False):
        return True

    # Decode uses the vendor kernel split (index_decode_vendor) -- the upstream
    # score kernel blows up once a DP rank serves more than one request.
    #
    # Prefill likewise uses the vendor scalar kernels (index_vendor). Measured
    # on the dual-machine A3 rig at C=64 (unique prompts), the local
    # ``torch.topk`` over the upstream score tensor dominated prefill: FL's
    # batch-process speedup on 64x1k prompts was 5.8x against the vendor's
    # 13.8x, showing up as +66% TTFT. The vendor prefill kernels are linear in
    # the block count and remove that nonlinearity.
    from vllm_fl.dispatch.backends.vendor.ascend.impl.indexer.index_decode_vendor import (
        minimax_m3_index_decode as _decode_vendor,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.indexer.index_vendor import (
        minimax_m3_index_score as _score_vendor,
        minimax_m3_index_topk as _topk_vendor,
    )

    originals = {
        "minimax_m3_index_topk": (_up.minimax_m3_index_topk, _topk_vendor),
        "minimax_m3_index_decode": (_up.minimax_m3_index_decode, _decode_vendor),
        "minimax_m3_index_score": (_up.minimax_m3_index_score, _score_vendor),
    }
    for name, (_, replacement) in originals.items():
        setattr(_up, name, replacement)
    _up._fl_ascend_topk = True  # type: ignore[attr-defined]

    # Patching the defining module is not enough: consumers do
    # ``from ...index_topk import minimax_m3_index_decode`` and therefore hold
    # their own reference, which keeps calling the upstream bitonic kernels.
    # Those cannot run on Ascend at all — the top-k merge kernel exceeds the
    # unified buffer (measured: "ub overflow, requires 2753792 bits while
    # 1572864 bits available"), so leaving them bound aborts graph capture.
    import sys

    rebound = []
    for mod_name, module in list(sys.modules.items()):
        if module is None or not mod_name.startswith("vllm.models.minimax_m3"):
            continue
        for name, (original, replacement) in originals.items():
            if getattr(module, name, None) is original:
                setattr(module, name, replacement)
                rebound.append(f"{mod_name}.{name}")

    logger.info(
        "MiniMax-M3: indexer switched to the vendor kernel chain "
        "(prefill score/top-k + decode split); rebound %d consumer "
        "reference(s). Measured reason: upstream's bitonic kernels cannot run "
        "on Ascend (UB overflow / impractically long compile), and the "
        "torch.topk-over-upstream-score fallback made prefill top-k nonlinear "
        "in block count (+66%% TTFT at C=64)",
        len(rebound),
    )
    if rebound:
        logger.debug("top-k rebound in: %s", ", ".join(sorted(rebound)))
    return True


__all__ = [
    "minimax_m3_index_topk",
    "minimax_m3_index_decode",
    "install_index_topk_replacement",
]
