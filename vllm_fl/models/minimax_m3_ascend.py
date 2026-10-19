# Copyright (c) 2026 BAAI. All rights reserved.
"""Vendor-owned MiniMax-M3 model classes for Ascend.

Why this file exists
--------------------
Upstream vLLM's MiniMax-M3 classes carry no ``@support_torch_compile``, and
vLLM auto-detects that *by architecture name* and routes the model to the
breakable-cudagraph path instead of the normal compile path. On Ascend that
means the whole M3 forward runs eagerly, which measured ~15% slower decode
(TPOT 51.3 ms vs 45.3 ms for vLLM-Ascend on 910C, DP2xTP8+EP,
FULL_DECODE_ONLY, identical serve flags).

Adding the decorator alone is not enough. The M3 forward performs several
in-place writes, and under the npugraph_ex backend those effects are not
preserved across graph replay, so the model emits garbled completions:

* ``fused_minimax_m3_qknorm_rope_kv_insert`` rewrites ``qkv`` in place and
  scatter-inserts K/V and the index key into the paged KV and indexer caches;
* the indexer fills a shared ``topk_indices_buffer`` and hands back views
  into it;
* the attend writes into a preallocated output buffer.

vLLM-Ascend solves the same problem by registering its attend as one opaque
custom op whose body reaches the real layer through
``forward_context.no_compile_layers``. This module follows that approach while
keeping the upstream model classes: the two attention classes are subclassed
so their in-place region goes through an opaque op, the top-level model
subclass adds the compile decorator, and the upstream module's own names are
rebound so the unmodified ``DecoderLayer`` / ``ForCausalLM`` construction
picks them up.

The opaque region is deliberately wider than vLLM-Ascend's. Upstream fuses
QK-norm/RoPE with the cache insert into a single operator that runs *before*
the attend, so the region has to start there; vLLM-Ascend's port performs the
insert inside its attention class instead and can therefore wrap only the
attend.

Narrowing the region to just the cache insert plus attend was tried and
reverted: it measured no faster (decode TPOT unchanged at C=2, ~3 ms worse at
C=4) and it hung once. With ``cudagraph_mode=FULL_DECODE_ONLY`` the whole
forward is captured either way, so an opaque op's kernels replay straight from
the graph and its Python body does not re-run per token -- the host-dispatch
saving that motivated the change does not exist in this configuration.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_SPARSE_OP = "minimax_m3_sparse_forward"

_installed = False
_ops_registered = False
_rmsnorm_patched = False


# ---------------------------------------------------------------------------
# Short-context dense-equivalent selection (``VLLM_FL_M3_SHORTCTX_DENSE=1``)
# ---------------------------------------------------------------------------
# With ``n`` visible blocks and ``n <= topk_blocks`` the lightning indexer
# selects **every** visible block, so it changes only the *order* in which the
# attend reads them (fp32 accumulation order -> <= 1 bf16 ULP), never the key
# set. Here ``n = ceil(seq_len / sparse_block_size)`` and the exact range is
# ``topk_blocks * sparse_block_size = 16 * 128 = 2048`` for MiniMax-M3 decode.
#
# This path writes that selection itself -- block ids ``[0, 1, ..., n-1, -1]``
# -- and skips the indexer and the index-key insert. The attend kernel
# (``npu_sparse_attention_score``, reached through the same
# ``layer.impl.forward``), the KV layout and the ``selectNumIdx`` contract
# (``attention/ascend/sparse_attn_m3.py:68-70``: ``count(idx >= 0)``) are
# untouched; only the producer of ``topk_indices_buffer`` changes.
#
# Why the identity set *is* the causal set (``npu_sparse_attention_score``,
# :196-223): the operator reads ``logicalId = idx[j]`` for ``j < selectNumIdx``,
# maps it through ``block_table``, skips negative ids, and truncates the block
# whose id equals ``(history_len + q_offset) // block_size`` to
# ``(history_len + q_offset) % block_size + 1`` tokens. Blocks ``0 .. n-1`` are
# exactly those that end at or before this query token, so their union is the
# full causal key set; no block past it is ever read.
#
# ``sparse_init_block`` / ``sparse_local_block`` (0 / 1 here) only *force*
# blocks into the selection (``index_topk.py:237-244``). Selecting every
# visible block already contains them, so nothing has to be re-added.
#
# Enabled at start-up only: under ``cudagraph_mode=FULL_DECODE_ONLY`` a Python
# branch inside the forward is captured as-is, so a per-request decision would
# freeze into the graph and silently change long-context output. The
# equivalence bound is enforced when the layers are built instead
# (:func:`_enforce_shortctx_dense_bound`).
_SHORTCTX_DENSE_SWITCH = "VLLM_FL_M3_SHORTCTX_DENSE"
_SHORTCTX_BLOCK_SIZE = 128  # SPARSE_BLOCK_SIZE, upstream ops/sparse_attn.py:25
_SHORTCTX_DEFAULT_TOPK = 16  # config.json "sparse_topk_blocks"


def _shortctx_dense_enabled() -> bool:
    """Read the start-up switch; resolved once, at module import."""
    import os

    return os.environ.get(_SHORTCTX_DENSE_SWITCH, "0").strip() == "1"


_SHORTCTX_DENSE = _shortctx_dense_enabled()

_shortctx_slots_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_shortctx_attend_impl = None
_shortctx_epoch = 0
_shortctx_filled_epoch = -1
_shortctx_last_ordinal: int | None = None
_shortctx_bound_checked = False


def _shortctx_attend(layer, q: torch.Tensor, attn_output: torch.Tensor) -> None:
    """Attend for the short-context path, keeping upstream's eager break.

    Upstream wraps indexer + attend in ``@eager_break_during_capture``
    (model.py:619-630) so the split-K attention kernels are never captured.
    This path has no indexer, so it applies the same decorator to the attend
    alone; outside breakable cudagraph capture (``FULL_DECODE_ONLY`` here) the
    decorator is a pass-through and the call is the plain ``impl.forward``.
    """
    global _shortctx_attend_impl

    if _shortctx_attend_impl is None:

        def _attend(layer, q: torch.Tensor, attn_output: torch.Tensor) -> None:
            layer.impl.forward(layer, q, layer.kv_cache, attn_output)

        try:
            from vllm_fl.compilation.break_graph import eager_break_during_capture
        except ImportError:  # pragma: no cover - upstream layout change
            _shortctx_attend_impl = _attend
        else:
            _shortctx_attend_impl = eager_break_during_capture(_attend)
    _shortctx_attend_impl(layer, q, attn_output)


def _shortctx_slots(
    topk: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[0 .. topk-1]`` and a scalar ``-1`` on ``device`` (cached per device)."""
    key = (device.type, str(device), topk)
    cached = _shortctx_slots_cache.get(key)
    if cached is None:
        slots = torch.arange(topk, dtype=torch.int32, device=device)
        cached = (slots, slots.new_full((), -1))
        _shortctx_slots_cache[key] = cached
    return cached


def _shortctx_blocks(kv_len: torch.Tensor, block_size: int) -> torch.Tensor:
    """``ceil(kv_len / block_size)`` as int32: the visible block count."""
    return ((kv_len + (block_size - 1)) // block_size).to(torch.int32)


def _shortctx_decode_counts(
    seq_lens: torch.Tensor, decode_query_len: int, block_size: int
) -> torch.Tensor:
    """Visible blocks per decode query token.

    Mirrors ``index_topk_ascend._decode_visible_blocks``: ``kv_len`` is plain
    ``seq_len`` when one query token per request, else
    ``seq_len - decode_query_len + q_offset + 1``.
    """
    seq_lens = seq_lens.to(torch.long)
    if decode_query_len == 1:
        kv_len = seq_lens
    else:
        total_q = seq_lens.shape[0] * decode_query_len
        ids = torch.arange(total_q, device=seq_lens.device, dtype=torch.long)
        req = ids // decode_query_len
        kv_len = seq_lens[req] - decode_query_len + (ids - req * decode_query_len) + 1
    return _shortctx_blocks(kv_len, block_size)


def _shortctx_prefill_counts(
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    num_prefill_tokens: int,
    block_size: int,
) -> torch.Tensor:
    """Visible blocks per prefill query token.

    Mirrors ``minimax_m3_index_topk`` (``index_topk_ascend.py:230-235``):
    ``valid = (context_len + q_offset + block) // block``, i.e.
    ``ceil((context_len + q_offset + 1) / block)``.
    """
    device = cu_seqlens_q.device
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.long)
    num_prefills = q_lens.shape[0]
    if num_prefills == 0 or num_prefill_tokens == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    req = torch.repeat_interleave(
        torch.arange(num_prefills, device=device),
        q_lens,
        output_size=num_prefill_tokens,
    )
    q_off = torch.arange(
        num_prefill_tokens, device=device, dtype=torch.long
    ) - cu_seqlens_q[req].to(torch.long)
    kv_len = context_lens[req].to(torch.long) + q_off + 1
    return _shortctx_blocks(kv_len, block_size)


def _shortctx_layer_ordinal(layer) -> int | None:
    """Decoder-layer index parsed from ``layer.layer_name`` (``...layers.N....``)."""
    name = getattr(layer, "layer_name", None)
    if not isinstance(name, str):
        return None
    parts = name.split(".")
    if "layers" not in parts:
        return None
    idx = parts.index("layers") + 1
    if idx >= len(parts) or not parts[idx].isdigit():
        return None
    return int(parts[idx])


def _shortctx_fill_due(layer) -> bool:
    """``True`` on the first sparse layer of each model forward.

    The identity buffer is layer-invariant, so one write per forward is enough.
    The forward boundary is detected without reading any tensor value: the
    layer loop restarts at the first sparse layer, so a non-increasing
    decoder-layer ordinal means a new forward started -- the same argument as
    ``attention/ascend/m3_step_cache.py``. Both replay modes stay correct:

    * eager / graph-break replay: this runs on every step and rewrites the
      buffer at the first sparse layer, before any attend reads it;
    * fully captured replay: this runs once while capturing, and the write is
      baked into the graph as device work, so every replay rewrites it.

    If the ordinal cannot be derived, write on every layer: correct, at the
    cost of one extra small copy per layer.
    """
    global _shortctx_epoch, _shortctx_filled_epoch, _shortctx_last_ordinal

    ordinal = _shortctx_layer_ordinal(layer)
    if ordinal is None:
        return True
    if _shortctx_last_ordinal is not None and ordinal <= _shortctx_last_ordinal:
        _shortctx_epoch += 1
    _shortctx_last_ordinal = ordinal
    if _shortctx_filled_epoch == _shortctx_epoch:
        return False
    _shortctx_filled_epoch = _shortctx_epoch
    return True


def _shortctx_write_identity_topk(layer, block_size: int) -> None:
    """Write ``[0 .. n-1, -1, ...]`` into ``layer.topk_indices_buffer``.

    Same buffer, same layout, same dtype the indexer wrote (decode tokens at
    ``[:, :num_decode_tokens]``, prefill tokens after them), so the attend sees
    exactly the values it already knows how to read and ``selectNumIdx`` --
    computed downstream as ``(idx >= 0).sum(-1)`` -- becomes ``n`` per token.
    """
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return  # memory-profiling run: no metadata, and no attend either
    main_md = attn_metadata[layer.layer_name]
    buf = layer.topk_indices_buffer
    if buf is None:
        return

    parts = []
    if main_md.num_decodes > 0:
        d = main_md.decode
        parts.append(
            _shortctx_decode_counts(d.seq_lens, d.decode_query_len, block_size)
        )
    if main_md.num_prefills > 0:
        p = main_md.prefill
        parts.append(
            _shortctx_prefill_counts(
                p.cu_seqlens_q,
                p.context_lens,
                main_md.num_prefill_tokens,
                block_size,
            )
        )
    if not parts:
        return
    counts = parts[0] if len(parts) == 1 else torch.cat(parts)
    if counts.numel() == 0:
        return

    topk = buf.shape[-1]
    counts = counts.clamp_(max=topk)
    slots, minus_one = _shortctx_slots(topk, buf.device)
    identity = torch.where(slots.unsqueeze(0) < counts.unsqueeze(1), slots, minus_one)
    total_q = main_md.num_actual_tokens
    buf[:, :total_q, :].copy_(identity.unsqueeze(0))


def _shortctx_selection_bound(config) -> int:
    """``topk_blocks * sparse_block_size`` -- the largest exact ``seq_len``."""
    sparse_cfg = getattr(config, "sparse_attention_config", None) or {}
    topk = int(sparse_cfg.get("sparse_topk_blocks", _SHORTCTX_DEFAULT_TOPK))
    block = int(sparse_cfg.get("sparse_block_size", _SHORTCTX_BLOCK_SIZE))
    return topk * block


def _enforce_shortctx_dense_bound(config) -> None:
    """Fail fast unless the whole context stays inside the identity bound.

    ``seq_len > topk_blocks * sparse_block_size`` is the only condition under
    which the identity selection differs from the indexer's (topk then drops
    blocks), so it must abort at start-up rather than degrade silently.
    """
    global _shortctx_bound_checked
    if _shortctx_bound_checked:
        return
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    max_model_len = int(vllm_config.model_config.max_model_len)
    bound = _shortctx_selection_bound(config)
    if max_model_len > bound:
        raise ValueError(
            f"{_SHORTCTX_DENSE_SWITCH}=1 replaces the block indexer with an "
            "identity selection of every visible block. That is equivalent to "
            "the block-sparse path only while max_model_len <="
            f" sparse_topk_blocks * sparse_block_size = {bound}, but got "
            f"--max-model-len {max_model_len}. Re-run with --max-model-len "
            f"{bound}, or unset the switch: above the bound the identity "
            "selection would attend blocks the indexer drops, silently "
            "changing the output."
        )
    _shortctx_bound_checked = True
    # Operator-visible confirmation that the switch took effect at start-up.
    # ``logger.info`` is NOT usable here: vLLM installs handlers for its own
    # "vllm" logger only, and this deployment's root logger sits at WARNING, so
    # the line was silently filtered in the field. ``print`` goes to the
    # serve driver's stdout (captured by its ``tee``) and needs no logging
    # configuration at all.
    print(
        f"[SHORTCTX_DENSE] MiniMax-M3: short-context dense-equivalent attention "
        f"enabled ({_SHORTCTX_DENSE_SWITCH}=1: identity block selection, "
        f"max_model_len {max_model_len} <= {bound})",
        flush=True,
    )


def _sparse_prepare(
    layer,
    qkv: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """QK-norm + partial RoPE + split for one sparse layer.

    Pure function of ``qkv`` -- it touches no cache and writes no persistent
    state -- so it runs inside the traced forward and therefore *inside* the ACL
    graph. That matters: the opaque custom op below is a single graph node whose
    Python body is re-entered on every replay, so anything left inside it pays
    host dispatch per decode step and issues kernels outside graph capture. A
    per-kernel profile showed exactly that -- FL issued 4549 kernels per decode
    step against vLLM-Ascend's 3017, with ``aclnnAdds`` at 472 versus 236. The
    doubled count is the ``1.0 + weight`` Gemma weights (4 per layer x 57), which
    the vendor computes inside its traced forward and this tree was computing
    inside the op body. vLLM-Ascend splits its own M3 the same way.

    The main ``q|k|v`` block and the indexer branch both go through the fused
    Triton kernel (``qkv_rmsnorm_rope``). Packing the index branch as
    ``[index_q | index_k | index_k]`` satisfies that kernel's ``[q|k|v]`` layout;
    the ``v`` output is discarded and repeating ``index_k`` there costs one small
    copy. Verified bitwise identical to the elementwise form (max_diff 0.0 at
    N=1/4/16).
    """
    head_dim = layer.head_dim
    main_size = layer.q_size + 2 * layer.kv_size
    cs_cache = layer.rotary_emb.cos_sin_cache
    eps = layer.q_norm.variance_epsilon

    q, k, v = _qkv_rmsnorm_rope(
        qkv.narrow(-1, 0, main_size).contiguous(),
        cs_cache,
        positions,
        q_weight=_gemma_plus_one(layer, "q_norm", "_fl_q_norm_plus_one"),
        k_weight=_gemma_plus_one(layer, "k_norm", "_fl_k_norm_plus_one"),
        q_hidden_size=layer.q_size,
        kv_hidden_size=layer.kv_size,
        head_dim=head_dim,
        eps=eps,
    )

    if _SHORTCTX_DENSE:
        # Short-context path: the index branch feeds only the indexer's block
        # scores, which the identity selection does not read. Alias q/k into the
        # two slots the opaque op still declares (it never dereferences them in
        # this mode); this drops the second fused norm/RoPE per sparse layer,
        # 50.94 us x 57 layers per decode step (measured on Ascend 910C).
        return q, k, v, q, k

    index_slab = qkv.narrow(-1, main_size, layer.index_q_size + head_dim)
    index_packed = torch.cat(
        (index_slab, index_slab.narrow(-1, layer.index_q_size, head_dim)), dim=-1
    ).contiguous()
    index_q, index_k, _ = _qkv_rmsnorm_rope(
        index_packed,
        cs_cache,
        positions,
        q_weight=_gemma_plus_one(layer, "index_q_norm", "_fl_index_q_norm_plus_one"),
        k_weight=_gemma_plus_one(layer, "index_k_norm", "_fl_index_k_norm_plus_one"),
        q_hidden_size=layer.index_q_size,
        kv_hidden_size=head_dim,
        head_dim=head_dim,
        eps=eps,
    )
    return q, k, v, index_q, index_k


def _sparse_forward_body(
    layer,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    attn_output: torch.Tensor,
) -> None:
    """Cache insert + attend for one sparse layer -- the stateful tail.

    Deliberately minimal: the K/V and index-key scatters write the paged caches
    and ``attn_output`` is written by the attend, so these effects have to stay
    outside the compiled graph. Everything up to this point runs in the graph
    (see :func:`_sparse_prepare`).
    """
    from vllm.forward_context import get_forward_context

    from vllm_fl.dispatch.backends.vendor.ascend.impl.linearnorm.minimax_m3_qknorm_rope import (
        _insert_index_k,
        _insert_kv,
    )

    slot_mapping = get_forward_context().slot_mapping
    if (
        not isinstance(slot_mapping, dict)
        or layer.layer_name not in slot_mapping
    ):
        # Memory-profiling run: caches are not bound yet, so there is nothing to
        # write and the attend has no metadata. Zero the output and stop.
        attn_output.zero_()
        return
    main_slots = slot_mapping[layer.layer_name]
    index_slots = slot_mapping[layer.indexer.index_cache.prefix]

    block_size = layer.kv_cache.size(2)
    _insert_kv(layer.kv_cache, main_slots, k, v, block_size)

    if _SHORTCTX_DENSE:
        # Short-context path: no indexer and no index-key insert. The selection
        # is the identity (every visible block, ascending), written into the
        # shared buffer the attend already reads; ``_shortctx_attend`` is the
        # same attend call upstream's ``_run_attention`` reaches (upstream
        # model.py:619-630 is exactly ``indexer(...)`` + ``impl.forward(...)``),
        # with the same eager break kept around it.
        if _shortctx_fill_due(layer):
            _shortctx_write_identity_topk(layer, block_size)
        _shortctx_attend(layer, q, attn_output)
        return

    _insert_index_k(
        layer.indexer.index_cache.kv_cache, index_slots, index_k, block_size
    )

    layer._run_attention(q, index_q, attn_output)


def minimax_m3_sparse_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
) -> None:
    """Opaque body for the sparse layer: cache writes and the attend.

    Padded rather than the whole per-layer region: the K/V and index-key scatters
    write the paged caches and ``attn_output`` is written by the attend, so only
    those effects must stay outside the compiled graph. QK-norm/RoPE runs in the
    traced forward (see :func:`_sparse_prepare`), which keeps it in the graph
    instead of being re-dispatched from Python on every replay -- the vendor's
    own M3 port is split the same way. The layer is reached through
    ``no_compile_layers``.
    """
    from vllm.forward_context import get_forward_context

    layer = get_forward_context().no_compile_layers[layer_name]
    _sparse_forward_body(layer, q, k, v, index_q, index_k, attn_output)


def minimax_m3_sparse_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


def _sparse_forward_inline(
    layer,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    index_query: torch.Tensor,
    attn_output: torch.Tensor,
) -> None:
    """Non-NPU path: fused norm/RoPE via the upstream operator, then attend.

    Used off-NPU and while the fusion pass traces its pattern; mirrors upstream's
    forward, which writes ``q``/``index_q`` through dedicated output buffers.
    """
    from vllm import _custom_ops as ops

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        layer.q_norm.weight,
        layer.k_norm.weight,
        layer.rotary_emb.cos_sin_cache,
        positions,
        layer.num_heads,
        layer.num_kv_heads,
        layer.rotary_emb.rotary_dim,
        layer.q_norm.variance_epsilon,
        layer.index_q_norm.weight,
        layer.index_k_norm.weight,
        layer.num_idx_heads,
        index_q_out=index_query,
        q_out=query,
    )
    main_size = layer.q_size + 2 * layer.kv_size
    q, k, v = qkv.narrow(-1, 0, main_size).split(
        [layer.q_size, layer.kv_size, layer.kv_size], dim=-1
    )
    layer._run_attention(query, index_query, attn_output)


def _register_ops() -> None:
    global _ops_registered
    if _ops_registered:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_SPARSE_OP,
        op_func=minimax_m3_sparse_forward,
        mutates_args=["attn_output"],
        fake_impl=minimax_m3_sparse_forward_fake,
        dispatch_key="PrivateUse1",
    )
    _ops_registered = True


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fused QK-norm / partial-RoPE / split (FL-owned kernel already in this tree)
# ---------------------------------------------------------------------------
# Gemma normalisation uses ``weight`` as ``1 + w``. The weights are model
# constants, but computing ``1 + w`` inside the traced forward means every
# decode step re-derives them for all 60 layers: a per-kernel profile showed
# ``aclnnAdds`` on the [128] head-dim tensors at 236 per step against the
# vendor runtime's 114, i.e. exactly double. The vendor avoids it by folding the
# add into its own ``npu_gemma_rms_norm`` operator; here the folded weights are
# cached on the layer once, so the forward only reads them.
_GEMMA_PLUS_ONE_ATTRS = (
    ("q_norm", "weight", "_fl_q_norm_plus_one"),
    ("k_norm", "weight", "_fl_k_norm_plus_one"),
    ("index_q_norm", "weight", "_fl_index_q_norm_plus_one"),
    ("index_k_norm", "weight", "_fl_index_k_norm_plus_one"),
)


def _gemma_plus_one(layer, module_name: str, cache_attr: str) -> torch.Tensor:
    """``1 + weight`` for a Gemma norm on ``layer``, cached if available.

    The fallback keeps this correct if a layer was built without the
    post-load caching pass having seen it; it just recomputes as before.
    """
    cached = getattr(layer, cache_attr, None)
    if cached is not None:
        return cached
    return 1.0 + getattr(layer, module_name).weight


def cache_gemma_plus_one_weights(model) -> int:
    """Precompute ``1 + w`` for every Gemma norm on ``model`` (idempotent).

    Called after weight loading. Falls back to computing on the fly (see
    :func:`_gemma_plus_one`) so a missed module is correct, only slower.
    """
    cached = 0
    for module in model.modules():
        for module_name, weight_attr, cache_attr in _GEMMA_PLUS_ONE_ATTRS:
            norm = getattr(module, module_name, None)
            weight = getattr(norm, weight_attr, None) if norm is not None else None
            # NOTE: do NOT skip already-cached modules. vLLM runs a memory-
            # profiling load before the real one, so the first call sees
            # zero-initialised parameters; skipping then would pin 1+w = 1 for
            # the whole run (measured gamma_delta 1.39 vs the live weight), and
            # the fused qkv_rmsnorm_rope path reads this cache. Recompute every
            # call -- idempotent in value, always matching the latest params.
            if weight is None:
                continue
            # Detach: this is a derived constant, not a trainable parameter.
            object.__setattr__(module, cache_attr, (1.0 + weight).detach())
            cached += 1
    return cached


def _qkv_rmsnorm_rope(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
):
    """Use FL's existing fused Triton kernel ``qkv_rmsnorm_rope``.

    The kernel already ships in this repository
    (``impl/linearnorm/split_qkv_rmsnorm_rope.py``) and is registered by
    ``graph_fusion_ops``; using it is what closes the per-layer gap against the
    vendor runtime. It is a value-returning op with a fake impl, so it is also
    safe inside ``torch.compile``.
    """
    from vllm_fl.dispatch.backends.vendor.ascend.impl.graph_fusion_ops import (
        ensure_graph_fusion_ops_registered,
    )

    ensure_graph_fusion_ops_registered()
    return torch.ops.vllm.qkv_rmsnorm_rope(
        input=qkv,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_dim,
        eps=eps,
    )


def _install_official_gemma_rmsnorm() -> bool:
    """Swap M3's ``MiniMAXGemmaRMSNorm`` for vLLM's ``GemmaRMSNorm``.

    ``MiniMAXGemmaRMSNorm`` is a plain ``nn.Module`` (upstream model.py:109) whose
    whole effect travels through in-place writes made by a flashinfer shim:

        gemma_fused_add_rmsnorm(x, residual, weight, eps)   # writes both in place
        return x, residual

    Being a bare ``nn.Module``, it sits *outside* vLLM's ``CustomOp`` contract.
    Every other layer type reaches the framework through that contract, which is
    what makes compilation safe in vLLM-Ascend: a ``CustomOp`` either compiles its
    own body in isolation (``CustomOp.maybe_compile``) or is excluded from the
    fused graph, so its side effects are always accounted for. A plain module has
    neither guarantee, so under ``torch.compile`` those in-place writes are not
    preserved and the residual stream is corrupted layer by layer -- the model
    then emits well-formed but meaningless text (measured: "The capital of France
    is" returns noise with finite logprobs).

    vLLM's ``GemmaRMSNorm`` has identical semantics (``x * (1 + w)``, optional
    fused residual add) and *is* a ``CustomOp``, and it is what vLLM-Ascend's own
    M3 port uses -- it never references ``MiniMAXGemmaRMSNorm`` at all. Substituting
    the class keeps upstream's model structure while putting this layer back inside
    the contract.
    """
    global _rmsnorm_patched
    if _rmsnorm_patched:
        return True
    try:
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm as _Official
        from vllm.models.minimax_m3.nvidia import model as _up
    except ImportError:
        return False

    if getattr(_up, "MiniMAXGemmaRMSNorm", None) is None:
        return False
    if getattr(_up.MiniMAXGemmaRMSNorm, "_fl_official_swap", False):
        _rmsnorm_patched = True
        return True

    _up.MiniMAXGemmaRMSNorm = _Official
    _Official._fl_official_swap = True  # type: ignore[attr-defined]
    _rmsnorm_patched = True
    logger.info(
        "MiniMax-M3: MiniMAXGemmaRMSNorm -> vLLM GemmaRMSNorm (CustomOp contract; "
        "identical x*(1+w) semantics, keeps in-place effects visible to compile)"
    )
    return True


class _GemmaPlusOneCacheMixin:
    """Cache ``1 + w`` for every Gemma norm right after weights are loaded.

    ``load_weights`` is the last point where the parameters are final and the
    forward has not been traced yet, so folding the constant there keeps the
    traced forward free of per-step ``Adds`` operators.
    """

    def load_weights(self, weights):
        result = super().load_weights(weights)
        try:
            n = cache_gemma_plus_one_weights(self)
            if n:
                logger.info(
                    "MiniMax-M3: precomputed 1+w for %d Gemma norms "
                    "(removes per-step Adds on constant weights)", n
                )
        except Exception as exc:  # pragma: no cover - cache is an optimisation
            logger.warning("MiniMax-M3: Gemma 1+w caching skipped: %s", exc)
        return result


def _build_classes():
    """Bind the upstream classes and derive our subclasses from them."""
    from vllm.models.minimax_m3.nvidia import model as _up

    class AscendMiniMaxM3Attention(_up.MiniMaxM3Attention):
        """Dense attention (layers 0-2): fused norm/RoPE through an op."""

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            qkv, _ = self.qkv_proj(hidden_states)
            # Same branch as vLLM-Ascend's own M3 port: on NPU the QK-norm /
            # partial-RoPE / split step is a single fused Triton kernel, which
            # measured ~5.8x faster than composing npu_rms_norm + rotation +
            # cat/copy for identical work (0.199 ms vs 1.151 ms per layer on A3,
            # dense shape, 4096 tokens). The else branch keeps upstream
            # behaviour for non-NPU devices and for pattern tracing.
            if (
                qkv.device.type == "npu"
                and qkv.dtype == torch.bfloat16
                and positions.ndim == 1
            ):
                q, k, v = _qkv_rmsnorm_rope(
                    qkv.contiguous(),
                    self.rotary_emb.cos_sin_cache,
                    positions,
                    # M3 uses Gemma-style normalisation: weight is 1 + w.
                    q_weight=_gemma_plus_one(
                        self, "q_norm", "_fl_q_norm_plus_one"
                    ),
                    k_weight=_gemma_plus_one(
                        self, "k_norm", "_fl_k_norm_plus_one"
                    ),
                    q_hidden_size=self.q_size,
                    kv_hidden_size=self.kv_size,
                    head_dim=self.head_dim,
                    eps=self.q_norm.variance_epsilon,
                )
            else:
                from vllm import _custom_ops as ops

                ops.fused_minimax_m3_qknorm_rope_kv_insert(
                    qkv,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    self.rotary_emb.cos_sin_cache,
                    positions,
                    self.num_heads,
                    self.num_kv_heads,
                    self.rotary_emb.rotary_dim,
                    self.q_norm.variance_epsilon,
                    kv_cache_dtype="auto",
                )
                q, k, v = qkv.split(
                    [self.q_size, self.kv_size, self.kv_size], dim=-1
                )
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

    class AscendMiniMaxM3SparseAttention(_up.MiniMaxM3SparseAttention):
        """Block-sparse attention (the 57 sparse layers).

        Norm/RoPE runs in the traced forward (``_sparse_prepare``); only the
        cache inserts and the attend go through one opaque op, so graph replay
        cannot lose the stateful effects while the pure-function work still runs
        inside the ACL graph. This is how vLLM-Ascend splits its own M3 port.

        With ``VLLM_FL_M3_SHORTCTX_DENSE=1`` the same forward runs without the
        indexer (see the module-level note); the only extra start-up step is
        :func:`_enforce_shortctx_dense_bound`, which refuses a context length
        the identity selection cannot reproduce.
        """

        def __init__(self, *args, **kwargs):
            config = kwargs.get("config", args[0] if args else None)
            super().__init__(*args, **kwargs)
            if _SHORTCTX_DENSE and config is not None:
                _enforce_shortctx_dense_bound(config)

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            qkv, _ = self.qkv_proj(hidden_states)
            num_tokens = qkv.shape[0]

            # The memory-profiling / unbound-cache case is decided INSIDE the
            # opaque op, not here. Upstream tested it with a data-dependent early
            # return (`if self.layer_name not in slot_mapping: return new_zeros`),
            # and Dynamo cannot trace that construct: it aborts the whole
            # attention forward, which is why all 57 sparse layers dropped out of
            # the compiled graph and their in-place KV writes went unprotected.
            # Keeping the branch inside the op makes this forward branch-free and
            # therefore traceable.

            if (
                qkv.device.type == "npu"
                and qkv.dtype == torch.bfloat16
                and positions.ndim == 1
            ):
                # Norm/RoPE stays in the traced graph; only the cache writes and
                # the attend go behind the opaque op. See _sparse_prepare.
                q, k, v, index_q, index_k = _sparse_prepare(self, qkv, positions)
                attn_output = torch.empty_like(q)
                torch.ops.vllm.minimax_m3_sparse_forward(
                    q,
                    k,
                    v,
                    index_q,
                    index_k,
                    attn_output,
                    self.layer_name,
                )
            else:
                # Non-NPU (and pattern tracing): same work inline.
                q = qkv.new_empty((num_tokens, self.q_size))
                index_q = qkv.new_empty(
                    (num_tokens, self.index_q_size),
                    dtype=self.indexer.index_cache.dtype,
                )
                attn_output = torch.empty_like(q)
                _sparse_forward_inline(
                    self, qkv, positions, q, index_q, attn_output
                )
            output, _ = self.o_proj(attn_output)
            return output

    import os as _os

    from vllm.compilation.decorators import support_torch_compile

    if _os.environ.get("M3_VENDOR_COMPILE", "1") == "1":
        # On by default, matching how vLLM-Ascend compiles its own M3 port, and
        # verified accurate: 10/10 on the semantic gate including 4.5K-token
        # prompts. The earlier garbled output was traced to a data-dependent
        # early return inside the traced attention forward, which made Dynamo
        # abandon that forward and dropped all 57 sparse layers out of the graph;
        # removing it (the profiling guard now lives inside the opaque op) fixed
        # it. Compilation is also what keeps the per-layer norm/RoPE in the ACL
        # graph instead of re-dispatching it from Python every decode step.
        @support_torch_compile(
            dynamic_arg_dims={
                "input_ids": 0,
                "positions": 0,
                "inputs_embeds": 0,
            },
        )
        class AscendMiniMaxM3Model(_GemmaPlusOneCacheMixin, _up.MiniMaxM3Model):
            """Upstream model plus the compile support Ascend needs."""

        _compiled_base = AscendMiniMaxM3Model
    else:
        class AscendMiniMaxM3Model(_GemmaPlusOneCacheMixin, _up.MiniMaxM3Model):
            """Upstream model, eager: compilation is not safe yet here.

            The class exists so the attention subclasses below stay
            independent of whether compilation is enabled; behaviour matches
            upstream exactly while ``M3_VENDOR_COMPILE`` is unset.
            """

    class AscendMiniMaxM3SparseForCausalLM(_up.MiniMaxM3SparseForCausalLM):
        """Registry entry point; ``__init__`` builds our model class."""

    return (
        AscendMiniMaxM3Attention,
        AscendMiniMaxM3SparseAttention,
        AscendMiniMaxM3Model,
        AscendMiniMaxM3SparseForCausalLM,
    )


(
    AscendMiniMaxM3Attention,
    AscendMiniMaxM3SparseAttention,
    AscendMiniMaxM3Model,
    AscendMiniMaxM3SparseForCausalLM,
) = _build_classes()


def install_ascend_minimax_m3_model() -> bool:
    """Register the opaque ops and rebind the upstream class names.

    Idempotent. Only the names the upstream construction path looks up at
    runtime are rebound (``DecoderLayer`` reads ``MiniMaxM3SparseAttention``,
    ``Model`` reads ``MiniMaxM3DecoderLayer``, ``ForCausalLM`` and the VL
    wrapper read ``MiniMaxM3Model``), so no upstream file is edited.
    """
    global _installed
    if _installed:
        return True

    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        logger.debug("MiniMax-M3 Ascend model classes skipped on %s", current_platform.device_type)
        return False

    _register_ops()
    _install_official_gemma_rmsnorm()

    from vllm.models.minimax_m3.nvidia import model as _up

    _up.MiniMaxM3Attention = AscendMiniMaxM3Attention
    _up.MiniMaxM3SparseAttention = AscendMiniMaxM3SparseAttention
    _up.MiniMaxM3Model = AscendMiniMaxM3Model
    _installed = True
    logger.info(
        "MiniMax-M3: installed Ascend model classes (opaque attention ops + "
        "@support_torch_compile); upstream files untouched"
    )
    return True


__all__ = [
    "AscendMiniMaxM3Attention",
    "AscendMiniMaxM3Model",
    "AscendMiniMaxM3SparseAttention",
    "AscendMiniMaxM3SparseForCausalLM",
    "install_ascend_minimax_m3_model",
]
