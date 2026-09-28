# Copyright (c) 2026 BAAI. All rights reserved.
"""Ascend replacement for vLLM's ``fused_minimax_m3_qknorm_rope_kv_insert``.

Upstream (CUDA) definition
--------------------------
``vllm._custom_ops.fused_minimax_m3_qknorm_rope_kv_insert`` (schema in
``csrc/libtorch_stable/torch_bindings.cpp``, reference math in
``tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py``).

```
fused_minimax_m3_qknorm_rope_kv_insert(
    Tensor! qkv, Tensor q_norm_weight, Tensor k_norm_weight,
    Tensor cos_sin_cache, Tensor positions, int num_heads, int num_kv_heads,
    int rotary_dim, float eps,
    Tensor? index_q_norm_weight, Tensor? index_k_norm_weight, int num_index_heads,
    Tensor? slot_mapping, Tensor? index_slot_mapping,
    Tensor!? kv_cache, Tensor!? index_cache, int block_size,
    Tensor!? q_out, Tensor!? index_q_out, str kv_cache_dtype) -> ()
```

Semantics (from the upstream docstring + kernel test):

* ``qkv`` is a single fused tensor laid out head-major as
  ``[q | k | v]`` (dense) or ``[q | k | v | index_q | index_k]`` (sparse),
  each group being ``n_heads * head_dim`` wide.
* Gemma RMSNorm (``x * (1 + w) * rsqrt(mean(x^2) + eps)``) is applied per head
  along the last dim, then NeoX-style **partial** RoPE is applied to the
  leading ``rotary_dim`` lanes (the remaining lanes pass through).
* ``q`` and ``index_q`` are written back **in place**, or into ``q_out`` /
  ``index_q_out`` when provided.
* When ``kv_cache`` is given, ``k``/``v`` are scattered into the paged cache by
  ``slot_mapping`` and ``index_k`` into ``index_cache`` by
  ``index_slot_mapping`` (falling back to ``slot_mapping`` when omitted).

Ascend implementation notes
---------------------------
Composed from our own operators (torch_npu fused + CANN), not FlagGems:

* normalisation -> ``torch.ops.npu.npu_rms_norm`` with ``gamma = 1 + w``
  (identical to :class:`AscendGemmaRMSNorm`, which is already validated for
  Qwen on this platform);
* RoPE -> ``torch.ops.npu.npu_rotary_mul`` in ``"half"`` (NeoX) mode, applied
  to the leading ``rotary_dim`` lanes only;
* cache insertion -> ``torch.ops._C_ascend.npu_scatter_nd_update_v2``, which is
  this repository's own CANN operator.

.. warning::
   The composition below has **not yet been validated numerically** against the
   CUDA kernel — the A3 devices were busy when it was written. Validation is
   the first item in the test plan (compare against the Python reference in
   ``tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py``).
"""

from __future__ import annotations

import torch


def _gemma_rms_norm_last_dim(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm over the last dim: ``x * (1 + w) * rsqrt(mean(x^2) + eps)``.

    ``x`` is ``[..., head_dim]``; ``weight`` is ``[head_dim]`` (M3 uses a shared
    per-lane weight — the kernel test builds it as ``randn(HEAD_DIM)``).
    """
    gamma = (1.0 + weight.to(torch.float32)).to(x.dtype)
    out, _ = torch.ops.npu.npu_rms_norm(x, gamma, eps)
    return out


def _partial_neox_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """NeoX-style partial RoPE on the leading ``rotary_dim`` lanes of ``x``.

    ``x`` is ``[num_tokens, n_heads, head_dim]``; ``cos_sin_cache`` is
    ``[max_pos, rotary_dim]`` holding ``cos || sin``.
    """
    head_dim = x.shape[-1]
    cs = cos_sin_cache[positions].to(torch.float32)  # [N, rotary_dim]
    half = rotary_dim // 2
    cos = cs[..., :half]
    sin = cs[..., half:]
    rot = x[..., :rotary_dim]
    rest = x[..., rotary_dim:]
    # Elementwise NeoX/"half" rotation. ATB's RopeOperation cannot be set up
    # inside the npugraph_ex compile region ("RopeOperation setup failed!"),
    # which aborts compilation for the whole model; this form was verified
    # bitwise identical to npu_rotary_mul("half") on A3.
    cos_h = cos.unsqueeze(1).to(torch.float32)
    sin_h = sin.unsqueeze(1).to(torch.float32)
    rot_f = rot.to(torch.float32)
    x1 = rot_f[..., :half]
    x2 = rot_f[..., half:]
    o1 = x1 * cos_h - x2 * sin_h
    o2 = x2 * cos_h + x1 * sin_h
    rot = torch.cat((o1, o2), dim=-1).to(x.dtype)
    if rest.shape[-1] == 0:
        return rot
    return torch.cat((rot, rest), dim=-1)


def _norm_rope_group(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    eps: float,
) -> torch.Tensor:
    """Gemma-norm then partial RoPE for one head group ``[N, n_heads, head_dim]``."""
    return _partial_neox_rope(
        _gemma_rms_norm_last_dim(x, weight, eps), positions, cos_sin_cache, rotary_dim
    )


def _insert_kv(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter ``k``/``v`` (``[N, num_kv_heads, head_dim]``) into the paged cache.

    Flattening the leading dims turns a slot into a flat row index, which our
    CANN op can write directly. Both paged layouts are accepted:

    * ``[2, num_blocks, block_size, num_kv_heads, head_dim]`` -- plane-major, so
      ``cache[0]``/``cache[1]`` are *contiguous* K and V. This is what the fused
      sparse attend needs (see ``sparse_attn_ascend``).
    * ``[num_blocks, 2, block_size, num_kv_heads, head_dim]`` -- upstream's fused
      layout, still used by the Triton fallback.
    """
    plane_major = kv_cache.shape[0] == 2
    num_blocks = kv_cache.shape[1] if plane_major else kv_cache.shape[0]
    num_kv_heads = kv_cache.shape[3]
    head_dim = kv_cache.shape[4]

    # `.view` (not `.reshape`): a silently-copied tensor would swallow the
    # in-place scatter below and corrupt the KV cache.
    flat = kv_cache.view(2 * num_blocks * block_size, num_kv_heads, head_dim)
    block = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offset = slot_mapping % block_size
    if plane_major:
        k_rows = block * block_size + offset
        v_rows = k_rows + num_blocks * block_size
    else:
        k_rows = block * (2 * block_size) + offset
        v_rows = k_rows + block_size
    # Graph capture pads unused slots with -1. ScatterNd skips negative indices,
    # so route every padding slot to -1 explicitly: in the plane-major layout
    # V's implicit row (-1 + num_blocks * block_size) is a *valid* row and would
    # otherwise be overwritten with padding.
    pad = slot_mapping < 0
    k_rows = torch.where(pad, -1, k_rows).to(torch.int32)
    v_rows = torch.where(pad, -1, v_rows).to(torch.int32)
    # npu_scatter_nd_update_v2 follows ScatterNd semantics: `indices` is
    # [N, K] where K is the coordinate rank. Passing a 1-D tensor of N slots
    # would be read as a *single* N-dimensional coordinate (verified on
    # hardware), so the row id must be given as [N, 1].
    torch.ops._C_ascend.npu_scatter_nd_update_v2(flat, k_rows.unsqueeze(-1), k)
    torch.ops._C_ascend.npu_scatter_nd_update_v2(flat, v_rows.unsqueeze(-1), v)


def _insert_index_k(
    index_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_k: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter ``index_k`` into the index-key side cache.

    The indexer's cache is **key-only and single-head**:
    ``(num_blocks, block_size, head_dim)`` — one ``head_dim`` vector per token
    (upstream kernel docstring and ``test_sparse_full`` both define it this way).
    ``index_k`` therefore arrives as ``[N, head_dim]``, not ``[N, n_heads, dim]``.
    """
    head_dim = index_cache.shape[-1]
    # See `_insert_kv`: the backing storage must be written, not a copy.
    flat = index_cache.view(-1, head_dim)
    # See `_insert_kv`: coordinates must be [N, 1], not [N].
    torch.ops._C_ascend.npu_scatter_nd_update_v2(
        flat, slot_mapping.to(torch.int32).unsqueeze(-1), index_k.reshape(-1, head_dim)
    )


def fused_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    """Ascend implementation; signature-compatible with the upstream op."""
    if qkv.dim() != 2:
        raise ValueError(f"qkv must be 2-D [num_tokens, width], got {tuple(qkv.shape)}")
    num_tokens = qkv.shape[0]
    head_dim = q_norm_weight.shape[-1]
    sparse = num_index_heads > 0

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    # Sparse packing is [q | k | v | index_q | index_k]. index_q spans
    # num_index_heads heads, but index_k is a SINGLE head (MQA-style): the index
    # cache is key-only with one head_dim vector per token. See the upstream
    # kernel test ``test_sparse_full``, which sets ``iksz = HEAD_DIM`` while
    # ``iqsz = num_idx_heads * HEAD_DIM``.
    idx_q_size = num_index_heads * head_dim
    idx_k_size = head_dim
    idx_size = idx_q_size  # kept for the in-place write-back below
    expected = q_size + 2 * kv_size + ((idx_q_size + idx_k_size) if sparse else 0)
    if qkv.shape[-1] != expected:
        raise ValueError(
            f"qkv width {qkv.shape[-1]} does not match head layout "
            f"(q + 2*kv{(' + index_q + index_k' if sparse else '')} = {expected})"
        )

    x = qkv.reshape(num_tokens, -1, head_dim)
    q = x[:, :num_heads]
    k = x[:, num_heads : num_heads + num_kv_heads]
    v = x[:, num_heads + num_kv_heads : num_heads + 2 * num_kv_heads]

    q = _norm_rope_group(q, q_norm_weight, positions, cos_sin_cache, rotary_dim, eps)
    k = _norm_rope_group(k, k_norm_weight, positions, cos_sin_cache, rotary_dim, eps)

    if q_out is not None:
        q_out.copy_(q.reshape(num_tokens, q_size))

    index_q = index_k = None
    if sparse:
        base = num_heads + 2 * num_kv_heads
        index_q = x[:, base : base + num_index_heads]
        # index_k occupies a single head's worth of lanes.
        ik_lo = base + num_index_heads
        index_k = x[:, ik_lo : ik_lo + 1]
        index_q = _norm_rope_group(
            index_q, index_q_norm_weight, positions, cos_sin_cache, rotary_dim, eps
        )
        index_k = _norm_rope_group(
            index_k, index_k_norm_weight, positions, cos_sin_cache, rotary_dim, eps
        )
        if index_q_out is not None:
            index_q_out.copy_(index_q.reshape(num_tokens, idx_q_size))

    # The upstream op is in-place: q, k *and* index_q are written back into the
    # fused buffer (only v is left alone). q/index_q go to their dedicated
    # output buffers instead when those are supplied.
    x[:, num_heads : num_heads + num_kv_heads].copy_(k)
    if q_out is None:
        x[:, :num_heads].copy_(q)
    if sparse:
        base = num_heads + 2 * num_kv_heads
        ik_lo = base + num_index_heads
        x[:, ik_lo : ik_lo + 1].copy_(index_k)
        if index_q_out is None:
            x[:, base : base + num_index_heads].copy_(index_q)

    if kv_cache is not None:
        _insert_kv(kv_cache, slot_mapping, k, v, block_size)
    if sparse and index_cache is not None:
        _insert_index_k(
            index_cache,
            index_slot_mapping if index_slot_mapping is not None else slot_mapping,
            index_k,
            block_size,
        )


__all__ = ["fused_minimax_m3_qknorm_rope_kv_insert"]
