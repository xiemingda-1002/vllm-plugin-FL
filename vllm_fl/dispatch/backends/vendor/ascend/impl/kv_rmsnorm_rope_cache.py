# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""npu_kv_rmsnorm_rope_cache -- torch_npu fused RMSNorm + RoPE + paged KV write

Source and contract
-------------------
Fused operator: split a fused KV projection row into its RMS part and its RoPE
part, RMS-normalise the first, rotate the second, and scatter both into the
paged KV caches. There is no AscendC source for the built-in under
`csrc/ascend`, so the contract is pinned by:

  * `csrc/ascend/attention/compressor/op_kernel/arch32/rms_norm.h` L36-85 for
    the RMSNorm half (x^2 -> RowSum -> /N -> +eps -> Sqrt -> x / Sqrt -> *gamma);
  * `csrc/ascend/attention/inplace_partial_rotary_mul/op_kernel/
    apply_rotary_pos_emb_common.h` L319-451 (`DeepSeekInterleaveModeVF`) for the
    RoPE half: read (x[2k], x[2k+1]), write two contiguous halves, cos/sin at
    the write indices;
  * FlagGems-vllm's validated Triton port
    `src/flaggems_vllm/runtime/backend/_ascend/ops/kv_rmsnorm_rope_cache.py`
    (rope kernel L52-169, rms kernel L213-290, host L293-387): same split, same
    paged scatter, and the same `(k_cache, ckv_cache, k_rope, c_norm)` return
    order;
  * vllm-ascend `vllm_ascend/attention/mla_v1.py` L1552-1581 (call shape
    [B, N, S, D], `cache_mode` "PA"/"PA_NZ", `c_kv_scale`, `is_output_kv`).

Call-site schema (every site in `vllm_fl`, verified by grep):
    npu_kv_rmsnorm_rope_cache(
        Tensor kv, Tensor gamma, Tensor cos, Tensor sin, Tensor index,
        Tensor k_cache, Tensor ckv_cache,
        Tensor? k_rope_scale=None, Tensor? c_kv_scale=None,
        Tensor? k_rope_offset=None, Tensor? c_kv_offset=None,
        float epsilon=1e-5, str cache_mode="Norm", bool is_output_kv=False)
      -> (k_cache, ckv_cache)                                    if not is_output_kv
      -> (k_cache, ckv_cache, k_rope, c_kv)                      if is_output_kv
      vllm_fl/attention/ascend/sfa_v1.py:1192  `_, _, k_pe, k_nope = ...`
      vllm_fl/attention/ascend/sfa_v1.py:1206  return value discarded
      both pass `slots.to(torch.int64)`, `kv_cache[1]`, `kv_cache[0]`,
      `epsilon=`, `cache_mode="PA"`, and (first site) `is_output_kv=True`.

  * kv     : [B, N, S, D], D = rms_size + rope_size, fp16/bf16/fp32;
             kv[..., :rms_size] is the RMS half, kv[..., rms_size:] the RoPE half.
  * gamma  : [rms_size], same dtype as kv.
  * cos/sin: last dim rope_size, one row per (b, s) token, contiguous.
  * index  : int64 slot ids, length B*S; slot = page_index * page_size + offset.
  * caches : paged [num_pages, page_size, ..., D_cache] with stride(-1) == 1.
  * k_rope / c_kv (only with is_output_kv) : [B, N, S, rope_size] /
             [B, N, S, rms_size], kv dtype. This matches the
             `(cache_k, cache_ckv, current_k_rope, current_c_kv)` order that
             FlagGems-vllm validated against torch_npu and that vllm_fl reads
             as `k_pe, k_nope`.

Explicitly not implemented (raised, never silently ignored):
  * `k_rope_scale` / `c_kv_scale` / `k_rope_offset` / `c_kv_offset` -- the
    quantise/dequantise path; no call site in this tree passes them.
  * every `cache_mode` except the paged BNSD layouts "PA" / "PA_BNSD"
    (torch_npu's default "Norm" layout and the fractal "PA_NZ" are unknown
    here, and PA_NZ is a storage format, not an addressing tweak).
  * `index` dtypes other than int64 (torch_npu requires int64; every call site
    casts explicitly).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["npu_kv_rmsnorm_rope_cache", "CACHE_MODE_PA", "CACHE_MODE_PA_BNSD"]

CACHE_MODE_PA = 1
CACHE_MODE_PA_BNSD = 1

_CACHE_MODES = {"pa": CACHE_MODE_PA, "pa_bnsd": CACHE_MODE_PA_BNSD}

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _kv_rms_norm_kernel(
    KV, GAMMA, OUT, CACHE, INDEX,
    N, S, D, RMS, EPS, PAGE, S0, S1,
    WRITE_OUT: tl.constexpr,
    OUT_FP32: tl.constexpr,
    BLK: tl.constexpr,
    NCH: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * D
    # kv row order is (b, n, s); index has one slot per (b, s) token.
    token = (row // (N * S)) * S + (row % S)

    ss = tl.zeros([BLK], dtype=tl.float32)
    for i in tl.static_range(NCH):
        cc = i * BLK + tl.arange(0, BLK)
        m = cc < RMS
        xv = tl.load(KV + base + cc, mask=m, other=0.0).to(tl.float32)
        ss += xv * xv
    rstd = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / RMS + EPS)

    slot = tl.load(INDEX + token)
    page = slot // PAGE
    off = slot % PAGE
    c_base = page * S0 + off * S1

    for i in tl.static_range(NCH):
        cc = i * BLK + tl.arange(0, BLK)
        m = cc < RMS
        xv = tl.load(KV + base + cc, mask=m, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + cc, mask=m, other=0.0).to(tl.float32)
        y = xv * rstd * g
        tl.store(CACHE + c_base + cc, y.to(CACHE.dtype.element_ty), mask=m)
        if WRITE_OUT:
            if OUT_FP32:
                tl.store(OUT + row * RMS + cc, y, mask=m)
            else:
                tl.store(OUT + row * RMS + cc, y.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _kv_rope_kernel(
    KV, COS, SIN, OUT, CACHE, INDEX,
    N, S, D, RMS, ROPE, PAGE, S0, S1,
    WRITE_OUT: tl.constexpr,
    BLK: tl.constexpr,
    NCH: tl.constexpr,
):
    row = tl.program_id(0)
    # kv row order is (b, n, s); cos/sin have one row per (b, s) token.
    token = (row // (N * S)) * S + (row % S)
    c_base = token * ROPE
    x_base = row * D + RMS
    half = ROPE // 2

    slot = tl.load(INDEX + token)
    page = slot // PAGE
    off = slot % PAGE
    slot_base = page * S0 + off * S1

    j = tl.arange(0, 2)
    m0 = j[None, :] == 0
    for i in tl.static_range(NCH):
        k = i * BLK + tl.arange(0, BLK)
        km = k < half
        off2 = 2 * k[:, None] + j[None, :]
        xv = tl.load(KV + x_base + off2, mask=km[:, None], other=0.0).to(tl.float32)
        x_even = tl.sum(tl.where(m0, xv, 0.0), axis=1)
        x_odd = tl.sum(tl.where(m0, 0.0, xv), axis=1)

        c1 = tl.load(COS + c_base + k, mask=km, other=0.0).to(tl.float32)
        s1 = tl.load(SIN + c_base + k, mask=km, other=0.0).to(tl.float32)
        c2 = tl.load(COS + c_base + half + k, mask=km, other=0.0).to(tl.float32)
        s2 = tl.load(SIN + c_base + half + k, mask=km, other=0.0).to(tl.float32)

        y_first = (x_even * c1 - x_odd * s1).to(CACHE.dtype.element_ty)
        y_second = (x_odd * c2 + x_even * s2).to(CACHE.dtype.element_ty)
        tl.store(CACHE + slot_base + k, y_first, mask=km)
        tl.store(CACHE + slot_base + half + k, y_second, mask=km)
        if WRITE_OUT:
            tl.store(OUT + row * ROPE + k, y_first.to(OUT.dtype.element_ty), mask=km)
            tl.store(
                OUT + row * ROPE + half + k,
                y_second.to(OUT.dtype.element_ty),
                mask=km,
            )


def _cache_geometry(cache: torch.Tensor, width: int, name: str):
    if cache.dim() < 2:
        raise ValueError(f"{name} must be at least 2-D [pages, page_size, ..., D]")
    if int(cache.shape[-1]) != width:
        raise ValueError(
            f"{name} last dim {int(cache.shape[-1])} != expected {width}"
        )
    if cache.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in its last dim")
    return int(cache.shape[1]), int(cache.stride(0)), int(cache.stride(1))


def npu_kv_rmsnorm_rope_cache(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    k_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    k_rope_scale: torch.Tensor | None = None,
    c_kv_scale: torch.Tensor | None = None,
    k_rope_offset: torch.Tensor | None = None,
    c_kv_offset: torch.Tensor | None = None,
    epsilon: float = 1e-5,
    cache_mode: str = "Norm",
    is_output_kv: bool = False,
):
    """Triton equivalent of `torch_npu.npu_kv_rmsnorm_rope_cache` (paged PA).

    Args:
        kv: [B, N, S, D] fused KV projection, rms half first.
        gamma: [rms_size] RMSNorm weight.
        cos, sin: rope_size-wide per-token tables.
        index: int64 slot ids, length B*S.
        k_cache: paged cache for the rope half (last dim rope_size).
        ckv_cache: paged cache for the normed half (last dim rms_size).
        k_rope_scale/c_kv_scale/k_rope_offset/c_kv_offset: must be None.
        epsilon: RMSNorm epsilon, default 1e-5.
        cache_mode: "PA" or "PA_BNSD" (the paged BNSD layout used by the call
            sites). Any other value raises NotImplementedError.
        is_output_kv: also return the per-token rotated / normed activations.

    Returns:
        `(k_cache, ckv_cache)` or `(k_cache, ckv_cache, k_rope, c_kv)`.
    """
    unsupported = {
        "k_rope_scale": k_rope_scale,
        "c_kv_scale": c_kv_scale,
        "k_rope_offset": k_rope_offset,
        "c_kv_offset": c_kv_offset,
    }
    for name, value in unsupported.items():
        if value is not None:
            raise NotImplementedError(
                f"npu_kv_rmsnorm_rope_cache: {name} is not implemented (the "
                f"quantise/dequantise path has no call site in this tree); got "
                f"shape {tuple(value.shape)} dtype {value.dtype}"
            )
    if cache_mode is None or str(cache_mode).strip().lower() not in _CACHE_MODES:
        raise NotImplementedError(
            "npu_kv_rmsnorm_rope_cache: cache_mode must be 'PA' or 'PA_BNSD' "
            f"(paged BNSD, the layout of both vllm_fl call sites); got "
            f"{cache_mode!r}. torch_npu's default 'Norm' layout and the fractal "
            "'PA_NZ' layout are not implemented."
        )

    if kv.dim() != 4:
        raise ValueError(
            f"kv must be 4-D [B, N, S, D] (the call sites reshape explicitly), "
            f"got {tuple(kv.shape)}"
        )
    if kv.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"kv dtype must be fp16/bf16/fp32, got {kv.dtype}")
    if gamma.dtype != kv.dtype:
        raise ValueError(f"gamma dtype {gamma.dtype} must equal kv dtype {kv.dtype}")
    if index.dtype != torch.int64:
        raise ValueError(
            f"index must be int64 (torch_npu contract, call sites cast explicitly), "
            f"got {index.dtype}"
        )
    if k_cache.dtype != kv.dtype or ckv_cache.dtype != kv.dtype:
        raise ValueError("k_cache/ckv_cache dtype must equal kv dtype")

    b, n, s, d = (int(kv.shape[0]), int(kv.shape[1]), int(kv.shape[2]), int(kv.shape[3]))
    rms = int(gamma.numel())
    rope = d - rms
    if rms <= 0 or rope <= 0 or rope % 2 != 0:
        raise ValueError(
            f"kv last dim {d} must split into a positive rms_size and a positive "
            f"even rope_size (gamma numel = {rms})"
        )
    tokens = b * s
    rows = tokens * n
    if index.numel() != tokens:
        raise ValueError(f"index numel {index.numel()} != B*S = {tokens}")
    for name, t in (("cos", cos), ("sin", sin)):
        if int(t.shape[-1]) != rope:
            raise ValueError(f"{name} last dim {int(t.shape[-1])} != rope_size {rope}")
        if t.numel() != tokens * rope:
            raise ValueError(
                f"{name} numel {t.numel()} != (B*S)*rope_size = {tokens * rope}"
            )
    page_size, k_s0, k_s1 = _cache_geometry(k_cache, rope, "k_cache")
    c_page_size, c_s0, c_s1 = _cache_geometry(ckv_cache, rms, "ckv_cache")
    if page_size != c_page_size:
        raise ValueError(
            f"k_cache page_size {page_size} != ckv_cache page_size {c_page_size}"
        )

    kv_c = kv.contiguous()
    g = gamma.contiguous()
    cos_c = cos.contiguous().view(tokens, rope)
    sin_c = sin.contiguous().view(tokens, rope)
    idx = index.contiguous().view(tokens)

    k_rope_out = None
    c_kv_out = None
    if is_output_kv:
        k_rope_out = torch.empty((b, n, s, rope), dtype=kv.dtype, device=kv.device)
        c_kv_out = torch.empty((b, n, s, rms), dtype=kv.dtype, device=kv.device)
    if rows == 0:
        if is_output_kv:
            return k_cache, ckv_cache, k_rope_out, c_kv_out
        return k_cache, ckv_cache

    blk_rms = min(_next_pow2(rms), 1024)
    _kv_rms_norm_kernel[(rows,)](
        kv_c, g,
        c_kv_out if is_output_kv else ckv_cache,
        ckv_cache, idx,
        n, s, d, rms, float(epsilon), page_size, c_s0, c_s1,
        WRITE_OUT=is_output_kv,
        OUT_FP32=(kv.dtype == torch.float32),
        BLK=blk_rms,
        NCH=_ceil_div(rms, blk_rms),
    )

    blk_rope = min(_next_pow2(rope // 2), 512)
    _kv_rope_kernel[(rows,)](
        kv_c, cos_c, sin_c,
        k_rope_out if is_output_kv else k_cache,
        k_cache, idx,
        n, s, d, rms, rope, page_size, k_s0, k_s1,
        WRITE_OUT=is_output_kv,
        BLK=blk_rope,
        NCH=_ceil_div(rope // 2, blk_rope),
    )

    if is_output_kv:
        return k_cache, ckv_cache, k_rope_out, c_kv_out
    return k_cache, ckv_cache
