# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""fusion_attention -- AscendC -> Triton (dense MHA/GQA, TND + BSND/BSH, minimal port)

Call sites (vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py):
    L759  (BaseDeviceAdaptor.npu_flash_attention, FP32 fallback -- `_npu_flash_attention_unpad`
           does not support FP32, so the FP32 branch of that helper calls this operator)
        npu_fusion_attention(query=query, key=key, value=value,
                             actual_seq_qlen=cumulative_seq_lens,
                             actual_seq_kvlen=cumulative_seq_lens,
                             head_num=head_num, scale=scale_value,
                             input_layout="TND")[0]
    L1901 (A5DeviceAdaptor.npu_flash_attention, the only branch there)
        the same call, again indexed with `[0]`
Both call sites pass a **flat TND** triple with cumulative sequence lengths, no mask, no
`sparse_mode` and no paged cache, and read only element 0 of the result.

Other in-tree call sites, used here only as evidence for the argument names and the mask
convention (they do not go through device_op):
    vllm_fl/attention/ascend/attention.py L1089-1112 `_forward_encoder_attention`
        sparse_mode=3 with atten_mask=attn_metadata.attn_mask, and the sparse_mode=0 branch

## Provenance of the semantics

There is no AscendC source for this operator in this repository (`csrc/ascend` has no
`fusion_attention` directory; a recursive search for the operator name over csrc/ hits only
an unrelated CMakeLists.txt and the tiling_sink implementation). It is a CANN built-in
reached through torch_npu, so the semantics are taken from the in-tree call sites:

  * `actual_seq_qlen` / `actual_seq_kvlen` are **cumulative** lists: device_op.py L757
    builds `cumulative_seq_lens = seq_lens_cpu.cumsum(0).tolist()`, so entry b is the
    inclusive prefix sum and the last entry is the total token count.
  * `head_num` is the query head count; the KV head count is **not** an argument of this
    operator and is inferred from the key shape (GQA).
  * `atten_mask` uses **1 = masked out** (standard_mask.py builds
    `torch.triu(torch.ones(2048, 2048), diagonal=1)`, i.e. 1 exactly when j > i).
  * `input_layout="TND"` is the flat `[tokens, heads, head_dim]` layout.

## Extracted computation

    b     = batch of token t, s1 = position of t inside its sequence
    n2    = n1 // (N1 / N2)
    j_hi  = per sparse_mode (see below); j_lo = 0 unless sparse_mode == 4
    valid(j) = (j <= j_hi) and (j >= j_lo) and (j < kvLen[b]) and (atten_mask[s1_abs, j] == 0)
    k_row(j) = TND kv  : kvBase[b] + j        (kvBase = cuKvLen[b])
               BSND kv : b * S2 + j
    score(j) = ( q[t, n1, :] . k[k_row(j), n2, :] ) * scale
    out[t,n1] = sum_j exp(score - max) * v[k_row(j), n2, :] / sum_j exp(score - max)

    sparse_mode 0  no causal limit        -> j_hi = kvLen[b] - 1
    sparse_mode 2  left-up causal         -> j_hi = s1
    sparse_mode 3  right-down causal      -> j_hi = s1 + (kvLen[b] - qLen[b])
    sparse_mode 4  band (sliding window)  -> j_hi = s1 + (kvLen[b] - qLen[b]) + next_tockens
                                             j_lo = j_hi - pre_tockens + 1
    s1_abs = s1 + (kvLen[b] - qLen[b]) for modes 3/4, else s1 (the atten_mask row index).

This is plain dense attention: the operator is `FlashAttentionScore` semantics, which is why
the same kernel body also serves `fused_infer_attention_score` -- the two AscendC operators
differ in their argument surface and in the paged-KV support, not in the arithmetic.

## Implemented branches (this is the minimal port; see "Not covered")

* `input_layout` in {"TND", "BSND", "BSH"} (BSND/BSH are the same memory order,
  `[B, S, H, D]` and the flattened `[B, S, H*D]`); the output takes the query's shape
* GQA, with the KV head count inferred from the key shape
* `sparse_mode` in {0, 2, 3, 4}; an optional 2-D int8/bool `atten_mask` (1 = masked out)
  applied in addition to the mode-derived mask
* `actual_seq_qlen` / `actual_seq_kvlen` as length-B cumulative vectors (the call-site form,
  device_op.py L757), length-B+1 cu_seqlens, or length-B per-sequence lengths that sum to
  the token count; when the two vectors are bit-identical (the call-site form) they are
  normalized together, so a self-attention call cannot be misread as per-request lengths
* the head dim is inferred from the query layout (`_head_dim`): `[B, S, N, D]` and
  `[T, N, D]` read `D` from the last dim, a flattened `[*, N*D]` divides by `head_num`; a
  3-D `[B, S, N*D]` whose `S` equals `head_num` is read as TND and then rejected instead of
  being silently mis-indexed
* `scale`, `head_num`, `inner_precise` (accepted; it is only a precision hint and cannot
  change the result of an fp32-accumulated reference); fp32 accumulation, online softmax
* returns `(attention_out, softmax_max, softmax_sum)`; the auxiliary two are **empty
  placeholders** because both call sites pass `return_softmax_lse=False` (default) and read
  only element 0

## Not covered (explicitly)

* `input_layout` in {"BNSD", "NTD", "TND_NTD", "NSD", "BSH_NBSD", ...} -- raises
* paged KV: `npu_fusion_attention` has no `block_table` argument, so no paged path exists;
  neither device_op call site uses one
* `sparse_mode` 1/5/6/7/8
* `pse` (position encoding), `padding_mask`, `keep_prob != 1.0` (dropout),
  `return_softmax_lse=True`: each raises NotImplementedError naming the value. The two
  auxiliary outputs (`softmax_max`, `softmax_sum`) are therefore **not computed** -- a
  caller that needs them cannot use this port as is
* `v` head dim different from `q` head dim (raises)
* ragged query lengths with `BSND` (ragged batches are a TND feature; raises)
* no Cube/Vector multi-core pipeline, no double buffering: one program walks whole
  (token, head) rows, so this is a reference implementation and not a speed match
* the exact arity of the torch_npu return tuple for this operator could not be verified
  from in-tree code (only `[0]` is ever read); this port returns three elements

## Capture note

The sequence lengths are needed on the host to build the token -> (batch, position) map and
the per-batch KV base. Device tensors are pulled with `.cpu()`, which is a device-to-host
copy and is **illegal inside ACL graph capture** (aclrtMemcpy 107030 / 107027). Both
device_op call sites already pass Python lists built from a CPU tensor (`seq_lens_cpu`), so
this normally costs nothing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["fusion_attention"]

_NEG = -1.0e30
_INT32_MAX = 2147483647
_BIG = _INT32_MAX // 4


@triton.jit
def _fa_kernel(
    Q, K, V, OUT, TB, TS, QLEN, KVLEN, KVBASE, MASK,
    N_ROWS, N1, N2, GS, N_PROGS,
    KV_STRIDE, MASK_STRIDE, MASK_M,
    SCALE, J_HI_SLACK, J_LO_SLACK,
    USE_MASK: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_K: tl.constexpr,
):
    """One program walks whole (query token, query head) rows.

    Dense attention with an online (running max / running sum) fp32 softmax, so no score
    buffer is allocated and the arithmetic is the same as the AscendC two-pass form.
    `Q/K/V/OUT` are `[rows, N, D]` contiguous, so the element offset of (row, head) is
    `(row * N + head) * D`.
    """
    pid = tl.program_id(0)
    d = tl.arange(0, BLK_D)
    dm = d < D
    dc = tl.where(dm, d, 0)

    for row in range(pid, N_ROWS, N_PROGS):
        n1 = row % N1
        t = row // N1
        n2 = n1 // GS
        b = tl.load(TB + t)
        s1 = tl.load(TS + t)
        kvlen = tl.load(KVLEN + b)
        qlen = tl.load(QLEN + b)

        if HAS_CAUSAL:
            s1_abs = s1 + (kvlen - qlen)
        else:
            s1_abs = s1
        j_hi = s1_abs + J_HI_SLACK
        j_lo = s1_abs - J_LO_SLACK

        q = tl.load(Q + row * D + dc, mask=dm, other=0.0).to(tl.float32)

        m = tl.full([1], NEG, tl.float32)
        l = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLK_D], dtype=tl.float32)

        for c in range(0, tl.cdiv(kvlen, BLK_K)):
            j = c * BLK_K + tl.arange(0, BLK_K)
            vld = (j < kvlen) & (j <= j_hi) & (j >= j_lo)
            # Invalid lanes must not form an out-of-range address: a masked load still
            # computes its address (see sparse_flash_attention.py).
            js = tl.where(vld, j, 0)
            kvrow = tl.load(KVBASE + b) + js
            if USE_MASK:
                mv = tl.load(MASK + s1_abs * MASK_STRIDE + js,
                             mask=vld & (js < MASK_M), other=1)
                vld = vld & (mv == 0)
            base = (kvrow * N2 + n2).to(tl.int64)

            k = tl.load(K + base[:, None] * KV_STRIDE + dc[None, :],
                        mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) * SCALE
            s = tl.where(vld, s, NEG)

            m_new = tl.maximum(m, tl.max(s, axis=0))
            p = tl.where(vld, tl.exp(s - m_new), 0.0)
            alpha = tl.exp(m - m_new)
            v = tl.load(V + base[:, None] * KV_STRIDE + dc[None, :],
                        mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = m_new

        res = tl.where(l > 0.0, acc / l, tl.zeros([BLK_D], dtype=tl.float32))
        tl.store(OUT + row * D + dc, res.to(OUT.dtype.element_ty), mask=dm)


def _host_lengths(seq, name):
    """Return the sequence lengths as a 1-D int64 CPU tensor (see the capture note)."""
    if seq is None:
        return None
    if not isinstance(seq, torch.Tensor):
        return torch.as_tensor(seq, dtype=torch.int64).flatten()
    return seq.detach().to(torch.int64).flatten().cpu()


def _as_cumulative(raw, B, name, T):
    """Cumulative / per-sequence lengths -> cu_seqlens (length B+1)."""
    if raw.numel() == B + 1:
        cu = raw
    elif raw.numel() == B:
        if int(raw[-1]) == int(T):
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), raw])
        elif int(raw.sum()) == int(T):
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), raw.cumsum(0)])
        else:
            raise ValueError(
                f"{name}: length {raw.numel()} (=B) is neither cumulative (last={int(raw[-1])} "
                f"!= T={int(T)}) nor per-sequence lengths (sum={int(raw.sum())} != T={int(T)})")
    else:
        raise ValueError(f"{name} length must be B={B} or B+1={B + 1}, got {raw.numel()}")
    if int(cu[-1]) != int(T):
        raise ValueError(f"{name} last entry {int(cu[-1])} != T={int(T)}")
    return cu


def _kv_cumulative(raw, B, cache_rows):
    """KV lengths -> cu_seqlens; a length-B non-decreasing vector ending at the cache size is
    read as cumulative, anything else of length B as per-request lengths."""
    if raw.numel() == B + 1:
        cu = raw
    elif raw.numel() == B:
        nondec = bool(torch.all(raw[1:] >= raw[:-1])) if raw.numel() > 1 else True
        if nondec and int(raw[-1]) == int(cache_rows):
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), raw])
        else:
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), raw.cumsum(0)])
    else:
        raise ValueError(
            f"actual_seq_kvlen length must be B={B} or B+1={B + 1}, got {raw.numel()}")
    if int(cu[-1]) > int(cache_rows):
        raise ValueError(
            f"actual_seq_kvlen asks for {int(cu[-1])} KV positions but only {int(cache_rows)} "
            f"exist")
    return cu


def _head_dim(query, head_num, tnd):
    """Head dim from the query layout.

      * `[B, S, N, D]` (BSND)      -> D = shape[-1]
      * `[T, N, D]`   (TND)        -> D = shape[-1]   (when shape[1] == head_num)
      * `[*, H * D]`   (flattened) -> D = shape[-1] // head_num
    A 3-D `[B, S, N*D]` whose `S` happens to equal `head_num` cannot be told apart from
    `[T, N, D]`; it is read as TND and then rejected by `_as_heads` instead of being
    silently mis-indexed.
    """
    if query.dim() == 4:
        return int(query.shape[-1])
    if query.dim() == 3 and tnd and int(query.shape[1]) == head_num:
        return int(query.shape[-1])
    if int(query.shape[-1]) % head_num != 0:
        raise ValueError(
            f"query last dim {query.shape[-1]} is not a multiple of head_num={head_num}")
    return int(query.shape[-1]) // head_num


def _infer_kv_heads(key, head_dim, tnd):
    """KV head count from the key layout (this operator has no num_key_value_heads argument).

      * `[B, S2, N2, D]`   -> N2 = shape[2]
      * `[T_kv, N2, D]`    -> N2 = shape[1]              (TND, 3-D and last dim == D)
      * `[rows, N2 * D]`   -> N2 = shape[-1] // D
      * `[rows, D]`        -> N2 = 1
    """
    if key.dim() == 4:
        n2 = int(key.shape[2])
    elif key.dim() == 3 and tnd and int(key.shape[-1]) == head_dim:
        n2 = int(key.shape[1])
    elif key.dim() >= 2 and int(key.shape[-1]) % head_dim == 0:
        n2 = int(key.shape[-1]) // head_dim
    else:
        raise ValueError(
            f"key shape {tuple(key.shape)} is inconsistent with head_dim={head_dim}")
    if n2 <= 0:
        raise ValueError(f"key shape {tuple(key.shape)} implies num_kv_heads={n2}")
    return n2


def _as_heads(x, n_heads, head_dim, name):
    """Normalize a q/k/v tensor to `[rows, n_heads, head_dim]`."""
    if x.shape[-1] == n_heads * head_dim and (x.dim() < 2 or x.shape[-2] != n_heads):
        return x.contiguous().reshape(-1, n_heads, head_dim)
    if x.dim() >= 2 and x.shape[-2] == n_heads and x.shape[-1] == head_dim:
        return x.contiguous().reshape(-1, n_heads, head_dim)
    raise ValueError(
        f"{name} shape {tuple(x.shape)} is inconsistent with n_heads={n_heads} and "
        f"head_dim={head_dim}")


def fusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_num: int,
    *,
    input_layout: str = "BSH",
    pse: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    atten_mask: torch.Tensor | None = None,
    scale: float = 1.0,
    keep_prob: float = 1.0,
    pre_tockens: int = _INT32_MAX,
    next_tockens: int = _INT32_MAX,
    inner_precise: int = 0,
    sparse_mode: int = 0,
    actual_seq_qlen: torch.Tensor | None = None,
    actual_seq_kvlen: torch.Tensor | None = None,
    return_softmax_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton equivalent of `torch_npu.npu_fusion_attention` (minimal port).

    Returns `(attention_out, softmax_max, softmax_sum)`; the last two are empty placeholders
    because `return_softmax_lse` is not implemented and both call sites read only `[0]`.
    Keyword names follow the torch_npu schema, including its `pre_tockens` / `next_tockens`
    spelling, and the order used by the call sites in the module docstring.
    """
    # ---- arguments that are not implemented: fail with the actual value, never silently ----
    if pse is not None:
        raise NotImplementedError("pse is not None: position-encoding attention is not ported")
    if padding_mask is not None:
        raise NotImplementedError("padding_mask is not None: padding masks are not ported")
    if float(keep_prob) != 1.0:
        raise NotImplementedError(f"keep_prob={keep_prob}: dropout is not ported")
    if bool(return_softmax_lse):
        raise NotImplementedError(
            "return_softmax_lse=True is not ported: the two auxiliary softmax outputs are "
            "returned empty")
    if int(sparse_mode) not in (0, 2, 3, 4):
        raise NotImplementedError(
            f"sparse_mode={sparse_mode} is not implemented (0 = no causal limit, 2 = left-up "
            f"causal, 3 = right-down causal, 4 = band/sliding window)")
    if input_layout not in ("TND", "BSND", "BSH"):
        raise NotImplementedError(
            f"input_layout={input_layout!r} is not implemented (TND / BSND / BSH)")

    n1 = int(head_num)
    if n1 <= 0:
        raise ValueError(f"head_num={n1} must be positive")
    d = _head_dim(query, n1, input_layout == "TND")
    tnd = input_layout == "TND"
    if tnd:
        T = int(query.shape[0])
    else:
        T = int(query.shape[0]) * int(query.shape[1])
    q3 = _as_heads(query, n1, d, "query")
    if q3.shape[0] != T:
        raise ValueError(f"query normalizes to {q3.shape[0]} rows, expected {T}")

    raw_q = _host_lengths(actual_seq_qlen, "actual_seq_qlen")
    raw_kv = _host_lengths(actual_seq_kvlen, "actual_seq_kvlen")
    if tnd and raw_q is None:
        raise ValueError("input_layout=TND requires actual_seq_qlen (length B or B+1)")

    if tnd:
        if raw_q.numel() >= 2 and int(raw_q[0]) != 0 and int(raw_q[-1]) == T:
            # length-B cumulative prefix (device_op.py L757 `seq_lens_cpu.cumsum(0).tolist()`)
            B = raw_q.numel()
        elif raw_q.numel() >= 2 and int(raw_q[0]) == 0 and int(raw_q[-1]) == T:
            # cu_seqlens of length B+1 (a cu_seqlens always starts at 0)
            B = raw_q.numel() - 1
        else:
            # per-sequence lengths
            B = raw_q.numel()
    else:
        B = int(query.shape[0])
    # The KV head count is not an argument of this operator: it follows from the key layout
    # once the head dim is known from the query.
    n2 = _infer_kv_heads(key, d, tnd)
    if n1 % n2 != 0:
        raise ValueError(
            f"key shape {tuple(key.shape)} implies num_kv_heads={n2}, which does not divide "
            f"head_num={n1}")
    gs = n1 // n2
    cache_rows = int(key.numel()) // (n2 * d)
    k3 = _as_heads(key, n2, d, "key")
    v3 = _as_heads(value, n2, d, "value")
    if v3.shape != k3.shape:
        raise ValueError(f"value {tuple(value.shape)} must match key {tuple(key.shape)}")

    cu_q = _as_cumulative(raw_q, B, "actual_seq_qlen", T)
    lens = cu_q[1:] - cu_q[:-1]
    if not tnd and not bool(torch.all(lens == lens[0])):
        raise NotImplementedError(
            f"ragged query lengths {lens.tolist()} with input_layout=BSND are not implemented "
            f"(ragged batches use TND)")
    tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
    tok_s1 = (torch.arange(T, dtype=torch.int64)
              - cu_q[:-1].repeat_interleave(lens)).to(torch.int32)
    qlen_b = lens.to(torch.int32)

    # device_op.py L759-764 passes the SAME cumulative vector for `actual_seq_qlen` and
    # `actual_seq_kvlen` (self attention over the same tokens).  Detecting that makes the KV
    # convention unambiguous instead of relying on the value heuristic below.
    same_as_q = (raw_kv is not None and raw_q is not None
                 and raw_kv.numel() == raw_q.numel() and bool(torch.equal(raw_kv, raw_q)))
    if tnd:
        if raw_kv is None:
            if cache_rows % B != 0:
                raise ValueError(
                    f"no actual_seq_kvlen given and the {cache_rows} KV rows are not evenly "
                    f"divisible by batch {B}")
            cu_kv = torch.arange(0, (B + 1) * (cache_rows // B), step=(cache_rows // B),
                                 dtype=torch.int64)
        elif same_as_q:
            cu_kv = cu_q
        else:
            cu_kv = _kv_cumulative(raw_kv, B, cache_rows)
        kvlen_b = (cu_kv[1:] - cu_kv[:-1]).to(torch.int32)
        kv_base = cu_kv[:-1].to(torch.int32)
    else:
        if cache_rows % B != 0:
            raise ValueError(
                f"BSND key holds {cache_rows} KV rows which is not a multiple of batch {B}")
        s2 = cache_rows // B
        if raw_kv is None:
            kvlen_b = torch.full((B,), s2, dtype=torch.int32)
        elif same_as_q:
            kvlen_b = (cu_q[1:] - cu_q[:-1]).to(torch.int32)
        else:
            cu_kv = _kv_cumulative(raw_kv, B, cache_rows)
            kvlen_b = (cu_kv[1:] - cu_kv[:-1]).to(torch.int32)
        kv_base = torch.arange(B, dtype=torch.int32) * s2

    # ---- mask / window ----
    sm = int(sparse_mode)
    has_causal = sm in (2, 3, 4)
    # The slacks stay below 2**29 so that `s1_abs +- slack` cannot overflow int32 even when
    # s1_abs is temporarily negative (kvLen < qLen during chunked prefill).
    if sm == 4:
        j_hi_slack = min(int(next_tockens), _BIG)
        j_lo_slack = min(int(pre_tockens) - 1, _BIG)
    elif sm in (2, 3):
        j_hi_slack, j_lo_slack = 0, _BIG
    else:
        j_hi_slack, j_lo_slack = _BIG, _BIG
    use_mask = atten_mask is not None
    mask_stride, mask_m, mask_t = 0, 0, None
    if use_mask:
        if atten_mask.dim() != 2:
            raise NotImplementedError(
                f"atten_mask must be 2-D (the in-tree masks are [2048, 2048] int8), got "
                f"{tuple(atten_mask.shape)}")
        mask_t = atten_mask.contiguous()
        mask_m = int(mask_t.shape[0])
        mask_stride = int(mask_t.shape[1])
        need = int(kvlen_b.max())
        if mask_m < need or mask_stride < need:
            raise ValueError(
                f"atten_mask {tuple(mask_t.shape)} cannot cover {need} KV positions: the mask "
                f"is indexed as mask[query_abs, kv_pos] and needs both dims >= {need}")

    # ---- kernel ----
    dev = query.device
    out = torch.empty((T, n1, d), dtype=query.dtype, device=dev)
    q3 = q3.contiguous()
    k3 = k3.contiguous()
    v3 = v3.contiguous()
    mask_arg = mask_t if use_mask else out

    n_rows = T * n1
    n_progs = max(1, min(n_rows, 1024))
    _fa_kernel[(n_progs,)](
        q3, k3, v3, out,
        tok_b.to(dev), tok_s1.to(dev), qlen_b.to(dev), kvlen_b.to(dev), kv_base.to(dev),
        mask_arg,
        n_rows, n1, n2, gs, n_progs,
        d, mask_stride, mask_m,
        float(scale), j_hi_slack, j_lo_slack,
        USE_MASK=bool(use_mask), HAS_CAUSAL=bool(has_causal), NEG=_NEG,
        D=d, BLK_D=_next_pow2(d), BLK_K=32,
    )

    empty = torch.empty((0,), dtype=torch.float32, device=dev)
    return out.reshape(query.shape), empty, empty
