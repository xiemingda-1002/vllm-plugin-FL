# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""fused_infer_attention_score (FIA) -- AscendC -> Triton (dense MHA/GQA, TND + BSND/BSH)

Call sites (vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py):
    L104  (BaseDeviceAdaptor)  npu_fused_infer_attention_score(
                                   query=..., key=key.contiguous(), value=value.contiguous(),
                                   num_key_value_heads=..., num_heads=..., scale=..., **kwargs)
    L1121 (A5DeviceAdaptor)    the same call after the `head_size` routing check
`**kwargs` is forwarded unchanged, so the argument surface implemented here is the full FIA
surface the in-tree callers use. Those callers are in this repository and are the authority
for the argument names and for which combinations occur:

    vllm_fl/attention/ascend/attention.py L835-849 (eager prefill / chunked prefill)
        query, key, value, block_table, input_layout="TND", block_size,
        actual_seq_lengths, actual_seq_lengths_kv, num_key_value_heads, num_heads, scale,
        sparse_mode in {0, 3, 4}, atten_mask, pre_tokens, next_tokens
    vllm_fl/attention/ascend/attention.py L719-737 (FULL-graph replay, same shape of call)
    vllm_fl/dispatch/backends/vendor/ascend/impl/mm_encoder_attention.py L167-195
        query, key, value, atten_mask=None, block_table=None, input_layout="TND", block_size,
        actual_seq_lengths, actual_seq_lengths_kv, num_key_value_heads, num_heads, scale,
        sparse_mode=0
    => the return value is unpacked as `attn_output, _ = ...`, i.e. this operator returns
       `(attention_out, softmax_lse)`.

## Provenance of the semantics

There is no AscendC source for FIA in this repository (`csrc/ascend` has no
`fused_infer_attention_score` directory; a recursive search for the operator name over csrc/
hits only an unrelated CMakeLists.txt and the tiling_sink implementation). FIA is a CANN
built-in reached through torch_npu, so the semantics below are extracted from how the
repository *calls* it -- the layouts, the sequence-length conventions and the mask polarity
are all pinned by in-tree code:

  * `actual_seq_lengths` is a **cumulative** vector: attention.py L391
    `actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()` (prefix sums, last = T).
  * `actual_seq_lengths_kv` is the **per-request KV length**: attention.py L392
    `seq_lens_list = seq_lens.tolist()`.
  * `atten_mask` uses **1 = masked out**: standard_mask.py `get_splitfuse_attn_mask`
    builds `torch.triu(torch.ones(2048, 2048), diagonal=1)`, i.e. entry (i, j) is 1 exactly
    when j > i, which is the causal (right-down) mask.
  * causal masks are selected with `sparse_mode` (attention.py L830-834):
    `sparse_mode = 4 if self.sliding_window is not None else 3`, and mode 4 additionally
    passes `pre_tokens=sliding_window, next_tokens=0`.

## Extracted computation

Standard grouped-query attention, one output row per (query token, query head):

    b     = batch of token t, s1 = position of t inside its sequence
    n2    = n1 // (N1 / N2)
    valid(j) = (j <= j_hi) and (j >= j_lo) and (j < kvLen[b]) and (atten_mask[s1_abs, j] == 0)
    k_row(j) = paged   : block_table[b, j // blockSize] * blockSize + (j % blockSize)
               TND kv  : kvBase[b] + j        (kvBase = cuKvLen[b], one packed KV sequence)
               BSND kv : b * S2 + j
    score(j) = ( q[t, n1, :] . k[k_row(j), n2, :] ) * scale
    out[t,n1] = sum_j exp(score - max) * v[k_row(j), n2, :] / sum_j exp(score - max)
    softmax_lse[t, n1] = max + log(sum_j exp(score - max))

`sparse_mode` (the only values the in-tree call sites use):
    0  no causal limit                     -> j_hi = kvLen[b] - 1  (only a user mask, if any)
    2  left-up causal                      -> j_hi = s1
    3  right-down causal                   -> j_hi = s1 + (kvLen[b] - qLen[b])
    4  band (sliding window)               -> j_hi = s1 + (kvLen[b] - qLen[b]) + next_tokens
                                              j_lo = j_hi - pre_tokens + 1
`s1_abs` (the `atten_mask` row a query token reads) is `s1 + (kvLen[b] - qLen[b])` for modes
3 and 4 and `s1` otherwise, so that a right-aligned causal mask lines up with the query
position -- which is exactly the alignment `standard_mask.py` relies on.

## Implemented branches

* `input_layout` in {"TND", "BSND", "BSH"} for query/key/value (BSND and BSH are the same
  memory order: `[B, S, H, D]` and the flattened `[B, S, H*D]`; both are accepted and the
  output is returned with the query's shape)
* paged KV (`block_table` given, `key` = `[numBlocks, blockSize, N2, D]` or
  `[numBlocks, blockSize, N2*D]`, `block_size` inferred from `key.shape[1]` when 0) and flat
  KV (TND concatenated sequences, or BSND `[B, S2, N2, D]`)
* GQA (`num_heads % num_key_value_heads == 0`), with the head dim inferred from the query
  layout (`_head_dim`; a 3-D `[B, S, N*D]` whose `S` equals `num_heads` is read as TND and
  then rejected rather than silently mis-indexed)
* `sparse_mode` in {0, 2, 3, 4}; an optional 2-D int8/bool `atten_mask` (1 = masked out)
  applied **in addition** to the mode-derived mask
* `actual_seq_lengths` as a length-B cumulative vector, a length-B+1 cu_seqlens, or
  length-B per-sequence lengths that sum to T; `actual_seq_lengths_kv` as a length-B
  per-request vector, a length-B cumulative vector when it is non-decreasing and ends at the
  cache size, a length-B+1 cu_seqlens, or -- when it is bit-identical to
  `actual_seq_lengths` -- the same normalized form (self attention over the same tokens)
* `softmax_lse_flag` (returns `(attention_out, softmax_lse)`); fp32 accumulation, online
  softmax, no score buffer
* defaults for `num_heads` / `num_key_value_heads` / `scale` / `block_table` /
  `input_layout` follow the torch_npu schema

## Not covered (explicitly)

* `input_layout` in {"BNSD", "NTD", "TND_NTD", "NSD", ...} -- raises naming the value
* `sparse_mode` 1/5/6/7/8 (all-mask, prefix, global, dilated, block-local)
* `pse_shift`, `query_rope`/`key_rope` (MLA-shaped FIA), `key_shared_prefix`,
  `value_shared_prefix`, `actual_shared_prefix_len`, `left_padding_size`,
  `query_padding_size`, `kv_padding_size`, and every quantized-attention argument
  (`dequant_scale*`, `quant_scale*`, `quant_offset2`, `key_antiquant_*`,
  `value_antiquant_*`): a non-None / non-default value raises NotImplementedError naming it
* `v` head dim different from `q` head dim (raises)
* ragged query lengths with `BSND` (ragged batches are a TND feature; raises)
* padded queries: a `BSND` call whose per-request query lengths do not fill `T` raises
  rather than guessing which lanes are padding
* no Cube/Vector multi-core pipeline, no double buffering: one program walks whole
  (token, head) rows, so this is a reference implementation and not a speed match

## Capture note

The sequence lengths are needed on the host to build the token -> (batch, position) map and
the per-batch KV base. Device tensors are pulled with `.cpu()`, which is a device-to-host
copy and is **illegal inside ACL graph capture** (aclrtMemcpy 107030 / 107027). The in-tree
callers already hold host values (`attention.py` passes `tolist()` results), so this normally
costs nothing; `cpu_actual_seq_lengths*` accepts an explicit host mirror, mirroring
`sparse_flash_attention.py`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["fused_infer_attention_score"]

_NEG = -1.0e30
_INT32_MAX = 2147483647
_BIG = _INT32_MAX // 4


@triton.jit
def _fia_kernel(
    Q, K, V, OUT, LSE, BT, TB, TS, QLEN, KVLEN, KVBASE, MASK,
    N_ROWS, N1, N2, GS, N_PROGS,
    KV_STRIDE, BT_STRIDE, BLOCK_SIZE, MASK_STRIDE, MASK_M,
    SCALE, J_HI_SLACK, J_LO_SLACK,
    PA: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    USE_MASK: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_K: tl.constexpr,
):
    """One program walks whole (query token, query head) rows.

    The KV loop runs up to the per-batch KV length, so the trip count is data dependent; the
    tail lanes of the last block are masked out. Softmax is online (running max / running
    sum) with fp32 accumulation, algebraically identical to the two-pass AscendC form.
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

        # Absolute (right-aligned) query position: the atten_mask row this token reads and
        # the anchor of the mode-3 / mode-4 window.
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
            # Never let an invalid lane form an out-of-range address (see
            # sparse_flash_attention.py: a masked load still computes its address).
            js = tl.where(vld, j, 0)
            if PA:
                page = tl.load(BT + b * BT_STRIDE + js // BLOCK_SIZE, mask=vld, other=0)
                kvrow = page * BLOCK_SIZE + (js % BLOCK_SIZE)
            else:
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
        if RETURN_LSE:
            tl.store(LSE + row, tl.where(l > 0.0, m + tl.log(l), NEG))


def _host_lengths(seq, cpu_lengths, name):
    """Return the sequence lengths as a 1-D int64 CPU tensor (see the capture note)."""
    if cpu_lengths is not None:
        if not isinstance(cpu_lengths, torch.Tensor):
            raise TypeError(f"cpu_{name} must be a torch.Tensor, got {type(cpu_lengths).__name__}")
        if cpu_lengths.device.type != "cpu":
            raise RuntimeError(f"cpu_{name} is on {cpu_lengths.device} but must be a host tensor")
        if isinstance(seq, torch.Tensor) and seq.numel() != cpu_lengths.numel():
            raise RuntimeError(
                f"cpu_{name} has {cpu_lengths.numel()} entries but {name} has {seq.numel()}: "
                f"the host copy is stale")
        return cpu_lengths.detach().to(torch.int64).flatten()
    if seq is None:
        return None
    if not isinstance(seq, torch.Tensor):
        return torch.as_tensor(seq, dtype=torch.int64).flatten()
    return seq.detach().to(torch.int64).flatten().cpu()


def _query_cumulative(raw, B, T):
    """Query lengths -> cu_seqlens. A length-B vector is cumulative (attention.py L391)."""
    if raw.numel() == B + 1:
        cu = raw
    elif raw.numel() == B:
        if int(raw[-1]) == int(T):
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), raw])
        elif int(raw.sum()) == int(T):
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), raw.cumsum(0)])
        else:
            raise ValueError(
                f"actual_seq_lengths: length {raw.numel()} (=B) is neither cumulative "
                f"(last={int(raw[-1])} != T={int(T)}) nor per-sequence lengths "
                f"(sum={int(raw.sum())} != T={int(T)})")
    else:
        raise ValueError(
            f"actual_seq_lengths length must be B={B} or B+1={B + 1}, got {raw.numel()}")
    if int(cu[-1]) != int(T):
        raise ValueError(f"actual_seq_lengths last entry {int(cu[-1])} != T={int(T)}")
    return cu


def _kv_cumulative(raw, B, cache_rows):
    """KV lengths -> cu_seqlens.

    A length-B vector is the per-request KV length (attention.py L392 passes
    `seq_lens.tolist()`), unless it is non-decreasing and ends exactly at the number of KV
    rows in the cache, which is the cumulative form.
    """
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
            f"actual_seq_lengths_kv length must be B={B} or B+1={B + 1}, got {raw.numel()}")
    if int(cu[-1]) > int(cache_rows):
        raise ValueError(
            f"actual_seq_lengths_kv asks for {int(cu[-1])} KV positions but only "
            f"{int(cache_rows)} exist")
    return cu


def _head_dim(query, n_heads, tnd):
    """Head dim from the query layout.

      * `[B, S, N, D]` (BSND)      -> D = shape[-1]
      * `[T, N, D]`   (TND)        -> D = shape[-1]   (when shape[1] == n_heads)
      * `[*, H * D]`   (flattened) -> D = shape[-1] // n_heads
    A 3-D `[B, S, N*D]` whose `S` happens to equal `n_heads` cannot be told apart from
    `[T, N, D]`; it is read as TND and then rejected by `_as_heads` instead of being
    silently mis-indexed.
    """
    if query.dim() == 4:
        return int(query.shape[-1])
    if query.dim() == 3 and tnd and int(query.shape[1]) == n_heads:
        return int(query.shape[-1])
    if int(query.shape[-1]) % n_heads != 0:
        raise ValueError(
            f"query last dim {query.shape[-1]} is not a multiple of num_heads={n_heads}")
    return int(query.shape[-1]) // n_heads


def _as_heads(x, n_heads, head_dim, name):
    """Normalize a q/k/v tensor to `[rows, n_heads, head_dim]` (rows = tokens or pages)."""
    if x.shape[-1] == n_heads * head_dim and (x.dim() < 2 or x.shape[-2] != n_heads):
        return x.contiguous().reshape(-1, n_heads, head_dim)
    if x.dim() >= 2 and x.shape[-2] == n_heads and x.shape[-1] == head_dim:
        return x.contiguous().reshape(-1, n_heads, head_dim)
    raise ValueError(
        f"{name} shape {tuple(x.shape)} is inconsistent with n_heads={n_heads} and "
        f"head_dim={head_dim}")


def fused_infer_attention_score(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    pse_shift: torch.Tensor | None = None,
    atten_mask: torch.Tensor | None = None,
    actual_seq_lengths: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    dequant_scale1: torch.Tensor | None = None,
    quant_scale1: torch.Tensor | None = None,
    dequant_scale2: torch.Tensor | None = None,
    quant_scale2: torch.Tensor | None = None,
    quant_offset2: torch.Tensor | None = None,
    num_heads: int = 1,
    num_key_value_heads: int = 1,
    scale: float = 1.0,
    block_table: torch.Tensor | None = None,
    input_layout: str = "BSH",
    key_antiquant_mode: int = 0,
    value_antiquant_mode: int = 0,
    sparse_mode: int = 0,
    pre_tokens: int = _INT32_MAX,
    next_tokens: int = _INT32_MAX,
    block_size: int = 0,
    softmax_lse_flag: bool = False,
    cpu_actual_seq_lengths: torch.Tensor | None = None,
    cpu_actual_seq_lengths_kv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton equivalent of `torch_npu.npu_fused_infer_attention_score`.

    Returns `(attention_out, softmax_lse)`; `softmax_lse` is empty unless
    `softmax_lse_flag` is set. Keyword names and order follow the torch_npu FIA schema and
    the in-tree call sites listed in the module docstring.
    """
    # ---- arguments that are not implemented: fail with the actual value, never silently ----
    for nm, v in (("pse_shift", pse_shift), ("dequant_scale1", dequant_scale1),
                  ("quant_scale1", quant_scale1), ("dequant_scale2", dequant_scale2),
                  ("quant_scale2", quant_scale2), ("quant_offset2", quant_offset2)):
        if v is not None:
            raise NotImplementedError(
                f"{nm} is not None: quantized attention and position-encoded attention are "
                f"not implemented by this port")
    if int(key_antiquant_mode) != 0 or int(value_antiquant_mode) != 0:
        raise NotImplementedError(
            f"key_antiquant_mode={key_antiquant_mode}, value_antiquant_mode="
            f"{value_antiquant_mode}: antiquantized (quantized) KV is not implemented")
    if int(sparse_mode) not in (0, 2, 3, 4):
        raise NotImplementedError(
            f"sparse_mode={sparse_mode} is not implemented (0 = no causal limit, 2 = left-up "
            f"causal, 3 = right-down causal, 4 = band/sliding window)")
    if input_layout not in ("TND", "BSND", "BSH"):
        raise NotImplementedError(
            f"input_layout={input_layout!r} is not implemented (TND / BSND / BSH)")

    n1 = int(num_heads)
    n2 = int(num_key_value_heads)
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"num_heads={n1} and num_key_value_heads={n2} must be positive")
    if n1 % n2 != 0:
        raise ValueError(f"num_heads={n1} must be divisible by num_key_value_heads={n2}")
    gs = n1 // n2
    tnd = input_layout == "TND"

    # ---- layout normalization to token-major [rows, N, D] ----
    d = _head_dim(query, n1, tnd)
    if tnd:
        T = int(query.shape[0])
    else:
        T = int(query.shape[0]) * int(query.shape[1])
    q3 = _as_heads(query, n1, d, "query")
    if q3.shape[0] != T:
        raise ValueError(f"query normalizes to {q3.shape[0]} rows, expected {T}")

    raw_q = _host_lengths(actual_seq_lengths, cpu_actual_seq_lengths, "actual_seq_lengths")
    raw_kv = _host_lengths(actual_seq_lengths_kv, cpu_actual_seq_lengths_kv,
                           "actual_seq_lengths_kv")

    if block_table is not None:
        if key.dim() not in (3, 4):
            raise ValueError(
                f"paged key must be [numBlocks, blockSize, N2, D] or "
                f"[numBlocks, blockSize, N2*D], got {tuple(key.shape)}")
        bs = int(block_size) if int(block_size) > 0 else int(key.shape[1])
        if int(key.shape[1]) != bs:
            raise ValueError(f"block_size={bs} does not match key.shape[1]={key.shape[1]}")
        bt_width = int(block_table.shape[1])
        cache_rows = int(key.shape[0]) * bs
        B = int(block_table.shape[0])
        if raw_kv is None:
            raise ValueError("a paged KV (block_table given) requires actual_seq_lengths_kv")
    else:
        bs, bt_width = 0, 0
        if tnd:
            if raw_q is None:
                raise ValueError("input_layout=TND requires actual_seq_lengths (length B or B+1)")
            if raw_q.numel() >= 2 and int(raw_q[0]) != 0 and int(raw_q[-1]) == T:
                # length-B cumulative prefix (attention.py L391 `query_start_loc_cpu[1:]`)
                B = raw_q.numel()
            elif raw_q.numel() >= 2 and int(raw_q[0]) == 0 and int(raw_q[-1]) == T:
                # cu_seqlens of length B+1 (a cu_seqlens always starts at 0)
                B = raw_q.numel() - 1
            else:
                # per-sequence lengths
                B = raw_q.numel()
        else:
            B = int(query.shape[0])
        if int(key.numel()) % (n2 * d) != 0:
            raise ValueError(
                f"key shape {tuple(key.shape)} is inconsistent with "
                f"num_key_value_heads={n2} and head_dim={d}")
        cache_rows = int(key.numel()) // (n2 * d)

    k3 = _as_heads(key, n2, d, "key")
    v3 = _as_heads(value, n2, d, "value")
    if v3.shape != k3.shape:
        raise ValueError(f"value {tuple(value.shape)} must match key {tuple(key.shape)}")

    # ---- sequence lengths ----
    if raw_q is None:
        cu_q = torch.arange(0, (B + 1) * int(query.shape[1]), step=int(query.shape[1]),
                            dtype=torch.int64)
    else:
        cu_q = _query_cumulative(raw_q, B, T)
    lens = cu_q[1:] - cu_q[:-1]
    if not tnd and not bool(torch.all(lens == lens[0])):
        raise NotImplementedError(
            f"ragged query lengths {lens.tolist()} with input_layout=BSND are not "
            f"implemented (ragged batches use TND)")
    tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
    tok_s1 = (torch.arange(T, dtype=torch.int64)
              - cu_q[:-1].repeat_interleave(lens)).to(torch.int32)
    qlen_b = lens.to(torch.int32)

    # The same vector may be passed for query and KV lengths (self attention over the same
    # tokens, as in fusion_attention's call site); that makes the KV convention unambiguous.
    same_as_q = (raw_kv is not None and raw_q is not None
                 and raw_kv.numel() == raw_q.numel() and bool(torch.equal(raw_kv, raw_q)))
    if block_table is not None or tnd:
        if raw_kv is None:
            if cache_rows % B != 0:
                raise ValueError(
                    f"no actual_seq_lengths_kv given and the {cache_rows} KV rows are not "
                    f"evenly divisible by batch {B}")
            cu_kv = torch.arange(0, (B + 1) * (cache_rows // B), step=(cache_rows // B),
                                 dtype=torch.int64)
        elif same_as_q:
            cu_kv = cu_q
        else:
            cu_kv = _kv_cumulative(raw_kv, B, cache_rows)
        kvlen_b = (cu_kv[1:] - cu_kv[:-1]).to(torch.int32)
        kv_base = cu_kv[:-1].to(torch.int32) if block_table is None else torch.zeros(
            B, dtype=torch.int32)
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

    if int(kvlen_b.max()) > cache_rows:
        raise ValueError(
            f"a KV length of {int(kvlen_b.max())} exceeds the {cache_rows} available KV rows")

    # ---- mask / window ----
    sm = int(sparse_mode)
    has_causal = sm in (2, 3, 4)
    # The slacks are kept below 2**29 so that `s1_abs +- slack` cannot overflow int32 even
    # when s1_abs is temporarily negative (kvLen < qLen during chunked prefill).
    if sm == 4:
        j_hi_slack = min(int(next_tokens), _BIG)
        j_lo_slack = min(int(pre_tokens) - 1, _BIG)
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
                f"atten_mask {tuple(mask_t.shape)} cannot cover {need} KV positions: the "
                f"mask is indexed as mask[query_abs, kv_pos] and needs both dims >= {need}")

    # ---- kernel ----
    dev = query.device
    out = torch.empty((T, n1, d), dtype=query.dtype, device=dev)
    if softmax_lse_flag:
        lse = torch.empty((T * n1,), dtype=torch.float32, device=dev)
    else:
        lse = torch.empty((1,), dtype=torch.float32, device=dev)

    q3 = q3.contiguous()
    k3 = k3.contiguous()
    v3 = v3.contiguous()
    bt = block_table.contiguous().to(torch.int32) if block_table is not None else out
    mask_arg = mask_t if use_mask else out

    n_rows = T * n1
    n_progs = max(1, min(n_rows, 1024))
    _fia_kernel[(n_progs,)](
        q3, k3, v3, out, lse, bt,
        tok_b.to(dev), tok_s1.to(dev), qlen_b.to(dev), kvlen_b.to(dev), kv_base.to(dev),
        mask_arg,
        n_rows, n1, n2, gs, n_progs,
        d, bt_width, bs, mask_stride, mask_m,
        float(scale), j_hi_slack, j_lo_slack,
        PA=bool(block_table is not None), RETURN_LSE=bool(softmax_lse_flag),
        USE_MASK=bool(use_mask), HAS_CAUSAL=bool(has_causal), NEG=_NEG,
        D=d, BLK_D=_next_pow2(d), BLK_K=32,
    )

    if softmax_lse_flag:
        lse = lse.reshape(T, n1).contiguous()
    else:
        lse = torch.empty((0,), dtype=torch.float32, device=dev)
    return out.reshape(query.shape), lse
