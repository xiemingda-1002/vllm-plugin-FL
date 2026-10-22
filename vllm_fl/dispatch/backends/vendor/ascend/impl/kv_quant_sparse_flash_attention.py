# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""kv_quant_sparse_flash_attention -- AscendC -> Triton (TND/BSND query, PA_BSND/BSND packed KV)

Call site (vllm_fl/dispatch/backends/vendor/ascend/device/device_op.py L733-752,
`BaseDeviceAdaptor.execute_kv_quant_sparse_flash_attention`):
    npu_kv_quant_sparse_flash_attention(
        query=torch.cat([ql_nope, q_pe], dim=-1).contiguous(),   # [T, N1, 512 + 64]
        key=kv, value=kv,                                       # packed quantized cache
        sparse_indices=topk_indices, scale_value=sfa_impl.scale,
        sparse_block_size=1, block_table=..., actual_seq_lengths_query=...,
        actual_seq_lengths_kv=..., layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=3, attention_mode=2, quant_scale_repo_mode=1,
        tile_size=128, key_quant_mode=2, value_quant_mode=2, rope_head_dim=64)
    `kv` here is `kv_cache[0]` and its dtype is int8 / float8_e4m3fn / float8_e5m2
    (device_op.py L678-682); the caller only routes here for those dtypes.

## Provenance of the semantics (read this before trusting the formulas)

The AscendC kernel of THIS operator is not part of this repository: `csrc/ascend` has
`sparse_flash_attention`, `sparse_attn_sharedkv`, `lightning_indexer`, `mla_preprocess`, ...
but no `kv_quant_sparse_flash_attention` directory (verified by a recursive search over
csrc/ for the operator name: the only hits are a CMakeLists.txt and the tiling_sink
implementation, neither of which defines this operator). Like
`npu_fused_infer_attention_score` / `npu_fusion_attention`, it is a CANN built-in exposed
through torch_npu, so the arithmetic is recovered from two in-repository sources instead:

1. `csrc/ascend/attention/sparse_flash_attention/op_kernel/arch35/
   sparse_flash_attention_service_vector_mla.h` -- the non-quantized sibling, whose Triton
   port (`impl/sparse_flash_attention.py`) documents the MLA sparse-attention core:
       dSizeRope = 64                          (L790) -> the KV row is [dense | rope]
       softmaxScale = scaleValue               (L791)
       online softmax (L470-529), output = acc / sum (L572 LastDivNew)
       page addressing through block_table     (L266-310)
2. The **producer** of the packed cache in this same repository, which is the only
   definition of the packed byte layout that exists in-tree:
       vllm_fl/attention/ascend/sfa_v1.py L1777-1784
           packed_kv = torch.cat([k_nope, k_pe, knope_scale], dim=-1)
           kv_cache[0].view(-1, packed_head_dim) <- scatter(packed_kv)
       vllm_fl/attention/ascend/sfa_v1.py L1992-2028 `custom_kv_rmsnorm_rope`
           k_nope, knope_scale = npu_dynamic_block_quant(k_nope.view(-1, 1, kv_lora_rank),
                                                         dst_type=int8,
                                                         row_block_size=1,
                                                         col_block_size=tile_size)
           return (k_rope.view(int8), k_nope, knope_scale.to(float32).view(int8))
       vllm_fl/attention/ascend/utils.py L82-93 `get_sfa_qsfa_packed_head_dim`
           scale_metadata_bytes = (kv_lora_rank // tile_size) * 4
           packed_head_dim = kv_lora_rank + qk_rope_head_dim * 2 + scale_metadata_bytes

So one KV row is a byte string (kv_lora_rank=512, qk_rope_head_dim=64, tile_size=128):

    byte offset  0   .. 512           int8 (or fp8_e4m3) ckv latent, quantized
    byte offset  512 .. 640           bf16 rope key, RAW bf16 bytes (not quantized)
    byte offset  640 .. 656           4 x fp32 dequant scale, one per 128-wide block

    656 = 512 + 64*2 + (512/128)*4    <- exactly `get_sfa_qsfa_packed_head_dim(512, 64, 128)`

`npu_dynamic_block_quant(row_block_size=1, col_block_size=128)` is symmetric per-token,
per-128-column-block int8 quantization, so the dequantization is
`k_dense[d] = q_int8[d] * scale[d // 128]` with no zero point. The rope part is stored as
raw bf16 and must NOT be dequantized; that is why this module loads the packed tensor as
bytes (int8 view) and only the dense block goes through a scale.

## Extracted computation (implemented here)

Let b = batch of token t, s1 = the token's position inside its sequence, n1 the query head,
n2 = n1 // (N1 / N2) the KV head. With `sparse_mode=3` (right-down causal) the causal limit is
the same one the non-quantized sibling uses:

    j_hi = s1 + (kvLen[b] - qLen[b])

For every index i0 in `sparse_indices[t, n2, :]` (K = 2048 in production, `-1` = invalid):

    valid(i0) = (i0 >= 0) and (i0 <= j_hi) and (i0 < kvLen[b])
    row(i0)   = PA_BSND: block_table[b, i0 // blockSize] * blockSize + (i0 % blockSize)
                BSND   : b * S2 + i0
    q_nope    = query[t, n1, 0:nope]            (fp16/bf16, not quantized)
    q_rope    = query[t, n1, nope:nope+rope]    (fp16/bf16)
    k_dense   = dequant(cache[row])             (nope wide, fp32)
    k_rope    = bf16(cache bytes [nope, nope+2*rope])
    score     = ( q_nope . k_dense + q_rope . k_rope ) * scale_value
    out[t,n1] = sum_valid exp(score - max) * k_dense  /  sum_valid exp(score - max)

The value tensor is the *same* dequantized dense block (`key` and `value` are the same
packed buffer at the call site, which is the MLA property: the ckv latent is both K and V).
The result is `[T, N1, nope]` in the query dtype, matching what `_v_up_proj` consumes
(sfa_v1.py L1965).

## Implemented branches

* `layout_query` in {"TND", "BSND"}; `layout_kv` in {"PA_BSND", "BSND"}
* `sparse_mode` in {0 (no causal limit), 3 (right-down causal)}
* packed KV in int8 (the A2/A3 production path) and fp8_e4m3fn / fp8_e5m2 (A5 path,
  decoded exactly to fp16 on the host -- see the cost note below)
* `sparse_block_size == 1`, `attention_mode == 2`, and
  `(quant_scale_repo_mode, key_quant_mode, value_quant_mode) == (1, 2, 2)`;
  every other value raises NotImplementedError naming the value
* `actual_seq_lengths_query` as a length-B cumulative vector (the convention of
  attention.py L391), a length-B+1 cu_seqlens, or length-B per-sequence lengths that sum to
  T; `actual_seq_lengths_kv` as length-B per-request lengths (attention.py L392), a length-B
  cumulative vector when it is non-decreasing and ends at the cache size, or a length-B+1
  cu_seqlens
* fp32 accumulation, online softmax (single pass, no score buffer)
* returns a single tensor (device_op.py L684 returns it directly)

## Not covered (explicitly)

* `sparse_block_size > 1` (each index expanding into several KV rows) -- the call site
  always passes 1
* `sparse_mode` 1/2/4/5/6/7/8, `attention_mode` != 2, `layout_query="TND_NTD"`
* attention sinks, `return_softmax_lse` (the operator exposes no such output here)
* `quant_scale_repo_mode`/`key_quant_mode`/`value_quant_mode` combinations other than
  (1, 2, 2). The numeric meanings of those three enums are **not documented in this
  repository** and the CANN documentation was not reachable from the working machine; the
  combination above is the only one the call site uses, and the byte layout it implies is
  the one the in-tree producer writes (`custom_kv_rmsnorm_rope` +
  `get_sfa_qsfa_packed_head_dim`). Anything else is rejected rather than guessed.
* no Cube/Vector multi-core pipeline and no double buffering: one program walks whole
  (token, head) rows, so this is a reference implementation, not a speed match

## Cost / capture notes

* The dense block is read **in place** from the packed cache with a byte row stride, so the
  int8 path allocates no extra KV-sized buffer. The fp8 path decodes the dense block to
  fp16 (an exact conversion: fp8 e4m3/e5m2 are representable in fp16) which does allocate
  one `rows * nope * 2` byte temporary; it exists because the rope part of the same byte
  row is raw bf16 and cannot go through a dtype conversion as a whole.
* The rope and scale slices are copied out (`rows * (2*rope + 4*nblk)` bytes), because a
  dtype reinterpretation of a strided slice is not possible.
* `actual_seq_lengths_*` are read on the host to build the token -> (batch, position) map.
  A device tensor is moved with `.cpu()`, which is a device-to-host copy and is **illegal
  during ACL graph capture** (aclrtMemcpy 107030 / "Not allow to synchronize captured
  stream" 107027). The optional `cpu_actual_seq_lengths_query` / `cpu_actual_seq_lengths_kv`
  arguments take the host mirror and are preferred under capture; this mirrors
  `sparse_flash_attention.py` (`_host_lengths`).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["kv_quant_sparse_flash_attention"]

_NEG = -1.0e30
_FP8_DTYPES = tuple(
    d for d in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
    if d is not None
)


@triton.jit
def _kvq_sfa_kernel(
    Q, KQ, KROPE, KSCL, IDX, BT, OUT,
    TOK_B, TOK_S1, QLEN, KVLEN,
    N_ROWS, N1, N2, GS, K_IDX, N_PROGS,
    KQ_STRIDE, KSCL_STRIDE, BT_STRIDE, BLOCK_SIZE, S2,
    SCALE,
    PA: tl.constexpr,
    CAUSAL: tl.constexpr,
    NEG: tl.constexpr,
    D: tl.constexpr,
    DR: tl.constexpr,
    BLK_D: tl.constexpr,
    BLK_R: tl.constexpr,
    BLK_K: tl.constexpr,
    TILE: tl.constexpr,
    NSC: tl.constexpr,
):
    """One program walks whole (token, query head) rows of the output.

    Row = one (token, head). The KV gather is driven by `sparse_indices`, so the trip count
    is `cdiv(K_IDX, BLK_K)` and does not depend on the sequence length. Softmax is online
    (running max / running sum) with fp32 accumulation, which is algebraically identical to
    the AscendC two-pass form and needs no score buffer.
    """
    pid = tl.program_id(0)
    d = tl.arange(0, BLK_D)
    dm = d < D
    # Every lane that reaches address arithmetic is clamped into its legal range: a masked
    # load on Ascend still forms the address, so a padded lane must not point outside the
    # buffer (same rule as sparse_flash_attention.py / sparse_attn_sharedkv.py).
    dc = tl.where(dm, d, 0)
    dr = tl.arange(0, BLK_R)
    drm = dr < DR
    drc = tl.where(drm, dr, 0)
    # Which per-block dequant scale each dense column uses (col_block_size = TILE).
    blk = tl.minimum(tl.where(dm, d, 0) // TILE, NSC - 1)
    qstride = D + DR

    for row in range(pid, N_ROWS, N_PROGS):
        n1 = row % N1
        t = row // N1
        n2 = n1 // GS
        b = tl.load(TOK_B + t)
        s1 = tl.load(TOK_S1 + t)

        qn = tl.load(Q + row * qstride + dc, mask=dm, other=0.0).to(tl.float32)
        qr = tl.load(Q + row * qstride + D + drc, mask=drm, other=0.0).to(tl.float32)

        kvlen = tl.load(KVLEN + b)
        if CAUSAL:
            j_hi = s1 + (kvlen - tl.load(QLEN + b))
        else:
            j_hi = S2

        m = tl.full([1], NEG, tl.float32)
        l = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLK_D], dtype=tl.float32)

        for c in range(0, tl.cdiv(K_IDX, BLK_K)):
            slot = c * BLK_K + tl.arange(0, BLK_K)
            sm = slot < K_IDX
            i0 = tl.load(IDX + (t * N2 + n2) * K_IDX + slot, mask=sm, other=-1)
            # Every index that reaches division/modulo/address arithmetic is clamped to a
            # non-negative value first; `vld` alone decides whether the lane contributes.
            # (Same rule as sparse_flash_attention.py: a masked load still forms an address,
            # and an out-of-range address is a hard device error, not a discarded load.)
            vld = sm & (i0 >= 0) & (i0 <= j_hi) & (i0 < kvlen)
            j = tl.where(vld, i0, 0)
            if PA:
                page = tl.load(BT + b * BT_STRIDE + j // BLOCK_SIZE, mask=vld, other=0)
                kvrow = page * BLOCK_SIZE + (j % BLOCK_SIZE)
            else:
                kvrow = b * S2 + j
            base = (kvrow * N2 + n2).to(tl.int64)

            kq = tl.load(KQ + base[:, None] * KQ_STRIDE + dc[None, :],
                         mask=vld[:, None] & dm[None, :], other=0).to(tl.float32)
            ks = tl.load(KSCL + base[:, None] * KSCL_STRIDE + blk[None, :],
                         mask=vld[:, None] & dm[None, :], other=0.0).to(tl.float32)
            kd = kq * ks
            kr = tl.load(KROPE + base[:, None] * DR + drc[None, :],
                         mask=vld[:, None] & drm[None, :], other=0.0).to(tl.float32)

            s = tl.sum(kd * qn[None, :], axis=1) + tl.sum(kr * qr[None, :], axis=1)
            s = tl.where(vld, s * SCALE, NEG)

            m_new = tl.maximum(m, tl.max(s, axis=0))
            p = tl.where(vld, tl.exp(s - m_new), 0.0)
            alpha = tl.exp(m - m_new)
            acc = acc * alpha + tl.sum(p[:, None] * kd, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = m_new

        res = tl.where(l > 0.0, acc / l, tl.zeros([BLK_D], dtype=tl.float32))
        tl.store(OUT + row * D + dc, res.to(OUT.dtype.element_ty), mask=dm)


def _host_lengths(seq, cpu_lengths, name):
    """Return the sequence lengths as a 1-D int64 CPU tensor.

    A device tensor is pulled with `.cpu()`, which is a device-to-host copy: correct in eager
    mode, illegal while an ACL graph is being captured. Callers that run under capture must
    pass the host mirror as `cpu_<name>`; the check is explicit so the hazard is visible.
    """
    if cpu_lengths is not None:
        if not isinstance(cpu_lengths, torch.Tensor):
            raise TypeError(f"cpu_{name} must be a torch.Tensor, got {type(cpu_lengths).__name__}")
        if cpu_lengths.device.type != "cpu":
            raise RuntimeError(
                f"cpu_{name} is on {cpu_lengths.device} but must be a host tensor")
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


def _as_cumulative(seq, B, name, T_expected=None):
    """Normalize `actual_seq_lengths*` into cu_seqlens (length B+1, last entry = total).

    Accepted conventions, disambiguated from the values when both are possible:
      * length B+1: already cu_seqlens;
      * length B  : cumulative exclusive prefix when `last == T_expected`, otherwise
                    per-sequence lengths (cumsum) when `sum == T_expected`.
    With `T_expected is None` a length-B vector is read as **cumulative**, which is the
    convention the query lengths use at the call sites (attention.py L391).
    """
    if seq is None:
        return None
    if seq.device.type != "cpu":
        raise RuntimeError(f"{name} reached _as_cumulative on {seq.device}; move it to the host first")
    s = seq.detach().to(torch.int64).flatten()
    if s.numel() == B + 1:
        cu = s
    elif s.numel() == B:
        if T_expected is None:
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), s])
        elif int(s[-1]) == int(T_expected):
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), s])
        elif int(s.sum()) == int(T_expected):
            cu = torch.cat([torch.zeros(1, dtype=torch.int64), s.cumsum(0)])
        else:
            raise ValueError(
                f"{name}: length {s.numel()} (=B) is neither cumulative (last={int(s[-1])} "
                f"!= T={int(T_expected)}) nor per-sequence lengths (sum={int(s.sum())} "
                f"!= T={int(T_expected)})")
    else:
        raise ValueError(
            f"{name} length must be B={B} (cumulative or per-sequence) or B+1={B + 1} "
            f"(cu_seqlens), got {s.numel()}")
    if T_expected is not None and int(cu[-1]) != int(T_expected):
        raise ValueError(f"{name} last entry {int(cu[-1])} != T={int(T_expected)}")
    return cu


def _kv_lengths_to_cumulative(raw, B, cache_rows):
    """KV lengths -> cu_seqlens.

    A length-B vector is the **per-request** KV length (attention.py L392 passes
    `seq_lens.tolist()`, and the SFA port consumes the same metadata), unless it is
    non-decreasing and ends exactly at the number of KV rows in the cache, which is the
    cumulative form.
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


def _unpack_packed_kv(packed, nope_dim, rope_dim, tile_size):
    """Split the packed byte row `[dense int8 | rope bf16 | scale fp32]`.

    Returns `(k_dense, k_rope, k_scale, dense_stride)` where
      * `k_dense` is 2-D `[rows * N2, dense_stride]`; for int8 it is a **view of the packed
        buffer** (`dense_stride == row_bytes`, only the first `nope_dim` columns are read),
        for fp8 it is an exact fp16 decode (`dense_stride == nope_dim`);
      * `k_rope` is 2-D `[rows * N2, rope_dim]` bf16 (a copy of the raw bf16 bytes);
      * `k_scale` is 2-D `[rows * N2, nope_dim // tile_size]` fp32 (a copy).
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if nope_dim % tile_size != 0:
        raise ValueError(
            f"packed KV: nope dim {nope_dim} must be divisible by tile_size {tile_size}")
    n_scale = nope_dim // tile_size
    rope_bytes = rope_dim * 2
    row_bytes = nope_dim + rope_bytes + n_scale * 4
    if packed.shape[-1] != row_bytes:
        raise ValueError(
            f"packed KV row has {packed.shape[-1]} bytes but [dense {nope_dim} | rope "
            f"{rope_dim}*2 | scale {n_scale}*4] needs {row_bytes}")
    if packed.dim() < 2:
        raise ValueError(f"packed KV must have at least 2 dims, got {tuple(packed.shape)}")
    if not packed.is_contiguous():
        packed = packed.contiguous()

    rows = int(packed.numel()) // row_bytes
    raw = packed.view(torch.int8).reshape(rows, row_bytes)
    # The rope bytes are raw bf16 and the scale bytes are raw fp32, so both slices are
    # copied out of the byte row and then reinterpreted.
    k_rope = raw[:, nope_dim:nope_dim + rope_bytes].contiguous().view(torch.bfloat16)
    k_scale = raw[:, nope_dim + rope_bytes:].contiguous().view(torch.float32)
    if packed.dtype in _FP8_DTYPES:
        # fp8 e4m3 / e5m2 are exactly representable in fp16, so this decode loses nothing.
        k_dense = packed[..., :nope_dim].contiguous().reshape(rows, nope_dim).to(torch.float16)
        dense_stride = nope_dim
    elif packed.dtype == torch.int8:
        # No copy: the dense block is read in place with the packed byte row as the stride.
        k_dense = raw
        dense_stride = row_bytes
    else:
        raise NotImplementedError(
            f"packed KV dtype {packed.dtype} is not supported (int8 / float8_e4m3fn / "
            f"float8_e5m2)")
    return k_dense, k_rope, k_scale, dense_stride


def kv_quant_sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float = 1.0,
    sparse_block_size: int = 1,
    sparse_mode: int = 3,
    attention_mode: int = 2,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    quant_scale_repo_mode: int = 1,
    tile_size: int = 128,
    key_quant_mode: int = 2,
    value_quant_mode: int = 2,
    rope_head_dim: int = 64,
    cpu_actual_seq_lengths_query: torch.Tensor | None = None,
    cpu_actual_seq_lengths_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton equivalent of `torch_npu.npu_kv_quant_sparse_flash_attention`.

    Signature matches the call site in device_op.py L733-752 argument for argument (the
    keyword names, their order and their values). Returns the attention output only, which
    is what the call site returns to its caller.

    See the module docstring for the extracted computation, the packed-byte layout and the
    list of branches that are not covered.
    """
    if int(sparse_block_size) != 1:
        raise NotImplementedError(
            f"sparse_block_size={sparse_block_size} is not implemented (only 1; a value > 1 "
            f"expands every index into several KV rows)")
    if int(attention_mode) != 2:
        raise NotImplementedError(
            f"attention_mode={attention_mode} is not implemented (2 is the MLA mode used by "
            f"the call site)")
    got_modes = (int(quant_scale_repo_mode), int(key_quant_mode), int(value_quant_mode))
    if got_modes != (1, 2, 2):
        raise NotImplementedError(
            f"(quant_scale_repo_mode, key_quant_mode, value_quant_mode)={got_modes} is not "
            f"implemented: only the (1, 2, 2) packed-cache combination that the call site "
            f"passes is covered")
    if int(sparse_mode) not in (0, 3):
        raise NotImplementedError(
            f"sparse_mode={sparse_mode} is not implemented (0 = no causal limit, "
            f"3 = right-down causal)")
    if layout_query not in ("TND", "BSND"):
        raise NotImplementedError(f"unsupported layout_query={layout_query!r} (TND/BSND)")
    if layout_kv not in ("PA_BSND", "BSND"):
        raise NotImplementedError(f"unsupported layout_kv={layout_kv!r} (PA_BSND/BSND)")
    if value.data_ptr() != key.data_ptr() or tuple(value.shape) != tuple(key.shape):
        raise NotImplementedError(
            "value must be the same packed tensor as key: the packed cache stores one "
            "quantized latent that serves as both K and V (MLA), so a distinct value buffer "
            "would need its own layout")
    if sparse_indices.dtype != torch.int32:
        raise ValueError(f"sparse_indices must be int32, got {sparse_indices.dtype}")

    rope = int(rope_head_dim)
    if query.dim() != 3:
        raise ValueError(f"query must be 3-D for this implementation, got {tuple(query.shape)}")
    if query.shape[-1] <= rope:
        raise ValueError(
            f"query last dim {query.shape[-1]} must be larger than rope_head_dim {rope}")
    nope = int(query.shape[-1]) - rope
    tnd = layout_query == "TND"
    pa = layout_kv == "PA_BSND"

    # ---------------- layout normalization (bookkeeping only) ----------------
    if tnd:
        T, N1, _ = query.shape
        if actual_seq_lengths_query is None:
            raise ValueError("layout_query=TND requires actual_seq_lengths_query (length B or B+1)")
        raw_q = _host_lengths(actual_seq_lengths_query, cpu_actual_seq_lengths_query,
                              "actual_seq_lengths_query")
        if block_table is not None:
            B = int(block_table.shape[0])
        elif raw_q.numel() >= 2 and int(raw_q[0]) != 0 and int(raw_q[-1]) == T:
            # length-B cumulative prefix (attention.py L391 `query_start_loc_cpu[1:]`)
            B = raw_q.numel()
        elif raw_q.numel() >= 2 and int(raw_q[0]) == 0 and int(raw_q[-1]) == T:
            # cu_seqlens of length B+1 (a cu_seqlens always starts at 0)
            B = raw_q.numel() - 1
        else:
            # per-sequence lengths
            B = raw_q.numel()
        cu_q = _as_cumulative(raw_q, B, "actual_seq_lengths_query", T_expected=T)
        lens = cu_q[1:] - cu_q[:-1]
        tok_b = torch.repeat_interleave(torch.arange(B, dtype=torch.int32), lens.to(torch.int32))
        tok_s1 = (torch.arange(T, dtype=torch.int64)
                  - cu_q[:-1].repeat_interleave(lens)).to(torch.int32)
        qlen_b = lens.to(torch.int32)
        S1 = None
    else:
        B, S1, N1, _ = query.shape
        T = B * S1
        tok_b = torch.arange(B, dtype=torch.int32).repeat_interleave(S1)
        tok_s1 = torch.arange(S1, dtype=torch.int32).repeat(B)
        qlen_b = torch.full((B,), S1, dtype=torch.int32)

    dev = query.device
    if pa:
        if block_table is None:
            raise ValueError("layout_kv=PA_BSND requires block_table")
        if key.dim() != 4:
            raise ValueError(
                f"PA_BSND packed KV must be [numBlocks, blockSize, N2, rowBytes], got "
                f"{tuple(key.shape)}")
        num_blocks, block_size, N2, _ = key.shape
        bt_width = int(block_table.shape[1])
        S2 = int(block_size) * bt_width
    else:
        if key.dim() != 4:
            raise ValueError(
                f"BSND packed KV must be [B, S2, N2, rowBytes], got {tuple(key.shape)}")
        Bk, S2, N2, _ = key.shape
        if int(Bk) != B:
            raise ValueError(f"key batch {Bk} != query batch {B}")
        block_size, bt_width = 1, 0

    kvlen_raw = None
    if actual_seq_lengths_kv is not None or cpu_actual_seq_lengths_kv is not None:
        kvlen_raw = _host_lengths(actual_seq_lengths_kv, cpu_actual_seq_lengths_kv,
                                  "actual_seq_lengths_kv")
    if kvlen_raw is None:
        kvlen_b = torch.full((B,), S2, dtype=torch.int32)
    else:
        cu_kv = _kv_lengths_to_cumulative(kvlen_raw, B, S2)
        kvlen_b = (cu_kv[1:] - cu_kv[:-1]).to(torch.int32)
        if int(kvlen_b.max()) > S2:
            raise ValueError(
                f"actual_seq_lengths_kv asks for {int(kvlen_b.max())} KV positions but the "
                f"cache only holds {S2}")

    if N1 % N2 != 0:
        raise ValueError(f"N1={N1} must be divisible by N2={N2}")
    gs = N1 // N2
    k_idx = int(sparse_indices.shape[-1])

    k_dense, k_rope, k_scale, dense_stride = _unpack_packed_kv(key, nope, rope, int(tile_size))
    n_scale = int(k_scale.shape[-1])

    idx = sparse_indices.contiguous().reshape(T, N2, k_idx)
    q = query.contiguous().reshape(T, N1, nope + rope)
    bt = block_table.contiguous().to(torch.int32) if pa else key
    out = torch.empty((T, N1, nope), dtype=query.dtype, device=dev)

    tok_b = tok_b.to(dev)
    tok_s1 = tok_s1.to(dev)
    kvlen_d = kvlen_b.to(dev)
    qlen_d = qlen_b.to(dev)

    n_rows = T * N1
    n_progs = max(1, min(n_rows, 1024))
    blk_d = _next_pow2(nope)
    blk_r = _next_pow2(rope)
    blk_k = 16
    _kvq_sfa_kernel[(n_progs,)](
        q, k_dense, k_rope, k_scale, idx, bt, out,
        tok_b, tok_s1, qlen_d, kvlen_d,
        n_rows, N1, N2, gs, k_idx, n_progs,
        dense_stride, n_scale, bt_width, block_size, S2,
        float(scale_value),
        PA=bool(pa), CAUSAL=bool(int(sparse_mode) == 3), NEG=_NEG,
        D=nope, DR=rope, BLK_D=blk_d, BLK_R=blk_r, BLK_K=blk_k, TILE=int(tile_size),
        NSC=n_scale,
    )
    if tnd:
        return out.reshape(T, N1, nope)
    return out.reshape(B, S1, N1, nope)
