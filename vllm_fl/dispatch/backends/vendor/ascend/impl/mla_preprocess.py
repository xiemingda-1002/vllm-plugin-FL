# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""mla_preprocess -- AscendC to Triton

Source: csrc/ascend/mla_preprocess/
Interface (mla_preprocess_torch_adpt.h L33-42 + op_kernel/mla_preprocess_kernel.cpp L23-28):
    Input : hiddenState  [N, hidden]                  bf16/fp16
            wdqkv        [qLora+kv+rope, hidden]      bf16   (transB matmul)
            gamma1       [qLora]                     bf16
            wuq          [headNum*(nope+rope), qLora] bf16  (transB matmul)
            gamma2       [kvLora]                    bf16
            wuk          [headNum, nope, kvLora]     bf16
            cos, sin     [N, rope]                   bf16
            slotMapping  [N]                         int32
            kv_cache / kv_cache_rope                 bf16 (input side, unused here)
    Output: q            [N, headNum, kvLora(+rope)]  bf16
            q2           [N*headNum, rope]            bf16 (when cache_mode="krope_ctkv")
            kv_cache_out0 / kv_cache_out1             bf16
            inner_out                                 (enable_inner_out, not covered here)
    Attr  : cache_mode in {kvcache, krope_ctkv, int8_nzcache, nzcache}
            quant_mode in {per_tensor_quant_asymm, per_token_quant_symm,
                           per_token_quant_asymm, no_quant}
            enable_inner_out

This file implements the **quant_mode="no_quant" bf16/fp16 path**
(op_kernel/mla_preprocess_mix_bf16_nq.hpp, tiling key KEY_BF16_CACHEMODE_1_QUANTMODE_3=281),
the only path without an int8 quantization branch.  The quantized paths (mix_bf16 / mix_fp16 /
mix_bf16_qdown) are not implemented (see "Simplifications" at the end of this docstring).

Formula (from mla_preprocess_mix_bf16_nq.hpp)
    ProcessCube (L1195-1211):
        mm1 : s3 = hiddenState @ wdqkv^T                    # [N, qLora+kv+rope]
                                                              (L972 mm_1.Init, transB=true)
        mm2 : s2 = rmsnorm_q(s3[:, kv+rope:], gamma1) @ wuq^T # [N, headNum*(nope+rope)]
                                                              (L1001-1003 rmsNormQuant2 + L973 mm_2)
        mm3 : q_nope = einsum('thk,hkn->thn', s2[:, :, :nope], wuk)
                                                              (L974 mm_ein_sum, ein=true)
    RmsNormQuant (L287-407):
        avg = 1/qLora;  ss = sum(x^2) * avg;  y = x / sqrt(ss + eps) * gamma1   # eps=1e-6
        When beta1 is non-empty, `+ beta1` is applied as well (AscendC `kernel_utils.h`
        L236-254 `ComputeRmsNorm<WITH_BETA>`: `Add(out,out,beta)` after `Mul(out,out,gamma)`;
        beta does **not** take part in the normalization).
        Call site: `beta1 = q_a_layernorm.bias` (`sfa_v1.py:877`), shaped like gamma1, `[qLora]`.
        (L375 Muls(avg) -> L377 ReduceSumCustom -> L379 Adds(eps) -> L381 Sqrt
          -> L384 factor=1/sqrt -> L387 Muls -> L390 Mul(g))
    RmsNormAndRopeConvergence1 (L1029-1134): rmsnorm over s3[:, 0:kv] (gamma2, eps=1e-6) and
        RoPE (HALF rotation) over s3[:, kv:kv+rope]:
        (L1081 rms = sqrt(sum(x^2)/kv + eps); L1094-1116 RoPE)
            revert[0:h]  = x[h:2h]
            revert[h:2h] = x[0:h]
            k = cos*x ;  revert = sin*revert ;  revert *= [-1]*h+[+1]*h ;  out = k + revert
        i.e. out[i]      = cos[i]  * x[i]     - sin[i]  * x[i+h]
             out[h + i] = cos[h+i]* x[h+i]   + sin[h+i]* x[i]
    RopeFp16 (L27-248): the same HALF RoPE over s2[:, :, nope:] (cos2/sin2 indexed by token);
        depending on cache_mode the result goes into the q row at offset concatSize
        (CACHE_MODE_KVCACHE) or into a separate q2 (CACHE_MODE_KROPE_CTKV).
    cache write-back (L1122-1130):
        CACHE_MODE_KVCACHE(0): kv_cache_out0[slot, 0:kv+rope] = [rmsnorm_kv | rope_k]
        CACHE_MODE_KROPE_CTKV(1): kv_cache_out0[slot, 0:kv] = rmsnorm_kv
                                  kv_cache_out1[slot, 0:rope] = rope_k
    q layout: CACHE_MODE_KVCACHE -> [N, headNum, kv+rope] (mm3 writes nope into the first kv
            columns and RopeFp16 writes rope at offset +kv); CACHE_MODE_KROPE_CTKV -> [N, headNum,
            kv] with rope written separately into q2[N*headNum, rope].

Simplifications (branches explicitly not covered)
    1. quant_mode supports "no_quant" (the original nq path) and "per_tensor_quant_asymm" (the
       int8 quantization path).  per_token_quant_symm(1) / per_token_quant_asymm(2) are not
       implemented.
    2. cache_mode supports "kvcache" / "krope_ctkv" / "nzcache"; "int8_nzcache"(2) and its
       EinSumQuant(q_nope_scale) are not implemented.
    3. enable_inner_out / inner_out are not implemented; the fp16 quantization branch
       (mix_fp16.hpp) is not implemented.
    4. The input-side kv_cache / kv_cache_rope take no part in the computation (matching the
       kernel: written only, never read).
    5. Every uncovered parameter raises an explicit NotImplementedError; nothing is silently
       ignored.

================================================================
The full formula and sources of the int8 quantization path (per_tensor_quant_asymm) are in the
module docstring of mla_preprocess_int8.py; the orchestration is in
mla_preprocess_int8.mla_preprocess_quant().

================================================================
Backend shape pitfalls (910_93 + BiShengIR/FlagTree, reproduced)
================================================================
1. "2-D block + 2-D masked store + indirect (runtime-loaded) row/column addressing" **silently
   writes wrong values** on this backend: it may leave whole lanes unwritten (keeping the initial
   output buffer value) or write **out-of-bounds garbage read from lanes outside the mask
   (fp16 overflow / NaN) into lanes inside the mask** -- with no error, no crash and a plausible
   mean error.  The original 2-D form of `_kv_cache_kernel` in this file left 18 / 52 lanes
   written with 1e3~6e4 / NaN at KVD=64/BLK_D=256 and KVD=512/BLK_D=256 respectively (sentinel
   detection: drop@expected=0, write@non-expected=0, but the values themselves are garbage).
   => Newly written kernels always use "**scalar row index + 1-D column tile + 1-D masked store**
   (runtime row loop)"; `_kv_cache_kernel` here has been rewritten that way.
2. Flattening a 2-D tile to 1-D with `tl.reshape` and then storing it **drops lanes just as
   badly** on this backend (896 of 1024 lanes were still dropped with cols=1024 and no mask at
   all) => **do not use reshape as a workaround**.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["mla_preprocess"]


# ---------------------------------------------------------------------------
# kernel 1: OUT[M,N] = A[M,K] @ W[N,K]^T   (plain matmul with transB)
# ---------------------------------------------------------------------------
@triton.jit
def _mm_nt_kernel(
    A, W, OUT, M, K, N,
    BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLK_M + tl.arange(0, BLK_M)
    rn = pn * BLK_N + tl.arange(0, BLK_N)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.float32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K)
        kmask = rk < K
        a = tl.load(A + rm[:, None] * K + rk[None, :],
                    mask=mmask[:, None] & kmask[None, :], other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :],
                    mask=nmask[:, None] & kmask[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(w))
    tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(OUT.dtype.element_ty),
             mask=mmask[:, None] & nmask[None, :])


# ---------------------------------------------------------------------------
# kernel 2: y = x / sqrt(sum(x^2)/D + eps) * gamma      (along the last dim)
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_kernel(
    X, GAMMA, BETA, Y, M, D, AVG, EPS,
    BLK_M: tl.constexpr, BLK_D: tl.constexpr, HAS_BETA: tl.constexpr = False,
):
    """y = x / sqrt(mean(x^2) + eps) * gamma (+ beta).

    Semantics of `beta` (from AscendC `op_kernel/kernel/kernel_utils.h` L236-254
    `ComputeRmsNorm<WITH_BETA>`): gamma/beta are cast to fp32 and then `Mul(out, out, gamma)`
    runs, followed by `if constexpr (WITH_BETA) Add(out, out, beta)`; the fp16 path is equivalent
    (`mix_fp16.hpp` L575-585, "quant beta is fp16 add").  That is,
    **y = rmsnorm(x) * gamma + beta** (beta takes no part in the normalization).
    Call site: `beta1 = q_a_layernorm.bias` (`sfa_v1.py:877`), applied only to the rmsnorm of the
    qLoRA segment.
    """
    pid = tl.program_id(0)
    rm = pid * BLK_M + tl.arange(0, BLK_M)
    mmask = rm < M
    ss = tl.zeros((BLK_M,), dtype=tl.float32)
    for d0 in range(0, D, BLK_D):
        rd = d0 + tl.arange(0, BLK_D)
        m2 = mmask[:, None] & (rd < D)[None, :]
        x = tl.load(X + rm[:, None] * D + rd[None, :], mask=m2, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    inv = 1.0 / tl.sqrt(ss * AVG + EPS)
    for d0 in range(0, D, BLK_D):
        rd = d0 + tl.arange(0, BLK_D)
        dm = rd < D
        m2 = mmask[:, None] & dm[None, :]
        x = tl.load(X + rm[:, None] * D + rd[None, :], mask=m2, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + rd, mask=dm, other=0.0).to(tl.float32)
        y = x * inv[:, None] * g[None, :]
        if HAS_BETA:
            b = tl.load(BETA + rd, mask=dm, other=0.0).to(tl.float32)
            y = y + b[None, :]
        tl.store(Y + rm[:, None] * D + rd[None, :], y.to(Y.dtype.element_ty), mask=m2)


# ---------------------------------------------------------------------------
# kernel 3: the [kv | rope] segment of s3 -- rmsnorm(gamma2) + RoPE(HALF) + KV cache write
# ---------------------------------------------------------------------------
@triton.jit
def _kv_cache_kernel(
    S3, GAMMA, COS, SIN, SLOT, KVOUT, KROPOUT,
    M, KVD, RD, D2, KLINE, AVG, EPS,
    SPLIT_OUT: tl.constexpr, NZ: tl.constexpr,
    BLK_M: tl.constexpr, BLK_D: tl.constexpr, HALF: tl.constexpr,
):
    """The [kv | rope] segment of s3: rmsnorm(gamma2) + RoPE(HALF) + KV cache write.

    Shape constraint (910_93 + BiShengIR/FlagTree, reproduced): this **must** use
    "scalar row index + 1-D column tile + 1-D masked store" and must not be written as a
    `[BLK_M, BLK_D]` 2-D tile with a 2-D masked store and indirect slot addressing.  When KVD is
    not a multiple of BLK_D (or smaller than BLK_D), the original 2-D form wrote **out-of-bounds
    garbage (fp16 overflow / NaN) into the offsets that should have been written** (sentinel
    detection: drop@expected=0, write@non-expected=0, yet 18~52 lanes held 1e3~6e4/NaN), with no
    error and no crash.  After the 1-D rewrite, 0 lanes are wrong.
    Unit-test cases: `small config bf16 krope_ctkv N=6` / `real dims bf16 krope_ctkv N=4`.
    """
    pid = tl.program_id(0)
    rd0 = tl.arange(0, BLK_D)
    h = tl.arange(0, HALF)
    hm = h < HALF
    for i in range(BLK_M):
        row = pid * BLK_M + i
        if row < M:
            ssv = tl.zeros([BLK_D], dtype=tl.float32)
            for d0 in range(0, KVD, BLK_D):
                rd = d0 + rd0
                x = tl.load(S3 + row * D2 + rd, mask=rd < KVD, other=0.0).to(tl.float32)
                ssv += x * x
            inv = 1.0 / tl.sqrt(tl.sum(ssv, axis=0) * AVG + EPS)

            slot = tl.load(SLOT + row)
            for d0 in range(0, KVD, BLK_D):
                rd = d0 + rd0
                dm = rd < KVD
                x = tl.load(S3 + row * D2 + rd, mask=dm, other=0.0).to(tl.float32)
                g = tl.load(GAMMA + rd, mask=dm, other=0.0).to(tl.float32)
                y = x * inv * g
                if NZ:
                    # nzcache: off(n) = outer*(128*KVD) + (n//16)*2048 + inner*16 + (n%16)
                    so = (slot // 128) * (128 * KVD) + (rd // 16) * 2048 \
                         + (slot % 128) * 16 + (rd % 16)
                elif SPLIT_OUT:
                    so = slot * KVD + rd
                else:
                    so = slot * KLINE + rd
                tl.store(KVOUT + so, y.to(KVOUT.dtype.element_ty), mask=dm)

            # ---- RoPE(HALF) on s3[row, KVD:KVD+RD] ----
            base = row * D2 + KVD
            xa = tl.load(S3 + base + h, mask=hm, other=0.0).to(tl.float32)
            xb = tl.load(S3 + base + HALF + h, mask=hm, other=0.0).to(tl.float32)
            ca = tl.load(COS + row * RD + h, mask=hm, other=0.0).to(tl.float32)
            cb = tl.load(COS + row * RD + HALF + h, mask=hm, other=0.0).to(tl.float32)
            sa = tl.load(SIN + row * RD + h, mask=hm, other=0.0).to(tl.float32)
            sb = tl.load(SIN + row * RD + HALF + h, mask=hm, other=0.0).to(tl.float32)
            oa = ca * xa - sa * xb
            ob = sb * xa + cb * xb
            if NZ:
                # rope: off(n) = outer*(128*RD) + (n//16)*2048 + inner*16 + (n%16)
                base2 = (slot // 128) * (128 * RD) + (slot % 128) * 16
                soa = base2 + (h // 16) * 2048 + (h % 16)
                sob = base2 + ((HALF + h) // 16) * 2048 + ((HALF + h) % 16)
                tl.store(KROPOUT + soa, oa.to(KROPOUT.dtype.element_ty), mask=hm)
                tl.store(KROPOUT + sob, ob.to(KROPOUT.dtype.element_ty), mask=hm)
            elif SPLIT_OUT:
                tl.store(KROPOUT + slot * RD + h, oa.to(KROPOUT.dtype.element_ty), mask=hm)
                tl.store(KROPOUT + slot * RD + HALF + h, ob.to(KROPOUT.dtype.element_ty), mask=hm)
            else:
                tl.store(KVOUT + slot * KLINE + KVD + h, oa.to(KVOUT.dtype.element_ty), mask=hm)
                tl.store(KVOUT + slot * KLINE + KVD + HALF + h, ob.to(KVOUT.dtype.element_ty), mask=hm)


# ---------------------------------------------------------------------------
# kernel 4: q_nope = s2[:,:, :nope] @ wuk[h]  + the rope part of q
# ---------------------------------------------------------------------------
@triton.jit
def _q_head_kernel(
    S2, WUK, COS, SIN, Q, Q2,
    M, HN, HD, NOPE, RD, KVD, QLINE,
    ROPE_IN_Q: tl.constexpr,
    BLK_M: tl.constexpr, BLK_K: tl.constexpr, BLK_N: tl.constexpr,
    HALF: tl.constexpr,
):
    pm = tl.program_id(0)
    h = tl.program_id(1)
    rm = pm * BLK_M + tl.arange(0, BLK_M)
    mmask = rm < M
    rk = tl.arange(0, BLK_K)          # BLK_K == NOPE
    kmask = rk < NOPE
    S2STRIDE = HN * HD
    QSTRIDE = HN * QLINE
    # A = s2[rm, h, 0:NOPE]
    a = tl.load(S2 + rm[:, None] * S2STRIDE + h * HD + rk[None, :],
                mask=mmask[:, None] & kmask[None, :], other=0.0)
    for n0 in range(0, KVD, BLK_N):
        rn = n0 + tl.arange(0, BLK_N)
        nmask = rn < KVD
        b = tl.load(WUK + h * NOPE * KVD + rk[:, None] * KVD + rn[None, :],
                    mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(a, b)
        tl.store(Q + rm[:, None] * QSTRIDE + h * QLINE + rn[None, :],
                 acc.to(Q.dtype.element_ty), mask=mmask[:, None] & nmask[None, :])

    hh = tl.arange(0, HALF)
    hm = mmask[:, None] & (hh < HALF)[None, :]
    base = rm[:, None] * S2STRIDE + h * HD + NOPE
    xa = tl.load(S2 + base + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    xb = tl.load(S2 + base + HALF + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    ca = tl.load(COS + rm[:, None] * RD + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    cb = tl.load(COS + rm[:, None] * RD + HALF + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    sa = tl.load(SIN + rm[:, None] * RD + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    sb = tl.load(SIN + rm[:, None] * RD + HALF + hh[None, :], mask=hm, other=0.0).to(tl.float32)
    oa = ca * xa - sa * xb
    ob = sb * xa + cb * xb
    if ROPE_IN_Q:
        tl.store(Q + rm[:, None] * QSTRIDE + h * QLINE + KVD + hh[None, :],
                 oa.to(Q.dtype.element_ty), mask=hm)
        tl.store(Q + rm[:, None] * QSTRIDE + h * QLINE + KVD + HALF + hh[None, :],
                 ob.to(Q.dtype.element_ty), mask=hm)
    else:
        row = rm[:, None] * HN + h
        tl.store(Q2 + row * RD + hh[None, :], oa.to(Q2.dtype.element_ty), mask=hm)
        tl.store(Q2 + row * RD + HALF + hh[None, :], ob.to(Q2.dtype.element_ty), mask=hm)


# ---------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------
_CACHE_MODES = {"kvcache": 0, "krope_ctkv": 1, "int8_nzcache": 2, "nzcache": 3}
_QUANT_MODES = {"per_tensor_quant_asymm": 0, "per_token_quant_symm": 1,
                "per_token_quant_asymm": 2, "no_quant": 3}

_UNSUPPORTED_HINT = (
    "\nThis Triton version of mla_preprocess currently only covers cache_mode in "
    "{kvcache, krope_ctkv} with quant_mode='no_quant' (the non-quantized branches of tiling "
    "keys 281 / 256 / 257).\n"
    "The int8 quantization paths (per_tensor_quant_asymm / per_token_quant_symm, including "
    "quant_scale/offset/bias, descale, ctkv_scale, q_nope_scale, the NZ write-back of nzcache "
    "and enable_inner_out) are not implemented yet.\n"
    "The kernel-side formulas have been extracted and are documented in the module docstring "
    "of mla_preprocess_int8.py."
)


def _reject_unsupported(**kw):
    """Parameters that call sites pass but this implementation does not cover: raise an explicit
    NotImplementedError rather than silently ignoring them."""
    bad = [k for k, v in kw.items() if v is not None and not (isinstance(v, bool) and v is False)]
    if bad:
        raise NotImplementedError(
            "mla_preprocess: the following arguments are not implemented yet and must not be "
            "silently ignored: " + ", ".join(sorted(bad))
            + _UNSUPPORTED_HINT)


def mla_preprocess(
    hidden_state: torch.Tensor,
    # positional argument order **exactly matches the real call site** (vllm_fl device_op.py:349):
    #   (hidden_states, wd_qkv, deq_scale_qkv, gamma1, beta1, wu_q, qb_deq_scl,
    #    gamma2, cos, sin, W_UK_T, decode_k_nope, decode_k_pe, slot_mapping)
    wdqkv: torch.Tensor | None = None,
    deq_scale_qkv: torch.Tensor | None = None,
    gamma1: torch.Tensor | None = None,
    beta1: torch.Tensor | None = None,
    wuq: torch.Tensor | None = None,
    qb_deq_scl: torch.Tensor | None = None,
    gamma2: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    wuk: torch.Tensor | None = None,
    decode_k_nope: torch.Tensor | None = None,
    decode_k_pe: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
    # keyword arguments of the call site
    *,
    quant_scale0: torch.Tensor | None = None,
    quant_offset0: torch.Tensor | None = None,
    bias0: torch.Tensor | None = None,
    quant_scale1: torch.Tensor | None = None,
    quant_offset1: torch.Tensor | None = None,
    bias1: torch.Tensor | None = None,
    ctkv_scale: torch.Tensor | None = None,
    q_nope_scale: torch.Tensor | None = None,
    cache_mode: str = "krope_ctkv",
    quant_mode: str = "no_quant",
    enable_inner_out: bool = False,
    inner_out: torch.Tensor | None = None,
    q_out0: torch.Tensor | None = None,
    kv_cache_out0: torch.Tensor | None = None,
    q_out1: torch.Tensor | None = None,
    kv_cache_out1: torch.Tensor | None = None,
    # internal parameters of this implementation (snake_case aliases are also accepted)
    # Dimensions default to None = **derived from the tensor shapes** (matching the native op,
    # see op_host/mla_preprocess.h:682-690).  Passing them explicitly only enables a
    # **consistency check**: a conflict with the shapes is an error instead of silently trusting
    # the caller's numbers.
    q: torch.Tensor | None = None,
    q2: torch.Tensor | None = None,
    q_lora_rank: int | None = None,
    kv_lora_rank: int | None = None,
    qk_nope_head_dim: int | None = None,
    qk_rope_head_dim: int | None = None,
    head_num: int | None = None,
    num_slots: int | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equivalent to the **no_quant** path of the AscendC mla_preprocess (see the module
    docstring).

    The positional argument order is **consistent** with the real call site (`vllm_fl`
    device_op.py:349) and the keyword arguments use the original call-site names; internal
    aliases such as `q=`/`q2=` are accepted as well.

    Only `quant_mode="no_quant"` with `cache_mode in {kvcache, krope_ctkv}` is runnable for now;
    **any non-empty quantization parameter raises `NotImplementedError`** (it is not silently
    ignored).

    The row order of the wdqkv weight is [kv_lora_rank | qk_rope_head_dim | q_lora_rank]
    (consistent with tiling: input_stride_ = splitSizeOne_ = kv+rope is the start of the qLoRA
    segment).

    Returns:
        (q, kv_cache_out0, q2, kv_cache_out1) -- corresponding to (q_out0, kv_cache_out0,
        q_out1, kv_cache_out1)
    """
    # ---- backfill call-site aliases ----
    if q_out0 is not None:
        q = q_out0
    if q_out1 is not None:
        q2 = q_out1
    if slot_mapping is None:
        raise ValueError("mla_preprocess requires slot_mapping")

    # ---- unimplemented parameters: fail explicitly ----
    # beta1 is implemented (paired with gamma1, see has_beta1 below; the semantics come from the
    # AscendC ComputeRmsNorm<WITH_BETA>)
    _reject_unsupported(decode_k_nope=decode_k_nope, decode_k_pe=decode_k_pe)
    if cache_mode != "int8_nzcache":   # ctkv_scale / q_nope_scale are only used by int8_nzcache
        _reject_unsupported(ctkv_scale=ctkv_scale, q_nope_scale=q_nope_scale)
    # enable_inner_out: inner_out = rmsnorm result of the qLoRA segment after M1
    # (bf16, [N, q_lora_rank])
    if enable_inner_out and (inner_out is None or inner_out.numel() == 0):
        raise ValueError("enable_inner_out=True requires a non-empty inner_out [N, q_lora_rank]")
    if quant_mode not in _QUANT_MODES:
        raise ValueError(f"unknown quant_mode {quant_mode!r}; valid values {sorted(_QUANT_MODES)}")
    qm = _QUANT_MODES[quant_mode]
    if qm in (1, 2):
        raise NotImplementedError(
            f"mla_preprocess: quant_mode={quant_mode!r} is not implemented yet" + _UNSUPPORTED_HINT)
    is_quant = (qm == 0)
    if not is_quant:
        # on the no_quant path these quantization parameters must be empty, otherwise they would
        # be silently ignored
        _reject_unsupported(
            deq_scale_qkv=deq_scale_qkv, qb_deq_scl=qb_deq_scl,
            quant_scale0=quant_scale0, quant_offset0=quant_offset0, bias0=bias0,
            quant_scale1=quant_scale1, quant_offset1=quant_offset1, bias1=bias1)
    if gamma1 is None or wuq is None or gamma2 is None or wuk is None or cos is None or sin is None:
        raise ValueError("mla_preprocess requires gamma1/wuq/gamma2/wuk/cos/sin")
    if cache_mode not in ("kvcache", "krope_ctkv", "nzcache", "int8_nzcache"):
        raise NotImplementedError(
            f"mla_preprocess: cache_mode={cache_mode!r} is not implemented" + _UNSUPPORTED_HINT)
    if cache_mode == "int8_nzcache" and (ctkv_scale is None or q_nope_scale is None):
        raise ValueError("cache_mode='int8_nzcache' requires ctkv_scale and q_nope_scale")
    # ---- dimensions: **always derived from the tensor shapes**, exactly like the native op
    #      (op_host/mla_preprocess.h:682-690) ----
    #   N = hiddenState.sizes()[0]; headNum = wuk.sizes()[0]; hidden = hiddenState.sizes()[-1]
    #   qkNopeHeadDim = wuk.sizes()[1]; kvLoraRank = wuk.sizes()[2]; qLoraRank = gamma1.sizes()[0]
    #   qkRopeHeadDim = kv_cache_rope.sizes()[-1] -> this implementation uses cos/sin's last dim
    # Why this is mandatory: the real call site (device_op.py:349/420) **passes no dimension
    # kwarg at all**, so the bridge defaults used to apply (q_lora_rank=1536 /
    # qk_nope_head_dim=128 / head_num=128, i.e. the DeepSeek family), whereas GLM-5.2
    # (glm_moe_dsa) uses 2048 / 192 / 64 => the result was **silently wrong** (it only surfaced
    # at model level, when beta1=(2048,) hit the default q_lora_rank=1536).  Passing an explicit
    # dimension that conflicts with the shapes is a hard error.
    N, Hd = int(hidden_state.shape[0]), int(hidden_state.shape[1])
    shape_HN, shape_Dnope, shape_Dkv = int(wuk.shape[0]), int(wuk.shape[1]), int(wuk.shape[2])
    shape_Dq = int(gamma1.numel())
    shape_Drd = int(cos.shape[-1])
    for nm, given, shape in (("q_lora_rank", q_lora_rank, shape_Dq),
                             ("kv_lora_rank", kv_lora_rank, shape_Dkv),
                             ("qk_nope_head_dim", qk_nope_head_dim, shape_Dnope),
                             ("qk_rope_head_dim", qk_rope_head_dim, shape_Drd),
                             ("head_num", head_num, shape_HN)):
        if given is not None and int(given) != shape:
            raise ValueError(
                f"mla_preprocess: {nm}={int(given)} does not match {shape} derived from shapes"
                f" (native op derives dims from shapes: op_host/mla_preprocess.h:682-690; "
                f"gamma1.numel()={shape_Dq}, wuk={tuple(wuk.shape)}, cos.shape[-1]={shape_Drd})")
    Dq, Dkv, Dnope, Drd, HN = shape_Dq, shape_Dkv, shape_Dnope, shape_Drd, shape_HN
    # beta1 (optional) = `q_a_layernorm.bias` (call site `sfa_v1.py:877`), paired with gamma1 and
    # **applied only to the rmsnorm of the qLoRA segment**:
    # y = x/sqrt(mean(x^2)+eps) * gamma1 + beta1 (AscendC `kernel_utils.h` L236-254
    # `ComputeRmsNorm<WITH_BETA>`: Mul(gamma) first, then Add(beta); the length follows the gamma
    # count, so it must match gamma1).
    has_beta1 = beta1 is not None
    if has_beta1:
        if int(beta1.numel()) != Dq:
            raise ValueError(
                f"mla_preprocess: beta1.numel()={int(beta1.numel())} != gamma1.numel()={Dq}"
                f" (ComputeRmsNorm<WITH_BETA> handles gamma/beta with one count, so equal length)")
        b1c = beta1.contiguous()
    else:
        b1c = None
    HD = Dnope + Drd
    Dkvr = Dkv + Drd
    dev = hidden_state.device
    dt = hidden_state.dtype
    if num_slots is None:
        num_slots = int(slot_mapping.detach().max().item()) + 1

    nz = (cache_mode in ("nzcache", "int8_nzcache"))
    i8nz = (cache_mode == "int8_nzcache")
    split_out = (cache_mode in ("krope_ctkv", "nzcache", "int8_nzcache"))
    qline = Dkv if split_out else (Dkv + Drd)
    nout = ceil_div(max(num_slots, 1), 128)
    if q is None:
        q = torch.empty((N, HN, Dkv if i8nz else qline),
                        dtype=(torch.int8 if i8nz else dt), device=dev)
    if split_out:
        if q2 is None:
            q2 = torch.empty((N * HN, Drd), dtype=dt, device=dev)
        if kv_cache_out0 is None:
            kv_cache_out0 = torch.empty(
                (nout * 128 * Dkv,) if nz else (num_slots, Dkv),
                dtype=(torch.int8 if i8nz else dt), device=dev)
        if kv_cache_out1 is None:
            kv_cache_out1 = torch.empty((nout * 128 * Drd,) if nz else (num_slots, Drd),
                                        dtype=dt, device=dev)
    else:
        if q2 is None:
            q2 = torch.empty((0,), dtype=dt, device=dev)
        if kv_cache_out0 is None:
            kv_cache_out0 = torch.empty((num_slots, Dkvr), dtype=dt, device=dev)
        if kv_cache_out1 is None:
            kv_cache_out1 = torch.empty((0,), dtype=dt, device=dev)

    if enable_inner_out:
        assert tuple(inner_out.shape) == (N, Dq) and inner_out.dtype == dt, \
            f"inner_out must be [{N},{Dq}] {dt}, got {tuple(inner_out.shape)} {inner_out.dtype}"
    hs = hidden_state.contiguous()
    wd = wdqkv.contiguous()
    wq = wuq.contiguous()
    wk = wuk.contiguous()
    cs = cos.contiguous()
    sn = sin.contiguous()
    slot = slot_mapping.to(torch.int32).contiguous()
    BLK_M_MM, BLK_N_MM, BLK_K_MM = 16, 128, 64
    MM1N = Dkvr + Dq
    D2 = HN * HD

    if is_quant:
        from .mla_preprocess_int8 import mla_preprocess_quant
        return mla_preprocess_quant(
            hs, wd, wq, wk, cs, sn, slot, dt, dev, N, Hd, Dq, Dkv, Dnope, Drd, HN,
            Dkvr, HD, qline, nz, q, q2, kv_cache_out0, kv_cache_out1,
            gamma1, gamma2, quant_scale0, quant_offset0, bias0, deq_scale_qkv,
            quant_scale1, quant_offset1, bias1, qb_deq_scl, float(eps),
            cache_mode=cache_mode, ctkv_scale=ctkv_scale, q_nope_scale=q_nope_scale,
            inner_out=inner_out if enable_inner_out else None,
            beta1=b1c)
    # --- mm1: s3 = hidden @ wdqkv^T  (split into the qLoRA segment and the kv/rope segment) ---
    w_q = wd[Dkvr:Dkvr + Dq].contiguous()
    w_kv = wd[0:Dkvr].contiguous()
    s3q = torch.empty((N, Dq), dtype=dt, device=dev)
    s3kv = torch.empty((N, Dkvr), dtype=dt, device=dev)

    _mm_nt_kernel[(ceil_div(N, BLK_M_MM), ceil_div(Dq, BLK_N_MM))](
        hs, w_q, s3q, N, Hd, Dq,
        BLK_M=BLK_M_MM, BLK_N=BLK_N_MM, BLK_K=BLK_K_MM)
    _mm_nt_kernel[(ceil_div(N, BLK_M_MM), ceil_div(Dkvr, BLK_N_MM))](
        hs, w_kv, s3kv, N, Hd, Dkvr,
        BLK_M=BLK_M_MM, BLK_N=BLK_N_MM, BLK_K=BLK_K_MM)

    # --- rmsnorm(s3q, gamma1) [+ beta1] ---
    q_lora = inner_out if (enable_inner_out and inner_out is not None) else torch.empty((N, Dq), dtype=dt, device=dev)
    BLK_M_R = 16
    BLK_D_R = 256
    _rmsnorm_kernel[(ceil_div(N, BLK_M_R),)](
        s3q, gamma1.contiguous(), (b1c if has_beta1 else gamma1.contiguous()), q_lora, N, Dq,
        1.0 / Dq, float(eps),
        BLK_M=BLK_M_R, BLK_D=BLK_D_R, HAS_BETA=bool(has_beta1))

    # --- mm2: s2 = q_lora @ wuq^T ---
    s2 = torch.empty((N, D2), dtype=dt, device=dev)
    _mm_nt_kernel[(ceil_div(N, BLK_M_MM), ceil_div(D2, BLK_N_MM))](
        q_lora, wq, s2, N, Dq, D2,
        BLK_M=BLK_M_MM, BLK_N=BLK_N_MM, BLK_K=BLK_K_MM)

    # --- mm3 (per-head) + q rope ---
    _q_head_kernel[(ceil_div(N, 16), HN)](
        s2, wk, cs, sn, q, q2,
        N, HN, HD, Dnope, Drd, Dkv, qline,
        ROPE_IN_Q=bool(not split_out),
        BLK_M=16, BLK_K=next_pow2(Dnope), BLK_N=128, HALF=Drd // 2)

    # --- kv / rope cache ---
    _kv_cache_kernel[(ceil_div(N, 16),)](
        s3kv, gamma2.contiguous(), cs, sn, slot, kv_cache_out0, kv_cache_out1,
        N, Dkv, Drd, Dkvr, qline, 1.0 / Dkv, float(eps),
        SPLIT_OUT=bool(split_out), NZ=bool(nz),
        BLK_M=16, BLK_D=256, HALF=Drd // 2)

    return q, kv_cache_out0, q2, kv_cache_out1
