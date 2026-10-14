# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""int8 quantization path of mla_preprocess (quant_mode="per_tensor_quant_asymm").

Source: csrc/ascend/mla_preprocess/op_kernel/mla_preprocess_mix_bf16.hpp
        MLAPO_BF16::MLAOperation<__bf16, CACHE_MODE, NZ, NZ, ND, PER_TENSOR_ASYMM_QUANT>
        (tiling key = (innerOut<<9)|(bf16<<8)|cacheMode|(quantMode<<3) => 257 / 259)

Pipeline (ProcessVector L2800-2939 + ProcessCube):
 1 Quant::Launch L332-496: **quantizes the raw hiddenState (without rmsnorm, see L2485)**,
   input_scale_=1/quantScale0(L373), input_offset_=quantOffset0(int8)(L374),
   y=x*input_scale_+input_offset_(L436-440), fp32->fp16->int8(min=-128)(L482-486)
   => x_i8 = saturate_i8(round_f16(x/s0 + o0))
 2 mm1 W8A8 (L2431): C_int32 = s1_i8 @ wdqkv_i8^T
 3 dequant PpMatmulW8a8Aiv::Process L2106-2369: bias=gmPerTensorBias_(**int32**, length n),
   descale=gmPerTensorScale_(fp32, length n), offsetScale=batch*n+n_idx*n0(L2140);
   L2227-2232 `C_int32 += bias[n]` (**bias is applied before dequantization**) -> int32->float
   -> L2277-2288 `*= descale[n]` -> float->bf16(CAST_RINT)
   => s3 = (c1 + bias0) * deq_scale_qkv
 4 RmsNormQuant L592-850 (config L2490-2493): s3[:,576:2112] goes through rmsnorm(gamma1,
   avg=1/Dq, eps=1e-6) and is then quantized like step 1 (quantScale1/Offset1) -> int8
 5 mm2 W8A8 (L2434) -> 6 dequant (L2469) => s4 = (c2 + bias1) * qb_deq_scl
 7 mm3 + q rope (reuses mla_preprocess._q_head_kernel)
 8 KV write-back RmsNormAndRopeConvergence1 L2526-2712:
   the NZ element offset of "nzcache"(3), L2686-2701
     off(n) = outer*(128*L) + (n//16)*2048 + inner*16 + (n%16)   (L = kv/rope length)
   Derivation: dstStride=(128*16-16)*sizeof(T1) is the **gap between blocks** and
   blockLen=16*sizeof(T1), hence block c starts at base + c*(16+2032) = base + c*2048;
   128*512 and 128*64 fill that exactly.

Not covered: per_token_quant_symm/asymm, int8_nzcache(2) and EinSumQuant(q_nope_scale),
enable_inner_out, and the fp16 quantization branch -- mla_preprocess() raises an explicit
NotImplementedError for those.

Backend behavior (910_93 + BiShengIR/FlagTree): with a **sentinel double-fill** the 2-D masked
stores of the three kernels in this file were measured to lose **no** writes:
    `_quant_asymm_kernel`   D in {17,24,33,48,100,127,129,256,1000,7168} x M in {16,19} -> dropped=0
    `_q_nope_quant_kernel`  KVD in {17,33,48,100,128,512}                              -> dropped=0
    `_kv_cache_i8_kernel`   (KVD,RD) in {(64,32),(17,16),(33,32),(100,32),(512,64)}     -> dropped=0
and `write@non-expected=0` (no out-of-range or misplaced writes). The stores in this file are
therefore **not** rewritten.
  CAUTION, counter-example 1: the "2-D masked store plus runtime-loaded slot used for indirect
  addressing" pattern of `mla_preprocess.py::_kv_cache_kernel` was measured to write garbage
  read out of range by masked-out lanes (1e3..6e4/NaN) into masked-in lanes; it has been
  rewritten into the 1-D form.
  CAUTION, counter-example 2: `tl.reshape` 2-D->1-D followed by a store drops lanes on a large
  scale; do not use it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .mla_preprocess import _q_head_kernel, _kv_cache_kernel, _rmsnorm_kernel
from .common import ceil_div, next_pow2

__all__ = ["mla_preprocess_quant"]


@triton.jit
def _quant_asymm_kernel(X, Y, S, O, M, D, BLK_M: tl.constexpr, BLK_D: tl.constexpr):
    """y_i8 = saturate_i8(round_f16(x/s + o))   (Quant::Launch L373-486)"""
    pid = tl.program_id(0)
    rm = pid * BLK_M + tl.arange(0, BLK_M)
    mmask = rm < M
    inv = 1.0 / tl.load(S).to(tl.float32)          # input_scale_ = 1/quantScale (L373)
    off = tl.load(O).to(tl.float32)                # quantOffset is an int8 tensor (L374)
    for d0 in range(0, D, BLK_D):
        rd = d0 + tl.arange(0, BLK_D)
        m2 = mmask[:, None] & (rd < D)[None, :]
        x = tl.load(X + rm[:, None] * D + rd[None, :], mask=m2, other=0.0).to(tl.float32)
        v = (x * inv + off).to(tl.float16).to(tl.float32)   # fp32->fp16 (L482-484)
        # fp16->int8 uses CAST_RINT (round-half-to-even, see CastFromF16ToI8)
        fl = tl.floor(v)
        r = v - fl
        odd = fl - 2.0 * tl.floor(fl * 0.5)                 # fl mod 2 in {0,1}
        v = tl.where(r > 0.5, fl + 1.0, tl.where(r < 0.5, fl, fl + odd))
        v = tl.minimum(tl.maximum(v, -128.0), 127.0)
        tl.store(Y + rm[:, None] * D + rd[None, :], v.to(tl.int8), mask=m2)


@triton.jit
def _mm_i8_kernel(A, W, C, M, K, N,
                  BLK_M: tl.constexpr, BLK_N: tl.constexpr, BLK_K: tl.constexpr):
    """C_int32 = A_i8 @ W_i8^T. The tiling is converted to int32 block by block: each block's
    integer sum is <= 128*128*BLK_K < 2^24, which fp32 represents exactly => bit-identical to
    the cube's int32 accumulation."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLK_M + tl.arange(0, BLK_M)
    rn = pn * BLK_N + tl.arange(0, BLK_N)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BLK_M, BLK_N), dtype=tl.int32)
    for k0 in range(0, K, BLK_K):
        rk = k0 + tl.arange(0, BLK_K)
        kmask = rk < K
        a = tl.load(A + rm[:, None] * K + rk[None, :],
                    mask=mmask[:, None] & kmask[None, :], other=0)
        w = tl.load(W + rn[:, None] * K + rk[None, :],
                    mask=nmask[:, None] & kmask[None, :], other=0)
        part = tl.dot(a.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
        acc += part.to(tl.int32)
    tl.store(C + rm[:, None] * N + rn[None, :], acc, mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _dequant_kernel(C, BIAS, DSC, OUT, M, N,
                    BLK_M: tl.constexpr, BLK_N: tl.constexpr):
    """out[m,n] = (C_int32[m,n] + bias[n]) * descale[n]
    (L2227-2232 adds the int32 bias first, L2277-2288 then multiplies by descale)"""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLK_M + tl.arange(0, BLK_M)
    rn = pn * BLK_N + tl.arange(0, BLK_N)
    mmask = rm < M
    nmask = rn < N
    m2 = mmask[:, None] & nmask[None, :]
    c = tl.load(C + rm[:, None] * N + rn[None, :], mask=m2, other=0)
    b = tl.load(BIAS + rn, mask=nmask, other=0)
    d = tl.load(DSC + rn, mask=nmask, other=0.0)
    v = (c + b[None, :]).to(tl.float32) * d[None, :]
    tl.store(OUT + rm[:, None] * N + rn[None, :], v.to(OUT.dtype.element_ty), mask=m2)


@triton.jit
def _q_nope_quant_kernel(S1, SCALE, QOUT, M, HN, KVD,
                         BLK_M: tl.constexpr, BLK_D: tl.constexpr):
    """int8_nzcache: q_nope_i8 = saturate_i8(round_f16(s1 * q_nope_scale[head]))

    EinSumQuant L899-1022: the scale uses only headNum elements, is Brcb-expanded to
    [headNum,8] and then broadcast per head (L953-955 src1BlkStride=0 / src1RepStride=1)
    => a **per-head multiplicative scale** (not per-token).
    """
    t = tl.program_id(0)
    h = tl.arange(0, 16)
    row = (t * HN + h)[:, None]
    sc = tl.load(SCALE + h, mask=h < HN, other=1.0).to(tl.float32)[:, None]
    for d0 in range(0, KVD, BLK_D):
        rd = d0 + tl.arange(0, BLK_D)
        m2 = (h[:, None] < HN) & (rd < KVD)[None, :]
        x = tl.load(S1 + row * KVD + rd[None, :], mask=m2, other=0.0).to(tl.float32)
        v = (x * sc).to(tl.float16).to(tl.float32)
        fl = tl.floor(v)
        r = v - fl
        odd = fl - 2.0 * tl.floor(fl * 0.5)
        v = tl.where(r > 0.5, fl + 1.0, tl.where(r < 0.5, fl, fl + odd))
        v = tl.minimum(tl.maximum(v, -128.0), 127.0)
        tl.store(QOUT + row * KVD + rd[None, :], v.to(tl.int8), mask=m2)


@triton.jit
def _kv_cache_i8_kernel(S3, GAMMA, COS, SIN, SLOT, CSCALE, KVOUT8, KROPOUT,
                        M, KVD, RD, D2, AVG, EPS,
                        BLK_M: tl.constexpr, BLK_D: tl.constexpr, HALF: tl.constexpr):
    """KV write-back for int8_nzcache (RmsNormAndRopeConvergence1 L2617-2685):
    nope: after rmsnorm, quantize with ctkv_scale (**reciprocal taken**, L2912
          scale3=1/quantScale3) to int8, written in the C0=32 NZ layout as
          off(n) = outer*(128*KVD) + (n//32)*4096 + inner*32 + (n%32)
    rope: not quantized, written in the C0=16 NZ layout as
          off(n) = outer*(128*RD) + (n//16)*2048 + inner*16 + (n%16)
    """
    pid = tl.program_id(0)
    rm = pid * BLK_M + tl.arange(0, BLK_M)
    mmask = rm < M
    ss = tl.zeros((BLK_M,), dtype=tl.float32)
    for d0 in range(0, KVD, BLK_D):
        rd = d0 + tl.arange(0, BLK_D)
        m2 = mmask[:, None] & (rd < KVD)[None, :]
        x = tl.load(S3 + rm[:, None] * D2 + rd[None, :], mask=m2, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    inv = 1.0 / tl.sqrt(ss * AVG + EPS)
    slot = tl.load(SLOT + rm, mask=mmask, other=0)
    out_ = (slot[:, None] // 128) * (128 * KVD)
    inn_ = (slot[:, None] % 128) * 32
    qinv = 1.0 / tl.load(CSCALE).to(tl.float32)
    for d0 in range(0, KVD, BLK_D):
        rd = d0 + tl.arange(0, BLK_D)
        dm = rd < KVD
        m2 = mmask[:, None] & dm[None, :]
        x = tl.load(S3 + rm[:, None] * D2 + rd[None, :], mask=m2, other=0.0).to(tl.float32)
        g = tl.load(GAMMA + rd, mask=dm, other=0.0).to(tl.float32)
        v = (x * inv[:, None] * g[None, :] * qinv).to(tl.float16).to(tl.float32)
        fl = tl.floor(v)
        r = v - fl
        odd = fl - 2.0 * tl.floor(fl * 0.5)
        v = tl.where(r > 0.5, fl + 1.0, tl.where(r < 0.5, fl, fl + odd))
        v = tl.minimum(tl.maximum(v, -128.0), 127.0)
        so = out_ + (rd[None, :] // 32) * 4096 + inn_ + (rd[None, :] % 32)
        tl.store(KVOUT8 + so, v.to(tl.int8), mask=m2)
    h = tl.arange(0, HALF)
    hm = mmask[:, None] & (h < HALF)[None, :]
    base = rm[:, None] * D2 + KVD
    xa = tl.load(S3 + base + h[None, :], mask=hm, other=0.0).to(tl.float32)
    xb = tl.load(S3 + base + HALF + h[None, :], mask=hm, other=0.0).to(tl.float32)
    ca = tl.load(COS + rm[:, None] * RD + h[None, :], mask=hm, other=0.0).to(tl.float32)
    cb = tl.load(COS + rm[:, None] * RD + HALF + h[None, :], mask=hm, other=0.0).to(tl.float32)
    sa = tl.load(SIN + rm[:, None] * RD + h[None, :], mask=hm, other=0.0).to(tl.float32)
    sb = tl.load(SIN + rm[:, None] * RD + HALF + h[None, :], mask=hm, other=0.0).to(tl.float32)
    oa = ca * xa - sa * xb
    ob = sb * xa + cb * xb
    base2 = (slot[:, None] // 128) * (128 * RD) + (slot[:, None] % 128) * 16
    soa = base2 + (h[None, :] // 16) * 2048 + (h[None, :] % 16)
    sob = base2 + ((HALF + h[None, :]) // 16) * 2048 + ((HALF + h[None, :]) % 16)
    tl.store(KROPOUT + soa, oa.to(KROPOUT.dtype.element_ty), mask=hm)
    tl.store(KROPOUT + sob, ob.to(KROPOUT.dtype.element_ty), mask=hm)


def _nz_to_kn(t):
    """Ascend NZ/fractal layout [1, in/32, out, 32] => row-major [out, in].

    Root-cause fix: the int8 weights passed by GLM-5.2-w8a8
    (wdqkv (1,192,2624,32) / wuq (1,64,1024,32)) are in the NZ fractal layout, while
    `_mm_i8_kernel` indexes them as row-major [out, K] => it read misaligned data
    => c1 was entirely wrong (bit-exact comparison against an independent fp64 oracle:
    exact=0%) => everything downstream was wrong => garbled model output.
    Verification: the quantized input s1 matches the oracle **100% bit-exactly**, so the
    single point of failure was the int8 matmul.
    """
    s = t.shape
    if t.dim() == 4 and s[0] == 1:
        return t.reshape(s[1], s[2], s[3]).permute(1, 0, 2).reshape(s[2], s[1] * s[3])
    return t.reshape(s[0], -1) if t.dim() == 3 else t


def mla_preprocess_quant(hs, wd, wq, wk, cs, sn, slot, dt, dev, N, Hd, Dq, Dkv, Dnope, Drd,
                         HN, Dkvr, HD, qline, nz, q, q2, kv_cache_out0, kv_cache_out1,
                         gamma1, gamma2, quant_scale0, quant_offset0, bias0, deq_scale_qkv,
                         quant_scale1, quant_offset1, bias1, qb_deq_scl, eps,
                         cache_mode="nzcache", ctkv_scale=None, q_nope_scale=None,
                         inner_out=None, beta1=None):
    """Full per_tensor_quant_asymm pipeline (every step matches the kernel, see the module
    docstring).

    inner_out (optional): `[N, q_lora_rank]` bf16. Its semantics are the **rmsnorm result of
    the qLoRA segment after M1** (i.e. `q_lora`), which is also the input of mm2. When passed
    in it is used directly as the destination buffer of the rmsnorm (saves one copy).
    Source: `mla_preprocess_def.cpp` L158 (innerOut is a REQUIRED output);
    the test assertion is in `test_mla_preprocess_qdown.py` L147-160 (NaN initial value ->
    must be overwritten).
    beta1 (optional): `[q_lora_rank]`, paired with gamma1,
    `y = rmsnorm(x)*gamma1 + beta1` (AscendC `ComputeRmsNorm<WITH_BETA>`, `kernel_utils.h`
    L236-254).
    """
    wd = _nz_to_kn(wd)
    wq = _nz_to_kn(wq)
    i8nz = (cache_mode == "int8_nzcache")
    D2 = HN * HD
    MM1N = Dkvr + Dq
    BM, BN, BK = 16, 128, 128
    s1 = torch.empty((N, Hd), dtype=torch.int8, device=dev)              # 1
    _quant_asymm_kernel[(ceil_div(N, BM),)](
        hs, s1, quant_scale0.contiguous(), quant_offset0.contiguous(), N, Hd, BLK_M=BM, BLK_D=256)
    c1 = torch.empty((N, MM1N), dtype=torch.int32, device=dev)           # 2
    _mm_i8_kernel[(ceil_div(N, BM), ceil_div(MM1N, BN))](
        s1, wd, c1, N, Hd, MM1N, BLK_M=BM, BLK_N=BN, BLK_K=BK)
    s3 = torch.empty((N, MM1N), dtype=dt, device=dev)                    # 3
    _dequant_kernel[(ceil_div(N, BM), ceil_div(MM1N, BN))](
        c1, bias0.to(torch.int32).contiguous(), deq_scale_qkv.to(torch.float32).contiguous(),
        s3, N, MM1N, BLK_M=BM, BLK_N=BN)
    q_lora = inner_out if inner_out is not None else torch.empty((N, Dq), dtype=dt, device=dev)  # 4
    b1c = beta1.contiguous() if beta1 is not None else None
    _rmsnorm_kernel[(ceil_div(N, BM),)](
        s3[:, Dkvr:MM1N].contiguous(), gamma1.contiguous(),
        (b1c if b1c is not None else gamma1.contiguous()), q_lora, N, Dq,
        1.0 / Dq, float(eps), BLK_M=BM, BLK_D=256, HAS_BETA=bool(b1c is not None))
    s1b = torch.empty((N, Dq), dtype=torch.int8, device=dev)
    _quant_asymm_kernel[(ceil_div(N, BM),)](
        q_lora, s1b, quant_scale1.contiguous(), quant_offset1.contiguous(), N, Dq,
        BLK_M=BM, BLK_D=256)
    c2 = torch.empty((N, D2), dtype=torch.int32, device=dev)             # 5
    _mm_i8_kernel[(ceil_div(N, BM), ceil_div(D2, BN))](
        s1b, wq, c2, N, Dq, D2, BLK_M=BM, BLK_N=BN, BLK_K=BK)
    s4 = torch.empty((N, D2), dtype=dt, device=dev)                      # 6
    _dequant_kernel[(ceil_div(N, BM), ceil_div(D2, BN))](
        c2, bias1.to(torch.int32).contiguous(), qb_deq_scl.to(torch.float32).contiguous(),
        s4, N, D2, BLK_M=BM, BLK_N=BN)
    if i8nz:
        # int8_nzcache: mm3 writes s1 (bf16), then EinSumQuant multiplies by the per-head
        # scale and quantizes into q_out0 (int8)
        s1m = torch.empty((N, HN, Dkv), dtype=dt, device=dev)
        _q_head_kernel[(ceil_div(N, 16), HN)](
            s4, wk, cs, sn, s1m, q2, N, HN, HD, Dnope, Drd, Dkv, Dkv,
            ROPE_IN_Q=False, BLK_M=16, BLK_K=next_pow2(Dnope), BLK_N=128, HALF=Drd // 2)
        _q_nope_quant_kernel[(N,)](
            s1m, q_nope_scale.to(dt).contiguous().reshape(-1), q,
            N, HN, Dkv, BLK_M=16, BLK_D=256)
        _kv_cache_i8_kernel[(ceil_div(N, 16),)](
            s3[:, :Dkvr].contiguous(), gamma2.contiguous(), cs, sn, slot,
            ctkv_scale.contiguous(), kv_cache_out0, kv_cache_out1,
            N, Dkv, Drd, Dkvr, 1.0 / Dkv, float(eps),
            BLK_M=16, BLK_D=256, HALF=Drd // 2)
        return q, kv_cache_out0, q2, kv_cache_out1

    _q_head_kernel[(ceil_div(N, 16), HN)](                               # 7
        s4, wk, cs, sn, q, q2, N, HN, HD, Dnope, Drd, Dkv, qline,
        ROPE_IN_Q=False, BLK_M=16, BLK_K=next_pow2(Dnope), BLK_N=128, HALF=Drd // 2)
    _kv_cache_kernel[(ceil_div(N, 16),)](                                # 8
        s3[:, :Dkvr].contiguous(), gamma2.contiguous(), cs, sn, slot,
        kv_cache_out0, kv_cache_out1,
        N, Dkv, Drd, Dkvr, qline, 1.0 / Dkv, float(eps),
        SPLIT_OUT=True, NZ=bool(nz), BLK_M=16, BLK_D=256, HALF=Drd // 2)
    return q, kv_cache_out0, q2, kv_cache_out1
