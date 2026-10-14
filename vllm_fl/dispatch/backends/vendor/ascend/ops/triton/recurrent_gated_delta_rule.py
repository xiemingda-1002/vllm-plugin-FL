# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""recurrent_gated_delta_rule -- AscendC to Triton

Source: csrc/ascend/attention/recurrent_gated_delta_rule/
Interface (op_host/recurrent_gated_delta_rule_def.cpp L21-81 + _infershape.cpp L38-81):
    Input : query              [T, NK, DK]  bf16
            key                [T, NK, DK]  bf16
            value              [T, NV, DV]  bf16
            beta               [T, NV]      bf16
            state              [S, NV, DV, DK]  bf16 / fp32     (S = number of state slots)
            actual_seq_lengths [B+1]        int32 (cumulative prefix sums, cu_seqlens)
            ssm_state_indices  [T]          int32 (state slot of each token)
            g                  [T, NV]      fp32 (OPTIONAL, log space)
            gk                 [T, NV, DK]  fp32 (OPTIONAL, log space, k-direction gate)
            num_accepted_tokens[B]          int32 (OPTIONAL)
    Output: out                [T, NV, DV]  same dtype as value
            state              [S, NV, DV, DK]  same dtype as the state input
    Attr  : scale_value = 1.0

Formula (op_kernel/recurrent_gated_delta_rule.h):
    Each (batch, v-head, v block) recurses independently, with the state denoted S[DV, DK]:
      gama  = exp(g[t, h])                    # L437 Exp(gamaInUb); L477 gama_
      gammaK= exp(gk[t, h, :])                # L226 Exp(gamaKInUb)
      S    *= gama                            # L366-368 Muls(stateInUb, stateInUb, gama_)
      S    *= gammaK                          # L369-371 MatVecMul(stateInUb, gamaKInUb, ..., isAdd=false)
      d     = v[t,h,:] - S @ k[t, hk, :]      # L375 MatVecMul + L377 ReduceSumDispatch + L379 Sub
      d    *= beta[t, h]                      # L381 Muls(delta, delta, beta_)
      S    += outer(d, k)                     # L383 Broadcast + L385 MatVecMul(..., isAdd=true)
      o     = S @ (q[t, hk, :] * scale)       # L387 + L389 + L242 Muls(q, q, scale_)
      out[t, h, :] = o                        # L398/L402-408 Cast(CAST_RINT) + DataCopyPad
      final_state[ssm_state_indices[t], h] = S     # L476 + L410-417, written for every token

    - hk = h / (NV / NK)                        # L445 uses head_i/(NV_/NK_) for qkOffset
    - Initial state slot: stateTokenIdx = seq0; with num_accepted_tokens it becomes seq0 + n - 1
      (L181-188)
    - g/gk are exponentiated right after loading (L437 / L226), i.e. the inputs are in log space
    - All intermediates are computed in fp32 (cast to fp32 first) and finally CAST_RINT back to
      the output dtype

Implementation: one program per (batch, v-head, v block), recursing serially over the tokens
inside the program.
(The AscendC kernel uses IsCurrentBlock to hand (batch,head) pairs to different cores, but each
block is computed by a single core, so the results are equivalent.)

Known limitation (Ascend bishengir compiler defect, not an algorithmic issue) -- **resolved**:
    Original symptom: with an fp32 state (hence an fp32 final state), bishengir crashed LLVM in
    the ConvertLinalgRToBinary pass ("PLEASE submit a bug report to llvm-project ... crash
    backtrace"), failing within 0.5s, reproducible 5/5, also in a standalone minimal process, and
    unrelated to load or OOM.
    Root-cause bisection (three variants, each in its own process):
      1. Removing the FINAL store from the dynamic loop => **it compiles** (proving that this
         store is the trigger);
      2. Keeping the store but rewriting it as a 1-D `tl.reshape` plus an arithmetic offset =>
         **still crashes** (earlier attempts stopped here and wrongly concluded that there was a
         second, unidentified trigger);
      3. Making the loop-carried state itself 1-D (reshaping back to 2-D for the math) and
         storing the 1-D carried value directly => **fp32/fp16/bf16 all compile**, while
         **keeping the per-token write-back semantics**.
    => The trigger is precisely "storing a 2-D loop-carried value inside a dynamic loop body",
       not a "2-D store" as such; changing only the store shape while the carried value stays 2-D
       is not enough.  `chunk_gated_delta_rule_fwd_h` was usable much earlier because its store
       sits outside the dynamic loop (or inside a statically unrolled loop).
    Numerics (host fp64 step-by-step recurrence reference):
      fp32 state: final max|delta| = 1.226e-07 (G) / 3.489e-07 (G+GK) <= fp32 floor 2e-6;
      bf16 state: final max|delta| = 1.942e-03 / 3.842e-03 <= bf16 floor 4e-3;
      out max|delta| ~ 9.6e-04 (bf16 output floor); **state slots that are never written stay
      unchanged = 0**.
    Workarounds tried and found **ineffective** (recorded here to avoid repeating the work):
    dropping the no-op cast, a separate constexpr branch for FINAL, unifying `.to(dtype)`,
    hoisting 2-D address pointers, changing only the store to 1-D, smaller tiles (BLK_V=8/16/32),
    various num_warps/num_stages values, and retries.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["recurrent_gated_delta_rule"]


@triton.jit
def _rgdr_kernel(
    Q, KEY, VAL, BETA, GAMA, GAMAK, STATE, CU, SSM, OUT, FINAL, NACC,
    T, NK, NV, DK, DV, SCALE,
    HAS_G: tl.constexpr,
    HAS_GK: tl.constexpr,
    HAS_ACC: tl.constexpr,
    FINAL_F32: tl.constexpr,
    STATE_F32: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_V: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_b = tl.program_id(2)

    seq0 = tl.load(CU + pid_b).to(tl.int32)
    seq1 = tl.load(CU + pid_b + 1).to(tl.int32)
    if seq1 <= seq0:
        return

    if HAS_ACC:
        nacc = tl.load(NACC + pid_b).to(tl.int32)
        state_tok = seq0 + nacc - 1
    else:
        state_tok = seq0
    slot = tl.load(SSM + state_tok).to(tl.int32)

    k_off = tl.arange(0, BLK_K)
    kmask = k_off < DK
    v_off = pid_v * BLK_V + tl.arange(0, BLK_V)
    vmask = v_off < DV

    hk = pid_h // (NV // NK)

    sptr = STATE + ((slot * NV + pid_h) * DV + v_off[:, None]) * DK + k_off[None, :]
    # Always go through .to(tl.float32) (a no-op for fp32, but the convert op is kept: without
    # this constexpr branch the IR matches the fixed chunk kernel and bishengir no longer crashes)
    sm2 = tl.load(sptr, mask=vmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)

    # Workaround for an Ascend bishengir bug: **the loop-carried state must be 1-D**.
    # Carrying the state as a 2-D [BLK_V, BLK_K] value and storing it inside the dynamic loop
    # made bishengir crash LLVM in the ConvertLinalgRToBinary pass; rewriting only the store as a
    # 1-D `tl.reshape` plus arithmetic offset is **not enough** (the carried value is still 2-D).
    # Carrying a 1-D `smf` and reshaping back to 2-D for the math makes fp32/fp16/bf16 all
    # compile.  Three-way experiment: dropping the store from the loop => compiles; a 2-D carried
    # value => crash; a 1-D carried value => compiles and **keeps the per-token write-back
    # semantics** (unlike moving the store out of the loop, which loses intermediate writes).
    i1 = tl.arange(0, BLK_V * BLK_K)
    row_i = i1 // BLK_K                     # row index inside the tile (= v direction)
    col_i = i1 % BLK_K                      # column index inside the tile (= k direction)
    vglob = pid_v * BLK_V + row_i
    off1 = vglob * DK + col_i
    fm1 = (vglob < DV) & (col_i < DK)
    smf = tl.reshape(sm2, [BLK_V * BLK_K])  # 1-D loop-carried variable

    t = seq0
    while t < seq1:
        sm = tl.reshape(smf, [BLK_V, BLK_K])    # 2-D view used by the math
        q = tl.load(Q + (t * NK + hk) * DK + k_off, mask=kmask, other=0.0).to(tl.float32) * SCALE
        kk = tl.load(KEY + (t * NK + hk) * DK + k_off, mask=kmask, other=0.0).to(tl.float32)
        vv = tl.load(VAL + (t * NV + pid_h) * DV + v_off, mask=vmask, other=0.0).to(tl.float32)
        bb = tl.load(BETA + t * NV + pid_h).to(tl.float32)

        if HAS_G:
            gm = tl.exp(tl.load(GAMA + t * NV + pid_h).to(tl.float32))
            sm = sm * gm
        if HAS_GK:
            gk = tl.exp(tl.load(GAMAK + (t * NV + pid_h) * DK + k_off, mask=kmask, other=0.0).to(tl.float32))
            sm = sm * gk[None, :]

        d = vv - tl.sum(sm * kk[None, :], axis=1)
        d = d * bb
        sm = sm + d[:, None] * kk[None, :]

        o = tl.sum(sm * q[None, :], axis=1)
        tl.store(OUT + (t * NV + pid_h) * DV + v_off, o.to(OUT.dtype.element_ty), mask=vmask)

        smf = tl.reshape(sm, [BLK_V * BLK_K])   # write back into the 1-D carried variable
        slot_t = tl.load(SSM + t).to(tl.int32)
        fbase = (slot_t * NV + pid_h) * DV * DK
        tl.store(FINAL + fbase + off1, smf.to(FINAL.dtype.element_ty), mask=fm1)
        t += 1


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    scale_value: float = 1.0,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent to the AscendC RecurrentGatedDeltaRule (gated delta rule for decoding).

    The argument order matches op_host/recurrent_gated_delta_rule_def.cpp
    (query, key, value, beta, state, actual_seq_lengths, ssm_state_indices, g, gk,
    num_accepted_tokens).
    `scale=` is also accepted as an alias for `scale_value`: the call site
    vllm_ascend/ops/gdn.py L556-567 / L581-593 passes `scale=key.shape[-1]**-0.5`, which
    corresponds to the op_host attribute `scale_value` (L81).

    Args:
        query/key: [T, NK, DK] bf16
        value:     [T, NV, DV] bf16
        beta:      [T, NV] bf16
        state:     [S, NV, DV, DK] -- bf16/fp16/fp32; any other dtype raises
                   NotImplementedError explicitly (see the state dtype support matrix below)
        actual_seq_lengths: [B+1] int32 cumulative sequence lengths
        ssm_state_indices:  [T] int32
        g:   [T, NV] fp32 (log space); None means no decay
        gk:  [T, NV, DK] fp32 (log space); None means no k-direction gating
        num_accepted_tokens: [B] int32; when given, the initial state comes from slot seq0+n-1
        scale_value: scalar attribute (may also be passed as the keyword `scale`)

    Returns:
        (out [T, NV, DV] = value.dtype, final_state [S, NV, DV, DK] = state.dtype)

    state dtype support matrix (declared explicitly, no silent fallback):
        bf16 / fp16 / **fp32** -> all usable (fp32 was enabled by carrying the loop state as a
                                  1-D value and reshaping it back to 2-D for the math, which
                                  avoids the bishengir LLVM crash in the ConvertLinalgRToBinary
                                  pass; see the "Known limitation" section at the top of the file)
    """
    if scale is not None:
        scale_value = scale

    if state.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError(f"unsupported state dtype {state.dtype} (only bf16/fp16/fp32)")
    assert query.shape == key.shape, f"query {tuple(query.shape)} != key {tuple(key.shape)}"
    T, NK, DK = (int(s) for s in query.shape)
    NV = int(value.shape[1])
    DV = int(value.shape[2])
    assert NV % NK == 0, f"NV={NV} must be a multiple of NK={NK}"
    S = int(state.shape[0])
    assert tuple(state.shape) == (S, NV, DV, DK), f"state shape {tuple(state.shape)} mismatch"
    assert tuple(beta.shape) == (T, NV), f"beta shape {tuple(beta.shape)} != {(T, NV)}"

    cu = actual_seq_lengths.contiguous().to(torch.int32)
    B = int(cu.numel()) - 1
    ssm = ssm_state_indices.contiguous().to(torch.int32)

    out = torch.empty((T, NV, DV), dtype=value.dtype, device=value.device)
    final = state.contiguous().clone()

    BLK_K = next_pow2(DK)
    BLK_V = min(next_pow2(DV), 64)
    grid = (NV, ceil_div(DV, BLK_V), max(B, 1))
    _rgdr_kernel[grid](
        query.contiguous(), key.contiguous(), value.contiguous(), beta.contiguous(),
        (g if g is not None else query).contiguous(),
        (gk if gk is not None else query).contiguous(),
        state.contiguous(), cu, ssm, out, final,
        (num_accepted_tokens if num_accepted_tokens is not None else cu).contiguous(),
        T, NK, NV, DK, DV, float(scale_value),
        HAS_G=bool(g is not None),
        HAS_GK=bool(gk is not None),
        HAS_ACC=bool(num_accepted_tokens is not None),
        FINAL_F32=bool(final.dtype == torch.float32),
        STATE_F32=bool(state.dtype == torch.float32),
        BLK_K=BLK_K,
        BLK_V=BLK_V,
    )
    return out, final
