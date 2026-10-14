# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""dequant_swiglu_quant -- AscendC -> Triton

Source: csrc/ascend/moe/dequant_swiglu_quant/
Interface (op_host/dequant_swiglu_quant_def.cpp L30-101 + infershape L44-69):
    Input : x                [M, 2H]  (int32 / bf16 / fp16)
            weight_scale     [2H] or [G, 2H]   fp32
            activation_scale [M] or [G, M]     fp32   (per-row scaling)
            bias             [2H] or [G, 2H]   fp32/fp16/bf16/int32
            quant_scale      [H]  or [G, H]    fp32   (per-channel smooth/quant scale)
            quant_offset     [H]  or [G, H]    fp32
            group_index      [G] or [G,2]      int32/int64 (token count per MoE group)
    Output: y     [M, H]       int8 (dst_type=2)
            scale [M]          fp32
    Attr  : activate_left=false, quant_mode="static", dst_type=2(int8),
            round_mode="rint", activate_dim=-1, swiglu_mode=0,
            clamp_limit=0.0, glu_alpha=1.0, glu_bias=0.0
   The split axis is activate_dim (the last dim by default), and along it the length of Y is
   halved (infershape L60-66).

Formula (op_kernel/dequant_swiglu_quant.h):
  Dequantization (L400-427, only when x is int32):
      xf = fp32(x)                                    # L401
      xf = weight_scale * xf                          # L405 (per-channel broadcast; with
                                                      #   several groups the row of that group
                                                      #   is used, see below)
      xf = activation_scale * xf                      # L413 (per-row broadcast, if provided)
      xf = xf + bias                                  # L424 (floating-point bias; an int32
                                                      #   bias is added earlier at L386-390)
  SwiGLU (L432-472, the plain path with swiglu_mode=0):
      actOffset_ = actRight * H, gateOffset_ = H - actOffset_     # L182-183
      activate_left=true  -> act = xf[:, :H], gate = xf[:, H:]    # tiling_base L155:
                                                                  #   false <-> activation on
                                                                  #   the right
      activate_left=false -> act = xf[:, H:], gate = xf[:, :H]
      y = gate * (act / (1 + exp(-act)))              # L459-472 (plain SiLU when glu_alpha=1)
  SwiGLU-OAI (L630-674, swiglu_mode=1, i.e. swigluoai_uninterleave):
      if clamp_limit > 0:  gate = min(max(gate, -c), c)        # L650-656 (two-sided clamp)
      gate = gate + glu_bias                                   # L657 (**unconditional** add)
      if clamp_limit > 0:  act  = min(act, c)                  # L659-663 (act: upper bound only)
      y = gate * (act / (1 + exp(-glu_alpha * act)))           # L664-672
      WARNING: **this implementation does not use the OAI semantics by default** (see the
      "semantics decision" below); set `DSQ_OAI=1` to enable it.
  Semantics decision (three independent pieces of measured evidence):
       (a) **the native payload completely ignores** swiglu_mode/clamp_limit/glu_alpha/
           glu_bias (changing only those five attributes for the same input gives
           **bit-identical output**; a hand computation resolves it: inverting native mode 1
           yields plain SiLU `80.000000`, whereas OAI by hand yields `49.999998`)
           => native == plain SiLU;
       (b) the model (DeepSeek-V4-Flash-w8a8-mtp) really calls it with
           `swiglu_mode=1, clamp_limit=10.0, glu_alpha=1.702, glu_bias=1.0,
           group_index=None, x=[8192,512] int32` (H=256, the shared-expert call site
           fused_moe.py:720, 40 times per rank -- the wrapper dump holds 320 entries, all with
           the same shape);
       (c) **the native variant produced correct output** (a fluent Chinese sentence) while the
           variant that faithfully implemented OAI produced garbled text; on the real shape the
           two semantics differ by a **median of 18.0%** in the dequantized values, with a
           scale max of 0.9914 => enough to break the model.
       => **the default matches the native behavior (plain SiLU, ignoring clamp/alpha/bias)**;
       the OAI semantics are kept behind `DSQ_OAI=1`.
  Dynamic quantization (L677-737, quant_mode="dynamic"):
      if smooth: y = y * quant_scale[H]               # L681-694
      rowmax = max_d |y|                              # L696-711
      scale  = rowmax * (1/127)                       # L714 (DYNAMIC_QUANT_FACTOR, L33)
      yq     = RINT(y / scale) -> int8                # L732-735 + L789-800 (CAST_RINT =
                                                      #   round-half-to-even)
  MoE multi-group (op_kernel/dequant_swiglu_quant_cut_group.h L38-58 + .h L224-238):
      group_index[g] = "the number of tokens in group g" (**not** a prefix sum; the call site
      is vllm_fl/.../moe_mlp.py L248 cumsum_group_list(group_list, group_list_type, 1), where
      dst_list_type=1 means per-group counts, see moe_mlp.py L54-83)
      Group g covers the output rows [sum_{k<g} count_k, sum_{k<=g} count_k) and uses row
      weight_scale[g] (L226-234)
      With speGroupType=1 (group_index is [G,2]): column 0 = the real group id, column 1 = the
      token count of that group (L44-47)

Triton implementation notes / semantic assumptions:
  1. Covered: x=int32 + quant_mode="dynamic" (or 1) + dst_type=int8 + activate_dim=-1 +
     swiglu_mode in {0,1} + group_index in {None, [G] counts, [G,2] (group id, count)} +
     activate_left in {T,F}.
     Not implemented: static quantization, the bf16/fp16 passthrough path (not verified, so not
     reproduced), int4/fp8 output, the int32 bias pre-add, activation_scale of shape [G,M], and
     activate_dim != the last dimension.
  2. Rounding uses round-half-to-even (CAST_RINT), consistent with torch.round.
  3. Rows with rowmax==0 get a scale of 0 (AscendC degenerates to 0/0 there; the test data
     avoids that case).
  4. The group-id mapping is produced by a separate small kernel that accumulates group by
     group (this avoids the Ascend backend pitfall of extracting a scalar from an int32 vector
     with tl.sum(tl.where(...))).

Backend behavior notes (910_93 + BiShengIR/FlagTree):
  The stores of both kernels in this file have the form "**scalar row index + 1-D column tile +
  1-D mask**" (`row = tl.program_id(0)`; `mask = hm = h < H`), which is not the
  "[BLK_D,BLK_C] 2-D block + 2-D masked store" shape. Measured with a double sentinel fill,
  every case gave dropped=0 for H in {17,24,33,48,100,127,129,256,1000,7168} (including
  activate_left / swiglu_mode1), the end-to-end result matched the fp64 reference element by
  element (0 values differing by more than 1 LSB), and two runs on the same input produced 0
  differences.
  => this file **needs no rewrite**.
"""

from __future__ import annotations

import os as _os

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["dequant_swiglu_quant"]

DYNAMIC_QUANT_FACTOR = 1.0 / 127.0

# The **default** semantics of `swiglu_mode=1` match the native payload (plain SiLU,
#    **ignoring** clamp_limit / glu_alpha / glu_bias). See the long comment inside
#    `_dsq_swiglu_kernel` for the reasoning (the native kernel has zero side effects when
#    measured + the model's real arguments + the native variant producing correct output while
#    the OAI variant produced garbage).
#    `DSQ_OAI=1` enables the SwiGLU-OAI semantics of the AscendC source at L630-674 (verified
#    in both directions and kept for the record).
#    WARNING: this switch is read at **module import time**, so pinning both semantics in a
#    unit test requires `importlib.reload` (see the test file).
_OAI = _os.environ.get("DSQ_OAI", "0") == "1"

# Official-mode switch + cache for the physical Vector core count.
#   The official Triton-Ascend guide (docs/en/programming_guide/vector_operator.md) says:
#     "not to create as many grid programs as possible, but to keep the launch close to the
#      number of physical Vector Cores and let each program process multiple tiles in an
#      inner loop"  => grid=(NUM_CORE,) + `for b in range(pid, NB, NUM_CORE)`
#   The default `DSQ_OFFICIAL=1` means grid = min(physical core count, row blocks) plus an
#   in-core runtime stride loop, with **no compile-time row unroll**, and it is **not bound by
#   the grid <= 65535 limit**; `DSQ_OFFICIAL=0` goes back to the old "large grid +
#   compile-time constant unroll" form, kept only as a single-variable control/regression.
#   NOTE (honest, measured): **non-determinism was not fixed by this mode** -- a
#   single-variable comparison shows that the pre-fix baseline code (BLK_M=8) was already
#   bit-identical with a **private cache** and only became non-deterministic with a **shared
#   cache** => the decisive variable is **compile-cache isolation**.
#   The real value of the official mode is that it follows the documented form, removes the
#   grid limit and removes the compile-time unroll (maintainability/scalability); it is also
#   **bit-identical** to the old mode (verified twice with the real shape and both sets of
#   semantics).
_OFFICIAL = _os.environ.get("DSQ_OFFICIAL", "1") == "1"
_NUM_CORE_CACHE: dict = {}


@triton.jit
def _dsq_group_ids_kernel(
    GROUP_INDEX, GID,
    M, G,
    GROUP_STRIDE: tl.constexpr,
    BLK: tl.constexpr,
):
    """Expand the "token count per group" into a per-row group id (the group id is the row of
    the group in weight_scale)."""
    pid = tl.program_id(0)
    rows = pid * BLK + tl.arange(0, BLK)
    gid = tl.zeros([BLK], dtype=tl.int32)
    off = 0
    for g in range(G):
        if GROUP_STRIDE == 2:
            c = tl.load(GROUP_INDEX + g * 2 + 1).to(tl.int32)   # [G,2]: column 1 = token count
            gv = tl.load(GROUP_INDEX + g * 2).to(tl.int32)      # column 0 = real group id
        else:
            c = tl.load(GROUP_INDEX + g).to(tl.int32)
            gv = tl.full([], g, tl.int32)
        gid = tl.where((rows >= off) & (rows < off + c), gv, gid)
        off += c
    tl.store(GID + rows, gid, mask=rows < M)


@triton.jit
def _dsq_swiglu_row(
    X, WSCALE, ASCALE, BIAS, QSCALE, GID, YBUF,
    M, H, row, dh, nloop,
    CLAMP_LIMIT, GLU_ALPHA, GLU_BIAS,
    IS_INT32: tl.constexpr,
    HAS_ASCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_QSCALE: tl.constexpr,
    HAS_GROUP: tl.constexpr,
    ACTIVATE_LEFT: tl.constexpr,
    SWIGLU_MODE1: tl.constexpr,
    OAI: tl.constexpr,
    BLK_H: tl.constexpr,
):
    """**Single-row** body (the inner loop body of the original `_dsq_swiglu_kernel`, kept
    verbatim).

    It is driven by `_dsq_swiglu_kernel` with two different outer loops (the official runtime
    stride loop / the old compile-time unroll); both modes share this one copy => any numerical
    difference can only come from the loop structure (a single-variable comparison).
    """
    if row < M:
        g = 0
        if HAS_GROUP:
            g = tl.load(GID + row)
        wbase = g * 2 * H
        for i in range(nloop):
            h = i * BLK_H + dh
            hm = h < H
            if ACTIVATE_LEFT:
                a = tl.load(X + row * 2 * H + h, mask=hm, other=0.0).to(tl.float32)
                gg = tl.load(X + row * 2 * H + H + h, mask=hm, other=0.0).to(tl.float32)
            else:
                gg = tl.load(X + row * 2 * H + h, mask=hm, other=0.0).to(tl.float32)
                a = tl.load(X + row * 2 * H + H + h, mask=hm, other=0.0).to(tl.float32)
            if IS_INT32:
                if HAS_GROUP:
                    ws_a = tl.load(WSCALE + wbase + h, mask=hm, other=0.0).to(tl.float32)
                    ws_g = tl.load(WSCALE + wbase + H + h, mask=hm, other=0.0).to(tl.float32)
                else:
                    if ACTIVATE_LEFT:
                        ws_a = tl.load(WSCALE + h, mask=hm, other=0.0).to(tl.float32)
                        ws_g = tl.load(WSCALE + H + h, mask=hm, other=0.0).to(tl.float32)
                    else:
                        ws_g = tl.load(WSCALE + h, mask=hm, other=0.0).to(tl.float32)
                        ws_a = tl.load(WSCALE + H + h, mask=hm, other=0.0).to(tl.float32)
                a = a * ws_a
                gg = gg * ws_g
            if HAS_ASCALE:
                asc = tl.load(ASCALE + row).to(tl.float32)
                a = a * asc
                gg = gg * asc
            if HAS_BIAS:
                if HAS_GROUP:
                    ba = tl.load(BIAS + wbase + h, mask=hm, other=0.0).to(tl.float32)
                    bg = tl.load(BIAS + wbase + H + h, mask=hm, other=0.0).to(tl.float32)
                elif ACTIVATE_LEFT:
                    ba = tl.load(BIAS + h, mask=hm, other=0.0).to(tl.float32)
                    bg = tl.load(BIAS + H + h, mask=hm, other=0.0).to(tl.float32)
                else:
                    bg = tl.load(BIAS + h, mask=hm, other=0.0).to(tl.float32)
                    ba = tl.load(BIAS + H + h, mask=hm, other=0.0).to(tl.float32)
                a = a + ba
                gg = gg + bg
            if SWIGLU_MODE1 and OAI:
                # L650-663: gate is clamped on both sides and then gets glu_bias; act is
                # clamped above only
                gg = tl.where(CLAMP_LIMIT > 0.0, tl.minimum(gg, CLAMP_LIMIT), gg)
                gg = tl.where(CLAMP_LIMIT > 0.0, tl.maximum(gg, -CLAMP_LIMIT), gg)
                gg = gg + GLU_BIAS
                a = tl.where(CLAMP_LIMIT > 0.0, tl.minimum(a, CLAMP_LIMIT), a)
                y = gg * (a / (1.0 + tl.exp(0.0 - GLU_ALPHA * a)))       # L664-672
            else:
                # **Default path: match the native payload = plain SiLU**
                #   Evidence (three independent, all measured):
                #     1) the native kernel **completely ignores** swiglu_mode/clamp_limit/
                #        glu_alpha/glu_bias (changing only those five attributes for the same
                #        input gives **bit-identical output**; a hand computation resolves it:
                #        inverting native mode 1 yields plain SiLU `80.000000`, whereas OAI by
                #        hand should yield `49.999998`);
                #     2) the real model call (320 wrapper dump entries, all the same shape)
                #        passes `swiglu_mode=1, clamp_limit=10.0, glu_alpha=1.702, glu_bias=1.0,
                #        group_index=None, x=[8192,512] int32` (H=256, fused_moe.py:720);
                #     3) **the native variant produced correct output** (a fluent Chinese
                #        sentence) while the variant faithfully implementing OAI produced
                #        garbled text; on the real shape the two semantics differ by a
                #        **median of 18.0%** in the dequantized values (scale max 0.9914)
                #        => enough to break the model.
                #   => **the model expects plain SiLU**. The OAI semantics are **kept** behind
                #   `DSQ_OAI=1` (verifiable in both directions and kept for the record).
                y = gg * (a / (1.0 + tl.exp(0.0 - a)))                   # L459-472
            if HAS_QSCALE:
                qs = tl.load(QSCALE + h, mask=hm, other=1.0).to(tl.float32)
                y = y * qs
            tl.store(YBUF + row * H + h, tl.where(hm, y, 0.0), mask=hm)


def _num_vectorcore(dev_idx):
    """Number of physical Vector cores (used as the fixed grid size in official mode). Falls
    back when it cannot be queried; can be overridden with DSQ_NUM_CORE."""
    env = _os.environ.get("DSQ_NUM_CORE")
    if env:
        return max(1, int(env))
    key = ("vc", dev_idx if dev_idx is not None else -1)
    if key in _NUM_CORE_CACHE:
        return _NUM_CORE_CACHE[key]
    n = None
    try:
        from triton.runtime import driver as _drv

        props = _drv.active.utils.get_device_properties(dev_idx)
        for k in ("num_vectorcore", "num_vector_core", "num_aivector_core", "vector_core_num"):
            if k in props:
                n = int(props[k])
                break
    except Exception:
        n = None
    if not n or n <= 0:
        n = 40          # measured 40 on this machine (same as the official doc example);
                        # fallback only, can be overridden with DSQ_NUM_CORE
    _NUM_CORE_CACHE[key] = n
    return n


@triton.jit
def _dsq_swiglu_kernel(
    X, WSCALE, ASCALE, BIAS, QSCALE, GID, YBUF,
    M, H,
    CLAMP_LIMIT, GLU_ALPHA, GLU_BIAS,
    NUM_CORE, NUM_RB, BLK_M_RT,
    IS_INT32: tl.constexpr,
    HAS_ASCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_QSCALE: tl.constexpr,
    HAS_GROUP: tl.constexpr,
    ACTIVATE_LEFT: tl.constexpr,
    SWIGLU_MODE1: tl.constexpr,
    OAI: tl.constexpr,
    BLK_H: tl.constexpr,
    BLK_M: tl.constexpr,
    OFFICIAL: tl.constexpr,
):
    """Switchable to the **official Triton-Ascend mode**.

    Official guidance (triton-ascend docs/en/programming_guide/vector_operator.md):
        "The key is **not to create as many grid programs as possible**, but to keep the
         launch **close to the number of physical Vector Cores** and let **each program
         process multiple tiles in an inner loop**."
        grid = (NUM_CORE,)
        for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):   # runtime stride loop
            ...

    * `OFFICIAL=True` (the default, `DSQ_OFFICIAL=1`): grid = min(physical core count, row
      blocks), with the **runtime** stride loop `for rb in range(pid, NUM_RB, NUM_CORE)`
      inside the core; within a row block, `for mi in range(BLK_M_RT)` is used (**BLK_M_RT is
      a runtime scalar => a runtime loop**) => there is **no compile-time constant unroll over
      rows** at all, which removes the possibility of a constant unroll being reordered in
      parallel by mistake (`import vllm_fl` sets `TRITON_ALL_BLOCKS_PARALLEL=1`, under which
      BLK_M=8 became non-deterministic).
    * `OFFICIAL=False` (`DSQ_OFFICIAL=0`, kept as a control/regression): grid =
      ceil_div(M, BLK_M) with the compile-time unroll `for mi in range(BLK_M)` (the old form,
      including the grid<=65535 assertion).
    Both modes run the **same** `_dsq_swiglu_row` => a single-variable comparison.
    """
    pid = tl.program_id(0)
    dh = tl.arange(0, BLK_H)
    nloop = tl.cdiv(H, BLK_H)
    if OFFICIAL:
        for rb in range(pid, NUM_RB, NUM_CORE):
            for mi in range(BLK_M_RT):
                row = rb * BLK_M + mi
                _dsq_swiglu_row(X, WSCALE, ASCALE, BIAS, QSCALE, GID, YBUF,
                                M, H, row, dh, nloop,
                                CLAMP_LIMIT, GLU_ALPHA, GLU_BIAS,
                                IS_INT32=IS_INT32, HAS_ASCALE=HAS_ASCALE, HAS_BIAS=HAS_BIAS,
                                HAS_QSCALE=HAS_QSCALE, HAS_GROUP=HAS_GROUP,
                                ACTIVATE_LEFT=ACTIVATE_LEFT, SWIGLU_MODE1=SWIGLU_MODE1,
                                OAI=OAI, BLK_H=BLK_H)
    else:
        # WARNING: the old form is kept for reference: dim 0 of the grid is limited to 65535
        # (measured: M=65536 fails while M=65537 passes again => non-monotonic)
        for mi in range(BLK_M):
            row = pid * BLK_M + mi
            _dsq_swiglu_row(X, WSCALE, ASCALE, BIAS, QSCALE, GID, YBUF,
                            M, H, row, dh, nloop,
                            CLAMP_LIMIT, GLU_ALPHA, GLU_BIAS,
                            IS_INT32=IS_INT32, HAS_ASCALE=HAS_ASCALE, HAS_BIAS=HAS_BIAS,
                            HAS_QSCALE=HAS_QSCALE, HAS_GROUP=HAS_GROUP,
                            ACTIVATE_LEFT=ACTIVATE_LEFT, SWIGLU_MODE1=SWIGLU_MODE1,
                            OAI=OAI, BLK_H=BLK_H)


@triton.jit
def _dsq_quant_row(
    YBUF, YOUT, SCALE,
    M, H, row, dh, nloop, QFACTOR,
    BLK_H: tl.constexpr,
):
    """**Single-row** dynamic quantization body (the inner loop body of the original
    `_dsq_quant_kernel`, kept verbatim)."""
    if row < M:
        mx = tl.zeros([1], dtype=tl.float32)
        for i in range(nloop):
            h = i * BLK_H + dh
            hm = h < H
            v = tl.load(YBUF + row * H + h, mask=hm, other=0.0)
            mx = tl.maximum(mx, tl.expand_dims(tl.max(tl.abs(v), axis=0), 0))
        sc = tl.sum(mx) * QFACTOR
        sc = tl.where(sc > 0.0, sc, 0.0)
        tl.store(SCALE + row, sc)
        for i in range(nloop):
            h = i * BLK_H + dh
            hm = h < H
            v = tl.load(YBUF + row * H + h, mask=hm, other=0.0)
            v = tl.where(sc > 0.0, v / sc, 0.0)
            f = tl.floor(v)
            fr = v - f
            od = (f * 0.5) != tl.floor(f * 0.5)          # RINT: round-half-to-even
            q = tl.where((fr > 0.5) | ((fr == 0.5) & od), f + 1.0, f)
            tl.store(YOUT + row * H + h, q.to(tl.int8), mask=hm)


@triton.jit
def _dsq_quant_kernel(
    YBUF, YOUT, SCALE,
    M, H, QFACTOR,
    NUM_CORE, NUM_RB, BLK_M_RT,
    BLK_H: tl.constexpr,
    BLK_M: tl.constexpr,
    OFFICIAL: tl.constexpr,
):
    """Dynamic quantization: scale = rowmax/127, yq = RINT(y/scale).

    The loop structure matches `_dsq_swiglu_kernel` (switchable between the official runtime
    stride loop and the old compile-time unroll), sharing `_dsq_quant_row`.
    """
    pid = tl.program_id(0)
    dh = tl.arange(0, BLK_H)
    nloop = tl.cdiv(H, BLK_H)
    if OFFICIAL:
        for rb in range(pid, NUM_RB, NUM_CORE):
            for mi in range(BLK_M_RT):
                row = rb * BLK_M + mi
                _dsq_quant_row(YBUF, YOUT, SCALE, M, H, row, dh, nloop, QFACTOR, BLK_H=BLK_H)
    else:
        for mi in range(BLK_M):
            row = pid * BLK_M + mi
            _dsq_quant_row(YBUF, YOUT, SCALE, M, H, row, dh, nloop, QFACTOR, BLK_H=BLK_H)


def _is_dynamic(quant_mode) -> bool:
    if isinstance(quant_mode, str):
        return quant_mode.lower() == "dynamic"
    return int(quant_mode) == 1


def dequant_swiglu_quant(
    x: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    quant_scale: torch.Tensor | None = None,
    quant_offset: torch.Tensor | None = None,
    group_index: torch.Tensor | None = None,
    activate_left: bool = True,
    quant_mode="dynamic",
    dst_type: torch.dtype = torch.int8,
    round_mode: str = "rint",
    activate_dim: int = -1,
    swiglu_mode: int = 0,
    clamp_limit: float | None = 0.0,
    glu_alpha: float | None = 1.0,
    glu_bias: float | None = 0.0,
):
    """Equivalent of the AscendC DequantSwigluQuant (the int32 + dynamic + int8 main path).

    Args:
        x: [M, 2H] int32
        weight_scale: [2H] (single group) or [G, 2H] (multi-group, same length as
            group_index), fp32
        activation_scale: [M] fp32, optional (per-row/per-token scaling)
        bias: [2H] or [G, 2H], optional (floating point)
        quant_scale: [H] fp32, optional (per-channel smooth scale, dynamic mode)
        quant_offset: unused in dynamic mode
        group_index: None / [G] int32|int64 (token count per group) / [G,2] (column 0 =
            group id, column 1 = token count)
        activate_left: True -> the activation applies to the left half (production call sites
            pass True)
        quant_mode: "dynamic"/1 ("static"/0 is not implemented)
        dst_type: only torch.int8
        swiglu_mode: 0 = standard SiLU SwiGLU;
                     1 = **matches the native payload by default = plain SiLU** (ignoring
                     clamp/glu_alpha/glu_bias, see the "semantics decision" section of the
                     module docstring); `DSQ_OAI=1` restores the AscendC SwiGLU-OAI semantics
                     (clamp/glu_alpha/glu_bias take effect)
        clamp_limit / glu_alpha / glu_bias: None is treated as the default 0.0 / 1.0 / 0.0;
                     they only participate when `swiglu_mode=1` and `DSQ_OAI=1`

    Returns:
        (y [M, H] int8, scale [M] fp32)
    """
    assert _is_dynamic(quant_mode), "only quant_mode='dynamic'(1) is implemented"
    assert dst_type == torch.int8, "only dst_type=int8 is implemented"
    assert swiglu_mode in (0, 1), "swiglu_mode must be 0 or 1"
    assert x.dim() == 2, f"x must be [M,2H], got {tuple(x.shape)}"
    if activate_dim not in (-1, 1):
        raise NotImplementedError("only activate_dim=-1 is implemented (split the last dim in half)")
    c_lim = 0.0 if clamp_limit is None else float(clamp_limit)
    g_alpha = 1.0 if glu_alpha is None else float(glu_alpha)
    g_bias = 0.0 if glu_bias is None else float(glu_bias)
    if swiglu_mode == 0:
        assert c_lim == 0.0 and g_alpha == 1.0 and g_bias == 0.0, \
            "clamp/glu_alpha/glu_bias only take effect with swiglu_mode=1 " \
            "(the plain AscendC path does not have them)"

    M, two_h = x.shape
    assert two_h % 2 == 0, "the last dimension must be even"
    H = two_h // 2
    assert x.dtype == torch.int32, "this implementation only covers the x=int32 dequantization main path"
    assert bias is None or bias.dtype != torch.int32, "the int32 bias pre-add is not implemented"

    has_group = group_index is not None
    if has_group:
        assert group_index.dim() in (1, 2), "group_index must be [G] or [G,2]"
        n_groups = group_index.shape[0]
        group_stride = 2 if group_index.dim() == 2 else 1
        assert weight_scale.numel() == n_groups * two_h, \
            f"with several groups, weight_scale must be [{n_groups},{two_h}], got {tuple(weight_scale.shape)}"
    else:
        group_stride = 1
        assert weight_scale.numel() == two_h, f"weight_scale must be [{two_h}]"
    for nm, t, n in (("activation_scale", activation_scale, M),
                     ("quant_scale", quant_scale, H)):
        if t is not None:
            assert t.numel() == n, f"{nm} length {t.numel()} != {n}"

    dev = x.device
    x = x.contiguous()
    # WARNING: use zeros rather than empty: `_dsq_quant_kernel` reads YBUF unconditionally, so
    # if the upstream `_dsq_swiglu_kernel` does not fill it completely for any reason (launch
    # failure / out of bounds), empty would **silently** fold uninitialized memory into the
    # result (measured: after an upstream launch failure the downstream region became 0 or
    # garbage without any error). zeros makes that case a deterministic 0, and it only costs
    # M*H*4 bytes.
    ybuf = torch.zeros((M, H), dtype=torch.float32, device=dev)
    y = torch.empty((M, H), dtype=torch.int8, device=dev)
    scale = torch.empty((M,), dtype=torch.float32, device=dev)
    gid = torch.zeros((M,), dtype=torch.int32, device=dev)
    if has_group:
        blk = 256
        _dsq_group_ids_kernel[(ceil_div(M, blk),)](
            group_index.contiguous(), gid, M, n_groups,
            GROUP_STRIDE=group_stride, BLK=blk,
        )

    blk_h = 128 if H > 128 else next_pow2(H)
    # WARNING: dim 0 of the grid is limited to 65535 (measured: M=65536 gives `KernelLaunch
    #    failed ... coreDim is invalid` while M=65537 passes again: **non-monotonic**). The
    #    **old mode** (OFFICIAL=False) is still bound by it; in **official mode** the grid is
    #    the physical core count, so there is no such limit.
    #
    # **Important correction**: this used to say "BLK_M must be 1/2/4, never 8", based on "when
    #     BLK_M=8 two calls with the same input give different results". **That attribution was
    #     wrong and is disproved by measurement**:
    #       single-variable comparison (pre-fix baseline code, same script, same seed,
    #       `import vllm_fl`):
    #         **shared cache** `.triton/cache` : M=512/1024/4096/16384/32768 -> **all
    #                                            non-deterministic** (max|delta scale| =
    #                                            0.496/0.5; mismatching rows 640/1024,
    #                                            3712/4096, 16000/16384, 32384/32768)
    #         **private cache** `TRITON_CACHE_DIR=...` : M=512/1024/4096/16384/32768 -> **all
    #                                            bit-identical** (ok)
    #       => **the real independent variable is "shared vs private compile cache", not
    #          BLK_M** (the shared cache holds **34** artifacts under the same kernel name, and
    #          resolving by name at run time can pick the wrong version, which produces both a
    #          "wrong tail" (M>12288) and "two runs that differ").
    #       => Attribution corrected: `BLK_M 8 -> 4` only **happened to change the cache key**
    #          (i.e. it masked the symptom); it is **not the root-cause fix**. The root-cause
    #          fix is the official mode (no compile-time unroll, see below) **plus** the
    #          requirement that the criterion be evaluated with a **private cache**.
    #       Discipline (upgraded): **every "bit-identical / numerically correct" criterion must
    #          (1) really `import vllm_fl` and (2) use a private `TRITON_CACHE_DIR`** -- either
    #          of the two alone can produce a false conclusion.
    blk_m = max(1, min(4, int(_os.environ.get("DSQ_BLK_M", "4"))))
    n_row_blocks = ceil_div(M, blk_m)

    # **Official Triton-Ascend mode** (grid close to the physical core count + an in-core
    #   runtime stride loop)
    #   `DSQ_OFFICIAL=1` (the default): grid = min(physical Vector cores, row blocks); inside
    #       the core, `for rb in range(pid, n_row_blocks, num_core)` (runtime stride) plus
    #       `for mi in range(BLK_M_RT)` (BLK_M_RT is a **runtime** scalar => a runtime loop)
    #       => **no compile-time constant unroll over rows**.
    #   `DSQ_OFFICIAL=0`: the old form (grid = number of row blocks, `for mi in range(BLK_M)`
    #       with a constant unroll), which keeps the coreDim guard of grid <= 65535.
    if _OFFICIAL:
        num_core = _num_vectorcore(x.device.index)
        grid = (max(1, min(num_core, n_row_blocks)),)
    else:
        num_core = 1
        grid = (n_row_blocks,)
        assert n_row_blocks <= 65535, (
            f"M={M} requires grid={n_row_blocks} > 65535 (the coreDim limit); "
            f"increase DSQ_BLK_M (currently {blk_m}), or use DSQ_OFFICIAL=1 (official mode "
            f"uses grid=core count and has no such limit)")
    _dsq_swiglu_kernel[grid](
        x,
        weight_scale.contiguous(),
        activation_scale.contiguous() if activation_scale is not None else x,
        bias.contiguous() if bias is not None else x,
        quant_scale.contiguous() if quant_scale is not None else x,
        gid,
        ybuf,
        M, H,
        c_lim, g_alpha, g_bias,
        num_core, n_row_blocks, blk_m,
        IS_INT32=True,
        HAS_ASCALE=activation_scale is not None,
        HAS_BIAS=bias is not None,
        HAS_QSCALE=quant_scale is not None,
        HAS_GROUP=has_group,
        ACTIVATE_LEFT=bool(activate_left),
        SWIGLU_MODE1=(swiglu_mode == 1),
        OAI=_OAI,                     # AscendC OAI semantics only when DSQ_OAI=1 (plain SiLU
                                      # by default)
        BLK_H=blk_h,
        BLK_M=blk_m,
        OFFICIAL=_OFFICIAL,
    )
    _dsq_quant_kernel[grid](ybuf, y, scale, M, H, DYNAMIC_QUANT_FACTOR,
                            num_core, n_row_blocks, blk_m,
                            BLK_H=blk_h, BLK_M=blk_m, OFFICIAL=_OFFICIAL)
    return y, scale
