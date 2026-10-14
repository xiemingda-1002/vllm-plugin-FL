# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""inplace_partial_rotary_mul -- AscendC to Triton.

Source: csrc/ascend/attention/inplace_partial_rotary_mul/
Call-site signature (csrc/ascend/torch_binding.cpp L278-286 / L1246):
    inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2,
                               str rotary_mode, int[] partial_slice) -> ()
    where rotary_mode is a **string** mapped onto the AscendC integer mode:
        {"half": 0, "interleave": 1, "quarter": 2, "interleave-half": 3}
    Real call site (vllm_fl/attention/ascend/dsa_v1.py L1762-1768):
        inplace_partial_rotary_mul(o_proj_input.unsqueeze(1), cos, -sin,
                                   rotary_mode="interleave",
                                   partial_slice=[nope_head_dim, head_dim])
    i.e. r2 receives **-sin** and only the [512, 576) segment is rotated.
Interface (op_host/inplace_partial_rotary_mul_def.cpp L22-46):
    Input : x [dim0, dim1, dim2, D_all]  (4D, fp16/fp32/bf16)
            cos [r0, r1, 1, headDim]     (same dtype as x, or fp32)
            sin [r0, r1, 1, headDim]     (same dtype as cos)
    Output: x (same-name output, written back in place; the non-sliced range is preserved)
    Attr  : mode         = 0 HALF / 1 INTERLEAVE / 2 QUARTER / 3 DEEPSEEK_INTERLEAVE
            partial_slice = [start, end]  -> only the [start, end) range of the last dim is rotated
                          -> headDim = end - start

Shape rules (op_host/inplace_partial_rotary_mul_a3_tiling.cpp):
    - x must be 4D (L279-281); cos/sin must be 4D with exactly the same shape (L282-292)
    - dim2 of cos/sin must be 1 (L330-332); dim3 == end - start == headDim (L324-326)
    - dim0 must equal dim0 of x (L327-329)
    - on the A2/A3 path only mode 1 (INTERLEAVE) is accepted (L302-305)
    - dim1 may be 1 or x.dim1 (L336-346: r1dim1_==1 goes through a broadcast branch,
      otherwise per head)
      -> cos/sin are broadcast over (dim0, dim1, dim2) of x in the standard way
         (dim2 is always 1 and is broadcast over the N heads)

Formula sources:

  (a) A2/A3 membase kernel op_kernel/inplace_partial_rotary_mul.h L227-299 (ABA Process)
        y   = x * cos                       # L277  ComputeMul(y, x, r1)
        x'  = gather(x, ids)  ids[i] = i^1  # L279-281 swap adjacent elements in pairs
        x'  = x' * sin                      # L283
        x'  = x' * (-1)  where i even       # L286 InterleavedInversion: mask 0x5555...
        y   = y + x'                        # L288
      -> INTERLEAVE:
        out[2k]   = x[2k]*cos[2k]   - x[2k+1]*sin[2k]
        out[2k+1] = x[2k+1]*cos[2k+1] + x[2k]*sin[2k+1]

  (b) arch35 regbase VF op_kernel/apply_rotary_pos_emb_common.h:
        L43-48   RotaryPosEmbeddingMode { HALF=0, INTERLEAVE=1, QUARTER=2, DEEPSEEK_INTERLEAVE=3 }
        L50-53   HALF formula comment (implementation L82-112)
        L115-120 QUARTER formula comment (implementation L157-201)
        L205-317 INTERLEAVE (DeInterleave/Muls(-1)/Interleave, identical to (a); the two
                 cross-check each other)
        L319-451 DEEPSEEK_INTERLEAVE (input interleaved in pairs; output first half = even
                 stream, second half = odd stream)

  Equivalent formulation of the four modes (as implemented here; P = length of one pairing block):
        out[a]  = x[a]*cos[a] - x[b]*sin[a]
        out[wa] = ... see the table below (read indices (a,b), write indices (wa,wb))
            HALF(0)    : a=k,   b=P+k,   wa=a,    wb=b
            QUARTER(2) : HALF applied once over x[0:2P] and once over x[2P:4P]
            INTERLEAVE(1) : a=2k, b=2k+1, wa=a, wb=b
            DEEPSEEK(3)   : a=2k, b=2k+1, wa=k, wb=P+k

  Note: the original AscendC operator writes back into x **in place**; this Triton
  implementation returns a new y (x is left unmodified) with equivalent semantics
  (the non-sliced range equals x element for element).

  NOTE **Known contract deviation (deliberately kept and documented here)**:
     this implementation does **not** modify `x` and returns a **new tensor**; if a caller
     **discards the return value** the behavior would not match what the function name
     (`inplace_*`) suggests.
     - the call site (`vllm_fl/attention/ascend/dsa_v1.py`) uses the return value, so it is
       **functionally correct**;
     - the unit test `test_inplace_partial_rotary_mul.py` pins this deviation with two
       assertions (`x` is unmodified + the return value is an independent buffer) rather
       than pinning the AscendC in-place semantics;
     - if true in-place behavior is ever required, this comment, those two assertions and
       every call site that discards the return value must be updated together.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import ceil_div, next_pow2

__all__ = ["inplace_partial_rotary_mul", "MODE_HALF", "MODE_INTERLEAVE",
           "MODE_QUARTER", "MODE_DEEPSEEK_INTERLEAVE"]

MODE_HALF = 0
MODE_INTERLEAVE = 1
MODE_QUARTER = 2
MODE_DEEPSEEK_INTERLEAVE = 3


@triton.jit
def _rope_pair(X, COS, SIN, Y, x_base, cs_base,
               a_off, b_off, wa_off, wb_off, km):
    """RoPE for one pairing (a,b) -> (wa,wb) (1-D vectors, no 2-D broadcast).

        out[wa] = x[a]*cos[wa] - x[b]*sin[wa]
        out[wb] = x[b]*cos[wb] + x[a]*sin[wb]

    cos/sin are sampled at the **write** indices (true for all four modes, see the module
    docstring).
    """
    xa = tl.load(X + x_base + a_off, mask=km, other=0.0).to(tl.float32)
    xb = tl.load(X + x_base + b_off, mask=km, other=0.0).to(tl.float32)
    ca = tl.load(COS + cs_base + wa_off, mask=km, other=0.0).to(tl.float32)
    sa = tl.load(SIN + cs_base + wa_off, mask=km, other=0.0).to(tl.float32)
    cb = tl.load(COS + cs_base + wb_off, mask=km, other=0.0).to(tl.float32)
    sb = tl.load(SIN + cs_base + wb_off, mask=km, other=0.0).to(tl.float32)
    oa = xa * ca - xb * sa
    ob = xb * cb + xa * sb
    tl.store(Y + x_base + wa_off, oa.to(Y.dtype.element_ty), mask=km)
    tl.store(Y + x_base + wb_off, ob.to(Y.dtype.element_ty), mask=km)


@triton.jit
def _rope_pair_ilv(X, COS, SIN, Y, x_base, cs_base, k, km, P, ILV_WRITE: tl.constexpr):
    """RoPE for the interleaved pairing (2k, 2k+1) -- avoids the "1-D strided + masked" form.

    NOTE why it is written this way: the Ascend/FlagTree FLIR pass
    `InterleaveStatusWithMaskOptimization` (`InterleaveOptimization.cpp:635`) fails an
    assert outright for accesses like `tl.load(X + 2*k, mask=km)` + `tl.load(X + 2*k+1,
    mask=km)` -- a "1-D strided + masked" access (`assert(llvm::isa<Attribute>(
    extractSizes.back()))`, triggered by a dynamic size) -- and aborts the process.
    Expanding the pairing into a 2-D [BLK, 2] block with the mask broadcast over rows
    compiles and runs correctly (verified on both sides with a minimal reproducer on
    910_93 + FlagTree).

    ILV_WRITE=True  : reads, writes and cos/sin are all interleaved as (2k, 2k+1) (MODE_INTERLEAVE)
    ILV_WRITE=False : only the reads are interleaved; cos/sin and the writes use the
                      contiguous (k, P+k) form (MODE_DEEPSEEK_INTERLEAVE)
    """
    j = tl.arange(0, 2)
    m0 = j[None, :] == 0
    m2 = km[:, None]
    off2 = 2 * k[:, None] + j[None, :]

    x2 = tl.load(X + x_base + off2, mask=m2, other=0.0).to(tl.float32)
    xa = tl.sum(tl.where(m0, x2, 0.0), axis=1)
    xb = tl.sum(tl.where(m0, 0.0, x2), axis=1)

    if ILV_WRITE:
        c2 = tl.load(COS + cs_base + off2, mask=m2, other=0.0).to(tl.float32)
        s2 = tl.load(SIN + cs_base + off2, mask=m2, other=0.0).to(tl.float32)
        ca = tl.sum(tl.where(m0, c2, 0.0), axis=1)
        cb = tl.sum(tl.where(m0, 0.0, c2), axis=1)
        sa = tl.sum(tl.where(m0, s2, 0.0), axis=1)
        sb = tl.sum(tl.where(m0, 0.0, s2), axis=1)
        oa = xa * ca - xb * sa
        ob = xb * cb + xa * sb
        y2 = tl.where(m0, oa[:, None], ob[:, None])
        tl.store(Y + x_base + off2, y2.to(Y.dtype.element_ty), mask=m2)
    else:
        ca = tl.load(COS + cs_base + k, mask=km, other=0.0).to(tl.float32)
        sa = tl.load(SIN + cs_base + k, mask=km, other=0.0).to(tl.float32)
        cb = tl.load(COS + cs_base + P + k, mask=km, other=0.0).to(tl.float32)
        sb = tl.load(SIN + cs_base + P + k, mask=km, other=0.0).to(tl.float32)
        oa = xa * ca - xb * sa
        ob = xb * cb + xa * sb
        tl.store(Y + x_base + k, oa.to(Y.dtype.element_ty), mask=km)
        tl.store(Y + x_base + P + k, ob.to(Y.dtype.element_ty), mask=km)


@triton.jit
def _inplace_partial_rotary_mul_kernel(
    X, COS, SIN, Y,
    N_HEADS, SEQ, D,
    C1, HEAD_DIM, START, P,
    BR: tl.constexpr,      # whether dim0 of cos/sin takes part in the broadcast (i.e. equals x.dim0)
    SR: tl.constexpr,      # whether dim1 of cos/sin takes part in the broadcast (i.e. equals x.dim1)
    MODE: tl.constexpr,
    BLK_P: tl.constexpr,
):
    # One program handles one row (dim0, dim1, dim2): everything is scalar/1-D, which avoids
    # 2-D broadcast masks
    row = tl.program_id(0)
    ss_ = (row // N_HEADS) % SEQ
    bb_ = row // (N_HEADS * SEQ)
    if BR:
        cbr = bb_ * C1
    else:
        cbr = 0
    if SR:
        csr = ss_
    else:
        csr = 0
    cs_base = (cbr + csr) * HEAD_DIM
    x_base = row * D + START

    k = tl.arange(0, BLK_P)
    km = k < P

    # MODE is compared against literals: module-level globals cannot be referenced inside
    # @triton.jit
    if MODE == 1:          # MODE_INTERLEAVE
        _rope_pair_ilv(X, COS, SIN, Y, x_base, cs_base, k, km, P, ILV_WRITE=True)
    elif MODE == 3:        # MODE_DEEPSEEK_INTERLEAVE
        _rope_pair_ilv(X, COS, SIN, Y, x_base, cs_base, k, km, P, ILV_WRITE=False)
    elif MODE == 0:        # MODE_HALF
        _rope_pair(X, COS, SIN, Y, x_base, cs_base, k, P + k, k, P + k, km)
    else:                  # MODE_QUARTER: length 4P, one HALF pairing over each half
        for j in tl.static_range(2):
            base = 2 * j * P
            _rope_pair(X, COS, SIN, Y, x_base, cs_base,
                       base + k, base + P + k, base + k, base + P + k, km)


def _parse_rotary_mode(rotary_mode) -> int:
    """Normalize the call-site rotary_mode into the AscendC mode enum value.

    Source: csrc/ascend/torch_binding.cpp L278-286
        static const std::unordered_map<std::string, int> modes{
            {"half", 0}, {"interleave", 1}, {"quarter", 2}, {"interleave-half", 3}};
    """
    if isinstance(rotary_mode, str):
        table = {"half": MODE_HALF, "interleave": MODE_INTERLEAVE,
                 "quarter": MODE_QUARTER, "interleave-half": MODE_DEEPSEEK_INTERLEAVE}
        key = rotary_mode.strip().lower()
        if key not in table:
            raise ValueError(
                f"rotary_mode only supports {sorted(table)} (or int 0..3), got {rotary_mode!r}")
        return table[key]
    if isinstance(rotary_mode, int):
        if rotary_mode not in (MODE_HALF, MODE_INTERLEAVE, MODE_QUARTER, MODE_DEEPSEEK_INTERLEAVE):
            raise ValueError(f"rotary_mode(int) only supports 0..3, got {rotary_mode}")
        return rotary_mode
    raise ValueError(f"rotary_mode must be a str or an int, got {type(rotary_mode)}")


def inplace_partial_rotary_mul(
    x: torch.Tensor,
    r1: torch.Tensor | None = None,
    r2: torch.Tensor | None = None,
    rotary_mode="interleave",
    partial_slice: tuple[int, int] | list[int] | None = None,
    *,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    mode: int | None = None,
) -> torch.Tensor:
    """Equivalent to the AscendC InplacePartialRotaryMul (returns the full updated tensor).

    Call-site signature (csrc/ascend/torch_binding.cpp L1246):
        inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2,
                                   str rotary_mode, int[] partial_slice) -> ()
    => the positional argument order is (x, r1, r2, rotary_mode, partial_slice), matching the
       vllm_fl call sites (e.g. vllm_fl/attention/ascend/dsa_v1.py L1762-1768: x, cos, -sin,
       rotary_mode="interleave", partial_slice=[nope_head_dim, head_dim]).
       Note that call sites pass **-sin** as r2; this implementation evaluates the generic
       r1/r2 formula and is therefore correct by construction.

    Args:
        x:   4D [dim0, dim1, dim2, D_all], fp16 / bf16 / fp32 (declared in op_host def.cpp L24/L54)
        r1:  cos, 4D [r0, r1, 1, headDim], same dtype as x or fp32 (mixed precision)
        r2:  sin (call sites usually pass -sin)
        rotary_mode: "half"(0) / "interleave"(1) / "quarter"(2) / "interleave-half"(3),
                     an int 0..3 is also accepted
        partial_slice: [start, end]; absent or [0,0] means the whole range [0, D_all)
        cos/sin: keyword aliases for r1/r2 (backward compatibility)
        mode:    alias for the AscendC mode enum (backward compatibility, highest priority)

    Returns:
        y: same shape/dtype as x; only the last dim over [start, end) is rotated, the rest
        equals x.
    """
    if mode is not None:
        rm = _parse_rotary_mode(mode)
        if rotary_mode not in (None, "interleave") and _parse_rotary_mode(rotary_mode) != rm:
            raise ValueError(f"mode={mode} conflicts with rotary_mode={rotary_mode!r}.")
    else:
        rm = _parse_rotary_mode(rotary_mode)

    if r1 is None:
        r1 = cos
    if r2 is None:
        r2 = sin
    if r1 is None or r2 is None:
        raise ValueError("r1/r2 (or cos/sin) must be provided")
    cos, sin = r1, r2
    if x.dim() != 4:
        raise ValueError(f"x must be 4D (required by the AscendC interface), got {tuple(x.shape)}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin must have the same shape: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if cos.dim() != 4:
        raise ValueError(f"cos/sin must be 4D, got {tuple(cos.shape)}")
    if cos.shape[2] != 1:
        raise ValueError(f"dim2 of cos/sin must be 1, got {cos.shape[2]}")
    if cos.shape[0] not in (1, x.shape[0]):
        raise ValueError(f"dim0 of cos/sin must be 1 or x.dim0={x.shape[0]}, got {cos.shape[0]}")
    if cos.shape[1] not in (1, x.shape[1]):
        raise ValueError(f"dim1 of cos/sin must be 1 or x.dim1={x.shape[1]}, got {cos.shape[1]}")
    if rm not in (MODE_HALF, MODE_INTERLEAVE, MODE_QUARTER, MODE_DEEPSEEK_INTERLEAVE):
        raise ValueError(f"mode must be 0/1/2/3, got {rm}")

    d_all = x.shape[3]
    # The AscendC attr default is {0, 0}, semantically equivalent to "no explicit slice"
    # -> the whole range
    if partial_slice is None or (int(partial_slice[0]) == 0 and int(partial_slice[1]) == 0):
        start, end = 0, d_all
    else:
        start, end = int(partial_slice[0]), int(partial_slice[1])
    if start < 0 or end > d_all or end <= start:
        raise ValueError(f"partial_slice=[{start},{end}] is invalid (last dim of x = {d_all})")
    head_dim = end - start
    if head_dim != cos.shape[3]:
        raise ValueError(f"headDim={head_dim} must equal the last dim of cos/sin {cos.shape[3]}")

    if rm == MODE_QUARTER:
        if head_dim % 4 != 0:
            raise ValueError(f"mode=2(QUARTER) requires headDim%4==0, got {head_dim}")
        p = head_dim // 4
    else:
        if head_dim % 2 != 0:
            raise ValueError(f"mode={rm} requires headDim%2==0, got {head_dim}")
        p = head_dim // 2

    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"x dtype only supports fp16/bf16/fp32 (op_host def.cpp L24), got {x.dtype}")
    if cos.dtype != x.dtype and cos.dtype != torch.float32:
        raise ValueError(f"cos/sin dtype must be the dtype of x or fp32, got {cos.dtype}")

    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    y = x.clone()

    dim0, dim1, dim2 = x.shape[0], x.shape[1], x.shape[2]
    n_rows = dim0 * dim1 * dim2
    grid = (n_rows,)
    _inplace_partial_rotary_mul_kernel[grid](
        x, cos, sin, y,
        dim2, dim1, d_all,
        cos.shape[1], head_dim, start, p,
        BR=bool(cos.shape[0] != 1),
        SR=bool(cos.shape[1] != 1),
        MODE=rm,
        BLK_P=next_pow2(p),
    )
    return y
