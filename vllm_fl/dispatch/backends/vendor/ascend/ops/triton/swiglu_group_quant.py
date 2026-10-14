# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""swiglu_group_quant -- AscendC -> Triton (**not implemented: the target environment is
unavailable; the adapter rejects explicitly per the spec**)

## Conclusion (read this first)

**This operator is unavailable on the target environment 910_93, so this file ships no runnable
Triton implementation: every entry point raises `NotImplementedError`.** This is not a shortcut;
it avoids "guess a layout and report a fake PASS", because one fork in the source -- whether the
two halves are contiguous or interleaved -- has two mutually exclusive pieces of evidence, and
guessing wrong yields numerically different results (silu(act)*gate vs silu(gate)*act is not a
precision issue, it is a wrong answer).

## Evidence of unavailability

1. `csrc/moe/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp` L61 registers only
   `this->AICore().AddConfig("ascend950");` (**not** ascend910b / ascend910_93), while the
   target container is Ascend910_9382 (910_93).
2. Checked inside the target container:
   ```
   torch 2.10.0+cpu npu ok True
   hasattr(torch.ops._C_ascend, npu_swiglu_group_quant) = False
   related ops available: []
   ```
   i.e. **`torch.ops._C_ascend.npu_swiglu_group_quant` does not exist in the container at all**,
   so the "reverse-engineer the layout from an identifiable input" experiment cannot be run.

## Semantics already confirmed from the source (ready to use when implementing on ascend950)

Call-site schema (`csrc/torch_binding.cpp` L3550-3558):
    npu_swiglu_group_quant(Tensor x, Tensor? topk_weight, Tensor? group_index,
                           ScalarType dst_type=39, int quant_mode=1, int group_size=128,
                           bool round_scale=False, bool ue8m0_scale=False,
                           bool output_origin=False, int group_list_type=0,
                           float clamp_value=0.0) -> (Tensor y, Tensor scale, Tensor y_origin)

1. **Shapes** (`op_host/swiglu_group_quant_proto.cpp` L48-87):
   x [.., 2H] -> y [.., H]; `scale` = x.shape[:-1] + [tailDim], where `tailDim = ceil(ceil(H/32)/2)`
   plus one extra dimension of size 2 when `quant_mode==2` (MX); otherwise `tailDim = ceil(H/128)`.
2. **group_index means "token count per group", and this operator only uses its sum**:
   the call site's original comment in `vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py`
   L408-416:
   "currently only supports the `count` type ... the operator performs a **summation** calculation
   on `group_index`. Therefore, under the cumsum type, the last value is directly taken"
   -> i.e. `sum(group_index) = number of valid tokens`. (Consistent with "tokens per group" in
   `dequant_swiglu_quant`, but this operator goes further: only the array sum is needed.)
3. **SwiGLU** (`op_kernel/swiglu_group_quant_base.h` L162-166):
   `silu(t) = t / (1 + exp(-t))`, `y = silu(one half) * other`; with `clamp_value>0` both sides are
   clamped to [-c, c] (L238-244); when `topk_weight` is non-empty it is multiplied into that row
   (L249-250).
4. **Per-token fp8 quantization** (L777-822): `amax = max|y|` -> `clampScale = max(amax, 1e-4)`
   (L789) -> `scale = clampScale / 448.0` (L791; 448 = E4M3FN max, L32)
   -> `y_q = y / scale` (L793/L817-822, i.e. multiply by `448/clampScale`).
5. **E8M0 shared exponent (MX / ue8m0_scale) -- the rule for `round_scale=True` (L795-813)**:
   ```
   exp   = (bits(scale) >> 23) & 0xFF          # L797-798
   man   = bits(scale) & 0x7FFFFF              # L799
   t     = (man != 0) ? 1 : 0                  # L800-801
   e     = (exp - 127) + t                     # L802-803 <- **non-zero mantissa rounds up**
   scale_e8m0 = e + 127                        # L805/L808 (e8m0 stores the raw bits of e+127)
   invScale   = ((127 - e) << 23) as float32   # L812-813 <- exactly 2^-e
   y_q        = y * invScale                   # L817/L821
   ```
   -> **`e = ceil(log2(sf))`** (round up to a power of two) and quantization multiplies by `2^-e`;
   with `round_scale=False` it is simply `y * (448/clampScale)` (L793).
   **Still open**: how MX mode tiles into blocks of 32, and how the extra trailing dimension of
   `scale` reshaped to (-1,2) (proto L76-80) aligns with `maybe_normalize_mxfp_scale_layout()` --
   both need hardware or a definite layout conclusion before they can be implemented.

## Two unresolved forks (**only hardware measurement can settle them, hence no implementation**)

**Fork A: are the two halves contiguous or interleaved?**
- Evidence for "interleaved": `swiglu_group_quant_base.h` L399 / L542 reads the two halves with
  `LoadDist::DIST_DINTLV_B16` (**de-interleave**: even/odd indices split into x0/x1);
- Evidence for "contiguous": the infershape (proto L60-62) merely halves the last dimension,
  matching the contiguous halving convention of `dequant_swiglu_quant`; the `Interleave(...)` at
  L551-586 may only be needed for packing the fp8 output (`StoreDist::DIST_PACK4_B32`, four fp8
  values into one 32-bit block).
- The two readings are numerically different, and even reading the assembly cannot settle it with
  certainty (DINTLV may act on another layout inside UB).

**Fork B: MX(E8M0) rounding direction and scale layout** (see section 5 above).

## Why there is no "runnable" implementation

**A PASS obtained by guessing the layout is a fake PASS.** The target environment cannot verify
either the layout or the E8M0 rule, so this file keeps only the specification and the rejecting
entry point; implement sections 1-5 above once an ascend950 environment (or a definitive layout
conclusion) is available.
"""

from __future__ import annotations

import torch

__all__ = ["swiglu_group_quant", "SWIGLU_GROUP_QUANT_UNAVAILABLE_REASON"]

SWIGLU_GROUP_QUANT_UNAVAILABLE_REASON = (
    "swiglu_group_quant is registered for ascend950 only (csrc/moe/swiglu_group_quant/op_host/"
    "swiglu_group_quant_def.cpp L61); on the target environment 910_93 torch.ops._C_ascend."
    "npu_swiglu_group_quant does not exist (hasattr=False in the container), and the kernel "
    "carries mutually exclusive evidence for 'contiguous vs interleaved halves' "
    "(DIST_DINTLV_B16 vs the infershape halving), so the layout cannot be decided without "
    "hardware; per the spec this file ships no unverified implementation. See the module "
    "docstring for details."
)


def swiglu_group_quant(*args, **kwargs):
    """Not implemented: see the module docstring (target environment 910_93 unavailable and the
    layout fork is unresolved)."""
    raise NotImplementedError(SWIGLU_GROUP_QUANT_UNAVAILABLE_REASON)
