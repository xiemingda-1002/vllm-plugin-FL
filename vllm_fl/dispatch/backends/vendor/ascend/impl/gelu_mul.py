# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the torch_npu operator of the same name.
# SPDX-License-Identifier: Apache-2.0
"""gelu_mul -- torch_npu.npu_gelu_mul -> Triton

Source:
    This tree carries no AscendC kernel for ``npu_gelu_mul``: ``torch_npu`` is not
    vendored and a case-insensitive search for ``gelu`` over all 979 files of
    ``csrc/`` returns 0 hits. ``csrc/ascend/moe`` holds causal_conv1d, chunk_fwd_o,
    chunk_gated_delta_rule_fwd_h, dequant_swiglu_quant, hc_post, hc_pre,
    hc_pre_inv_rms, hc_pre_sinkhorn, moe_gating_top_k, moe_gating_top_k_hash,
    moe_init_routing_custom and scatter_nd_update_v2, i.e. no activation operator.
    The semantics are therefore pinned by the sources below instead of by a single
    kernel file.

    1. The eager fallback for this exact operator, in the same backend:
           impl/activation.py L46-54
       L46-48 states "``npu_gelu_mul`` splits x into two halves exactly like
       GeluAndMul"; L52-53 spells the split out as
       ``d = x.shape[-1] // 2; return F.gelu(x[..., :d], approximate=approximate)
       * x[..., d:]``.
    2. Two MoE implementations of the same activation, both spelling the split out:
           impl/fused_moe.py L134-136
               gate, up = gate_up.chunk(2, dim=-1); gate_up = F.gelu(gate) * up
           impl/moe/moe_mlp.py L303-305
               gate, up = hidden_states.chunk(2, dim=-1)
               hidden_states = F.gelu(gate, approximate=approximate) * up
       and the same pair again at moe_mlp.py L420-422 and L538-543.
    3. The other backends' reference for the same dispatch op (``gelu_and_mul``):
           dispatch/backends/reference/impl/activation.py L40-43
               x1, x2 = x[..., :d], x[..., d:]
               return F.gelu(x1, approximate=approximate) * x2
           dispatch/backends/flaggems/impl/activation.py L50-54
               x1, x2 = x[..., :d], x[..., d:]; gelu_and_mul(x1, x2, approximate)
    4. Independent Triton kernels with the same layout and both GELU forms:
           FlagGems-vllm/src/flaggems_vllm/ops/gelu_and_mul.py
               L29-35  erf form: 0.5 * x * (1 + erf(x / sqrt(2))) * y
               L66-78  tanh form
               L110-121 ``gelu_and_mul(x, y, approximate="none")``
    5. The gelu-and-mul kernel triton-ascend itself ships as its reference test
       (Huawei copyright), which fixes the layout independently:
           triton-ascend/ascend/test/sglang/v0.4.8/test_gelu_and_mul_kernel.py
               L49-62 x1 is loaded at column 0 (the gate half), x3 at column H
               (the up half), ``gelu_x1 = 0.5 * (1 + erf(x1 * 0.7071067811865475))
               * x1``, ``out = x3 * gelu_x1``.

Call sites (this repository):
    vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe_layer.py L168
        gate_up = torch_npu.npu_gelu_mul(gate_up)        (activation == "gelu")
    vllm_fl/dispatch/backends/vendor/ascend/impl/activation.py L54
        return torch.ops.npu.npu_gelu_mul(x, approximate=approximate)
    The first passes one positional argument and no keyword; the second passes
    ``approximate`` as a keyword. The signature below accepts both spellings.

Formula:
    ``x`` is split along the last dimension into two contiguous halves of width
    ``H = d // 2`` -- not into even/odd lanes::

        a   = x[..., :H]        # GELU operand (the "gate" half)
        b   = x[..., H:]        # multiplicand (the "up" half)
        out = gelu(a) * b

    The gate half is the first one. All five sources above agree on both points:
    the sglang kernel indexes the gate at offset 0 and the up half at offset H, and
    impl/activation.py L53 writes the same expression out.

    ``approximate`` selects the GELU form, exactly like ``torch.nn.functional.gelu``:
      * ``"none"`` (default) -- the erf form::

            0.5 * a * (1 + erf(a / sqrt(2)))

      * ``"tanh"`` -- the tanh form::

            0.5 * a * (1 + tanh(0.7978845608028654 * (a + 0.044715 * a ** 3)))

    Which form is the default is pinned by the call sites, not assumed:
      * the only keyword the repository passes explicitly is
        ``GeluAndMul.approximate`` (activation.py L45-54), whose in-tree default is
        ``"none"`` (ops/activation.py L90-91, dispatch/ops.py L93);
      * every MoE path that spells ``gelu`` out uses the erf form: fused_moe.py L136
        ``F.gelu(gate)`` (PyTorch's default is ``approximate="none"``),
        moe_mlp.py L304-305 with
        ``approximate = "tanh" if activation == GELU_TANH else "none"``, and
        fused_moe_layer.py accepts only ``"silu"``/``"gelu"`` (L113) -- there is no
        ``gelu_tanh`` path that could reach ``npu_gelu_mul``;
      * the tanh call sites (moe_mlp.py L543) call
        ``F.gelu(..., approximate="tanh")`` directly and never ``npu_gelu_mul``.
    torch_npu is not vendored, so ``npu_gelu_mul``'s own declared default is not
    independently readable here; ``"none"`` is the only value consistent with the
    rest of this repository. ``"tanh"`` is implemented as well, so an explicit call
    with either legal value is exact and neither silently degrades.

Coverage:
    Implemented       float16 / bfloat16 / float32, arbitrary leading dimensions,
                      arbitrary H (including H that is not a power of two), empty input.
    NotImplementedError
                      a float dtype other than those three, and inputs whose element
                      count exceeds int32 (the kernel's element offsets are int32).
    ValueError        an odd last dimension, and an ``approximate`` value other than
                      ``"none"`` / ``"tanh"``.

Tiling:
    One program owns whole rows and walks each row in 1-D column chunks of
    ``BLK_N = min(_next_pow2(H), 1024)``, floored at 2. The grid is
    ``(min(rows, 65535),)`` and every program strides over the remaining rows, so the
    AI Core ``coreDim`` limit is never exceeded. The row count is
    ``num_tokens * top_k``: ``fused_moe_layer.py`` L143 sets
    ``num_tokens = x.shape[:-1].numel()`` and L144-155 expands it through
    ``moe_init_routing_custom`` with ``active_num=num_tokens * top_k``, so an 8192
    token prefill at top-k 8 already produces 65536 rows, i.e. one row past the limit
    without the stride loop.

    A chunk holds at most 1024 float32 values per operand: a, b, the GELU value and
    the product are about 16 KiB of UB, about 32 KiB with the compiler's default
    double buffering, well under the 192 KiB of the 910B/910C family.

    Every load and the store are 1-D and masked. This is deliberate: a 2-D masked
    store at a non-power-of-two width is the shape that has been observed to drop
    lanes silently on this backend, and the MoE widths this operator runs at
    (768 / 1536 / 2304) are exactly non-powers-of-two. The neighbouring trap -- a
    store mask that does not cover the tail block -- is TRITON_OPERATOR_PITFALLS.md
    C4, and every store here carries the same mask as its loads.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _next_pow2(n: int) -> int:
    return max(1, 1 << (int(n) - 1).bit_length())



__all__ = ["gelu_mul", "npu_gelu_mul"]

#: Largest column chunk one inner iteration handles; 1024 float32 lanes is ~16 KiB.
_BLK_N_MAX = 1024
#: 1-D grid limit of the backend (AI Core ``coreDim``); each program strides over rows.
_MAX_GRID = 65535
#: Above this element count the kernel's int32 element offsets would overflow.
_MAX_ELEMS = 2 ** 31 - 1
#: tanh form constants, identical to ``torch.nn.functional.gelu(approximate="tanh")``.
_TANH_C0 = 0.7978845608028654  # sqrt(2 / pi)
_TANH_C1 = 0.044715
#: 1 / sqrt(2), the argument scale of the erf form.
_RCP_SQRT_2 = 0.7071067811865476


@triton.jit
def _tanh_via_sigmoid(z):
    """``tanh(z)`` as ``2 * sigmoid(2 * z) - 1``.

    ``tl.math.tanh`` is absent from triton-ascend's supported-math table, while
    ``sigmoid`` is listed, so the identity is spelled out. The gelu-and-mul kernel
    that triton-ascend ships defines tanh the same way for this same activation
    (ascend/test/sglang/v0.4.8/test_gelu_and_mul_triton_kernel.py L31-33). The form
    is saturating in both tails, so it neither overflows nor loses the sign.
    """
    return 2.0 * tl.sigmoid(2.0 * z) - 1.0


@triton.jit
def _gelu_mul_kernel(
    X, Y,
    R, H,
    NPROG,
    BLK_N: tl.constexpr,
    APPROX_TANH: tl.constexpr,
):
    """out[r, c] = gelu(x[r, c]) * x[r, H + c] for the flat [R, 2H] view of x.

    ``X`` is the contiguous ``[R, 2H]`` input (row stride ``2 * H``), ``Y`` the
    ``[R, H]`` output. Program ``pid`` owns the rows ``pid, pid + NPROG, ...``; each
    row is walked in 1-D column chunks so that no 2-D masked store is emitted.
    """
    pid = tl.program_id(0)
    nloop = tl.cdiv(H, BLK_N)
    nrow = tl.cdiv(R - pid, NPROG)
    d = tl.arange(0, BLK_N)
    # ``pid`` is always < R (the host sets NPROG = min(R, 65535)), so nrow >= 1. The
    # stride is spelled as an index step inside a constant-step loop: a runtime loop
    # step is not portable across Triton front ends.
    for k in range(0, nrow):
        row = pid + k * NPROG
        in_base = row * (2 * H)
        out_base = row * H
        for i in range(0, nloop):
            cols = i * BLK_N + d
            mask = cols < H
            # Masked loads: lanes past H must not read the next row's data (or past
            # the buffer). The gate half is the first half, the up half the second.
            a = tl.load(X + in_base + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(X + in_base + H + cols, mask=mask, other=0.0).to(tl.float32)
            if APPROX_TANH:
                z = _TANH_C0 * (a + _TANH_C1 * a * a * a)
                g = 0.5 * a * (1.0 + _tanh_via_sigmoid(z))
            else:
                g = 0.5 * a * (1.0 + tl.erf(a * _RCP_SQRT_2))
            tl.store(Y + out_base + cols, (g * b).to(Y.dtype.element_ty), mask=mask)


def gelu_mul(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Triton equivalent of ``torch_npu.npu_gelu_mul``.

    Args:
        x: ``[..., 2H]`` with ``x.shape[-1]`` even; float16, bfloat16 or float32.
            Made contiguous internally (the kernel assumes a row stride of exactly
            ``2 * H``).
        approximate: ``"none"`` for the erf form (the default, and the only value the
            repository's GELU paths use) or ``"tanh"`` for the tanh form.

    Returns:
        ``[..., H]`` tensor with ``x``'s dtype and device, where
        ``out[..., c] = gelu(x[..., c]) * x[..., H + c]``.
    """
    if x.dim() < 1:
        raise ValueError(f"gelu_mul expects at least 1-D input, got shape {tuple(x.shape)}")
    if approximate not in ("none", "tanh"):
        raise ValueError(
            f"approximate must be either 'none' or 'tanh', got {approximate!r}"
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            f"gelu_mul input dtype {x.dtype} is not implemented: npu_gelu_mul is a "
            f"float16/bfloat16/float32 activation and the MoE call site feeds it a "
            f"bfloat16 gate_up (fused_moe_layer.py L157-168)"
        )

    d = int(x.shape[-1])
    if d % 2 != 0:
        raise ValueError(f"the last dimension of x must be even, got {d}")
    H = d // 2
    numel = int(x.numel())
    if numel > _MAX_ELEMS:
        raise NotImplementedError(
            f"gelu_mul input with {numel} elements exceeds the int32 element-offset "
            f"range (shape {tuple(x.shape)})"
        )
    rows = 0 if d == 0 else numel // d

    xc = x.contiguous()
    y = torch.empty(x.shape[:-1] + (H,), dtype=x.dtype, device=x.device)
    if rows > 0 and H > 0:
        blk_n = max(2, min(_next_pow2(H), _BLK_N_MAX))
        nprog = min(rows, _MAX_GRID)
        _gelu_mul_kernel[(nprog,)](
            xc.view(rows, 2 * H),
            y.view(rows, H),
            rows,
            H,
            nprog,
            BLK_N=blk_n,
            APPROX_TANH=(approximate == "tanh"),
        )
    return y


#: The call sites spell the operator ``torch_npu.npu_gelu_mul`` (fused_moe_layer.py
#: L168) and ``torch.ops.npu.npu_gelu_mul`` (activation.py L54); the deployment shim
#: imports the module-level name, so both spellings are kept.
npu_gelu_mul = gelu_mul
