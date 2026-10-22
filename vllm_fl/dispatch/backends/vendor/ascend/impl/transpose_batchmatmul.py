# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the AscendC operator of the same name (see csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""transpose_batchmatmul -- AscendC to Triton.

Source: this operator is the CANN / torch_npu built-in ``aclnnTransposeBatchMatMul``
behind ``torch_npu.npu_transpose_batchmatmul``. **There is no AscendC kernel body for
it in this tree**:

    grep -rn "transpose_batchmatmul\\|TransposeBatchMatmul" csrc   ->   no match

(unlike the custom operators of ``csrc/ascend``, which are registered at
``torch.ops._C_ascend.<op>``). The semantics below are therefore pinned from (a) the
call sites, (b) the torch reference that upstream vllm-ascend's unit tests substitute
for the operator, and (c) the parameter set the call sites actually pass -- not from a
kernel body.

------------------------------------------------------------------------------
Call sites -- 6 occurrences of the name in the vllm_fl tree
------------------------------------------------------------------------------
    grep -rn "npu_transpose_batchmatmul" vllm_fl --include=*.py

  [1] attention/ascend/sfa_v1.py L1245   ``hasattr(torch_npu, "npu_transpose_batchmatmul")``
                                          availability guard (no call)
  [2] attention/ascend/sfa_v1.py L1249   ``_v_up_proj`` fallback
  [3] attention/ascend/dsa_v1.py L1797   ``_forward_o_proj``, OTP / all-to-all branch
  [4] attention/ascend/dsa_v1.py L1826   ``_forward_o_proj``, default branch
  [5] attention/ascend/context_parallel/dsa_cp.py L1270   PCP copy of [3]
  [6] impl/batch_matmul_transpose.py L73 and the identical line in
      ops/triton/batch_matmul_transpose.py -- a docstring cross-check of the same
      einsum against the custom ``batch_matmul_transpose`` operator, not a call path.

  [2]  torch_npu.npu_transpose_batchmatmul(x, self.W_UV,
                                          perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
  [3][4][5]
       torch_npu.npu_transpose_batchmatmul(
           o_proj_input, self.wo_a.weight,
           bias=None, scale=None,
           perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2),
           batch_split_factor=1)

  => the entry signature below has to accept every one of those keywords
     (``bias``, ``scale``, ``perm_x1``, ``perm_x2``, ``perm_y``,
     ``batch_split_factor``), including the two that this port cannot honour.

------------------------------------------------------------------------------
Semantics (pinned)
------------------------------------------------------------------------------
    y = bmm(x1.permute(perm_x1), x2.permute(perm_x2)).permute(perm_y)

i.e. ``perm_x1``/``perm_x2`` move the batch axis of each operand to the front, the
ordinary batched matmul contracts the last axis of the first operand with the middle
axis of the second one, and ``perm_y`` moves the axes of the [B, M, N] result back.

Evidence, in order of strength:

  1. The reference implementation that upstream vllm-ascend's unit test substitutes
     for the operator (_repos/vllm-ascend/tests/ut/attention/test_mla_v1.py L48-49,
     L69-70):

         def fused_bmm(input, weight, *, perm_x1=(0, 1, 2), perm_y):
             return torch.bmm(input.permute(perm_x1), weight).permute(perm_y)

     Two facts come out of it: ``x2`` is used **as it is** (no transpose, so it is the
     [B, K, N] operand), and a permutation of ``None`` means the identity.

  2. The same reference, applied to the two production shapes
     (_repos/vllm-ascend/tests/ut/attention/test_dsa_v1.py L2663-2664 and L2725-2728):

         def matmul(x, weight, **kwargs):
             return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)

     with ``x`` [T, G, D] and ``weight`` [G, D, R] -> [T, G, R], which is exactly the
     ``einsum("tgd,gdr->tgr")`` the o_proj call sites need.

  3. The operand layout that (2) requires is the layout the weight loaders actually
     produce:
       - ``vllm_fl/dispatch/backends/vendor/ascend/ops/linear.py`` L474-476 and
         L484-492 reshape ``wo_a.weight`` to
         ``view(n_local_groups, o_lora_rank, -1).transpose(2, 1)``, i.e.
         **[n_local_groups, hidden_size, o_lora_rank] = [B, K, N]**;
       - upstream says the same in two comments: ``_repos/vllm-ascend/vllm_ascend/
         ops/linear.py`` L84 and L125-127 ("DSV4 wo_a is consumed by
         npu_transpose_batchmatmul in the 3D layout
         [n_local_groups, hidden_size, o_lora_rank]").
     (The ``# o = torch.einsum("tgd,grd->tgr", ...)`` comment next to the dsa_cp call
     site describes the *checkpoint* [G, R, D] layout before the loader transposes it,
     and is stale relative to the operator call it sits next to.)

  4. ``sfa_v1.py`` L1245-1258 offers the same operator versus a hand-written
     ``torch.bmm(x.view(-1, N, L).transpose(0, 1), W_UV).transpose(0, 1)`` fallback, so
     the two are the same einsum: out[b, n, v] = sum_l x[b, n, l] * W_UV[n, l, v], with
     x1 = [B=num_input_tokens, M=local_num_heads, K=kv_lora_rank] and
     W_UV = [local_num_heads, kv_lora_rank, v_head_dim] (built at
     ``sfa_v1.py`` L717-724 as ``kv_b_proj_weight.split(...); W_UV.transpose(0, 1)``).

------------------------------------------------------------------------------
Production shapes
------------------------------------------------------------------------------
  [2]  B = local_num_heads, M = num_input_tokens, K = kv_lora_rank = 512,
       N = v_head_dim (128 for DeepSeek-V3/R1 MLA, 256 for HY4 -- see
       ``vllm_fl/models/hy4/config.py`` L77 and the note in the sibling
       ``impl/batch_matmul_transpose.py`` L113-121).
  [3]-[5]
       B = n_local_groups (8 in the DeepSeek-V3/R1 MLA family, ``o_groups``),
       M = num_tokens (or ``oproj_tp_size * exchange_num_tokens`` in the OTP branch),
       K = group_hidden_dim = n_local_heads * head_dim / n_local_groups,
       N = o_lora_rank.
  dtype fp16 / bf16 on both operands; output dtype = input dtype.

------------------------------------------------------------------------------
Parameter coverage
------------------------------------------------------------------------------
  bias                : only ``None`` is implemented. Every call site passes ``None``; a
                        non-None bias raises NotImplementedError rather than being
                        dropped (the bias layout for a transposed batch matmul -- [N] or
                        [B, M, N] -- is not pinned by any source in this tree).
  scale               : only ``None`` is implemented, same reasoning (it is a post-matmul
                        dequant scale whose layout is not pinned here).
  perm_x1 / perm_x2   : a permutation of (0, 1, 2), or ``None`` for the identity.
                        ``perm_x2`` defaults to the identity, which is what the calls
                        that omit it rely on ([2] passes no ``perm_x2``).
  perm_y              : a permutation of (0, 1, 2) whose **last entry is 2**, i.e. the N
                        axis stays the innermost axis of the result. All call sites pass
                        (1, 0, 2); another permutation would need a strided store and
                        raises NotImplementedError instead.
  batch_split_factor  : only 1 is implemented (the value at all call sites). It splits
                        the batch over cores for scheduling; no source in this tree pins
                        its effect, so a different value raises instead of being
                        silently ignored.
  m <= 1024 / 65536   : the operand-size guards at the call sites
                        (``BMM_TRANS_MAX_SUPPORTED_TOKENS``, ``N*L < 65536``,
                        ``B < TRANSPOSE_BMM_MAX_SUPPORTED_DIM``) are aclnn tiling /
                        operand-check artefacts of the native kernel (see the comment
                        block at ``sfa_v1.py`` L1354-1363). This implementation does not
                        inherit them and is correct for any size; the callers' branch
                        selection is unaffected.

------------------------------------------------------------------------------
Implementation notes
------------------------------------------------------------------------------
* ``grid = (B, <=65535//(B*ceil(N/BLOCK_N)), ceil(N/BLOCK_N))``: one program per
  (batch, column tile), and the row tiles are folded into a strided loop inside the
  program so the program count can never exceed the backend's 65535 (block_dim) cap --
  ``B * ceil(M/BLOCK_M) * ceil(N/BLOCK_N)`` reaches 131 072 for the [2] shape alone at
  65536 tokens (128 heads x 1024 row tiles).
* Nothing is materialised for the permutations: the operands are permuted as *views*
  and only their strides are handed to the kernel, so no extra copy and no extra
  allocation appears on the forward path (a non-unit innermost stride, which no
  production permutation produces, falls back to ``contiguous()``).
* The result is allocated directly in the ``perm_y`` layout and the kernel writes into
  it through the inverse permutation's strides, so the returned tensor is contiguous
  with exactly the shape the native operator returns -- ``reshape(num_tokens, -1)`` at
  the call sites stays a view.
* Accumulation is fp32 (``out_dtype=tl.float32``) and the store casts back to the
  output dtype, the pattern already verified in the sibling
  ``batch_matmul_transpose.py`` (whose tile sizes are reused: BLOCK_M=64, BLOCK_K=64,
  BLOCK_N=64/128). Loads and the store are masked, so M, N and K need not be multiples
  of the tiles.

------------------------------------------------------------------------------
Unconfirmed (no CANN header or NPU in this container)
------------------------------------------------------------------------------
* ``bias`` / ``scale`` semantics (see the coverage table) -- unimplemented, not guessed.
* whether the native returns a *contiguous* tensor: this port does, which is what the
  call sites' immediate ``reshape`` needs, but it was not measured against the native.
* ``batch_split_factor`` -- assumed to be a scheduling knob only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b



__all__ = ["transpose_batchmatmul", "npu_transpose_batchmatmul"]

# Tile sizes, reused from impl/batch_matmul_transpose.py (the same einsum, verified
# PASS there): BLOCK_K = 64 keeps every live fp16/bf16 tile far below the UB limit.
_BLOCK_M = 64
_BLOCK_K = 64
_BLOCK_N_WIDE = 128
_BLOCK_N = 64

#: AI Core block_dim limit of this backend (a larger grid was measured to crash, see
#: _agent_logs/TRITON_OPERATOR_PITFALLS.md C1 and impl/grouped_matmul.py L132-139).
_MAX_PROGRAMS = 65535

_IDENTITY = (0, 1, 2)


@triton.jit
def _tbmm_kernel(
    A, B, C,
    M, N, K,
    num_mt,
    stride_ab, stride_am,
    stride_bb, stride_bk,
    stride_cb, stride_cm,
    grid_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[b, m, n] = sum_k A[b, m, k] * B[b, k, n] for the (batch, row tile, col tile)
    triple this program owns.

    ``A`` is [B, M, K] with element strides (stride_ab, stride_am, 1);
    ``B`` is [B, K, N] with element strides (stride_bb, stride_bk, 1);
    ``C`` is [B, M, N] with element strides (stride_cb, stride_cm, 1).

    ``grid_m`` row tiles are owned by the grid; a program strides over the row tiles it
    owns (``pid_m``, ``pid_m + grid_m``, ...) so the program count stays under the
    backend cap without changing any value.
    """
    batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    mt = pid_m
    while mt < num_mt:
        offs_m = mt * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + tl.arange(0, BLOCK_K)
            mask_k = kk < K
            a = tl.load(
                A + batch * stride_ab + offs_m[:, None] * stride_am + kk[None, :],
                mask=mask_m[:, None] & mask_k[None, :], other=0.0,
            )
            b = tl.load(
                B + batch * stride_bb + kk[:, None] * stride_bk + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :], other=0.0,
            )
            acc += tl.dot(a, b, out_dtype=tl.float32)
        tl.store(
            C + batch * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :],
            acc.to(C.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )
        mt += grid_m


def _norm_perm(perm, name: str) -> tuple[int, int, int]:
    """A 3-element permutation of (0, 1, 2); ``None`` means the identity."""
    if perm is None:
        return _IDENTITY
    try:
        p = tuple(int(v) for v in perm)
    except (TypeError, ValueError):
        raise ValueError(f"{name}={perm!r} is not a permutation of (0, 1, 2)") from None
    if len(p) != 3 or sorted(p) != [0, 1, 2]:
        raise ValueError(f"{name}={p} is not a permutation of (0, 1, 2)")
    return p


def _inverse_perm(p: tuple[int, int, int]) -> tuple[int, int, int]:
    q = [0, 0, 0]
    for j, v in enumerate(p):
        q[v] = j
    return (q[0], q[1], q[2])


def transpose_batchmatmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    batch_split_factor: int = 1,
) -> torch.Tensor:
    """Equivalent of ``torch_npu.npu_transpose_batchmatmul`` (aclnn TransposeBatchMatMul).

    ``y = bmm(x1.permute(perm_x1), x2.permute(perm_x2)).permute(perm_y)`` -- see the
    module docstring for the evidence behind that formula.

    Args:
        x1: [B, M, K] fp16/bf16, up to the permutations of ``perm_x1``.
        x2: [B, K, N] fp16/bf16, up to the permutations of ``perm_x2``.
        bias: must be ``None`` (unimplemented, see the module docstring).
        scale: must be ``None`` (unimplemented, see the module docstring).
        perm_x1: permutation of (0, 1, 2) applied to ``x1``; ``None`` = identity.
        perm_x2: permutation of (0, 1, 2) applied to ``x2``; ``None`` = identity.
        perm_y: permutation of (0, 1, 2) applied to the [B, M, N] result; ``None`` =
            identity. Its last entry must be 2 (the N axis stays innermost).
        batch_split_factor: must be 1.

    Returns:
        A **new contiguous** tensor of shape ``tuple((B, M, N)[i] for i in perm_y)``.
    """
    if bias is not None:
        raise NotImplementedError(
            "bias is not None: no call site in the vllm_fl tree passes a bias to "
            "npu_transpose_batchmatmul (all four sites pass bias=None) and no source in "
            "this tree pins its layout ([N] vs [B, M, N]), so it is not implemented "
            "rather than silently dropped."
        )
    if scale is not None:
        raise NotImplementedError(
            "scale is not None: no call site in the vllm_fl tree passes a scale to "
            "npu_transpose_batchmatmul (all four sites pass scale=None) and the "
            "post-matmul scale semantics are not pinned by any source in this tree."
        )
    if int(batch_split_factor) != 1:
        raise NotImplementedError(
            f"batch_split_factor={batch_split_factor!r}: only the value used at every "
            "call site (1) is implemented; the knob's effect on the result is not "
            "pinned by any source in this tree, so it is not ignored silently."
        )

    if x1.dim() != 3:
        raise ValueError(f"x1 must be 3-D [B, M, K], got {tuple(x1.shape)}")
    if x2.dim() != 3:
        raise ValueError(f"x2 must be 3-D [B, K, N], got {tuple(x2.shape)}")
    if x1.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"x1 must be fp16 or bf16, got {x1.dtype}")
    if x2.dtype != x1.dtype:
        raise ValueError(f"x1 and x2 must share one dtype, got {x1.dtype} / {x2.dtype}")
    if x1.device != x2.device:
        raise ValueError("x1 and x2 must live on the same device")

    px1 = _norm_perm(perm_x1, "perm_x1")
    px2 = _norm_perm(perm_x2, "perm_x2")
    py = _norm_perm(perm_y, "perm_y")
    if py[2] != 2:
        raise NotImplementedError(
            f"perm_y={py}: this implementation keeps the N axis innermost (it writes the "
            "result through the inverse permutation's strides); the call sites only use "
            "(1, 0, 2)."
        )

    a = x1.permute(px1)
    b = x2.permute(px2)
    batch, m, k = (int(s) for s in a.shape)
    batch_b, k_b, n = (int(s) for s in b.shape)
    if batch != batch_b:
        raise ValueError(
            f"the batch axis disagrees after perm_x1/perm_x2: {batch} vs {batch_b}"
        )
    if k != k_b:
        raise ValueError(
            f"the contraction axis disagrees after perm_x1/perm_x2: {k} vs {k_b}"
        )

    # Permutations are views; only an exotic (non-unit innermost) layout forces a copy.
    if a.stride(2) != 1:
        a = a.contiguous()
    if b.stride(2) != 1:
        b = b.contiguous()

    out_shape = (batch, m, n)
    y = torch.empty(
        tuple(out_shape[i] for i in py), dtype=x1.dtype, device=x1.device
    )
    # y == out.permute(py), so out == y.permute(inverse(py)): writing through those
    # strides fills y contiguously instead of filling out and permuting afterwards.
    c = y.permute(_inverse_perm(py))
    if c.stride(2) != 1:
        raise NotImplementedError(
            f"perm_y={py}: the result's N axis is not its innermost axis; a strided "
            "store is not implemented."
        )
    if batch == 0 or m == 0 or n == 0:
        return y
    if k == 0:
        y.zero_()
        return y

    block_n = _BLOCK_N_WIDE if n >= _BLOCK_N_WIDE else _BLOCK_N
    num_mt = _ceil_div(m, _BLOCK_M)
    num_nt = _ceil_div(n, block_n)
    if batch * num_nt > _MAX_PROGRAMS:
        raise NotImplementedError(
            f"batch({batch}) * column tiles({num_nt}) = {batch * num_nt} exceeds the "
            f"backend's {_MAX_PROGRAMS}-program cap; folding the row axis cannot help."
        )
    grid_m = max(1, min(num_mt, _MAX_PROGRAMS // (batch * num_nt)))
    _tbmm_kernel[(batch, grid_m, num_nt)](
        a, b, y,
        m, n, k,
        num_mt,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        grid_m,
        BLOCK_M=_BLOCK_M, BLOCK_N=block_n, BLOCK_K=_BLOCK_K,
    )
    return y


#: ``torch_npu`` spelling of the same operator.
npu_transpose_batchmatmul = transpose_batchmatmul
