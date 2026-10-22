# Copyright (c) 2026 BAAI. All rights reserved.
# Triton implementation of the CANN operator of the same name (NOT vendored in csrc/ascend).
# SPDX-License-Identifier: Apache-2.0
"""format_cast -- ``torch_npu.npu_format_cast`` (ACL ND <-> NZ relayout)

## Verdict: the NZ direction is **NOT PORTABLE to Triton**. This module therefore does
## not contain a kernel; it contains the evidence, plus the one case that is well defined.

Evidence that this operator does not exist in the tree:

    grep -rn "format_cast|FormatCast" csrc                  -> no hit
    find csrc -iname "*format_cast*"                        -> no hit
    grep -rn "FRACTAL|format_cast" over _repos/triton-ascend (*.py, *.pyi, *.h, *.cpp,
                                                             *.md, *.rst)  -> no hit
        (triton-ascend has no NZ / tensor-format API at all)

Evidence about where the format lives (all in this repository):

    csrc/ascend/aclnn_torch_adapter/NPUStorageImpl.h
        L29-37  struct NPUStorageDesc { base_sizes_, base_strides_, storage_sizes_,
                                        origin_format_, npu_format_ }   // aclFormat fields
        L52     NPUStorageDesc npu_desc_;   // carried by every NPU storage
    csrc/ascend/aclnn_torch_adapter/op_api_common.h
        L149-157  IsOpInputBaseFormat() returns whether
                  NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_ is ND/NCHW/NHWC/NCDHW
        L393-399  when it is NOT a base format, the aclTensor handed to aclnn is created
                  with  format = npu_desc_.npu_format_
                  and   storageDims = npu_desc_.storage_sizes_
                  while the logical sizes stay the tensor's own sizes
    csrc/ascend/common/include/fallback/fallback.h
        L299-304  aclCreateTensor(..., acl_format = ACL_FORMAT_FRACTAL_NZ, ...) for an NZ
                  GE tensor; shape/strides remain the logical ND ones

So an "NZ tensor" is the triple (data_ptr, storage_sizes_, npu_format_=29). The physical
layout is a 16x16 (M0 x N0) fractal blocking with padding; the logical [M, N] shape is kept
as a *view* over that buffer. Confirmation that the physical rank actually changes comes
from the NZ branch of the sibling operator contract, quoted in
``impl/batch_matmul_transpose.py`` L32-34 (vllm-ascend
``batch_matmul_transpose/op_host/batch_matmul_transpose.h`` L73-76):
    "NZ: tensor_b is 4-D, tensor_a.shape[2] == tensor_b.shape[2] and
     n = tensor_b.shape[1] * tensor_b.shape[3]"
i.e. an [K, N] weight becomes a 4-D [N1, K1, K0, N0] storage.

Why Triton cannot do it:

1. **The format is metadata, not data.** A Triton program receives pointers, sizes and
   strides; the ``npu_format_`` / ``storage_sizes_`` fields of ``NPUStorageDesc`` are
   C++ storage fields with no Python or Triton surface. A kernel can permute bytes but it
   cannot make the result *be* an NZ tensor.
2. **A padded fractal layout is not expressible as strides.** PyTorch describes a tensor
   as (storage, offset, sizes, strides); a blocked layout with an intra-block transposition
   and padding cannot be written as a stride tuple, so even the pure data transform has no
   view form -- it can only be a byte permutation into a fresh buffer, which is where (1)
   stops it.
3. **Every consumer in this tree dispatches on the format field.** ``npu_quant_matmul``,
   ``npu_grouped_matmul``, ``batch_matmul_transpose`` and ``npu_mla_preprocess`` all read
   the aclFormat of their arguments; a Triton-produced flat buffer is an ND tensor and
   would be silently mis-read as if it were ND (exactly the "silently wrong" outcome this
   migration must avoid).
4. **In a Triton-only backend NZ is meaningless, and actively harmful.** A Triton matmul
   reads ND strides, so an NZ weight is unreadable (the values are permuted). The correct
   action is to *never produce* NZ: the policy gate ``maybe_trans_nz``
   (``impl/quantization/linear_utils.py`` L80-111, ``_should_trans_nz``) must return the
   weight unchanged. That is a one-line change in an existing file, which this task is not
   allowed to make; it is recorded in RESULT.md as the required wiring step.
5. **The exact geometry is not documented anywhere in the tree.** M0/N0 per dtype, the
   padding value, the behaviour for 3-D weights ([E, K, N] in ``impl/moe/fused_moe.py``
   L148-149, ``impl/quantization/moe.py`` L270-273) and the reverse NZ -> ND direction are
   all unstated; writing an encoder would be guessing a byte layout that no in-tree test
   pins. Per the task rules ("do not hard-write a wrong implementation") this is refused.

## Call sites (20, all in vllm_fl)

    ND   (ACL_FORMAT_FRACTAL_ND = 2, sfa_v1.py L50):
        vllm_fl/attention/ascend/sfa_v1.py:700   npu_format_cast(kv_b_proj.weight.data, 2).T
    NZ   (ACL_FORMAT_FRACTAL_NZ = 29):
        vllm_fl/attention/ascend/sfa_v1.py:836, 837, 838, 883, 905, 948, 955, 958, 972,
            977, 981
        vllm_fl/dispatch/backends/vendor/ascend/dsa_compat.py:157, 163
        vllm_fl/dispatch/backends/vendor/ascend/impl/moe/compat.py:104
        vllm_fl/dispatch/backends/vendor/ascend/impl/moe/fused_moe.py:148, 149
        vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/linear_utils.py:111
        vllm_fl/dispatch/backends/vendor/ascend/impl/quantization/moe.py:270, 273

## What this module implements

``format_cast(tensor, ACL_FORMAT_FRACTAL_ND)`` -- the ND -> ND request -- is an identity
of the *values*: the tensor already carries ND, so the cast cannot change anything. That is
returned as the input tensor itself (no copy). It is only meaningful as an assertion, and
this module states the precondition explicitly: it is correct **iff** the input is already
ND. The only ND call site satisfies that by construction (a freshly loaded 2-D
``nn.Linear`` weight, ``sfa_v1.py:700``).

``format_cast(tensor, ACL_FORMAT_FRACTAL_NZ)`` raises ``NotImplementedError`` naming the
format and the reason. It does **not** return the input (that would be a silent
ND-for-NZ swap and would corrupt every downstream CANN matmul).
"""

from __future__ import annotations

import torch

__all__ = [
    "format_cast",
    "npu_format_cast",
    "ACL_FORMAT_FRACTAL_ND",
    "ACL_FORMAT_FRACTAL_NZ",
]

#: Values used by the tree: ``impl/quantization/linear_utils.py`` L32, ``impl/moe/compat.py``
#: L16, ``impl/quantization/moe.py`` L43 (NZ = 29) and ``sfa_v1.py`` L50-51 (ND = 2, NZ = 29).
ACL_FORMAT_FRACTAL_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29

_NZ_REASON = (
    "The NZ (ACL_FORMAT_FRACTAL_NZ, 29) relayout is not portable to Triton. NZ is not a "
    "value transform but a storage format: the format and the physical storage dims live "
    "in NPUStorageDesc (csrc/ascend/aclnn_torch_adapter/NPUStorageImpl.h L29-37, read back "
    "at op_api_common.h L149-157 and L393-399), while the logical [M, N] shape stays a view "
    "over a padded 16x16-fractal buffer. A Triton kernel only ever sees pointers, sizes and "
    "strides, so it can neither set npu_format_ nor describe the blocked padded layout as "
    "strides, and every consumer of the result in this tree (npu_quant_matmul, "
    "npu_grouped_matmul, batch_matmul_transpose, npu_mla_preprocess) dispatches on exactly "
    "that format field -- an ND buffer would be silently mis-read. A Triton-only backend "
    "must therefore never produce NZ: make maybe_trans_nz "
    "(impl/quantization/linear_utils.py L80-111) return the weight unchanged and keep the "
    "weights in ND. The NZ geometry (M0/N0 per dtype, padding, 3-D [E, K, N] weights, the "
    "reverse NZ -> ND direction) is not documented anywhere in this tree, so no encoder is "
    "included rather than shipping an unverifiable byte layout. See this module's docstring "
    "and RESULT.md for the full evidence."
)


def format_cast(tensor: torch.Tensor, acl_format) -> torch.Tensor:
    """``torch_npu.npu_format_cast`` for the two ACL formats this tree uses.

    Args:
        tensor: the tensor to relayout (the op is applied to weights only in this tree).
        acl_format: ``ACL_FORMAT_FRACTAL_ND`` (2) / ``"ND"`` / ``None`` -- identity, the
            values are unchanged; ``ACL_FORMAT_FRACTAL_NZ`` (29) / ``"NZ"`` -- raises,
            see the module docstring.

    Returns:
        ``tensor`` itself for the ND request.

    Raises:
        NotImplementedError: for the NZ request and for any other format value.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"format_cast expects a torch.Tensor, got {type(tensor).__name__}")

    if isinstance(acl_format, str):
        fmt = acl_format.upper()
        if fmt == "ND":
            return tensor
        if fmt == "NZ":
            raise NotImplementedError(_NZ_REASON)
        raise NotImplementedError(
            f"acl_format={acl_format!r} is not one of the formats used in this tree "
            "(ND, NZ)"
        )

    if acl_format is None or acl_format == ACL_FORMAT_FRACTAL_ND:
        # ND -> ND. The values cannot change; this is an assertion that the tensor is
        # already ND, which holds at the only ND call site (a freshly loaded nn.Linear
        # weight, sfa_v1.py:700).
        return tensor

    if acl_format == ACL_FORMAT_FRACTAL_NZ:
        raise NotImplementedError(_NZ_REASON)

    raise NotImplementedError(
        f"acl_format={acl_format!r} is not implemented; this tree only uses "
        f"ACL_FORMAT_FRACTAL_ND={ACL_FORMAT_FRACTAL_ND} and "
        f"ACL_FORMAT_FRACTAL_NZ={ACL_FORMAT_FRACTAL_NZ}"
    )


#: ``torch_npu`` spelling of the same operator.
npu_format_cast = format_cast
