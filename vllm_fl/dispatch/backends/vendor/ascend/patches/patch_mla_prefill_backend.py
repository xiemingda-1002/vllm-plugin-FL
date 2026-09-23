# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install vLLM-Ascend's MLA prefill construction boundary.

Upstream MLAAttention always constructs a separate prefill backend. Ascend's
MLA/SFA implementations own prefill inside their attention implementation, so
that auxiliary backend is never executed. The upstream automatic selector can
fall back to CUDA FlashAttention when Ascend has no CUDA-style device
capability, which fails while the model is being constructed.

Keep this patch vendor-scoped and mirror vLLM-Ascend 0.24.0rc1: construct a
no-op backend and leave execution to AscendMLAImpl/AscendSFAImpl.
"""

import torch
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend


class AscendMLAPrefillBackend(MLAPrefillBackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Ascend MLA prefill is handled by AscendSFAImpl/AscendMLAImpl"
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Ascend MLA prefill is handled by AscendSFAImpl/AscendMLAImpl"
        )


def apply_ascend_mla_prefill_backend_patch() -> None:
    """Patch the symbol captured by ``MLAAttention.__init__``.

    This is invoked only from Ascend vendor initialization. Assigning the
    class repeatedly is harmless and avoids modifying the upstream selector
    used by other vendors.
    """
    import vllm.model_executor.layers.attention.mla_attention as mla_attention

    mla_attention.get_mla_prefill_backend = (
        lambda _vllm_config: AscendMLAPrefillBackend
    )
