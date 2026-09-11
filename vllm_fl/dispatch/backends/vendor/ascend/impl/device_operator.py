"""Small current-rc1 device boundary used by the FL-owned Qwen GDN path."""

from __future__ import annotations

import torch


class DeviceOperator:
    """A2/A3 implementations needed by the eager GDN closure.

    Keeping this boundary local avoids importing vllm_ascend's monolithic
    device module, which also initializes unrelated quant, sparse-attention
    and graph capabilities.
    """

    @staticmethod
    def npu_mm_reduce_scatter_base(
        x1: torch.Tensor,
        x2: torch.Tensor,
        hcom: str,
        world_size: int,
        *,
        reduce_op: str = "sum",
        bias: torch.Tensor | None = None,
        x1_scale: torch.Tensor | None = None,
        x2_scale: torch.Tensor | None = None,
        comm_turn: int = 0,
        output_dtype: torch.dtype | None = None,
        comm_mode: str = "aiv",
    ) -> torch.Tensor:
        """Delegate rc1's MMRS boundary without import-time NPU effects."""
        import torch_npu

        return torch_npu.npu_mm_reduce_scatter_base(
            x1,
            x2,
            hcom,
            world_size,
            reduce_op=reduce_op,
            bias=bias,
            x1_scale=x1_scale,
            x2_scale=x2_scale,
            comm_turn=comm_turn,
            output_dtype=output_dtype,
            comm_mode=comm_mode,
        )

    @staticmethod
    def chunk_scaled_dot_kkt_fwd(
        num_core,
        bh_step,
        task_num,
        k,
        beta,
        g_cumsum,
        A,
        cu_seqlens,
        chunk_indices,
        T,
        B,
        H,
        Hg,
        K,
        BT,
        BK,
    ):
        from .fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel

        chunk_scaled_dot_kkt_fwd_kernel[(num_core,)](
            k=k,
            beta=beta,
            g_cumsum=g_cumsum,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            B=B,
            H=H,
            Hg=Hg,
            K=K,
            BT=BT,
            BK=BK,
            bh_step=bh_step,
            task_num=task_num,
            num_core=num_core,
            num_warps=8,
            num_stages=3,
            multibuffer=True,
        )
        return A

    @staticmethod
    def solve_tril_16x16(A, Ad, cu_seqlens, chunk_indices, T, H, BT, LARGE_BLOCK_T, NT, B):
        from .fla.solve_tril import solve_tril_16x16_kernel

        solve_tril_16x16_kernel[NT, B * H](
            A=A,
            Ad=Ad,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            LARGE_BLOCK_T=LARGE_BLOCK_T,
            EXTRACT_SLICE_STRIDE_1=LARGE_BLOCK_T // 32,
            num_warps=1,
            num_stages=4,
        )
        return Ad

    @staticmethod
    def fused_gdn_gating(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_fused_gdn_gating(
            A_log, a, b, dt_bias.to(A_log.dtype)
        )

    @staticmethod
    def moe_gating_top_k(
        x: torch.Tensor,
        *,
        k: int,
        k_group: int,
        group_count: int,
        group_select_mode: int,
        renorm: int,
        norm_type: int,
        out_flag: bool,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
        bias_opt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call FL's A2/A3 native router without importing vllm-ascend."""
        topk_weights, topk_ids, auxiliary = (
            torch.ops._C_ascend.moe_gating_top_k(
                x,
                k=k,
                k_group=k_group,
                group_count=group_count,
                group_select_mode=group_select_mode,
                renorm=renorm,
                norm_type=norm_type,
                out_flag=out_flag,
                routed_scaling_factor=routed_scaling_factor,
                eps=eps,
                bias_opt=bias_opt,
            )
        )
        return topk_weights, topk_ids.to(torch.int32), auxiliary

    @staticmethod
    def npu_moe_init_routing(
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        active_num: int,
        expert_num: int,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        active_expert_range: list[int] | None = None,
        quant_mode: int = -1,
        act_quant_type: torch.dtype | None = None,
    ):
        if act_quant_type is not None or quant_mode != -1:
            raise NotImplementedError(
                "FL Ascend A2 BF16 MoE does not support quantized routing"
            )
        return torch.ops._C_ascend.npu_moe_init_routing_custom(
            hidden_states,
            topk_ids,
            scale=scale,
            active_num=active_num,
            expert_num=expert_num,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            active_expert_range=active_expert_range,
            quant_mode=quant_mode,
        )

    @staticmethod
    def npu_moe_token_unpermute(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        probs: torch.Tensor | None,
    ) -> torch.Tensor:
        import torch_npu

        return torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permuted_tokens,
            sorted_indices=torch.abs(sorted_indices),
            probs=probs,
        )

    @staticmethod
    def maybe_normalize_mxfp_scale_layout(
        scale: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return scale

    @staticmethod
    def npu_dynamic_quant(*_args, **_kwargs):
        raise NotImplementedError(
            "FL Ascend A2 BF16 MoE does not support MXFP MoE quantization"
        )

    @staticmethod
    def npu_grouped_matmul_swiglu_quant(*_args, **_kwargs):
        raise NotImplementedError(
            "FL Ascend A2 BF16 MoE does not support MXFP MoE quantization"
        )

    @staticmethod
    def npu_grouped_matmul_gmm2(*_args, **_kwargs):
        raise NotImplementedError(
            "FL Ascend A2 BF16 MoE does not support MXFP MoE quantization"
        )

    @staticmethod
    def split_qkv_rmsnorm_rope(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        q_hidden_size: int,
        kv_hidden_size: int,
        head_dim: int,
        eps: float,
        q_bias: torch.Tensor | None,
        k_bias: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Delegate the fused graph target through its canonical vLLM op."""
        from .graph_fusion_ops import ensure_graph_fusion_ops_registered

        ensure_graph_fusion_ops_registered()
        return torch.ops.vllm.qkv_rmsnorm_rope(
            input=input,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=q_hidden_size,
            kv_hidden_size=kv_hidden_size,
            head_dim=head_dim,
            eps=eps,
            q_bias=q_bias,
            k_bias=k_bias,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
        )
