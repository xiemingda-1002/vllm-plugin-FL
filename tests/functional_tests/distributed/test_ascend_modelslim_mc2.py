# SPDX-License-Identifier: Apache-2.0
"""Installed-package framework fixture: torchrun --nproc-per-node=16 this_file.

Real DP2/TP8/EP16 groups; no model-service or performance acceptance claim.
"""

import gc
import json
import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.npu.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 16,
        reason="Requires torchrun with sixteen Ascend processes",
    ),
]

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_dp_group,
    get_ep_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from vllm_fl import _patch_ascend_torch_accelerator
from vllm_fl.dispatch import get_default_manager, with_strict_mode
from vllm_fl.dispatch.backends.vendor.ascend.impl.w8a8_moe import (
    modelslim_w8a8_experts,
    modelslim_w8a8_prepare_experts,
)
from vllm_fl.forward_context import set_forward_context
from vllm_fl.ops.fused_moe.prepare_finalize import NpuMoEPrepareAndFinalize
from vllm_fl.quantization.w8a8.modelslim import AscendModelSlimMoEMethod

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
HIDDEN, INTERMEDIATE, EXPERTS, LOCAL_EXPERTS = 1024, 64, 32, 2
DTYPE = torch.bfloat16


def record(phase, **values):
    print(json.dumps({"phase": phase, "rank": RANK, **values}), flush=True)


def test_dynamic_experts_mc2_and_alltoall():
    assert int(os.environ["WORLD_SIZE"]) == 16
    torch.npu.set_device(LOCAL_RANK)
    _patch_ascend_torch_accelerator()
    config = VllmConfig(
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[8, 16, 32],
            max_cudagraph_capture_size=32,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=8,
            data_parallel_size=2,
            data_parallel_rank=RANK // 8,
            enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
            all2all_backend="allgather_reducescatter",
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
            max_num_seqs=32,
            max_model_len=8192,
            is_encoder_decoder=False,
        ),
    )
    graph = None
    runtime = None
    with set_current_vllm_config(config), torch.inference_mode():
        try:
            init_distributed_environment(
                world_size=16, rank=RANK, local_rank=LOCAL_RANK, backend="hccl"
            )
            initialize_model_parallel(8, backend="hccl")
            init_workspace_manager(torch.device(f"npu:{LOCAL_RANK}"))
            ep, dp, tp = get_ep_group(), get_dp_group(), get_tp_group()
            assert (ep.world_size, dp.world_size, tp.world_size) == (16, 2, 8)
            # DP assignments can map a logical local ordinal to another NPU.
            # A coordinator's logical device must not own model-side buffers.
            coordinator_device = ep.device_communicator.device
            if dp.rank_in_group == 1:
                ep.device_communicator.device = torch.device("npu", tp.rank_in_group)
            topology = {
                name: {"ranks": group.ranks, "rank_in_group": group.rank_in_group}
                for name, group in (("ep", ep), ("dp", dp), ("tp", tp))
            }
            record("groups", topology=topology)
            parallel = FusedMoEParallelConfig.make(8, 1, 2, 1, config.parallel_config)
            assert parallel.tp_size == parallel.sp_size == 1
            assert parallel.ep_size == 16 and parallel.dp_size == 2
            assert parallel.ep_rank == ep.rank_in_group
            moe = FusedMoEConfig(
                num_experts=EXPERTS,
                experts_per_token=2,
                hidden_dim=HIDDEN,
                intermediate_size=INTERMEDIATE,
                num_local_experts=LOCAL_EXPERTS,
                num_logical_experts=EXPERTS,
                activation=MoEActivation.SILU,
                device=torch.device(f"npu:{LOCAL_RANK}"),
                in_dtype=DTYPE,
                routing_method=RoutingMethodType.Default,
                moe_parallel_config=parallel,
            )
            method = AscendModelSlimMoEMethod(moe)
            layer = torch.nn.Module()
            layer._expert_routing_tables = lambda: None
            layer.activation = moe.activation
            layer.global_num_experts = EXPERTS
            layer.apply_router_weight_on_input = False
            first = ep.rank_in_group * LOCAL_EXPERTS
            layer.expert_map = torch.full(
                (EXPERTS,), -1, dtype=torch.int32, device="npu"
            )
            layer.expert_map[first : first + LOCAL_EXPERTS] = torch.arange(
                LOCAL_EXPERTS, dtype=torch.int32, device="npu"
            )
            method.create_weights(
                layer,
                LOCAL_EXPERTS,
                HIDDEN,
                INTERMEDIATE,
                DTYPE,
                weight_loader=lambda *args: None,
            )
            layer.to("npu")
            # Shared deterministic full weights; local slicing uses active EP rank.
            w1 = (
                torch.arange(EXPERTS * 2 * INTERMEDIATE * HIDDEN, device="npu")
                .reshape(EXPERTS, 2 * INTERMEDIATE, HIDDEN)
                .remainder(47)
                .sub(23)
                .to(torch.int8)
            )
            w2 = (
                torch.arange(EXPERTS * HIDDEN * INTERMEDIATE, device="npu")
                .reshape(EXPERTS, HIDDEN, INTERMEDIATE)
                .add(11)
                .remainder(47)
                .sub(23)
                .to(torch.int8)
            )
            s1 = torch.full(
                (EXPERTS, 2 * INTERMEDIATE, 1),
                0.015625,
                dtype=DTYPE,
                device="npu",
            )
            s2 = torch.full((EXPERTS, HIDDEN, 1), 0.015625, dtype=DTYPE, device="npu")
            for destination, source in (
                (layer.w13_weight, w1),
                (layer.w2_weight, w2),
                (layer.w13_weight_scale, s1),
                (layer.w2_weight_scale, s2),
            ):
                destination.copy_(source[first : first + LOCAL_EXPERTS])
            full_weights = modelslim_w8a8_prepare_experts(w1, w2, s1, s2)
            with with_strict_mode():
                manager = get_default_manager()
                for operation in (
                    "modelslim_w8a8_prepare_experts",
                    "modelslim_w8a8_grouped_experts",
                ):
                    assert manager.get_selected_impl_id(operation) == "vendor.ascend"
                method.process_weights_after_loading(layer)
            prepare = method.moe_kernel.prepare_finalize
            assert isinstance(prepare, NpuMoEPrepareAndFinalize)
            assert method.moe_kernel.output_is_reduced()
            runtime = ep.device_communicator.moe_runtime
            assert prepare.runtime is runtime
            assert runtime.device == runtime._mask.device == layer.w13_weight.device
            record(
                "device_ownership",
                coordinator_device=str(ep.device_communicator.device),
                weight_device=str(layer.w13_weight.device),
                mask_device=str(runtime._mask.device),
                coordinator_is_remapped=dp.rank_in_group == 1,
            )
            assert runtime.mc2_tokens_capacity == 32
            assert dist.get_world_size(runtime._group) == ep.world_size
            assert dist.get_rank(runtime._group) == ep.rank_in_group

            def inputs(rows, step=0):
                index = torch.arange(rows * HIDDEN, device="npu").reshape(rows, HIDDEN)
                x = ((index + 7 * dp.rank_in_group + step * 3) % 29 - 14).float()
                x = (x * 0.02).to(DTYPE)
                row = torch.arange(rows, device="npu", dtype=torch.int32)
                ids = torch.stack((row % EXPERTS, (row + 13) % EXPERTS), dim=1)
                weights = torch.empty((rows, 2), device="npu", dtype=torch.float32)
                weights[:, 0], weights[:, 1] = 0.375, 0.625
                return x, ids.contiguous(), weights

            def oracle(x, ids, weights, actual):
                return modelslim_w8a8_experts(
                    x[:actual],
                    *full_weights,
                    weights[:actual],
                    ids[:actual],
                    global_num_experts=EXPERTS,
                )

            def scope(rows, actual, vector, mode):
                return set_forward_context(
                    None,
                    config,
                    num_tokens=rows,
                    num_actual_tokens=actual,
                    num_tokens_across_dp=vector,
                    cudagraph_runtime_mode=mode,
                    batch_descriptor=BatchDescriptor(
                        num_tokens=rows, num_reqs=rows, uniform=True
                    )
                    if mode == CUDAGraphMode.FULL
                    else None,
                )

            def apply(x, ids, weights):
                metadata = get_forward_context().dp_metadata
                assert metadata is not None
                with metadata.sp_local_sizes(1):
                    return method.apply(layer, x, weights, ids, None, None)

            def check(
                phase,
                output,
                x,
                original,
                ids,
                weights,
                actual,
                vector,
                original_vector,
                selected,
                replay=0,
            ):
                expected = oracle(x, ids, weights, actual)
                assert output.shape == x.shape and output.dtype == DTYPE
                assert torch.isfinite(output[:actual]).all()
                torch.testing.assert_close(
                    output[:actual], expected, rtol=0.03, atol=0.06
                )
                assert torch.equal(x, original)
                assert torch.equal(vector, original_vector)
                error = (
                    float((output[:actual].float() - expected.float()).abs().max())
                    if actual
                    else 0.0
                )
                nonzero = int(torch.count_nonzero(output[:actual]))
                assert actual == 0 or nonzero > 0
                record(
                    phase,
                    dp=dp.rank_in_group,
                    tp=tp.rank_in_group,
                    selectedcomm=selected,
                    capture=int(phase == "mc2_capture"),
                    replay=replay,
                    actualtokens=actual,
                    maxerror=error,
                    nonzero=nonzero,
                    dp_vector=vector.tolist(),
                    dp_vector_invariant=True,
                    input_unchanged=True,
                    output_reduced=method.moe_kernel.output_is_reduced(),
                )

            # Stable local 16-row buffers, independent true activity per DP rank.
            x, ids, weights = inputs(16)
            vector = torch.tensor([16, 16], dtype=torch.int32)
            original_vector = vector.clone()
            initial_counts = (5, 11)
            actual = initial_counts[dp.rank_in_group]
            for _ in range(3):
                with scope(16, actual, vector, CUDAGraphMode.FULL):
                    assert get_forward_context().additional_kwargs["npu_moe"].use_mc2
                    eager = apply(x, ids, weights)
            torch.npu.synchronize()
            dist.barrier(group=ep.cpu_group)
            original = x.clone()
            check(
                "mc2_eager",
                eager,
                x,
                original,
                ids,
                weights,
                actual,
                vector,
                original_vector,
                "mc2",
            )
            graph = torch.npu.NPUGraph()
            # Context entry updates the device mask OUTSIDE graph capture.
            with scope(16, actual, vector, CUDAGraphMode.FULL):
                state = get_forward_context().additional_kwargs["npu_moe"]
                mask_pointer = state.active_mask.data_ptr()
                with torch.npu.graph(graph):
                    output = apply(x, ids, weights)
            graph.replay()
            torch.npu.synchronize()
            check(
                "mc2_capture",
                output,
                x,
                original,
                ids,
                weights,
                actual,
                vector,
                original_vector,
                "mc2",
                replay=1,
            )
            for replay, counts in enumerate(((0, 7), (13, 1), (5, 11)), 1):
                x.copy_(inputs(16, replay)[0])
                original = x.clone()
                actual = counts[dp.rank_in_group]
                with scope(16, actual, vector, CUDAGraphMode.FULL):
                    state = get_forward_context().additional_kwargs["npu_moe"]
                    assert (
                        state.use_mc2 and state.active_mask.data_ptr() == mask_pointer
                    )
                    # No Python prepare/finalize or experts invocation on replay.
                    graph.replay()
                torch.npu.synchronize()
                assert torch.all(state.active_mask[:actual])
                assert not torch.any(state.active_mask[actual:])
                record(
                    "mask",
                    dp=dp.rank_in_group,
                    tp=tp.rank_in_group,
                    replay=replay,
                    maskprefix=actual,
                    active_mask=state.active_mask.cpu().tolist(),
                )
                check(
                    "mc2_replay",
                    output,
                    x,
                    original,
                    ids,
                    weights,
                    actual,
                    vector,
                    original_vector,
                    "mc2",
                    replay,
                )
            graph = None
            gc.collect()
            torch.npu.synchronize()
            dist.barrier(group=ep.cpu_group)

            for counts in ((43, 51), (0, 80)):
                actual = counts[dp.rank_in_group]
                x, ids, weights = inputs(actual)
                original = x.clone()
                vector = torch.tensor(counts, dtype=torch.int32)
                original_vector = vector.clone()
                with scope(actual, actual, vector, CUDAGraphMode.NONE):
                    state = get_forward_context().additional_kwargs["npu_moe"]
                    assert not state.use_mc2
                    output = apply(x, ids, weights)
                torch.npu.synchronize()
                check(
                    "alltoall_eager",
                    output,
                    x,
                    original,
                    ids,
                    weights,
                    actual,
                    vector,
                    original_vector,
                    "alltoall",
                )
                dist.barrier(group=ep.cpu_group)

            # Prove explicit private-group retirement and communicator recreation.
            old_runtime = runtime
            old_mask = old_runtime._mask
            old_runtime.destroy()
            assert old_runtime._closed and old_runtime._group is None
            try:
                old_runtime.mc2_dispatcher(EXPERTS)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Closed MoE runtime accepted a dispatcher")
            ep.device_communicator.moe_runtime = None
            runtime = ep.device_communicator.get_moe_runtime(
                config, device=layer.w13_weight.device
            )
            assert runtime is not old_runtime
            assert runtime._mask.data_ptr() != old_mask.data_ptr()
            assert dist.get_world_size(runtime._group) == 16
            runtime.destroy()
            assert runtime._closed and runtime._group is None
            ep.device_communicator.moe_runtime = None
            ep.device_communicator.device = coordinator_device
            record(
                "runtime_cleanup",
                private_runtime_closed=True,
                private_group_released=True,
                recreation=True,
            )
            dist.barrier(group=ep.cpu_group)
        finally:
            graph = None
            gc.collect()
            torch.npu.synchronize()
            try:
                if runtime is not None:
                    runtime.destroy()
            finally:
                try:
                    reset_workspace_manager()
                finally:
                    try:
                        destroy_model_parallel()
                    finally:
                        destroy_distributed_environment()
            record(
                "cleanup",
                destroyed_model_parallel=True,
                distributed_initialized=dist.is_initialized(),
            )


if __name__ == "__main__":
    test_dynamic_experts_mc2_and_alltoall()
