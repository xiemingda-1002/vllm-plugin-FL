# SPDX-License-Identifier: Apache-2.0
"""Explicit Ascend component-test runtime; never initialized on import."""

from contextlib import contextmanager


@contextmanager
def single_rank_runtime(config_factory, *, workspace=False):
    """Run components with real HCCL groups and a lazily constructed vLLM config."""
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.network_utils import get_open_port

    from vllm_fl import _patch_ascend_torch_accelerator

    torch.npu.set_device(0)
    _patch_ascend_torch_accelerator()
    config = config_factory()
    with set_current_vllm_config(config), torch.inference_mode():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="hccl",
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        try:
            initialize_model_parallel(1, backend="hccl")
            if workspace:
                from vllm.v1.worker.workspace import (
                    init_workspace_manager,
                    reset_workspace_manager,
                )

                init_workspace_manager(torch.device("npu:0"))
            try:
                yield config
            finally:
                if workspace:
                    reset_workspace_manager()
        finally:
            try:
                destroy_model_parallel()
            finally:
                destroy_distributed_environment()
