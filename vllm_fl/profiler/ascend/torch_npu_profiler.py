# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Adapted from vLLM-Ascend v0.24.0rc1
# https://github.com/vllm-project/vllm-ascend/blob/v0.24.0rc1/vllm_ascend/profiler/torch_npu_profiler.py
# SPDX-License-Identifier: Apache-2.0

"""Ascend implementation of vLLM's worker profiler lifecycle."""

import os
from collections.abc import Mapping
from typing import Any

import torch_npu
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler


def resolve_msmonitor_use_daemon(
    additional_config: Mapping[str, Any] | None = None,
) -> bool:
    """Resolve rc1's config-over-environment profiler/daemon switch."""
    if additional_config is not None and "msmonitor_use_daemon" in additional_config:
        return bool(additional_config["msmonitor_use_daemon"])
    return bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0")))


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of ``WorkerProfiler`` using ``torch_npu.profiler``."""

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        trace_name: str,
        additional_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(profiler_config)
        self.profiler: Any = self._create_profiler(
            profiler_config, trace_name, additional_config
        )

    @staticmethod
    def _create_profiler(
        profiler_config: ProfilerConfig,
        trace_name: str,
        additional_config: Mapping[str, Any] | None = None,
    ) -> Any:
        if profiler_config.profiler != "torch":
            raise RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")
        if resolve_msmonitor_use_daemon(additional_config):
            raise RuntimeError(
                "MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time."
            )

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=False,
            profile_memory=profiler_config.torch_profiler_with_memory,
            # torch_npu's with_modules is the equivalent of torch profiler's
            # with_stack. rc1 avoids with_stack due to trace overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                worker_name=trace_name,
            ),
        )

    def _start(self) -> None:
        self.profiler.start()

    def _stop(self) -> None:
        self.profiler.stop()

    def _profiler_step(self) -> bool:
        return True
