# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable graph lifecycle primitives for FL model runners.

The vLLM runner owns the shapes and contents of its persistent input buffers.
This module owns the device graph lifecycle around those buffers: device-class
resolution, capture-stream selection, wrapper dispatch, stable input bindings,
capture/replay state, and decoder-wrapper cleanup.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import GraphCaptureContext
from vllm.platforms import current_platform

_GRAPH_CLASS_NAMES = {
    "cuda": "CUDAGraph",
    "npu": "NPUGraph",
    "musa": "MUSAGraph",
    "ptpu": "PTPUGraph",
}


class GraphPhase(Enum):
    """Current execution phase of one graph wrapper."""

    IDLE = auto()
    CAPTURING = auto()
    REPLAYING = auto()


@dataclass(frozen=True)
class StaticInputBindings:
    """Tensor addresses captured from vLLM-owned persistent input buffers."""

    addresses: tuple[int, ...]

    @classmethod
    def from_args(cls, args: tuple[Any, ...]) -> "StaticInputBindings":
        return cls(
            tuple(arg.data_ptr() for arg in args if isinstance(arg, torch.Tensor))
        )

    def validate(self, args: tuple[Any, ...]) -> None:
        current = self.from_args(args).addresses
        assert current == self.addresses, (
            "Input addresses for graphs changed during replay. "
            f"Expected {self.addresses}, got {current}"
        )


class GraphRuntimeController:
    """Own graph capture/replay and wrapper lifecycle for an FL runner.

    Device resolution is deliberately lazy so importing the plugin does not
    require the accelerator runtime to be initialized. Attention metadata is
    passed unchanged to lifecycle hooks through vLLM's ``ForwardContext``;
    its construction remains in vLLM 0.24, where persistent input buffers and
    padding semantics are defined.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        device_type: str | None = None,
        platform: Any | None = None,
        full_graph_wrapper_type: type[Any] | None = None,
        breakable_graph_wrapper_type: type[Any] | None = None,
        ubatch_wrapper_type: type[Any] | None = None,
    ) -> None:
        self._vllm_config = vllm_config
        self._device_type = device_type
        self._platform = platform
        self._full_graph_wrapper_type = full_graph_wrapper_type
        self._breakable_graph_wrapper_type = breakable_graph_wrapper_type
        self._ubatch_wrapper_type = ubatch_wrapper_type
        self.phase = GraphPhase.IDLE
        self._update_stream: Any | None = None

    @property
    def platform(self) -> Any:
        return self._platform if self._platform is not None else current_platform

    @property
    def device_type(self) -> str:
        return self._device_type or self.platform.device_type

    def resolve_graph_class(self) -> type[Any] | None:
        """Resolve the active torch graph class only when capture needs it."""
        if self.device_type == "txda":
            return None
        class_name = _GRAPH_CLASS_NAMES.get(self.device_type)
        if class_name is None:
            raise NotImplementedError(
                f"Static graph is not supported on device type {self.device_type!r}"
            )
        try:
            device_module = getattr(torch, self.device_type)
            return getattr(device_module, class_name)
        except AttributeError as exc:
            raise NotImplementedError(
                f"Torch does not provide {class_name} for "
                f"device type {self.device_type!r}"
            ) from exc

    def create_graph(self) -> Any:
        graph_class = self.resolve_graph_class()
        if graph_class is None:
            raise NotImplementedError(
                f"Static graph capture is unavailable for {self.device_type!r}"
            )
        return graph_class()

    def prepare_model_compile(self) -> int:
        """Resolve FL operator dispatch before Ascend fullgraph tracing."""
        if self.device_type != "npu":
            return 0
        from vllm_fl.dispatch import prewarm_cached_ops

        return prewarm_cached_ops()

    def bind_static_inputs(self, args: tuple[Any, ...]) -> StaticInputBindings:
        return StaticInputBindings.from_args(args)

    @contextmanager
    def capture_scope(self, forward_context: Any) -> Iterator[None]:
        assert self.phase is GraphPhase.IDLE
        self.phase = GraphPhase.CAPTURING
        began = False
        try:
            self.on_capture_begin(forward_context)
            began = True
            yield
        finally:
            try:
                if began:
                    self.on_capture_end(forward_context)
            finally:
                self.phase = GraphPhase.IDLE

    @contextmanager
    def replay_scope(self, forward_context: Any) -> Iterator[None]:
        assert self.phase is GraphPhase.IDLE
        self.phase = GraphPhase.REPLAYING
        began = False
        replay_enqueued = False
        try:
            self.before_replay(forward_context)
            began = True
            yield
            replay_enqueued = True
        finally:
            try:
                if began and replay_enqueued:
                    self.after_replay(forward_context)
            finally:
                self.phase = GraphPhase.IDLE

    def on_capture_begin(self, forward_context: Any) -> None:
        """Device extension hook invoked with vLLM's live ForwardContext."""
        if self.device_type == "npu":
            from vllm_fl.compilation.graph_params import prepare_graph_params

            num_tokens = self._num_tokens(forward_context)
            prepare_graph_params(num_tokens)
            # vllm-ascend 0.24rc1 uses this bit to distinguish the one real
            # capture from preceding warmups in attention/model components.
            forward_context.capturing = True

    def on_capture_end(self, forward_context: Any) -> None:
        """Device extension hook that also runs when graph capture fails."""
        if self.device_type == "npu":
            try:
                from vllm_fl.compilation.graph_params import weak_ref_workspace

                weak_ref_workspace(self._num_tokens(forward_context))
            finally:
                forward_context.capturing = False

    def before_replay(self, forward_context: Any) -> None:
        """Wait for the previous replay before enqueueing the next graph."""
        if self.device_type == "npu":
            self.platform.torch_device_fn.current_stream().synchronize()

    def after_replay(self, forward_context: Any) -> None:
        """Update NPU task parameters after replay enqueue releases its event."""
        if self.device_type == "npu":
            if self._vllm_config is None:
                raise RuntimeError(
                    "Ascend full-graph replay requires GraphRuntimeController "
                    "to own the active VllmConfig"
                )
            if self._update_stream is None:
                self._update_stream = self.platform.torch_device_fn.Stream()
            from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
                AscendAttentionBackendImpl,
            )

            AscendAttentionBackendImpl.update_graph_params(
                self._update_stream,
                forward_context,
                self._num_tokens(forward_context),
                self._vllm_config,
            )

    def wrap_model(
        self,
        model: Any,
        vllm_config: VllmConfig,
        *,
        cudagraph_mode: CUDAGraphMode,
        use_ubatching: bool,
        device: torch.device,
        drafter: Any | None,
        breakable_enabled: bool,
    ) -> Any:
        """Apply the vLLM 0.24 wrapper policy without runner-local branches."""
        if breakable_enabled and cudagraph_mode != CUDAGraphMode.NONE and not use_ubatching:
            wrapper_type = self._require_wrapper_type(
                self._breakable_graph_wrapper_type, "breakable graph"
            )
            model = wrapper_type(model, vllm_config)
            if drafter is not None and hasattr(drafter, "model"):
                drafter.model = wrapper_type(drafter.model, vllm_config)
        elif cudagraph_mode.has_full_cudagraphs() and not use_ubatching:
            wrapper_type = self._require_wrapper_type(
                self._full_graph_wrapper_type, "full graph"
            )
            model = wrapper_type(
                model,
                vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                graph_runtime=self,
            )
        elif use_ubatching:
            wrapper_type = self._require_wrapper_type(
                self._ubatch_wrapper_type, "ubatch"
            )
            runtime_mode = (
                CUDAGraphMode.FULL
                if cudagraph_mode.has_full_cudagraphs()
                else CUDAGraphMode.NONE
            )
            model = wrapper_type(model, vllm_config, runtime_mode, device)
        return model

    @staticmethod
    def _require_wrapper_type(
        wrapper_type: type[Any] | None, description: str
    ) -> type[Any]:
        if wrapper_type is None:
            raise RuntimeError(f"No {description} wrapper type was configured")
        return wrapper_type

    def decoder_graph_wrappers(self) -> list[Any]:
        """Return full and breakable wrappers tracked by vLLM 0.24."""
        wrappers: list[Any] = []
        for wrapper_type in (
            self._full_graph_wrapper_type,
            self._breakable_graph_wrapper_type,
        ):
            instances = getattr(wrapper_type, "_all_instances", None)
            if instances is not None:
                wrappers.extend(list(instances))
        return wrappers

    def set_decoder_graph_pool(self, graph_pool: Any) -> dict[int, Any]:
        original_pools: dict[int, Any] = {}
        for wrapper in self.decoder_graph_wrappers():
            original_pools[id(wrapper)] = wrapper.graph_pool
            wrapper.graph_pool = graph_pool
        return original_pools

    def restore_decoder_graph_pools(self, original_pools: dict[int, Any]) -> None:
        for wrapper in self.decoder_graph_wrappers():
            if id(wrapper) in original_pools:
                wrapper.graph_pool = original_pools[id(wrapper)]

    def clear_decoder_graphs(self) -> None:
        for wrapper_type in (
            self._full_graph_wrapper_type,
            self._breakable_graph_wrapper_type,
        ):
            clear = getattr(wrapper_type, "clear_all_graphs", None)
            if clear is not None:
                clear()
        if self.device_type == "npu":
            from vllm_fl.compilation.graph_params import clear_graph_params

            clear_graph_params()

    @staticmethod
    def _num_tokens(forward_context: Any) -> int:
        descriptor = getattr(forward_context, "batch_descriptor", None)
        num_tokens = getattr(descriptor, "num_tokens", None)
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            raise RuntimeError(
                "Ascend full-graph capture/replay requires a BatchDescriptor "
                "with an integer num_tokens"
            )
        return num_tokens


def get_graph_capture(
    default_capture: Callable[[torch.device], Any],
) -> Callable[[torch.device], Any]:
    """Wrap vLLM's capture context with the FL multi-device stream policy."""

    @contextmanager
    def graph_capture(device: torch.device) -> Iterator[GraphCaptureContext]:
        platform = current_platform
        if platform.dist_backend != "flagcx" and platform.device_type not in {
            "musa",
            "npu",
        }:
            with default_capture(device=device) as capture_context:
                yield capture_context
            return

        capture_context = GraphCaptureContext(
            platform.torch_device_fn.Stream(device=device)
        )
        capture_stream = capture_context.stream
        current_stream = platform.torch_device_fn.current_stream()
        if current_stream != capture_stream:
            capture_stream.wait_stream(current_stream)

        with platform.torch_device_fn.stream(capture_stream):
            yield capture_context

    return graph_capture


def get_graph_class(device_type: str | None = None) -> type[Any] | None:
    """Compatibility helper for callers that only need class resolution."""
    return GraphRuntimeController(device_type=device_type).resolve_graph_class()
