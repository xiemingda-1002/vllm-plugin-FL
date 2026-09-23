# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend graph compiler integration owned by the FL plugin.

This module implements the current Ascend npugraph_ex path and its fusion-only
fallback. Other Ascend compilation modes fail closed until their matching
dependency closures are migrated independently.
"""

from __future__ import annotations

import copy
import functools
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
from torch._inductor.decomposition import select_decomp_table
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger

from vllm_fl.compilation.config import (
    ASCEND_COMPILATION_DEFAULTS,
    COMPILATION_PASS_KEY,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class AscendCompilerOptions:
    """The supported subset of the current Ascend compilation config."""

    enable_npugraph_ex: bool = True
    enable_static_kernel: bool = False
    fuse_norm_quant: bool = True
    fuse_qknorm_rope: bool = True
    fuse_allreduce_rms: bool = False
    fuse_muls_add: bool = True

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "AscendCompilerOptions":
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        raw_options = additional_config.get("ascend_compilation_config", {})
        if not isinstance(raw_options, dict):
            raise TypeError("additional_config.ascend_compilation_config must be a dict")
        supported = set(ASCEND_COMPILATION_DEFAULTS)
        unknown = sorted(set(raw_options) - supported)
        if unknown:
            raise ValueError(f"Unsupported FL Ascend compiler options: {unknown}")
        for name, value in raw_options.items():
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        values = {
            name: raw_options.get(name, default)
            for name, default in ASCEND_COMPILATION_DEFAULTS.items()
        }
        return cls(**values)

    def validate_supported(self) -> None:
        if self.enable_static_kernel:
            raise NotImplementedError(
                "FL AscendCompiler does not yet support enable_static_kernel=True"
            )


def _configure_npugraph_ex(config: Any, process_kwargs_options: Callable) -> None:
    options: dict[str, Any] = {
        "force_eager": True,
        "inplace_pass": False,
        "clone_input": False,
        "clone_output": False,
    }
    process_kwargs_options(config, {"options": options})


def _compile_with_npugraph_ex(
    graph: fx.GraphModule,
    example_inputs: list[Any],
) -> Callable[..., Any]:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "FL AscendCompiler requires an importable torch_npu runtime"
        ) from exc

    try:
        import npugraph_ex as nge
    except ImportError as exc:
        raise RuntimeError(
            "FL AscendCompiler requires npugraph_ex; refusing to fall back to Inductor"
        ) from exc

    try:
        from npugraph_ex.configs.compiler_config import _process_kwargs_options
    except ImportError:
        try:
            from npugraph_ex.configs.npugraphex_config import (
                _process_kwargs_options,
            )
        except ImportError as exc:
            raise RuntimeError(
                "npugraph_ex does not expose a supported _process_kwargs_options API"
            ) from exc

    if not hasattr(torch, "npu"):
        raise RuntimeError("torch_npu did not register torch.npu")

    torch.npu.set_compile_mode(jit_compile=False)
    config = nge.CompilerConfig()
    _configure_npugraph_ex(config, _process_kwargs_options)
    backend = nge.get_npu_backend(compiler_config=config)

    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs, backend)
    return backend(graph, example_inputs)


def _compile_fx(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    inner_compile: Callable[..., Callable[..., Any]],
    decompositions: dict[Any, Any],
) -> Callable[..., Any]:
    """Run the current vLLM-Ascend fusion-only AOTAutograd pipeline."""
    recursive_compile_fx = functools.partial(
        _compile_fx,
        inner_compile=inner_compile,
        decompositions=decompositions,
    )
    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs, recursive_compile_fx)
    return aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)


def _compile_with_fusion_passes(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
) -> Callable[..., Any]:
    """Apply FL's Ascend fusion manager without invoking npugraph_ex."""

    def compile_inner(
        graph_module: fx.GraphModule, inner_example_inputs: list[Any]
    ) -> Callable[..., Any]:
        del inner_example_inputs
        try:
            pass_manager = compiler_config[COMPILATION_PASS_KEY]
        except KeyError as exc:
            raise RuntimeError(
                "Ascend fusion-only compilation requires compiler config key "
                f"{COMPILATION_PASS_KEY!r}"
            ) from exc
        # AOTAutograd supplies a GraphModule, while vLLM 0.24 custom graph
        # passes consume its mutable Graph. Recompile after applying the manager.
        pass_manager(graph_module.graph)
        graph_module.recompile()
        return graph_module

    return _compile_fx(
        graph=graph,
        example_inputs=example_inputs,
        inner_compile=compile_inner,
        decompositions=select_decomp_table(),
    )


class AscendCompiler(CompilerInterface):
    """Current-version graph compiler for the FL Ascend platform."""

    name = "FLAscendCompiler"

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None:
        # This first migration slice deliberately does not persist compiled code.
        self.cache_dir = cache_dir
        self.disable_cache = disable_cache
        self.cache_prefix = prefix

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        self.vllm_config = vllm_config
        options = AscendCompilerOptions.from_vllm_config(vllm_config)
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError(
                "FL AscendCompiler requires an importable torch_npu runtime"
            ) from exc

        factors = {
            "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
            **asdict(options),
        }
        logger.info("FL AscendCompiler hash factors: %s", factors)
        return sha256(
            json.dumps(factors, sort_keys=True).encode(), usedforsecurity=False
        ).hexdigest()[:10]

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any], None]:
        del compile_range, key
        if not hasattr(self, "vllm_config"):
            raise RuntimeError("compute_hash must be called before compile")

        options = AscendCompilerOptions.from_vllm_config(self.vllm_config)
        options.validate_supported()

        # Keep the caller's FX graph intact and normalize FakeTensor ownership,
        # matching the current vLLM-Ascend 0.24 compiler contract.
        graph = copy.deepcopy(graph)
        from torch._guards import detect_fake_mode

        current_fake_mode = detect_fake_mode()
        if current_fake_mode is not None:
            example_inputs = [
                current_fake_mode.from_tensor(inp)
                if (
                    isinstance(inp, torch.Tensor)
                    and hasattr(inp, "fake_mode")
                    and inp.fake_mode is not current_fake_mode
                )
                else inp
                for inp in example_inputs
            ]

        if options.enable_npugraph_ex:
            logger.info_once(
                "FL AscendCompiler is compiling with npugraph_ex.", scope="global"
            )
            compiled = _compile_with_npugraph_ex(graph, example_inputs)
        else:
            logger.info_once(
                "FL AscendCompiler is compiling with Ascend fusion passes only.",
                scope="global",
            )
            compiled = _compile_with_fusion_passes(
                graph,
                example_inputs,
                compiler_config,
            )
        return compiled, None
