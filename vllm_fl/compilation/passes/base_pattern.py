# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig

from .utils.npugraph_ex_utils_check import extra_stream_scope_check

_registered_npugraph_patterns: set[tuple[object, ...]] = set()


class BasePattern(ABC):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    @abstractmethod
    def get_inputs(self) -> list[torch.Tensor]: ...

    @abstractmethod
    def get_pattern(self) -> Callable: ...

    @abstractmethod
    def get_replacement(self) -> Callable: ...

    def get_extra_stream_scope_check(self):
        return extra_stream_scope_check

    def registration_key(self) -> tuple[object, ...]:
        """Describe every pattern dimension that changes traced semantics."""
        dimensions = tuple(
            (name, getattr(self, name))
            for name in (
                "eps",
                "dtype",
                "head_dim",
                "num_heads",
                "num_kv_heads",
                "q_size",
                "kv_size",
                "rope_dim",
                "scale",
            )
            if hasattr(self, name)
        )
        return (self.__class__.__module__, self.__class__.__qualname__, *dimensions)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        # Each manager owns its PatternMatcherPass and must always receive the
        # local registration, even after the process-global npugraph registry
        # has already seen an equivalent pattern.
        pm.register_replacement(
            pattern_fn,
            replacement_fn,
            example_inputs,
            pm.fwd_only,
            pm_pass,
        )

        key = self.registration_key()
        if key in _registered_npugraph_patterns:
            return
        try:
            import npugraph_ex as nge
        except ImportError as exc:
            raise RuntimeError(
                "FL Ascend graph fusion requires importable npugraph_ex"
            ) from exc
        nge.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=example_inputs,
            extra_check=self.get_extra_stream_scope_check(),
        )
        _registered_npugraph_patterns.add(key)
