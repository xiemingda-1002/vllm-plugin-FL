# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig

from .utils.stream_scope import extra_stream_scope_check


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

    def register(self, pm_pass: PatternMatcherPass) -> None:
        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        # Each manager owns its matcher; registrations must remain local even
        # when another manager has already traced an equivalent pattern.
        pm.register_replacement(
            pattern_fn,
            replacement_fn,
            example_inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=self.get_extra_stream_scope_check(),
        )
