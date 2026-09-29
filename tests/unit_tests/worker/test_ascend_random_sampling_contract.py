# Copyright (c) 2026 BAAI. All rights reserved.
"""Keep native random sampling safe when enabling FlagGems on Ascend."""

import importlib.util
from pathlib import Path

import pytest
import yaml


def test_native_exponential_dependencies_are_excluded():
    config = Path(__file__).parents[3] / "vllm_fl/dispatch/config/ascend.yaml"
    blacklist = yaml.safe_load(config.read_text())["flagos_blacklist"]
    # Native exponential_ invokes log_ internally; excluding only its outer
    # operator does not prevent FlagGems from replacing that dependency.
    assert {"exponential_", "log_"}.issubset(blacklist)


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None
    or importlib.util.find_spec("flag_gems") is None,
    reason="requires Ascend and FlagGems",
)
def test_flaggems_random_sampling_preserves_one_hot_distribution():
    import torch
    import torch_npu  # noqa: F401
    import flag_gems

    from vllm_fl.sample.sampler import random_sample
    from vllm_fl.utils import get_flag_gems_whitelist_blacklist

    _, blacklist = get_flag_gems_whitelist_blacklist()
    with flag_gems.use_gems(exclude=blacklist):
        noise = torch.empty((1, 151936), device="npu").exponential_()
        torch.npu.synchronize()
        noise_cpu = noise.cpu()
        assert noise_cpu.isfinite().all()
        assert (noise_cpu > 0).all()
        probabilities = torch.zeros((1, 151936), device="npu")
        probabilities[0, 42] = 1
        for _ in range(3):
            assert random_sample(probabilities.clone(), {}).item() == 42
