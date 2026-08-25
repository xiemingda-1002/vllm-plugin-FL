# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for worker module.

Note: These tests require vllm >= 0.13.0 with profiler support.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def has_vllm_profiler():
    """Check if vllm profiler is available."""
    try:
        from vllm.profiler.wrapper import TorchProfilerWrapper  # noqa: F401

        return True
    except ImportError:
        return False


# Skip all tests if vllm profiler is not available
pytestmark = pytest.mark.skipif(
    not has_vllm_profiler(),
    reason="vllm.profiler.wrapper not available (requires vllm >= 0.13.0)",
)


class TestAscendCpuBindingSwitch:
    @patch("vllm_fl.cpu_binding.bind_cpus")
    @patch("vllm_fl.worker.worker.current_platform")
    def test_cpu_binding_defaults_to_enabled_on_npu(
        self, mock_platform, mock_bind_cpus
    ):
        from vllm_fl.worker.worker import _maybe_bind_ascend_cpus

        mock_platform.device_type = "npu"
        _maybe_bind_ascend_cpus(SimpleNamespace(additional_config={}), 1)

        mock_bind_cpus.assert_called_once_with(1)

    @patch("vllm_fl.cpu_binding.bind_cpus")
    @patch("vllm_fl.worker.worker.current_platform")
    def test_cpu_binding_honors_explicit_disable(self, mock_platform, mock_bind_cpus):
        from vllm_fl.worker.worker import _maybe_bind_ascend_cpus

        mock_platform.device_type = "npu"
        config = SimpleNamespace(additional_config={"enable_cpu_binding": False})
        _maybe_bind_ascend_cpus(config, 0)

        mock_bind_cpus.assert_not_called()

    @patch("vllm_fl.cpu_binding.bind_cpus")
    @patch("vllm_fl.worker.worker.current_platform")
    def test_cpu_binding_is_ascend_only(self, mock_platform, mock_bind_cpus):
        from vllm_fl.worker.worker import _maybe_bind_ascend_cpus

        mock_platform.device_type = "cuda"
        _maybe_bind_ascend_cpus(SimpleNamespace(additional_config={}), 0)

        mock_bind_cpus.assert_not_called()

    @patch("vllm_fl.cpu_binding.bind_cpus", side_effect=RuntimeError("topology"))
    @patch("vllm_fl.worker.worker.current_platform")
    def test_cpu_binding_failure_is_non_fatal(
        self, mock_platform, mock_bind_cpus, caplog
    ):
        from vllm_fl.worker.worker import _maybe_bind_ascend_cpus

        mock_platform.device_type = "npu"
        _maybe_bind_ascend_cpus(SimpleNamespace(additional_config={}), 0)

        mock_bind_cpus.assert_called_once_with(0)
        assert "Skipping CPU binding" in caplog.text


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass behavior."""

    def test_default_values_without_auto_measure(self):
        """Test MemorySnapshot initializes with correct default values."""
        from vllm_fl.worker.worker import MemorySnapshot

        snapshot = MemorySnapshot(auto_measure=False)

        assert snapshot.torch_peak == 0
        assert snapshot.free_memory == 0
        assert snapshot.total_memory == 0
        assert snapshot.cuda_memory == 0
        assert snapshot.torch_memory == 0
        assert snapshot.non_torch_memory == 0

    def test_subtraction_computes_difference(self):
        """Test MemorySnapshot subtraction operator computes correct differences."""
        from vllm_fl.worker.worker import MemorySnapshot

        snapshot1 = MemorySnapshot(auto_measure=False)
        snapshot1.torch_peak = 1000
        snapshot1.free_memory = 5000
        snapshot1.total_memory = 10000
        snapshot1.cuda_memory = 5000
        snapshot1.torch_memory = 3000
        snapshot1.non_torch_memory = 2000
        snapshot1.timestamp = 10.0

        snapshot2 = MemorySnapshot(auto_measure=False)
        snapshot2.torch_peak = 500
        snapshot2.free_memory = 6000
        snapshot2.total_memory = 10000
        snapshot2.cuda_memory = 4000
        snapshot2.torch_memory = 2000
        snapshot2.non_torch_memory = 2000
        snapshot2.timestamp = 5.0

        diff = snapshot1 - snapshot2

        assert diff.torch_peak == 500
        assert diff.free_memory == -1000
        assert diff.cuda_memory == 1000
        assert diff.torch_memory == 1000
        assert diff.timestamp == 5.0


class TestMemoryProfilingResult:
    """Test MemoryProfilingResult dataclass behavior."""

    def test_default_values(self):
        """Test MemoryProfilingResult initializes with correct default values."""
        from vllm_fl.worker.worker import MemoryProfilingResult

        result = MemoryProfilingResult()

        assert result.weights_memory == 0
        assert result.torch_peak_increase == 0
        assert result.non_torch_increase == 0
        assert result.non_kv_cache_memory == 0
        assert result.profile_time == 0.0

    def test_creates_default_snapshots(self):
        """Test MemoryProfilingResult creates default snapshot objects."""
        from vllm_fl.worker.worker import MemoryProfilingResult

        result = MemoryProfilingResult()

        assert result.before_profile is not None
        assert result.after_profile is not None
