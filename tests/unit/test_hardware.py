"""
Tests for the hardware profiler and resource allocation logic.

Verifies batch size computation across different memory scenarios,
clamping behaviour, memory pressure detection, and the full
profile_resources() integration.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.utils.hardware import (
    compute_batch_size,
    detect_system_memory,
    get_memory_pressure,
    profile_resources,
    ResourceProfile,
    MODEL_MEMORY_GB,
)


# ---- compute_batch_size unit tests ----

class TestComputeBatchSize:
    """Test the core batch sizing formula across hardware tiers."""

    # Model = 6.8GB float32, headroom = 20%, per_segment = 1.5GB
    MODEL = MODEL_MEMORY_GB["float32"]  # 6.8
    COST = 1.5
    HEADROOM = 0.20

    def _calc(self, available, total, **kw):
        return compute_batch_size(
            available_memory_gb=available,
            model_memory_gb=kw.get("model", self.MODEL),
            per_segment_cost_gb=kw.get("cost", self.COST),
            headroom_fraction=kw.get("headroom", self.HEADROOM),
            total_memory_gb=total,
            min_batch=kw.get("min_batch", 1),
            max_batch=kw.get("max_batch", 12),
        )

    def test_8gb_system(self):
        """8GB: almost no headroom, should clamp to min=1."""
        # available ~6GB, model 6.8, headroom 1.6 -> usable = negative
        result = self._calc(available=6.0, total=8.0)
        assert result == 1

    def test_16gb_system(self):
        """16GB: tight but workable."""
        # available ~12GB, model 6.8, headroom 3.2 -> usable = 2.0 -> 1 segment
        result = self._calc(available=12.0, total=16.0)
        assert result == 1

    def test_32gb_system(self):
        """32GB: comfortable for small batches."""
        # available ~26GB, model 6.8, headroom 6.4 -> usable = 12.8 -> 8
        result = self._calc(available=26.0, total=32.0)
        assert result == 8

    def test_64gb_system(self):
        """64GB: plenty of room."""
        # available ~55GB, model 6.8, headroom 12.8 -> usable = 35.4 -> 23 -> clamped to 12
        result = self._calc(available=55.0, total=64.0)
        assert result == 12  # clamped to max

    def test_128gb_system(self):
        """128GB M4 Max: lots of room, should still clamp to max."""
        # available ~110GB, model 6.8, headroom 25.6 -> usable = 77.6 -> 51 -> clamped to 12
        result = self._calc(available=110.0, total=128.0)
        assert result == 12

    def test_clamp_to_min(self):
        """Should never go below min_batch."""
        result = self._calc(available=1.0, total=4.0, min_batch=2)
        assert result == 2

    def test_clamp_to_max(self):
        """Should never exceed max_batch."""
        result = self._calc(available=200.0, total=256.0, max_batch=6)
        assert result == 6

    def test_zero_available(self):
        """Degenerate: 0 available -> min batch."""
        result = self._calc(available=0.0, total=16.0)
        assert result == 1

    def test_bfloat16_model(self):
        """CUDA bfloat16 uses smaller model footprint."""
        # 24GB VRAM, model 3.4, headroom 4.8 -> usable = 15.8 -> 10
        result = self._calc(
            available=24.0, total=24.0,
            model=MODEL_MEMORY_GB["bfloat16"],
        )
        assert result == 10

    def test_custom_per_segment_cost(self):
        """Higher per-segment cost should yield smaller batches."""
        # 64GB, big cost
        small = self._calc(available=55.0, total=64.0, cost=3.0)
        large = self._calc(available=55.0, total=64.0, cost=1.0)
        assert small < large


# ---- Memory pressure ----

class TestMemoryPressure:

    def test_returns_fraction(self):
        """get_memory_pressure should return a value between 0 and 1."""
        pressure = get_memory_pressure()
        assert 0.0 <= pressure <= 1.0

    def test_detect_system_memory_positive(self):
        """detect_system_memory should return positive values."""
        total, available = detect_system_memory()
        assert total > 0
        assert available > 0
        assert available <= total


# ---- profile_resources integration ----

class TestProfileResources:

    def test_auto_mode_returns_valid_profile(self):
        """Auto mode should produce a profile with reasonable defaults."""
        profile = profile_resources()
        assert isinstance(profile, ResourceProfile)
        assert profile.batch_size >= 1
        assert profile.memory_total_gb > 0
        assert profile.flush_interval > 0

    def test_manual_override(self):
        """Manual batch_size_override should be respected."""
        profile = profile_resources(batch_size_override=3, max_batch_size=12)
        assert profile.batch_size == 3

    def test_override_clamped_to_max(self):
        """Override exceeding max should be clamped."""
        profile = profile_resources(batch_size_override=99, max_batch_size=8)
        assert profile.batch_size == 8

    def test_override_clamped_to_min(self):
        """Override below min should be clamped."""
        profile = profile_resources(batch_size_override=0, min_batch_size=1)
        # batch_size_override=0 is treated as falsy, so auto-mode kicks in
        # but any positive override below min should clamp
        profile2 = profile_resources(batch_size_override=1, min_batch_size=2)
        assert profile2.batch_size == 2

    def test_flush_interval_passed_through(self):
        profile = profile_resources(flush_interval=25)
        assert profile.flush_interval == 25

    @patch("src.utils.hardware.detect_system_memory", return_value=(8.0, 5.0))
    @patch("src.utils.hardware.detect_gpu", return_value={
        "gpu_type": "cpu", "gpu_name": "", "gpu_vram_gb": 0.0,
        "device": "cpu", "dtype": "float32",
    })
    def test_low_memory_system(self, mock_gpu, mock_mem):
        """On an 8GB CPU-only system, batch size should be 1."""
        profile = profile_resources()
        assert profile.batch_size == 1
        assert profile.gpu_type == "cpu"
