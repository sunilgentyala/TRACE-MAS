"""Tests for Theorem 3: Temporal Memory-Validation Boundary."""

import math
import numpy as np
import pytest

from trace_mas.temporal import (
    TemporalAlarm,
    TemporalMemoryMonitor,
    TemporalMonitorConfig,
    _kl_divergence_histogram,
)


class TestKLDivergence:
    def test_identical_distributions_zero_kl(self):
        rng = np.random.default_rng(0)
        samples = rng.standard_normal(500).tolist()
        kl = _kl_divergence_histogram(samples, samples)
        assert kl < 0.05

    def test_shifted_distributions_positive_kl(self):
        rng = np.random.default_rng(1)
        p = rng.standard_normal(500).tolist()
        q = (rng.standard_normal(500) + 3.0).tolist()
        kl = _kl_divergence_histogram(p, q)
        assert kl > 0.1

    def test_empty_returns_zero(self):
        assert _kl_divergence_histogram([], [1.0, 2.0]) == 0.0
        assert _kl_divergence_histogram([1.0], []) == 0.0


class TestMissProbability:
    def test_longer_window_lower_miss_prob(self):
        cfg_short = TemporalMonitorConfig(window_size=5, phi_B=0.2, nu=0.01, sigma=0.05)
        cfg_long = TemporalMonitorConfig(window_size=50, phi_B=0.2, nu=0.01, sigma=0.05)
        assert cfg_long.miss_probability() < cfg_short.miss_probability()

    def test_miss_prob_formula(self):
        cfg = TemporalMonitorConfig(window_size=10, phi_B=0.3, nu=0.05, sigma=0.1)
        gap = 0.3 - 0.05
        expected = math.exp(-10 * gap**2 / (2 * 0.1**2))
        assert abs(cfg.miss_probability() - expected) < 1e-12

    def test_zero_gap_returns_one(self):
        cfg = TemporalMonitorConfig(window_size=10, phi_B=0.0, nu=0.1, sigma=0.1)
        assert cfg.miss_probability() == 1.0


class TestBenignTrajectory:
    def test_benign_trajectory_passes(self):
        # Use a very high phi_B so benign trajectories never trigger alarm.
        cfg = TemporalMonitorConfig(window_size=5, phi_B=1000.0, nu=0.0, sigma=0.1)
        monitor = TemporalMemoryMonitor(cfg)
        rng = np.random.default_rng(42)

        for t in range(15):
            # Memory vectors with enough spread so KL is well-defined
            memory = rng.standard_normal(64).tolist()
            result = monitor.add_snapshot(memory, t)
            assert result is None

    def test_certified_rounds_tracked(self):
        cfg = TemporalMonitorConfig(window_size=5, phi_B=1000.0)
        monitor = TemporalMemoryMonitor(cfg)
        rng = np.random.default_rng(7)
        for t in range(10):
            monitor.add_snapshot(rng.standard_normal(32).tolist(), t)
        assert monitor.last_certified_round() == 9


class TestAdversarialTrajectory:
    def test_large_shift_triggers_alarm(self):
        """A sudden large distributional shift should trigger TemporalAlarm."""
        cfg = TemporalMonitorConfig(
            window_size=5, phi_B=0.1, nu=0.0, sigma=0.01, n_bins=16
        )
        monitor = TemporalMemoryMonitor(cfg)
        rng = np.random.default_rng(0)

        # First few rounds: benign (near zero)
        for t in range(4):
            monitor.add_snapshot([0.0] * 8, t)

        # Adversarial shift: large change
        adversarial = (rng.standard_normal(8) * 10.0).tolist()
        with pytest.raises(TemporalAlarm) as exc_info:
            monitor.add_snapshot(adversarial, 4)

        assert exc_info.value.I_value > cfg.phi_B

    def test_alarm_contains_window_info(self):
        cfg = TemporalMonitorConfig(window_size=3, phi_B=0.01, nu=0.0, sigma=0.01)
        monitor = TemporalMemoryMonitor(cfg)
        for t in range(2):
            monitor.add_snapshot([0.0] * 4, t)
        with pytest.raises(TemporalAlarm) as exc_info:
            monitor.add_snapshot([100.0] * 4, 2)
        assert exc_info.value.window == 3
        assert exc_info.value.t == 2
