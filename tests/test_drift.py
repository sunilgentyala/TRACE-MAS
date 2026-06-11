"""Tests for Theorem 1: Drift-Decay under Contractive Aggregation."""

import math
import numpy as np
import pytest

from trace_mas.drift import (
    ContractiveDriftCorrector,
    DriftAlarm,
    DriftCorrectorConfig,
)


def _perfect_verifier(alpha_0: np.ndarray):
    """Verifier that always returns alpha_0 exactly (rho=0)."""
    def fn(output):
        return alpha_0.copy()
    return fn


def _noisy_verifier(alpha_0: np.ndarray, rho: float = 0.03):
    """Verifier with bounded residual rho."""
    rng = np.random.default_rng(42)
    def fn(output):
        noise = rng.standard_normal(alpha_0.shape)
        noise = noise / np.linalg.norm(noise) * rho
        return alpha_0 + noise
    return fn


class TestTheoreticalEnvelope:
    def test_envelope_formula(self):
        cfg = DriftCorrectorConfig(gamma_min=0.4, rho=0.05, epsilon=0.02, delta_star=1.0)
        lam = 1.0 - 0.4
        expected = 0.05 + lam * 0.02 / (1.0 - lam)
        assert abs(cfg.asymptotic_envelope() - expected) < 1e-12

    def test_envelope_shrinks_with_larger_gamma(self):
        cfg_low = DriftCorrectorConfig(gamma_min=0.2, rho=0.05, epsilon=0.02, delta_star=1.0)
        cfg_high = DriftCorrectorConfig(gamma_min=0.8, rho=0.05, epsilon=0.02, delta_star=1.0)
        assert cfg_high.asymptotic_envelope() < cfg_low.asymptotic_envelope()

    def test_bound_n_steps_tight(self):
        alpha_0 = np.zeros(8)
        cfg = DriftCorrectorConfig(gamma_min=0.5, rho=0.0, epsilon=0.05, delta_star=1.0)
        corrector = ContractiveDriftCorrector(cfg, alpha_0, _perfect_verifier(alpha_0))
        for n in [1, 5, 10, 20]:
            bound = corrector.theoretical_bound_n(n)
            assert bound >= 0


class TestContractiveAggregation:
    """Verify that empirical drift stays within the Theorem 1 envelope."""

    def _run_chain(self, n_steps, gamma_min, rho, epsilon, seed=0):
        rng = np.random.default_rng(seed)
        d = 16
        alpha_0 = rng.standard_normal(d)
        cfg = DriftCorrectorConfig(
            gamma_min=gamma_min, rho=rho, epsilon=epsilon, delta_star=10.0
        )
        corrector = ContractiveDriftCorrector(cfg, alpha_0, _noisy_verifier(alpha_0, rho))

        current = alpha_0.copy()
        max_delta = 0.0
        for i in range(n_steps):
            noise = rng.standard_normal(d)
            noise = noise / np.linalg.norm(noise) * epsilon
            agent_out = current + noise
            current, delta = corrector.step(agent_out, step_idx=i)
            max_delta = max(max_delta, delta)

        return max_delta, cfg.asymptotic_envelope()

    def test_drift_stays_within_envelope(self):
        for gamma_min in [0.3, 0.5, 0.7]:
            max_delta, envelope = self._run_chain(200, gamma_min, rho=0.03, epsilon=0.02)
            assert max_delta <= envelope * 2.5, (
                f"gamma={gamma_min}: max_delta={max_delta:.4f} > 2.5*envelope={envelope*2.5:.4f}"
            )

    def test_chain_length_independence(self):
        """Longer chains should not produce larger final drift under contraction."""
        _, env_short = self._run_chain(20, 0.5, rho=0.03, epsilon=0.02, seed=1)
        _, env_long = self._run_chain(200, 0.5, rho=0.03, epsilon=0.02, seed=1)
        assert abs(env_short - env_long) < 1e-10, "Envelope must be independent of n"

    def test_drift_alarm_fires(self):
        alpha_0 = np.zeros(4)
        cfg = DriftCorrectorConfig(gamma_min=0.05, rho=0.0, epsilon=0.5, delta_star=0.01)
        corrector = ContractiveDriftCorrector(cfg, alpha_0, _perfect_verifier(alpha_0))
        large_output = np.ones(4) * 5.0
        with pytest.raises(DriftAlarm):
            corrector.step(large_output, step_idx=0)
