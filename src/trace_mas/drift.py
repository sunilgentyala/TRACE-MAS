"""
Drift-Decay Bounding Function (Theorem 1).

Implements verifier-mediated contractive aggregation that bounds
cascading alignment drift independently of pipeline length n.

Theorem 1 guarantee:
    delta_inf <= rho + lambda * epsilon / (1 - lambda)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


class DriftAlarm(Exception):
    """Raised when accumulated drift exceeds the configured budget delta_star."""
    def __init__(self, step: int, drift: float, budget: float):
        self.step = step
        self.drift = drift
        self.budget = budget
        super().__init__(
            f"DriftAlarm at step {step}: drift={drift:.4f} > delta_star={budget:.4f}"
        )


@dataclass
class DriftCorrectorConfig:
    gamma_min: float = 0.3       # minimum verifier weight; lambda = 1 - gamma_min
    delta_star: float = 0.15     # maximum allowed asymptotic drift
    rho: float = 0.05            # verifier residual bound: ||xi - alpha_0||
    epsilon: float = 0.02        # per-agent noise bound

    def __post_init__(self):
        if not 0 < self.gamma_min < 1:
            raise ValueError("gamma_min must be in (0, 1)")
        lam = 1.0 - self.gamma_min
        theoretical_max = self.rho + lam * self.epsilon / (1.0 - lam)
        if self.delta_star < theoretical_max:
            import warnings
            warnings.warn(
                f"delta_star={self.delta_star} is below the theoretical envelope "
                f"{theoretical_max:.4f}. Some benign trajectories will trigger DriftAlarm."
            )

    @property
    def lambda_(self) -> float:
        return 1.0 - self.gamma_min

    def asymptotic_envelope(self) -> float:
        """Returns the chain-length-independent drift bound from Theorem 1."""
        return self.rho + self.lambda_ * self.epsilon / (1.0 - self.lambda_)


class ContractiveDriftCorrector:
    """
    Verifier-mediated contractive aggregation operator (Definition 2).

    For each handoff i, blends the agent output with a verifier reference
    signal to produce a corrected representation that provably contracts
    toward the user intent alpha_0.

    Usage:
        corrector = ContractiveDriftCorrector(config, alpha_0, verifier_fn)
        alpha_next, delta = corrector.step(agent_output, risk_score)
    """

    def __init__(
        self,
        config: DriftCorrectorConfig,
        alpha_0: np.ndarray,
        verifier_fn: Callable[[np.ndarray], np.ndarray],
    ):
        self.config = config
        self.alpha_0 = alpha_0.copy()
        self.verifier_fn = verifier_fn
        self.alpha_current = alpha_0.copy()
        self._history: list[float] = []

    def _compute_gamma(self, risk_score: float) -> float:
        """Adaptive verifier weight: higher risk -> more verifier correction."""
        return max(self.config.gamma_min, min(0.95, self.config.gamma_min + 0.2 * risk_score))

    def step(
        self,
        agent_output: np.ndarray,
        risk_score: float = 0.0,
        step_idx: int = 0,
    ) -> tuple[np.ndarray, float]:
        """
        Apply one contractive aggregation step.

        Returns:
            alpha_next: corrected representation
            delta: drift from alpha_0

        Raises:
            DriftAlarm: if drift exceeds config.delta_star
        """
        xi = self.verifier_fn(agent_output)
        gamma = self._compute_gamma(risk_score)
        alpha_next = (1.0 - gamma) * agent_output + gamma * xi

        delta = float(np.linalg.norm(alpha_next - self.alpha_0))
        self._history.append(delta)
        self.alpha_current = alpha_next

        if delta > self.config.delta_star:
            raise DriftAlarm(step=step_idx, drift=delta, budget=self.config.delta_star)

        return alpha_next, delta

    def drift_history(self) -> list[float]:
        return list(self._history)

    def theoretical_bound_n(self, n: int) -> float:
        """Upper bound on drift after n steps (Eq. 3 in the paper)."""
        lam = self.config.lambda_
        delta_0 = float(np.linalg.norm(self.alpha_0))
        envelope = (lam * self.config.epsilon + (1.0 - lam) * self.config.rho) / (1.0 - lam)
        return lam**n * delta_0 + envelope * (1.0 - lam**n)
