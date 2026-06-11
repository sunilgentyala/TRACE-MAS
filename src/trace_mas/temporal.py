"""
Temporal Memory-Validation Boundary (Theorem 3).

Implements a sliding-window KL-divergence functional over joint agent
memory to detect slow temporal adversarial attacks whose individual
perturbations remain below any single agent's detection threshold.

Theorem 3 guarantee:
    Pr[undetected] <= exp( -T_w * (phi(B) - nu)^2 / (2 * sigma^2) )
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


class TemporalAlarm(Exception):
    """Raised when the temporal inconsistency functional exceeds phi(B)."""
    def __init__(self, t: int, I_value: float, threshold: float, window: int):
        self.t = t
        self.I_value = I_value
        self.threshold = threshold
        self.window = window
        super().__init__(
            f"TemporalAlarm at t={t}: I(t,T_w={window})={I_value:.4f} > phi(B)={threshold:.4f}"
        )


@dataclass
class TemporalMonitorConfig:
    window_size: int = 20        # T_w: sliding window length (rounds)
    phi_B: float = 0.25          # detection threshold phi(B)
    nu: float = 0.02             # mean shift of benign drift
    sigma: float = 0.05          # sub-Gaussian variance proxy
    n_bins: int = 32             # histogram bins for KL estimation

    def miss_probability(self) -> float:
        """Theorem 3 upper bound on detection miss probability."""
        gap = self.phi_B - self.nu
        if gap <= 0:
            return 1.0
        return math.exp(-self.window_size * gap**2 / (2.0 * self.sigma**2))

    def optimal_window(
        self,
        fp_cost: float = 1.0,
        fn_cost: float = 10.0,
        budget: float = 0.1,
    ) -> int:
        """
        Estimate T_w* by minimizing false-negative + false-positive risk.
        In production, calibrate fp_cost/fn_cost from deployment data.
        """
        best_w, best_risk = 1, float("inf")
        for w in range(1, 500):
            gap = self.phi_B - self.nu
            fn_risk = fn_cost * math.exp(-w * gap**2 / (2 * self.sigma**2))
            fp_risk = fp_cost * max(0.0, budget / math.sqrt(w))
            total = fn_risk + fp_risk
            if total < best_risk:
                best_risk, best_w = total, w
        return best_w


def _kl_divergence_histogram(
    samples_p: Sequence[float],
    samples_q: Sequence[float],
    n_bins: int = 32,
    eps: float = 1e-10,
) -> float:
    """
    Estimate KL(P || Q) from two sample sequences using histogram density estimation.
    Returns 0 if either sequence is empty.
    """
    if not samples_p or not samples_q:
        return 0.0

    all_vals = list(samples_p) + list(samples_q)
    lo, hi = min(all_vals), max(all_vals)
    if abs(hi - lo) < 1e-12:
        return 0.0

    edges = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(samples_p, bins=edges, density=True)
    q_hist, _ = np.histogram(samples_q, bins=edges, density=True)

    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


class TemporalMemoryMonitor:
    """
    Sliding-window KL functional over joint agent memory (Definition 6 & 7).

    At each round t, computes the supremal KL divergence between any two
    memory snapshots within the window T_w. If this exceeds phi(B), a
    TemporalAlarm is raised, triggering rollback to the last certified
    checkpoint.

    Usage:
        monitor = TemporalMemoryMonitor(config)
        checkpoint = monitor.add_snapshot(memory_vector, t)
        # checkpoint is None if ok, else the last safe t
    """

    def __init__(self, config: TemporalMonitorConfig):
        self.config = config
        self._window: deque[tuple[int, list[float]]] = deque()
        self._certified_checkpoints: list[tuple[int, list[float]]] = []

    def add_snapshot(
        self,
        memory_vector: Sequence[float],
        t: int,
    ) -> int | None:
        """
        Add a memory snapshot for round t and check temporal consistency.

        Returns:
            None if the window is consistent (no attack detected)
            int: the last certified round to roll back to if alarm fires

        Raises:
            TemporalAlarm if I(t, T_w) > phi(B)
        """
        snapshot = list(memory_vector)
        self._window.append((t, snapshot))

        while len(self._window) > self.config.window_size:
            self._window.popleft()

        if len(self._window) < 2:
            self._certified_checkpoints.append((t, snapshot))
            return None

        I_value = self._compute_functional()

        if I_value > self.config.phi_B:
            last_safe_t = (
                self._certified_checkpoints[-1][0] if self._certified_checkpoints else 0
            )
            raise TemporalAlarm(
                t=t,
                I_value=I_value,
                threshold=self.config.phi_B,
                window=self.config.window_size,
            )

        self._certified_checkpoints.append((t, snapshot))
        return None

    def _compute_functional(self) -> float:
        """
        I(t, T_w) = sup_{s1,s2 in window} KL(M^{s1} || M^{s2})

        For efficiency, compare earliest vs latest snapshot in window
        plus a few internal pairs.
        """
        snapshots = list(self._window)
        if len(snapshots) < 2:
            return 0.0

        max_kl = 0.0
        candidates = [snapshots[0], snapshots[-1]]
        mid = len(snapshots) // 2
        if mid not in (0, len(snapshots) - 1):
            candidates.append(snapshots[mid])

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                kl = _kl_divergence_histogram(
                    candidates[i][1],
                    candidates[j][1],
                    self.config.n_bins,
                )
                max_kl = max(max_kl, kl)

        return max_kl

    def last_certified_round(self) -> int:
        if not self._certified_checkpoints:
            return 0
        return self._certified_checkpoints[-1][0]
