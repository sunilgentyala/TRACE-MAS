"""
TRACE-MAS Unified Runtime (Algorithm 1).

Orchestrates the three-phase per-round security loop:
  Phase 1: Attestation gate (ZKP + behavioral score)
  Phase 2: Contractive drift correction
  Phase 3: Temporal window monitoring

Reference: Gentyala, S. (2026). TRACE-MAS.
GitHub: https://github.com/sunilgentyala/TRACE-MAS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

import numpy as np

from .attestation import (
    AgentAttestationGate,
    AgentAttestation,
    InjectionAlarm,
    PolicyCommitment,
    SpoofAlarm,
)
from .drift import ContractiveDriftCorrector, DriftAlarm, DriftCorrectorConfig
from .temporal import TemporalMemoryMonitor, TemporalAlarm, TemporalMonitorConfig


class CertifiedTrajectory(NamedTuple):
    """Outputs from a successful TRACE-MAS workflow run."""
    alphas: list[np.ndarray]   # certified representations per step
    deltas: list[float]        # drift values per step
    rounds: int                # number of rounds completed


@dataclass
class TraceMASConfig:
    drift: DriftCorrectorConfig = field(default_factory=DriftCorrectorConfig)
    temporal: TemporalMonitorConfig = field(default_factory=TemporalMonitorConfig)
    attestation_tau_0: float = 0.6
    attestation_kappa: float = 0.15
    replay_window_sec: float = 30.0


@dataclass
class AgentSpec:
    """Specification for one agent in the pipeline."""
    agent_id: str
    secret: bytes
    policy: PolicyCommitment
    run_fn: Callable[[np.ndarray], tuple[np.ndarray, str]]  # returns (output, message)
    risk_score_fn: Callable[[], float] | None = None


class TraceMASRuntime:
    """
    TRACE-MAS unified runtime.

    Instantiates and coordinates the attestation gate, drift corrector,
    and temporal monitor for a fixed pipeline of agents.

    Usage:
        config = TraceMASConfig()
        runtime = TraceMASRuntime(config, agents, verifier_fn, alpha_0)
        trajectory = runtime.run(T=10)
    """

    def __init__(
        self,
        config: TraceMASConfig,
        agents: list[AgentSpec],
        verifier_fn: Callable[[np.ndarray], np.ndarray],
        alpha_0: np.ndarray,
    ):
        self.config = config
        self.agents = agents
        self.alpha_0 = alpha_0.copy()

        self._gate = AgentAttestationGate(
            tau_0=config.attestation_tau_0,
            kappa=config.attestation_kappa,
            replay_window_sec=config.replay_window_sec,
        )
        self._corrector = ContractiveDriftCorrector(
            config=config.drift,
            alpha_0=alpha_0,
            verifier_fn=verifier_fn,
        )
        self._monitor = TemporalMemoryMonitor(config=config.temporal)

        self._memory: list[np.ndarray] = []
        self._certified_alphas: list[np.ndarray] = []
        self._drift_history: list[float] = []

    def run(self, T: int = 1) -> CertifiedTrajectory:
        """
        Execute T rounds of the TRACE-MAS workflow.

        Returns CertifiedTrajectory on success.
        Raises DriftAlarm, SpoofAlarm, InjectionAlarm, or TemporalAlarm on failure.
        """
        alpha_current = self.alpha_0.copy()

        for t in range(1, T + 1):
            for idx, spec in enumerate(self.agents):

                # Phase 1: Attestation gate
                agent_output, message = spec.run_fn(alpha_current)

                attestation = _build_dummy_attestation_for_honest_agent(
                    spec, message, [str(a) for a in self._memory[-10:]]
                )
                risk_score = spec.risk_score_fn() if spec.risk_score_fn else 0.0

                self._gate.verify(
                    attestation=attestation,
                    agent_secret=spec.secret,
                    message_content=message,
                    risk_score=risk_score,
                )

                # Phase 2: Drift correction
                alpha_next, delta = self._corrector.step(
                    agent_output=agent_output,
                    risk_score=risk_score,
                    step_idx=t * len(self.agents) + idx,
                )
                alpha_current = alpha_next
                self._drift_history.append(delta)
                self._memory.append(alpha_next)
                self._certified_alphas.append(alpha_next)

            # Phase 3: Temporal monitoring
            flat_memory = self._memory_as_flat_vector()
            self._monitor.add_snapshot(flat_memory, t)

        return CertifiedTrajectory(
            alphas=list(self._certified_alphas),
            deltas=list(self._drift_history),
            rounds=T,
        )

    def _memory_as_flat_vector(self) -> list[float]:
        window = self._memory[-self.config.temporal.window_size:]
        if not window:
            return [0.0]
        return [float(v) for arr in window for v in arr.flatten()]


def _build_dummy_attestation_for_honest_agent(
    spec: AgentSpec,
    message: str,
    history: list[str],
) -> AgentAttestation:
    """Build a valid attestation for an honest agent (production: use ZKP prover)."""
    from .attestation import make_attestation
    return make_attestation(
        agent_id=spec.agent_id,
        message=message,
        policy=spec.policy,
        agent_secret=spec.secret,
        history=history,
    )
