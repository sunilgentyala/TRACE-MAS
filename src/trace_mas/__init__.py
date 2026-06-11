"""
TRACE-MAS: Tri-vector Resilient Algorithm for Cooperative Embodied
Multi-Agent Security.

Provides provable security guarantees for multi-agent LLM pipelines
against cascading alignment drift, identity spoofing, cross-agent
prompt injection, and temporal adversarial attacks.

Paper: Gentyala, S. (2026). TRACE-MAS: Securing the Inter-Agent
Surface in Multi-Agent LLM Pipelines.
GitHub: https://github.com/sunilgentyala/TRACE-MAS
"""

from .drift import ContractiveDriftCorrector, DriftAlarm
from .attestation import AgentAttestationGate, SpoofAlarm, InjectionAlarm
from .temporal import TemporalMemoryMonitor, TemporalAlarm
from .runtime import TraceMASRuntime, TraceMASConfig, CertifiedTrajectory

__version__ = "1.0.0"
__author__ = "Sunil Gentyala"
__email__ = "sugentyala@ieee.org"

__all__ = [
    "TraceMASRuntime",
    "TraceMASConfig",
    "CertifiedTrajectory",
    "ContractiveDriftCorrector",
    "AgentAttestationGate",
    "TemporalMemoryMonitor",
    "DriftAlarm",
    "SpoofAlarm",
    "InjectionAlarm",
    "TemporalAlarm",
]
