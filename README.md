# TRACE-MAS

**Tri-vector Resilient Algorithm for Cooperative Embodied Multi-Agent Security**

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-26%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![Validation](https://img.shields.io/badge/Validation-24%2C300%20trials-blueviolet)](experiments/RESULTS.md)
[![Research](https://img.shields.io/badge/Research-Under%20Submission-orange)](https://github.com/sunilgentyala/TRACE-MAS)
[![Website](https://img.shields.io/badge/Website-Live-brightgreen)](https://sunilgentyala.github.io/TRACE-MAS/)

Open-source Python implementation of a security layer for multi-agent LLM pipelines. Addresses four structural attack vectors at the inter-agent surface: cascading alignment drift, identity spoofing, cross-agent prompt injection, and temporal adversarial manipulation.

---

## The Problem

```
Agent A₁ ──► [SEAM] ──► Agent A₂ ──► [SEAM] ──► Agent A₃ ──► Outcome
              │                        │
              ▼                        ▼
      Drift / Injection        Spoof / Poison / Temporal
```

Each agent boundary in a multi-agent LLM pipeline is an unverified handoff. A single compromised or drifted output propagates silently to every downstream agent. TRACE-MAS intercepts each handoff with three sequential checks before the next agent receives it.

---

## Install

```bash
git clone https://github.com/sunilgentyala/TRACE-MAS
cd TRACE-MAS
pip install -e ".[dev]"
```

---

## Quick Start

```python
from trace_mas import TraceMASRuntime, TraceMASConfig
from trace_mas.drift import DriftCorrectorConfig
from trace_mas.runtime import AgentSpec, PolicyCommitment

config = TraceMASConfig(
    drift=DriftCorrectorConfig(gamma_min=0.4, rho=0.04, epsilon=0.02),
)

runtime = TraceMASRuntime(config, agents, verifier_fn, alpha_0)

# Raises DriftAlarm, SpoofAlarm, InjectionAlarm, or TemporalAlarm on attack
trajectory = runtime.run(T=10)
```

Run the full demo:

```bash
python examples/pipeline_demo.py
```

```
BENIGN PIPELINE RUN (3 agents, 5 rounds)
  Max drift:        0.0629
  Theoretical bound: 0.0700   ← empirically confirmed
  STATUS: PASSED

ADVERSARIAL RUN: Prompt Injection Attack
  CAUGHT: InjectionAlarm — behavioral score 0.10 < threshold 0.40
  STATUS: BLOCKED
```

---

## Validation Results

24,300 trials across four experiment families, run against this reference
implementation. Full tables and methodology: **[experiments/RESULTS.md](experiments/RESULTS.md)**.

| Experiment | Headline result |
|---|---|
| Drift envelope (Theorem 1) | Empirical drift flat from n=5 to n=200 agents; 100% of 3,000 trials within the analytic bound, with 3.4x-5.3x headroom |
| Attestation gate (Theorem 2) | 100% spoofing detection, 0% false positives on benign traffic, 100% injection detection after a discovered scorer evasion was patched |
| Temporal detector (Theorem 3) | 0% empirical miss rate across 6,000 trials, including a near-threshold "stealth" attack budget |
| Per-round overhead | 423 &micro;s mean (reference-implementation logic only) |

Validation surfaced and fixed a real evasion vector: the behavioral scorer's
keyword matcher used exact token equality, letting `eval(malicious_payload)` slip past
a rule for `eval(`. Fixed by switching to substring matching — see
[RESULTS.md](experiments/RESULTS.md#2-attestation-gate-detection-performance-theorem-2)
and the regression test in `tests/test_attestation.py`.

Reproduce: `python experiments/validation_suite.py`

---

## Repository Structure

```
TRACE-MAS/
├── src/trace_mas/
│   ├── drift.py         # Phase 2: contractive aggregation, DriftAlarm
│   ├── attestation.py   # Phase 1: ZKP gate, SpoofAlarm, InjectionAlarm
│   ├── temporal.py      # Phase 3: KL-window monitor, TemporalAlarm
│   └── runtime.py       # Unified TraceMASRuntime
├── tests/               # 26 tests covering all three security phases
├── examples/
│   └── pipeline_demo.py # Benign + adversarial 3-agent demo
├── experiments/
│   ├── validation_suite.py  # 24,300-trial empirical validation campaign
│   ├── results.json         # Raw output
│   └── RESULTS.md            # Tables, discussion, threats to validity
├── docs/                # GitHub Pages website
├── CITATION.cff
└── pyproject.toml
```

---

## Tests

```bash
python -m pytest tests/ -v
# 26 passed
```

Covers: drift bound correctness, theorem envelope independence of chain length, spoofing rejection, injection detection (including the patched evasion), temporal alarm under distributional shift.

---

## How It Works

Each round of the runtime runs three sequential phases per agent:

```
Phase 1 — Attestation Gate
  Agent generates ZKP: π = Prove(key, message, policy_commit, history)
  Dynamic threshold: τ = τ₀ + κ·log(1 + risk_score)
  If Verify(π) fails or behavioral_score < τ → SpoofOrInjectAlarm

Phase 2 — Drift Correction
  Verifier quorum produces reference signal ξ
  α_next = (1-γ)·agent_output + γ·ξ    [contractive aggregation]
  If ‖α_next - α₀‖ > δ* → DriftAlarm

Phase 3 — Temporal Monitoring (per round)
  I = sup KL(M^s1 ‖ M^s2) over sliding window
  If I > φ(B) → TemporalAlarm; rollback to last certified checkpoint
```

---

## Author

**Sunil Gentyala**
Independent Research, MAS Security and Applied Cryptography
IEEE Member | sugentyala@ieee.org
[LinkedIn](https://www.linkedin.com/in/sunil-gentyala/) | [GitHub](https://github.com/sunilgentyala)

---

## License

MIT License. See [LICENSE](LICENSE).
