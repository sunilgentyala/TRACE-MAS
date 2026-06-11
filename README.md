# TRACE-MAS

**Tri-vector Resilient Algorithm for Cooperative Embodied Multi-Agent Security**

[![IEEE](https://img.shields.io/badge/Venue-IEEE%20Conference-blue)](https://github.com/sunilgentyala/TRACE-MAS)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/Paper-TRACE--MAS-orange)](https://github.com/sunilgentyala/TRACE-MAS/blob/main/paper/MAS_Complexity_Gaps_TRACE_v3.tex)
[![GitHub Pages](https://img.shields.io/badge/Website-Live-brightgreen)](https://sunilgentyala.github.io/TRACE-MAS/)

---

## The Problem

Modern AI systems rarely rely on a single model. A coder agent drafts a module, a quality-assurance agent reviews it, and a deployment agent ships it. Three distinct LLMs. One shared pipeline. Zero cryptographic cross-verification between them.

When that pipeline is attacked, the consequences do not stay isolated. They cascade.

This repository accompanies the research paper **"TRACE-MAS: Securing the Inter-Agent Surface Against Cascading Drift, Identity Spoofing, Prompt Injection, and Temporal Adversarial Attacks in Multi-Agent LLM Pipelines"** by Sunil Gentyala.

---

## What TRACE-MAS Addresses

The inter-agent surface — the seam between cooperating agents — is the least defended layer of the modern agentic stack. TRACE-MAS identifies and closes four structural failure modes that arise at that surface:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THE INTER-AGENT SURFACE                         │
│                                                                     │
│   Agent A₁ ──► [SEAM] ──► Agent A₂ ──► [SEAM] ──► Agent A₃        │
│                  │                        │                         │
│                  ▼                        ▼                         │
│          Cascading Drift          Identity Spoofing                 │
│          Prompt Injection         Temporal Attacks                  │
└─────────────────────────────────────────────────────────────────────┘
```

| Attack Vector | Description | TRACE-MAS Defense |
|---|---|---|
| **Cascading Alignment Drift** | Small errors in early agents compound silently into large semantic deviations downstream | Verifier-mediated contractive aggregation (Theorem 1) |
| **Inter-Agent Identity Spoofing** | Rogue agents admitted to open swarms via forged capability descriptors | Dynamic ZKP verification threshold (Theorem 2) |
| **Cross-Agent Prompt Injection** | Adversarial directives embedded in processed content hijack downstream agents | Policy-commitment attestation blocks messages inconsistent with declared policy |
| **Temporal Adversarial Attacks** | Patient, staged perturbations below any single agent's detection horizon drive the system toward a malicious end-state | Sliding-window KL-divergence functional (Theorem 3) |

---

## Framework at a Glance

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRACE-MAS RUNTIME                             │
│                                                                      │
│  Per-Round Loop:                                                     │
│                                                                      │
│  1. ATTESTATION GATE                                                 │
│     Agent Aᵢ generates ZKP π(sk, message, policy, history)          │
│     Dynamic threshold τ = τ₀ + κ·log(1 + risk_score)               │
│     → Rejects: spoofed identity, injected messages                  │
│                                                                      │
│  2. DRIFT CORRECTION                                                 │
│     Verifier quorum 𝒱 produces reference signal ξ                   │
│     α_{i+1} = (1-γ)·â_{i+1} + γ·ξ      [contractive aggregation]   │
│     → Guarantees: δ_∞ ≤ ρ + λε/(1-λ)  [chain-length-independent]   │
│                                                                      │
│  3. TEMPORAL MONITORING                                              │
│     Sliding window: I(t, T_w) = sup KL(M^s₁ ∥ M^s₂)               │
│     Trigger quarantine if I(t, T_w*) > φ(B)                        │
│     → Miss probability: exp(-T_w·(φ(B)-ν)²/2σ²)                   │
└──────────────────────────────────────────────────────────────────────┘
```

### The Three Guarantees

**Theorem 1 (Drift-Decay):** Under verifier-mediated aggregation with contraction factor λ ∈ (0,1):
```
δ_∞  ≤  ρ  +  λε / (1 - λ)
```
The asymptotic drift is **independent of chain length n**. Longer pipelines are not inherently riskier.

**Theorem 2 (Spoofing-Acceptance Bound):** For a ZKP scheme with soundness error 2⁻ˢ:
```
Pr[rogue agent accepted]  ≤  2⁻ˢ  +  e^{-τ/μ}
```
Over N rounds: union-bounded probability ≤ N(2⁻ˢ + ε).

**Theorem 3 (Temporal Detection):** Under sub-Gaussian benign drift:
```
Pr[undetected temporal attack]  ≤  exp( -T_w · (φ(B) - ν)² / 2σ² )
```

**Composite Failure Bound:**
```
P_fail  ≤  1[δ_∞ > δ*]  +  T(2⁻ˢ + ε)  +  T·exp(−T_w*(φ(B)−ν)²/2σ²)
```
Each term is independently controllable.

---

## Repository Structure

```
TRACE-MAS/
├── paper/
│   ├── MAS_Complexity_Gaps_TRACE_v3.tex   ← Full LaTeX source (v3)
│   ├── IEEEtran.cls                        ← IEEE conference format
│   ├── algorithm.sty                       ← Algorithm package
│   └── algorithmic.sty                     ← Algorithmic package
├── docs/
│   └── index.html                          ← GitHub Pages website
├── README.md
├── CITATION.cff
└── LICENSE
```

---

## Compiling the Paper

Requirements: `pdflatex` (TeX Live or MiKTeX)

```bash
cd paper
pdflatex MAS_Complexity_Gaps_TRACE_v3.tex
pdflatex MAS_Complexity_Gaps_TRACE_v3.tex   # second pass for citations
```

---

## Algorithm Pseudocode

The unified runtime (Algorithm 1 in the paper) processes each round as follows:

```
TRACE-MAS(agents A, verifiers V, intent α₀, ZKP scheme, params):
  For each round t = 1..T:
    For each agent Aᵢ acting at t:
      1. Aᵢ produces output â and message m
      2. Aᵢ generates proof π = Prove(sk, m, policy_commit, history)
      3. Compute risk score ρ; set threshold τ = τ₀ + κ·log(1+ρ)
      4. IF Verify(π) fails OR behavioral_score(π) < τ:
            → HALT with SpoofOrInjectAlarm
      5. Verifier quorum V produces reference ξ
      6. Apply contractive aggregation: α_{i+1} = (1-γ)·â + γ·ξ
      7. IF drift δ_{i+1} > δ*: → HALT with DriftAlarm
      8. Append α_{i+1} to joint memory M
    Compute temporal functional I(t, T_w*)
    IF I(t, T_w*) > φ(B): → HALT with TemporalAlarm, roll back
  RETURN certified trajectory (α₁,...,αₙ)
```

**Complexity per round:** O(log|A|) ZKP verification + O(d) aggregation + O(T_w*) window functional.

---

## Key References

| # | Reference | Role in TRACE-MAS |
|---|---|---|
| [1] | Li et al., JAS 2025 (Embodied MAS review) | Motivation for cross-agent verification |
| [2] | Ma & Li, TIFS 2025 (Grey-box attacks on MARL) | Attack surface evidence |
| [3] | Lv et al., TIFS 2024 (Safe MARL vs adversarial comms) | Baseline defense limits |
| [4] | Liu et al., IEEE SP 2025 (DataSentinel) | Prompt injection detection baseline |
| [5] | Lee & Tiwari, arXiv 2024 (Prompt Infection) | Cross-agent injection formalization |
| [6] | Chen et al., NeurIPS 2024 (AgentPoison) | Memory poisoning attack model |
| [7] | Pathak et al., IEEE Access 2024 (ZKP for IoT) | ZKP authentication backbone |
| [8] | Li et al., IEEE TC 2024 (PBFT for IoT) | Byzantine quorum primitive |
| [9] | Zhang et al., TDSC 2024 (PrivacyAsst) | Tool-use privacy in LLM agents |
| [10] | Louck et al., ACM TAISP 2025 | Agent protocol security analysis |

Full bibliography in the LaTeX source.

---

## Citation

If you build on this work, please cite:

```bibtex
@inproceedings{gentyala2026tracemas,
  author    = {Sunil Gentyala},
  title     = {{TRACE-MAS}: Securing the Inter-Agent Surface Against Cascading
               Drift, Identity Spoofing, Prompt Injection, and Temporal
               Adversarial Attacks in Multi-Agent {LLM} Pipelines},
  booktitle = {Proceedings of an IEEE Conference},
  year      = {2026},
  note      = {Available: https://github.com/sunilgentyala/TRACE-MAS}
}
```

---

## Author

**Sunil Gentyala**  
Independent Research, MAS Security and Applied Cryptography  
IEEE Member | sugentyala@ieee.org  
[LinkedIn](https://www.linkedin.com/in/sunil-gentyala/) | [GitHub](https://github.com/sunilgentyala)

---

## License

This repository is released under the MIT License. The paper content is the intellectual property of the author. Please cite appropriately if you use this work.
