# TRACE-MAS Empirical Validation Results

Reproducible via `python experiments/validation_suite.py` (raw output:
`experiments/results.json`). Four experiment families, 24,300 total trials, validating
Theorems 1-3 against the reference implementation.

## 1. Drift-envelope validation (Theorem 1) — chain-length independence

Params: rho=0.05, epsilon=0.02, gamma_min=0.3 (lambda=0.7) &rarr; analytic envelope
delta&#8734; &le; **0.0967**. 500 Monte Carlo trials per chain length (3,000 total).

| n (chain length) | Empirical max drift (mean) | p95 | Analytic envelope | Naive unprotected bound (n&middot;epsilon) | % trials within envelope |
|---:|---:|---:|---:|---:|---:|
| 5   | 0.0183 | 0.0241 | 0.0967 | 0.10 | 100% |
| 10  | 0.0212 | 0.0263 | 0.0967 | 0.20 | 100% |
| 25  | 0.0237 | 0.0282 | 0.0967 | 0.50 | 100% |
| 50  | 0.0253 | 0.0297 | 0.0967 | 1.00 | 100% |
| 100 | 0.0266 | 0.0302 | 0.0967 | 2.00 | 100% |
| 200 | 0.0280 | 0.0315 | 0.0967 | 4.00 | 100% |

**Takeaway:** empirical drift stays flat across a 40x increase in chain length (n=5 to
n=200) while the naive unprotected bound grows linearly, reaching 41x the TRACE-MAS
envelope at n=200. 100% of 3,000 trials stayed within the analytic bound, with
3.4x-5.3x headroom, indicating Theorem 1 is a valid but conservative guarantee under
this noise model.

## 2. Attestation gate detection performance (Theorem 2)

Params: tau_0=0.4, kappa=0.1. 1,000 trials per traffic class (4,000 total).

| Traffic class | Detection / FP rate |
|---|---:|
| Benign (false positives) | 0.0% |
| Forged signature (spoofing) | 100.0% |
| Prompt injection — pre-fix scorer (exact token match) | 75.0% |
| Prompt injection — current scorer (substring match) | 100.0% |
| Replay (resend valid attestation) | Caught |

### Discovered evasion (now fixed)

Of four injection payload templates, the pre-fix `behavioral_score()` missed one:
`"eval(malicious_payload) system(cat /etc/passwd)"` scored 0.500 against a 0.4
threshold and passed, because forbidden-keyword matching used exact whitespace-token
equality — `eval(malicious_payload)` as a single token is not equal to the forbidden
entry `eval(`. Since token boundaries are attacker-controlled, this is a real evasion
vector, not a corner case.

**Fix** (see `src/trace_mas/attestation.py::behavioral_score`): forbidden markers are
now matched as substrings of the lowercased message (`kw in text`) instead of via
token-set intersection. Detection rose to 100% with no change in the benign
false-positive rate. Regression test: `tests/test_attestation.py::
TestPromptInjectionDetection::test_substring_adjacent_keyword_is_caught`.

## 3. Temporal detector miss probability vs. window size (Theorem 3)

Params: nu=0.02 (benign mean drift), sigma=0.05. 4 budgets x 5 window sizes = 20
configurations, 300 trials each (6,000 total), staging a constant per-round mean-shift
attack.

| Budget &phi;(B) | Window T<sub>w</sub> | Theorem 3 bound | Empirical miss rate |
|---|---:|---:|---:|
| 0.03 (stealth, 1.5x nu) | 5  | 0.905 | 0.0% |
| 0.03 (stealth) | 80 | 0.202 | 0.0% |
| 0.05 | 5  | 0.407 | 0.0% |
| 0.05 | 80 | 5.6e-7 | 0.0% |
| 0.10 | 5  | 1.7e-3 | 0.0% |
| 0.20 | 5  | 8.5e-15 | 0.0% |

(Full 20-row table in `experiments/results.json`.)

**Takeaway:** empirical miss rate is 0% across all 6,000 trials, including the
near-threshold "stealth" budget where Theorem 3's own worst-case bound concedes up to
90.5% miss probability at the smallest window. This confirms the bound *holds*
(empirical &le; theoretical, as required) — the KL-divergence functional is sensitive
to any distributional shift, not just the mean-shift adversary model Theorem 3
analyzes, so it outperforms the proven floor here. It does **not** confirm the bound is
*tight*: no configuration produced a non-zero miss rate, so bound tightness (does miss
probability actually shrink with T<sub>w</sub> the way Eq. 7 predicts, for an
implementation that can fail) remains open. An adversary model closer to Theorem 3's
assumptions is needed to observe that — flagged as future work.

## 4. Per-round wall-clock overhead

d=1,536 (GPT-family embedding dim), T<sub>w</sub>=50, 2,000 sequential rounds, full
three-phase check (attestation + drift + temporal) per round.

| Metric | Value |
|---|---:|
| Mean | 423 &micro;s |
| p50 | 404 &micro;s |
| p95 | 622 &micro;s |
| p99 | 823 &micro;s |
| Max | 1,536 &micro;s |

This is reference-implementation logic overhead (HMAC-based attestation stand-in,
vector arithmetic, histogram KL estimation) — it excludes real zk-SNARK proving. The
Groth16 (~22 ms generation, ~3 ms verification) and PLONK figures cited in the paper
are literature values for the cryptographic layer this stand-in represents, not a
benchmark of this codebase. Sub-millisecond logic overhead is negligible next to both
the tens-to-hundreds-of-milliseconds round-trip time of an actual inter-agent LLM call
and the ~22 ms Groth16 proof-generation cost it would sit alongside in a full
deployment.

## Threats to validity

1. HMAC-SHA256 stands in for the ZKP layer of Definition 3 — this validates protocol
   *logic* (gating, thresholds, drift correction, windowing), not zk-SNARK soundness
   itself.
2. Drift experiments use synthetic Gaussian perturbations bounded by rho/epsilon, not
   measured drift from real LLM embeddings in a live pipeline. The qualitative claim
   (chain-length independence) is a structural property of Eq. (1) for any noise model
   satisfying Definition 2's bounds; the *quantitative* headroom (3.4x-5.3x) is specific
   to this noise model.
3. Temporal experiments confirm bound validity but not tightness (see above).
4. All timing is single-machine; no measurement under network partitioning, concurrent
   multi-tenant load, or timing attacks against the verifier sidecar itself.

Closing gap (2) — replacing synthetic drift with logged embeddings from a real
coder-QA-deploy pipeline — is the highest-priority next step.
