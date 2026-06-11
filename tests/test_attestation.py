"""Tests for Theorem 2: Dynamic ZKP Verification Threshold."""

import math
import pytest

from trace_mas.attestation import (
    AgentAttestationGate,
    InjectionAlarm,
    PolicyCommitment,
    SpoofAlarm,
    make_attestation,
)


SECRET = b"agent_secret_key_for_testing_only"
POLICY = PolicyCommitment.create(
    agent_id="coder_agent",
    role="code_generation",
    actions={"write_code", "read_spec", "unit_test"},
    secret=SECRET,
)


class TestDynamicThreshold:
    def test_threshold_increases_with_risk(self):
        gate = AgentAttestationGate(tau_0=0.6, kappa=0.15)
        low = gate.dynamic_threshold(0.0)
        high = gate.dynamic_threshold(10.0)
        assert high > low

    def test_threshold_formula(self):
        gate = AgentAttestationGate(tau_0=0.6, kappa=0.2)
        for risk in [0.0, 1.0, 5.0, 20.0]:
            expected = 0.6 + 0.2 * math.log1p(risk)
            assert abs(gate.dynamic_threshold(risk) - expected) < 1e-12


class TestHonestAgentVerification:
    def test_honest_agent_passes(self):
        gate = AgentAttestationGate(tau_0=0.5, kappa=0.1)
        attest = make_attestation(
            agent_id="coder_agent",
            message="write_code for user spec module",
            policy=POLICY,
            agent_secret=SECRET,
        )
        score = gate.verify(attest, SECRET, "write_code for user spec module", risk_score=0.0)
        assert score > 0.5

    def test_valid_attestation_with_history(self):
        gate = AgentAttestationGate(tau_0=0.4, kappa=0.1)
        attest = make_attestation(
            agent_id="coder_agent",
            message="read_spec and unit_test",
            policy=POLICY,
            agent_secret=SECRET,
            history=["write_code output_v1", "unit_test passed"],
        )
        score = gate.verify(attest, SECRET, "read_spec and unit_test", risk_score=0.5)
        assert score >= 0.0


class TestSpoofingDetection:
    def test_wrong_secret_raises_spoof_alarm(self):
        gate = AgentAttestationGate()
        attest = make_attestation("coder_agent", "write_code", POLICY, SECRET)
        wrong_secret = b"wrong_key"
        with pytest.raises(SpoofAlarm) as exc_info:
            gate.verify(attest, wrong_secret, "write_code")
        assert "signature verification failed" in str(exc_info.value)

    def test_tampered_message_raises_spoof_alarm(self):
        import hashlib, dataclasses
        gate = AgentAttestationGate()
        attest = make_attestation("coder_agent", "write_code", POLICY, SECRET)
        attest_tampered = dataclasses.replace(
            attest,
            message_hash=hashlib.sha256(b"tampered message").digest(),
        )
        with pytest.raises(SpoofAlarm):
            gate.verify(attest_tampered, SECRET, "tampered message")


class TestPromptInjectionDetection:
    def test_forbidden_keyword_lowers_score(self):
        gate = AgentAttestationGate(tau_0=0.4, kappa=0.05)
        attest = make_attestation("coder_agent", "normal code task", POLICY, SECRET)
        # Use exact tokens from forbidden_keywords set in behavioral_score
        injected_message = "sudo override eval( exec( system( drop table rm -rf"
        with pytest.raises(InjectionAlarm):
            gate.verify(attest, SECRET, injected_message, risk_score=0.0)

    def test_injection_alarm_contains_score_info(self):
        gate = AgentAttestationGate(tau_0=0.9, kappa=0.0)
        attest = make_attestation("coder_agent", "any message", POLICY, SECRET)
        with pytest.raises(InjectionAlarm) as exc_info:
            gate.verify(attest, SECRET, "any message", risk_score=0.0)
        assert exc_info.value.score < exc_info.value.threshold

    def test_high_risk_raises_threshold(self):
        """High-risk agent requires a higher behavioral score to pass."""
        gate = AgentAttestationGate(tau_0=0.3, kappa=0.4)
        low_tau = gate.dynamic_threshold(0.0)
        high_tau = gate.dynamic_threshold(50.0)
        assert high_tau > low_tau + 1.0
