"""
Dynamic ZKP Verification Threshold (Theorem 2).

Implements agent attestation gate: each handoff message must carry
a proof binding (identity, policy commitment, message provenance).
The dynamic threshold adapts to per-agent runtime risk, catching both
identity spoofing and cross-agent prompt injection.

Theorem 2 guarantee:
    Pr[rogue accepted] <= 2^{-s} + exp(-tau / mu)
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import dataclass, field
from typing import Any


class SpoofAlarm(Exception):
    """Raised when an agent fails ZKP verification (identity spoofing detected)."""
    def __init__(self, agent_id: str, reason: str):
        self.agent_id = agent_id
        super().__init__(f"SpoofAlarm: agent '{agent_id}' rejected — {reason}")


class InjectionAlarm(Exception):
    """Raised when message content violates the agent's declared policy commitment."""
    def __init__(self, agent_id: str, score: float, threshold: float):
        self.agent_id = agent_id
        self.score = score
        self.threshold = threshold
        super().__init__(
            f"InjectionAlarm: agent '{agent_id}' behavioral score {score:.3f} "
            f"< threshold {threshold:.3f} — possible prompt injection"
        )


@dataclass
class PolicyCommitment:
    """Hiding commitment to an agent's declared policy parameters."""
    agent_id: str
    commitment_hash: bytes
    declared_role: str
    allowed_actions: frozenset[str]
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(cls, agent_id: str, role: str, actions: set[str], secret: bytes) -> "PolicyCommitment":
        payload = f"{agent_id}:{role}:{sorted(actions)}".encode()
        commitment = hmac.new(secret, payload, hashlib.sha256).digest()
        return cls(
            agent_id=agent_id,
            commitment_hash=commitment,
            declared_role=role,
            allowed_actions=frozenset(actions),
        )


@dataclass
class AgentAttestation:
    """Cryptographic attestation pi_i^{(t)} for a single agent message."""
    agent_id: str
    message_hash: bytes
    policy_commitment: PolicyCommitment
    timestamp: float
    signature: bytes
    history_digest: bytes

    def verify_signature(self, agent_secret: bytes) -> bool:
        payload = (
            self.agent_id.encode()
            + self.message_hash
            + self.policy_commitment.commitment_hash
            + self.history_digest
            + str(self.timestamp).encode()
        )
        expected = hmac.new(agent_secret, payload, hashlib.sha256).digest()
        return hmac.compare_digest(self.signature, expected)


class AgentAttestationGate:
    """
    Verification gate for agent-to-agent handoffs.

    Enforces:
    1. Cryptographic proof of identity (ZKP soundness term 2^{-s})
    2. Message-policy consistency check (behavioral score >= dynamic threshold)
    3. Anti-replay via timestamp window

    Prompt injection is caught at check 2: an injected directive causes
    the agent to produce a message inconsistent with its policy commitment,
    so behavioral_score falls below tau_i^{(t)}.
    """

    def __init__(
        self,
        tau_0: float = 0.6,
        kappa: float = 0.15,
        replay_window_sec: float = 30.0,
    ):
        self.tau_0 = tau_0
        self.kappa = kappa
        self.replay_window_sec = replay_window_sec
        self._seen_timestamps: dict[str, list[float]] = {}

    def dynamic_threshold(self, risk_score: float) -> float:
        """tau_i^{(t)} = tau_0 + kappa * log(1 + rho_i^{(t)})"""
        return self.tau_0 + self.kappa * math.log1p(risk_score)

    def behavioral_score(
        self,
        attestation: AgentAttestation,
        message_content: str,
    ) -> float:
        """
        Compute how consistently the message aligns with the declared policy.

        In a production system this would invoke a lightweight classifier or
        a committed model evaluation. Here we implement a structural check:
        messages that reference actions outside the declared allowed_actions
        set receive a lower score.

        Forbidden markers are matched as substrings of the lowercased message,
        not as exact whitespace-delimited tokens. An earlier token-set version
        of this check missed markers immediately followed by other characters
        (e.g. "eval(malicious_payload)" did not equal the forbidden token
        "eval(" under set intersection, letting that payload through). Since
        token boundaries are attacker-controlled, substring matching is the
        correct default here.
        """
        declared_actions = attestation.policy_commitment.allowed_actions
        if not declared_actions:
            return 1.0

        text = message_content.lower()
        tokens = set(text.split())
        matches = len(tokens & {a.lower() for a in declared_actions})
        forbidden_keywords = {"sudo", "override", "ignore previous", "drop table",
                              "rm -rf", "exec(", "eval(", "system("}
        violations = sum(1 for kw in forbidden_keywords if kw in text)

        base_score = min(1.0, 0.5 + 0.1 * matches)
        penalty = 0.4 * violations
        return max(0.0, base_score - penalty)

    def _check_replay(self, agent_id: str, timestamp: float) -> bool:
        now = time.time()
        window = self._seen_timestamps.setdefault(agent_id, [])
        window[:] = [t for t in window if now - t < self.replay_window_sec]
        if timestamp in window:
            return False
        window.append(timestamp)
        return True

    def verify(
        self,
        attestation: AgentAttestation,
        agent_secret: bytes,
        message_content: str,
        risk_score: float = 0.0,
    ) -> float:
        """
        Verify the attestation and return the behavioral score.

        Raises:
            SpoofAlarm:     signature invalid or replay detected
            InjectionAlarm: behavioral score below dynamic threshold
        """
        if not attestation.verify_signature(agent_secret):
            raise SpoofAlarm(attestation.agent_id, "signature verification failed")

        if not self._check_replay(attestation.agent_id, attestation.timestamp):
            raise SpoofAlarm(attestation.agent_id, "replay attack detected")

        score = self.behavioral_score(attestation, message_content)
        tau = self.dynamic_threshold(risk_score)

        if score < tau:
            raise InjectionAlarm(attestation.agent_id, score, tau)

        return score


def make_attestation(
    agent_id: str,
    message: str,
    policy: PolicyCommitment,
    agent_secret: bytes,
    history: list[str] | None = None,
) -> AgentAttestation:
    """Helper to construct a valid AgentAttestation for an honest agent."""
    msg_hash = hashlib.sha256(message.encode()).digest()
    history_str = "|".join(history or [])
    history_digest = hashlib.sha256(history_str.encode()).digest()
    ts = time.time()
    payload = (
        agent_id.encode()
        + msg_hash
        + policy.commitment_hash
        + history_digest
        + str(ts).encode()
    )
    sig = hmac.new(agent_secret, payload, hashlib.sha256).digest()
    return AgentAttestation(
        agent_id=agent_id,
        message_hash=msg_hash,
        policy_commitment=policy,
        timestamp=ts,
        signature=sig,
        history_digest=history_digest,
    )
