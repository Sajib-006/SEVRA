from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .actions import Action, active_verification_prompt
from .gates import RecoverabilityGate
from .schema import Attempt, VerificationOutput

Verifier = Callable[[str, Attempt], VerificationOutput | str]


@dataclass(frozen=True)
class Decision:
    action: Action
    recoverability: float
    threshold: float

    @property
    def should_verify(self) -> bool:
        return self.action is Action.ACTIVE_VERIFY


@dataclass(frozen=True)
class Result:
    final_response: str
    decision: Decision
    verification_tokens: int = 0
    verification_metadata: Mapping[str, Any] = field(default_factory=dict)


class SEVRAController:
    """Route completed attempts to accept or candidate-specific verification."""

    def __init__(self, gate: RecoverabilityGate, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.gate = gate
        self.threshold = float(threshold)

    def decide(self, attempt: Attempt) -> Decision:
        score = float(self.gate.score(attempt))
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"gate score must be in [0, 1], got {score}")
        action = Action.ACTIVE_VERIFY if score >= self.threshold else Action.ACCEPT
        return Decision(action=action, recoverability=score, threshold=self.threshold)

    def run(self, attempt: Attempt, verifier: Verifier) -> Result:
        decision = self.decide(attempt)
        if not decision.should_verify:
            return Result(final_response=attempt.base_response, decision=decision)

        prompt = active_verification_prompt(attempt.query, attempt.base_response)
        output = verifier(prompt, attempt)
        if isinstance(output, str):
            output = VerificationOutput(response=output)
        return Result(
            final_response=output.response,
            decision=decision,
            verification_tokens=output.actual_tokens,
            verification_metadata=output.metadata,
        )
