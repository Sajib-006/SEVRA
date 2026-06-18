from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .schema import Attempt


FEATURE_NAMES = (
    "difficulty",
    "verification_need",
    "constraint_density",
    "ambiguity_score",
    "retrieval_need",
    "base_finalizer_used",
    "base_done_reason_length",
    "log_base_actual_tokens",
)


class RecoverabilityGate(Protocol):
    """Scores the expected usefulness of active verification in [0, 1]."""

    def score(self, attempt: Attempt) -> float: ...


def observable_features(attempt: Attempt) -> tuple[float, ...]:
    """Return the cheap serving-layer feature vector used by the baseline gate."""

    return (
        attempt.difficulty,
        attempt.verification_need,
        attempt.constraint_density,
        attempt.ambiguity_score,
        attempt.retrieval_need,
        float(attempt.base_finalizer_used),
        float(attempt.base_done_reason == "length"),
        math.log1p(attempt.base_actual_tokens),
    )


def format_gate_input(attempt: Attempt) -> str:
    """Format an attempt for the learned sequence-classification gate."""

    return (
        "Predict whether active verification will correct this attempted solution. "
        "Use only observable information; the gold answer is unavailable.\n\n"
        f"Task type: {attempt.task_type}\n"
        f"Estimated difficulty: {attempt.difficulty:.3f}\n"
        f"Verification need: {attempt.verification_need:.3f}\n"
        f"Constraint density: {attempt.constraint_density:.3f}\n\n"
        f"Problem:\n{attempt.query}\n\n"
        f"Base attempt:\n{attempt.base_response}"
    )


def attempt_from_row(row: Mapping[str, Any]) -> Attempt:
    """Convert the canonical recovery JSONL schema into an :class:`Attempt`."""

    features = row.get("features") or {}
    usage = row.get("base_usage") or {}
    return Attempt(
        query=str(row["query"]),
        base_response=str(row["base_response"]),
        task_type=str(row.get("task_type", "unknown")),
        difficulty=float(features.get("difficulty", 0.0)),
        verification_need=float(features.get("verification_need", 0.0)),
        constraint_density=float(features.get("constraint_density", 0.0)),
        ambiguity_score=float(features.get("ambiguity_score", 0.0)),
        retrieval_need=float(features.get("retrieval_need", 0.0)),
        base_actual_tokens=int(row.get("base_actual_tokens", 0)),
        base_finalizer_used=bool(row.get("base_finalizer_used", False)),
        base_done_reason=usage.get("done_reason"),
    )


@dataclass(frozen=True)
class CallableGate:
    """Adapter for an arbitrary application scorer."""

    scorer: Callable[[Attempt], float]

    def score(self, attempt: Attempt) -> float:
        return _validate_probability(self.scorer(attempt))


@dataclass(frozen=True)
class LinearGate:
    """Portable logistic gate over cheap observable features."""

    weights: Mapping[str, float]
    intercept: float = 0.0

    def score(self, attempt: Attempt) -> float:
        values = dict(zip(FEATURE_NAMES, observable_features(attempt)))
        logit = self.intercept + sum(
            float(self.weights.get(name, 0.0)) * value for name, value in values.items()
        )
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearGate":
        payload = json.loads(Path(path).read_text())
        return cls(weights=payload["weights"], intercept=float(payload.get("intercept", 0.0)))


def _validate_probability(value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"gate score must be in [0, 1], got {probability}")
    return probability
