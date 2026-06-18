from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PolicyMetrics:
    examples: int
    accuracy: float
    intervention_rate: float
    helpful_fix_rate: float
    harmful_flip_rate: float
    average_verification_tokens: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_policy(
    base_correct: Sequence[bool],
    verified_correct: Sequence[bool],
    gate_scores: Sequence[float],
    verification_tokens: Sequence[int],
    threshold: float,
) -> PolicyMetrics:
    """Evaluate a frozen threshold on paired base/verification outcomes."""

    lengths = {
        len(base_correct),
        len(verified_correct),
        len(gate_scores),
        len(verification_tokens),
    }
    if len(lengths) != 1:
        raise ValueError("all inputs must have the same length")
    if not base_correct:
        raise ValueError("at least one example is required")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    selected = [float(score) >= threshold for score in gate_scores]
    final = [
        bool(verified) if choose else bool(base)
        for base, verified, choose in zip(
            base_correct, verified_correct, selected, strict=True
        )
    ]
    fixes = [
        choose and not bool(base) and bool(verified)
        for base, verified, choose in zip(
            base_correct, verified_correct, selected, strict=True
        )
    ]
    flips = [
        choose and bool(base) and not bool(verified)
        for base, verified, choose in zip(
            base_correct, verified_correct, selected, strict=True
        )
    ]
    selected_tokens = [
        int(tokens) if choose else 0
        for tokens, choose in zip(verification_tokens, selected, strict=True)
    ]
    n = len(final)
    return PolicyMetrics(
        examples=n,
        accuracy=sum(final) / n,
        intervention_rate=sum(selected) / n,
        helpful_fix_rate=sum(fixes) / n,
        harmful_flip_rate=sum(flips) / n,
        average_verification_tokens=sum(selected_tokens) / n,
    )


def calibrate_threshold(
    base_correct: Sequence[bool],
    verified_correct: Sequence[bool],
    gate_scores: Sequence[float],
    verification_tokens: Sequence[int],
) -> tuple[float, PolicyMetrics]:
    """Choose accuracy first, then lower token cost and fewer harmful flips."""

    candidates: Iterable[float] = (1.0, *sorted(set(map(float, gate_scores)), reverse=True))
    best: tuple[float, PolicyMetrics] | None = None
    for threshold in candidates:
        metrics = evaluate_policy(
            base_correct,
            verified_correct,
            gate_scores,
            verification_tokens,
            threshold,
        )
        rank = (
            metrics.accuracy,
            -metrics.average_verification_tokens,
            -metrics.harmful_flip_rate,
        )
        if best is None:
            best = (threshold, metrics)
            best_rank = rank
        elif rank > best_rank:
            best = (threshold, metrics)
            best_rank = rank
    assert best is not None
    return best
