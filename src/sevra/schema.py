from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Attempt:
    """Runtime-observable state for one completed base attempt."""

    query: str
    base_response: str
    task_type: str = "unknown"
    difficulty: float = 0.0
    verification_need: float = 0.0
    constraint_density: float = 0.0
    ambiguity_score: float = 0.0
    retrieval_need: float = 0.0
    base_actual_tokens: int = 0
    base_finalizer_used: bool = False
    base_done_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.base_response.strip():
            raise ValueError("base_response must not be empty")
        for name in (
            "difficulty",
            "verification_need",
            "constraint_density",
            "ambiguity_score",
            "retrieval_need",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.base_actual_tokens < 0:
            raise ValueError("base_actual_tokens must be non-negative")


@dataclass(frozen=True)
class VerificationOutput:
    """Output returned by an application-provided verifier."""

    response: str
    actual_tokens: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.response.strip():
            raise ValueError("verification response must not be empty")
        if self.actual_tokens < 0:
            raise ValueError("actual_tokens must be non-negative")
