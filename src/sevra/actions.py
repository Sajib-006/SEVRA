from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    ACCEPT = "accept"
    ACTIVE_VERIFY = "active_verify"


def active_verification_prompt(query: str, base_attempt: str) -> str:
    """Build the candidate-specific verification prompt used by SEVRA."""

    if not query.strip() or not base_attempt.strip():
        raise ValueError("query and base_attempt must not be empty")
    return (
        "Create and execute at least two candidate-specific checks for the attempted "
        "solution, such as reconstructing the governing equations, checking units or "
        "bounds, substituting the result back, or solving by an independent route. "
        "Preserve the answer if all checks pass; otherwise repair it. Finish with "
        "exactly: Final answer: <answer>.\n\n"
        f"Problem:\n{query}\n\nAttempted solution:\n{base_attempt}"
    )
