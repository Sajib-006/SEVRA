"""SEVRA: selective verification for reasoning allocation."""

from .actions import Action, active_verification_prompt
from .controller import Decision, Result, SEVRAController
from .gates import (
    CallableGate,
    LinearGate,
    RecoverabilityGate,
    attempt_from_row,
    format_gate_input,
)
from .hub import HuggingFaceGate
from .metrics import PolicyMetrics, calibrate_threshold, evaluate_policy
from .schema import Attempt, VerificationOutput

__all__ = [
    "Action",
    "Attempt",
    "CallableGate",
    "Decision",
    "LinearGate",
    "HuggingFaceGate",
    "PolicyMetrics",
    "RecoverabilityGate",
    "Result",
    "SEVRAController",
    "VerificationOutput",
    "active_verification_prompt",
    "attempt_from_row",
    "calibrate_threshold",
    "evaluate_policy",
    "format_gate_input",
]

__version__ = "0.1.0"
