from sevra import Attempt, CallableGate, SEVRAController, VerificationOutput


# Replace this function with a classifier, calibrated feature gate, or hosted model.
gate = CallableGate(
    lambda attempt: 0.92 if attempt.base_done_reason == "length" else 0.15
)
controller = SEVRAController(gate=gate, threshold=0.60)

attempt = Attempt(
    query="A shop discounts a $50 item by 20%. What is the final price?",
    base_response="20% of 50 is 10, so the final price is $40. Final answer: 40",
    task_type="math",
    base_actual_tokens=83,
    base_done_reason="stop",
)


def verifier(prompt: str, _: Attempt) -> VerificationOutput:
    # Send `prompt` to your existing model client. This stub keeps the example offline.
    assert "candidate-specific checks" in prompt
    return VerificationOutput(
        response="The arithmetic and substitution checks pass. Final answer: 40",
        actual_tokens=61,
    )


result = controller.run(attempt, verifier)
print(result.decision.action.value, result.decision.recoverability)
print(result.final_response)
