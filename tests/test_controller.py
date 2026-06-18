import unittest

from sevra import Attempt, CallableGate, SEVRAController, VerificationOutput


def make_attempt() -> Attempt:
    return Attempt(query="What is 2 + 2?", base_response="Final answer: 4")


class ControllerTests(unittest.TestCase):
    def test_accept_does_not_call_verifier(self) -> None:
        controller = SEVRAController(CallableGate(lambda _: 0.2), threshold=0.6)

        def verifier(*_):
            raise AssertionError("verifier should not run")

        result = controller.run(make_attempt(), verifier)
        self.assertEqual(result.decision.action.value, "accept")
        self.assertEqual(result.final_response, "Final answer: 4")
        self.assertEqual(result.verification_tokens, 0)

    def test_selected_attempt_runs_active_verification(self) -> None:
        controller = SEVRAController(CallableGate(lambda _: 0.9), threshold=0.6)

        def verifier(prompt, _):
            self.assertIn("at least two candidate-specific checks", prompt)
            return VerificationOutput("Checks pass. Final answer: 4", actual_tokens=73)

        result = controller.run(make_attempt(), verifier)
        self.assertTrue(result.decision.should_verify)
        self.assertTrue(result.final_response.endswith("Final answer: 4"))
        self.assertEqual(result.verification_tokens, 73)

    def test_threshold_is_inclusive(self) -> None:
        controller = SEVRAController(CallableGate(lambda _: 0.6), threshold=0.6)
        self.assertTrue(controller.decide(make_attempt()).should_verify)


if __name__ == "__main__":
    unittest.main()
