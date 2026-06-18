import math
import unittest

from sevra import Attempt, CallableGate, LinearGate, format_gate_input
from sevra.gates import FEATURE_NAMES, observable_features


class GateTests(unittest.TestCase):
    def test_observable_features_have_stable_order(self) -> None:
        attempt = Attempt(
            query="q",
            base_response="a",
            difficulty=0.2,
            verification_need=0.3,
            constraint_density=0.4,
            ambiguity_score=0.5,
            retrieval_need=0.6,
            base_finalizer_used=True,
            base_done_reason="length",
            base_actual_tokens=9,
        )
        values = observable_features(attempt)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertEqual(values[:7], (0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.0))
        self.assertAlmostEqual(values[-1], math.log(10))

    def test_linear_gate_returns_logistic_probability(self) -> None:
        gate = LinearGate(weights={"difficulty": 2.0}, intercept=-1.0)
        score = gate.score(Attempt(query="q", base_response="a", difficulty=0.5))
        self.assertAlmostEqual(score, 0.5)

    def test_callable_gate_rejects_invalid_probability(self) -> None:
        gate = CallableGate(lambda _: 1.2)
        with self.assertRaisesRegex(ValueError, "gate score must be"):
            gate.score(Attempt(query="q", base_response="a"))

    def test_gate_text_never_contains_gold_answer(self) -> None:
        text = format_gate_input(Attempt(query="Question", base_response="Attempt"))
        self.assertIn("Question", text)
        self.assertIn("Attempt", text)
        self.assertIn("gold answer is unavailable", text)


if __name__ == "__main__":
    unittest.main()
