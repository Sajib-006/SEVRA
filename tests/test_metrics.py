import unittest

from sevra import calibrate_threshold, evaluate_policy


class MetricsTests(unittest.TestCase):
    def test_policy_metrics_count_fixes_flips_and_tokens(self) -> None:
        metrics = evaluate_policy(
            base_correct=[True, False, True, False],
            verified_correct=[True, True, False, False],
            gate_scores=[0.1, 0.9, 0.2, 0.8],
            verification_tokens=[100, 200, 300, 400],
            threshold=0.6,
        )
        self.assertAlmostEqual(metrics.accuracy, 0.75)
        self.assertAlmostEqual(metrics.intervention_rate, 0.5)
        self.assertAlmostEqual(metrics.helpful_fix_rate, 0.25)
        self.assertEqual(metrics.harmful_flip_rate, 0.0)
        self.assertAlmostEqual(metrics.average_verification_tokens, 150.0)

    def test_calibration_prefers_lower_cost_on_accuracy_tie(self) -> None:
        threshold, metrics = calibrate_threshold(
            base_correct=[True, False],
            verified_correct=[True, True],
            gate_scores=[0.2, 0.9],
            verification_tokens=[1000, 100],
        )
        self.assertAlmostEqual(threshold, 0.9)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.intervention_rate, 0.5)

    def test_mismatched_inputs_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            evaluate_policy([True], [True, False], [0.5], [10], threshold=0.5)


if __name__ == "__main__":
    unittest.main()
