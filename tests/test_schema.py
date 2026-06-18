import unittest

from sevra import Attempt, VerificationOutput


class SchemaTests(unittest.TestCase):
    def test_attempt_rejects_out_of_range_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "difficulty"):
            Attempt(query="q", base_response="a", difficulty=1.1)

    def test_verification_output_rejects_negative_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            VerificationOutput(response="answer", actual_tokens=-1)


if __name__ == "__main__":
    unittest.main()
