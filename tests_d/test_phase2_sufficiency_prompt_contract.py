"""Static guardrails for the Phase 2 sufficiency prompt.

These tests enforce wording/contract only; real LLM behaviour requires an API regression.
"""
from pathlib import Path
import unittest

PROMPT = Path(__file__).resolve().parents[1] / 'prompts' / 'verification' / 'verify_sufficiency.txt'


class SufficiencyPromptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = PROMPT.read_text(encoding='utf-8')

    def test_distinguishes_short_answer_from_explanation(self):
        self.assertIn('draft.short_answer', self.prompt)
        self.assertIn('draft.explanation', self.prompt)
        self.assertIn('NOT automatically INCOMPLETE', self.prompt)

    def test_unsupported_filled_answer_is_uncertain(self):
        self.assertIn('UNCERTAIN', self.prompt)
        self.assertIn('unsupported short_answer', self.prompt)

    def test_preserves_json_contract(self):
        for field in ('answer_sufficiency', 'missing_information', 'rationale', 'evidence_ids'):
            self.assertIn(field, self.prompt)
        self.assertIn('COMPLETE requires at least one valid citation', self.prompt)


if __name__ == '__main__':
    unittest.main()
