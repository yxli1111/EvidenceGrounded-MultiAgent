"""Offline V3 diagnostic tests; no network, no shared schema edits."""
import json
import unittest

from tests_d.support import ScriptedLLM, mock_contracts, sample_inputs
from src.verification import configure
from src.verification.errors import VerificationError
from src.verification.dual_dimension import (
    verify_answer_sufficiency, verify_dual_dimension, verify_draft_dual_dimension,
)


def suff(label="INCOMPLETE", missing="Second person's birth year", cites=None):
    return {"answer_sufficiency": label, "missing_information": missing,
            "rationale": "The evidence does not establish the requested comparison.",
            "evidence_ids": ["S1E1"] if cites is None else cites}


def claim_verdict():
    return {"items": [{"target_id": "claim-1", "label": "SUPPORTED",
                       "rationale": "The cited sentence supports the claim.",
                       "evidence_ids": ["C1E1"], "suggested_correction": None}],
            "overall_comment": "Claim checked."}


class DualDimensionTests(unittest.TestCase):
    def setUp(self):
        manager = mock_contracts()
        manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        configure(model="mock-model")
        self.example, self.draft, self.claims, self.evidence = sample_inputs()

    def test_supported_claim_can_have_incomplete_answer(self):
        client = ScriptedLLM(claim_verdict(), suff())
        result = verify_dual_dimension(self.example, self.draft, self.claims,
                                       {"claim-1": self.evidence}, client)
        self.assertEqual(result.claim_feedback["items"][0]["label"], "SUPPORTED")
        self.assertEqual(result.sufficiency.answer_sufficiency, "INCOMPLETE")
        self.assertEqual(result.sufficiency.evidence_ids, ["Ada Lovelace::0"])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result.to_dict()["sufficiency"]["prompt_version"], "d-answer-sufficiency-v3")
        payload = json.loads(client.calls[1]["messages"][1]["content"])
        self.assertNotIn("gold_answer", json.dumps(payload))
        self.assertEqual(payload["allowed_evidence_ids"], ["S1E1", "S1E2"])

    def test_draft_dual_dimension_preserves_v2_feedback(self):
        draft_verdict = {"items": [{"target_id": self.draft.draft_id, "label": "SUPPORTED",
                                    "rationale": "Checked.", "evidence_ids": ["D1E1"],
                                    "suggested_correction": None}], "overall_comment": "Done."}
        client = ScriptedLLM(draft_verdict, suff("COMPLETE", None))
        result = verify_draft_dual_dimension(self.example, self.draft, self.evidence, client)
        self.assertEqual(result.claim_feedback["items"][0]["scope"], "draft")
        self.assertEqual(result.sufficiency.answer_sufficiency, "COMPLETE")

    def test_invalid_citation_rejected(self):
        client = ScriptedLLM(suff(cites=["S1E99"]))
        with self.assertRaisesRegex(VerificationError, "not in supplied evidence"):
            verify_answer_sufficiency(self.example, self.draft, self.evidence, client)

    def test_missing_information_required_for_incomplete(self):
        client = ScriptedLLM(suff(missing=None))
        with self.assertRaisesRegex(VerificationError, "must specify"):
            verify_answer_sufficiency(self.example, self.draft, self.evidence, client)

    def test_complete_requires_citation(self):
        client = ScriptedLLM(suff("COMPLETE", None, []))
        with self.assertRaisesRegex(VerificationError, "requires at least one"):
            verify_answer_sufficiency(self.example, self.draft, self.evidence, client)

    def test_no_evidence_is_uncertain_without_api_call(self):
        client = ScriptedLLM()
        result = verify_answer_sufficiency(self.example, self.draft, [], client)
        self.assertEqual(result.answer_sufficiency, "UNCERTAIN")
        self.assertEqual(client.calls, [])

    def test_canonical_citation_accepted_and_deduplicated(self):
        client = ScriptedLLM(suff(cites=["S1E1", "Ada Lovelace::0", "S1E1"]))
        result = verify_answer_sufficiency(self.example, self.draft, self.evidence, client)
        self.assertEqual(result.evidence_ids, ["Ada Lovelace::0"])

    def test_unexpected_fields_rejected(self):
        bad = suff()
        bad["gold_answer"] = "leak"
        with self.assertRaisesRegex(VerificationError, "exactly four"):
            verify_answer_sufficiency(self.example, self.draft, self.evidence, ScriptedLLM(bad))


if __name__ == "__main__":
    unittest.main()
