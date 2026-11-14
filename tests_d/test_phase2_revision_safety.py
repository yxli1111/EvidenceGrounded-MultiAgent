"""Scripted protocol tests. These do not establish real LLM factual reliability."""
import unittest
from tests_d.support import ScriptedLLM, mock_contracts, sample_inputs
from src.verification import configure
from src.verification.dual_dimension import SufficiencyResult, DualDimensionResult
from src.agents.revision_v3 import revise_with_sufficiency_v3
from src.verification.errors import RevisionError


def diagnostic(example, draft, label="INCOMPLETE", citations=None):
    return DualDimensionResult(
        {"feedback_id": "feedback-v2", "items": [], "overall_comment": ""},
        SufficiencyResult(example.example_id, draft.draft_id, label,
                          "Country of birth is missing" if label == "INCOMPLETE" else None,
                          "Diagnostic", ["Ada Lovelace::0"] if citations is None else citations,
                          "d-answer-sufficiency-v3"))


def candidate(answer="England", citations=None):
    return {"final_answer": answer,
            "final_explanation": "Ada Lovelace was born in London, which is in England.",
            "evidence_ids": ["R1E1", "R1E2"] if citations is None else citations}


def suff_response(label="COMPLETE", missing=None):
    return {"answer_sufficiency": label, "missing_information": missing,
            "rationale": "Checked the requested country.", "evidence_ids": ["S1E1", "S1E2"]}


def support_response(label="SUPPORTED"):
    return {"items": [{"target_id": "example-1:draft", "label": label,
                       "rationale": "Checked against both sentences.",
                       "evidence_ids": ["D1E1", "D1E2"], "suggested_correction": None}],
            "overall_comment": "Checked."}


class V3RevisionSafetyTests(unittest.TestCase):
    def setUp(self):
        manager = mock_contracts()
        manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        configure(model="mock-model")
        self.example, self.draft, _, self.evidence = sample_inputs()

    def run_v3(self, diagnostic_value, *responses):
        client = ScriptedLLM(*responses)
        result = revise_with_sufficiency_v3(self.example, self.draft, diagnostic_value,
                                            self.evidence, client)
        return result, client

    def test_complete_and_uncertain_never_call_model(self):
        for label in ("COMPLETE", "UNCERTAIN"):
            with self.subTest(label=label):
                result, client = self.run_v3(diagnostic(self.example, self.draft, label))
                self.assertFalse(result.attempted)
                self.assertEqual(client.calls, [])

    def test_missing_citations_does_not_revise(self):
        result, client = self.run_v3(diagnostic(self.example, self.draft, citations=[]))
        self.assertEqual(result.reason, "insufficient_evidence_for_repair")
        self.assertEqual(client.calls, [])

    def test_invalid_candidate_citation_rejected(self):
        with self.assertRaisesRegex(RevisionError, "outside the supplied set"):
            self.run_v3(diagnostic(self.example, self.draft), candidate(citations=["R1E99"]))

    def test_incomplete_postcheck_rolls_back(self):
        result, client = self.run_v3(diagnostic(self.example, self.draft), candidate(),
                                     suff_response("INCOMPLETE", "Another missing link"))
        self.assertFalse(result.accepted)
        self.assertEqual(result.revision.final_answer, self.draft.short_answer)
        self.assertEqual(len(client.calls), 2)

    def test_unsupported_postcheck_rolls_back(self):
        result, client = self.run_v3(diagnostic(self.example, self.draft), candidate(),
                                     suff_response(), support_response("UNSUPPORTED"))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "post_support_not_supported")
        self.assertEqual(len(client.calls), 3)

    def test_both_postchecks_accept_candidate(self):
        result, client = self.run_v3(diagnostic(self.example, self.draft), candidate(),
                                     suff_response(), support_response())
        self.assertTrue(result.accepted)
        self.assertEqual(result.revision.final_answer, "England")
        self.assertEqual(result.candidate_citations, ["Ada Lovelace::0", "London::0"])
        self.assertEqual(result.to_dict()["post_support"], "SUPPORTED")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual([c["stage"] for c in client.calls],
                         ["revise_sufficiency_v3", "verify_sufficiency", "verify_draft"])

    def test_unchanged_candidate_no_postcheck(self):
        result, client = self.run_v3(diagnostic(self.example, self.draft), {
            "final_answer": self.draft.short_answer,
            "final_explanation": self.draft.explanation,
            "evidence_ids": ["R1E1"]})
        self.assertFalse(result.accepted)
        self.assertEqual(len(client.calls), 1)

    def test_wrong_question_diagnostic_rejected(self):
        bad = diagnostic(self.example, self.draft)
        bad = DualDimensionResult(bad.claim_feedback,
            SufficiencyResult("wrong", self.draft.draft_id, "INCOMPLETE", "missing", "r", [], "v3"))
        with self.assertRaisesRegex(RevisionError, "different question"):
            self.run_v3(bad)


if __name__ == "__main__":
    unittest.main()
