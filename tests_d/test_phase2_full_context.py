"""Offline Phase 2 M4 evidence-volume and citation regression tests.

Place in tests_d/test_phase2_full_context.py and run:
    python -m unittest tests_d.test_phase2_full_context -v
"""
import json
import unittest

from tests_d.support import AnswerDraft, Evidence, QAInput, Sentence, ScriptedLLM, mock_contracts
from src.verification import configure, verify_draft
from src.verification.errors import VerificationError
from src.agents.revision import revise_answer


class FullContextCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.contracts = mock_contracts()
        self.contracts.__enter__()
        self.addCleanup(self.contracts.__exit__, None, None, None)
        configure(model="mock-model", temperature=0.0,
                  verification_max_tokens=2048, revision_max_tokens=2048)
        sentences = [Sentence("Document A", i, f"Fact sentence number {i}.") for i in range(15)]
        self.example = QAInput("phase2-example", "Which fact is correct?", sentences)
        self.draft = AnswerDraft("phase2-draft", "Old answer", "Old explanation.")
        self.all_evidence = [Evidence(s.title, s.sentence_id, s.text, 1.0 - i / 100,
                                      self.example.question, i + 1)
                             for i, s in enumerate(sentences)]

    def response(self, citation, label="SUPPORTED"):
        return {"items": [{"target_id": self.draft.draft_id, "label": label,
                           "rationale": "This evidence was checked.",
                           "evidence_ids": [citation], "suggested_correction": None}],
                "overall_comment": "Checked."}

    def test_all_three_evidence_volumes_preserved(self):
        for count in (5, 10, 15):
            with self.subTest(count=count):
                client = ScriptedLLM(self.response(f"D1E{count}"))
                feedback = verify_draft(self.example, self.draft,
                                        self.all_evidence[:count], client)
                self.assertEqual(feedback.items[0].evidence_ids,
                                 [f"Document A::{count - 1}"])
                payload = json.loads(client.calls[0]["messages"][1]["content"])
                target = payload["targets"][0]
                self.assertEqual(len(target["evidence"]), count)
                self.assertEqual(len(target["allowed_evidence_ids"]), count)
                self.assertEqual(target["allowed_evidence_ids"][-1], f"D1E{count}")

    def test_full_context_citation_passes_to_revision(self):
        client = ScriptedLLM(self.response("D1E15"))
        feedback = verify_draft(self.example, self.draft, self.all_evidence, client)
        self.assertEqual(feedback.items[0].evidence_ids, ["Document A::14"])
        # SUPPORTED is not actionable; revision must preserve the original draft.
        revision_client = ScriptedLLM()
        result = revise_answer(self.example, self.draft, feedback, revision_client)
        self.assertEqual(result.final_answer, self.draft.short_answer)
        self.assertEqual(revision_client.calls, [])

    def test_invalid_alias_rejected(self):
        client = ScriptedLLM(self.response("D1E16"))
        with self.assertRaisesRegex(VerificationError, "Citation is not in this target"):
            verify_draft(self.example, self.draft, self.all_evidence, client)

    def test_context_sentence_not_in_retrieved_subset_rejected(self):
        client = ScriptedLLM(self.response("Document A::14"))
        with self.assertRaisesRegex(VerificationError, "Citation is not in this target"):
            verify_draft(self.example, self.draft, self.all_evidence[:5], client)

    def test_duplicate_evidence_rejected(self):
        client = ScriptedLLM()
        with self.assertRaisesRegex(VerificationError, "duplicate sentences"):
            verify_draft(self.example, self.draft,
                         self.all_evidence[:5] + [self.all_evidence[0]], client)
        self.assertEqual(client.calls, [])

    def test_nonmatching_full_context_text_rejected(self):
        bad = list(self.all_evidence)
        original = bad[14]
        bad[14] = Evidence(original.doc_title, original.sentence_id,
                           "Changed text", original.score, original.retrieval_query, original.rank)
        client = ScriptedLLM()
        with self.assertRaisesRegex(VerificationError, "does not match"):
            verify_draft(self.example, self.draft, bad, client)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
