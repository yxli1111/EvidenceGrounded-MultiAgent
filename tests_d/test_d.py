"""Deterministic interface tests, not measurements of real model quality.

Run from the package root: python -m unittest discover -s tests_d -v
"""

from copy import deepcopy
from dataclasses import dataclass, field, is_dataclass, make_dataclass
import json
import sys
import unittest
from unittest.mock import patch

from tests_d.support import (
    AnswerDraft, AtomicClaim, Evidence, FeedbackItem, QAInput, RevisionResult,
    ScriptedLLM, Sentence, VerificationFeedback, mock_contracts, sample_inputs,
)

with mock_contracts():
    from src.verification import configure, runtime_metadata, verify_claims, verify_draft
    from src.verification.errors import RevisionError, VerificationError
    from src.agents.revision import revise_answer


def verdict(target_id, label="CONTRADICTED", evidence_ids=None, correction=None):
    return {
        "target_id": target_id,
        "label": label,
        "rationale": "The supplied sentences establish London and then England.",
        "evidence_ids": ["Ada Lovelace::0", "London::0"]
        if evidence_ids is None else evidence_ids,
        "suggested_correction": correction,
    }


def verification_response(*items):
    return {"items": list(items), "overall_comment": "Checked supplied evidence."}


def corrected_response(target_id):
    return {
        "final_answer": "England",
        "final_explanation": "Ada Lovelace was born in London, which is in England.",
        "edits": [
            {"field": "final_answer", "feedback_target_ids": [target_id],
             "reason": "The feedback links London to England."},
            {"field": "final_explanation", "feedback_target_ids": [target_id],
             "reason": "Replace the contradicted location with the supplied evidence."},
        ],
    }


class DTests(unittest.TestCase):
    def setUp(self):
        self.contract_context = mock_contracts()
        self.contract_context.__enter__()
        self.addCleanup(self.contract_context.__exit__, None, None, None)
        configure(model="mock-model", temperature=0.0,
                  verification_max_tokens=2048, revision_max_tokens=2048)
        self.example, self.draft, self.claims, self.evidence = sample_inputs()

    def feedback(self, label="CONTRADICTED", scope="draft", correction="Use England."):
        target = self.draft.draft_id if scope == "draft" else self.claims[0].claim_id
        citations = [] if label is None or label == "UNSUPPORTED" else [
            "Ada Lovelace::0", "London::0"
        ]
        return VerificationFeedback(
            "feedback-1", [FeedbackItem(scope, target, "Check the location in context.",
                                        label, citations, correction)], "Review location."
        )

    def assert_unchanged(self, result, feedback):
        self.assertEqual(result.final_answer, self.draft.short_answer)
        self.assertEqual(result.final_explanation, self.draft.explanation)
        self.assertEqual(result.draft_id, self.draft.draft_id)
        self.assertEqual(result.feedback_id, feedback.feedback_id)
        self.assertIsInstance(result, RevisionResult)

    def test_draft_verification_returns_shared_dataclass_and_multisentence_citations(self):
        llm = ScriptedLLM(verification_response(verdict(self.draft.draft_id)))
        feedback = verify_draft(self.example, self.draft, self.evidence, llm)
        self.assertIsInstance(feedback, VerificationFeedback)
        self.assertTrue(is_dataclass(feedback.items[0]))
        self.assertEqual(feedback.items[0].scope, "draft")
        self.assertEqual(feedback.items[0].target_id, self.draft.draft_id)
        self.assertEqual(feedback.items[0].label, "CONTRADICTED")
        self.assertEqual(feedback.items[0].evidence_ids, ["Ada Lovelace::0", "London::0"])
        self.assertTrue(feedback.feedback_id)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["stage"], "verify_draft")

    def test_claim_verification_accepts_three_labels(self):
        for label in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED"):
            with self.subTest(label=label):
                citations = [] if label == "UNSUPPORTED" else ["Ada Lovelace::0", "London::0"]
                llm = ScriptedLLM(verification_response(
                    verdict("claim-1", label=label, evidence_ids=citations)
                ))
                feedback = verify_claims(self.example, self.draft, self.claims,
                                         {"claim-1": self.evidence}, llm)
                self.assertEqual(feedback.items[0].scope, "claim")
                self.assertEqual(feedback.items[0].label, label)
                self.assertEqual(llm.calls[0]["stage"], "verify_claims")

    def test_no_evidence_is_unsupported_without_model_call(self):
        for mode in ("draft", "claim"):
            with self.subTest(mode=mode):
                llm = ScriptedLLM()
                feedback = verify_draft(self.example, self.draft, [], llm) if mode == "draft" else (
                    verify_claims(self.example, self.draft, self.claims, {"claim-1": []}, llm)
                )
                self.assertEqual(len(feedback.items), 1)
                self.assertEqual(feedback.items[0].label, "UNSUPPORTED")
                self.assertEqual(feedback.items[0].evidence_ids, [])
                self.assertIsNone(feedback.items[0].suggested_correction)
                self.assertEqual(llm.calls, [])

    def test_empty_claims_is_success_without_model_call(self):
        llm = ScriptedLLM()
        result = verify_claims(self.example, self.draft, [], {}, llm)
        self.assertEqual(result.items, [])
        self.assertEqual(llm.calls, [])

    def test_invalid_citation_is_rejected(self):
        llm = ScriptedLLM(verification_response(
            verdict(self.draft.draft_id, evidence_ids=["Nonexistent::0"])
        ))
        with self.assertRaises(VerificationError):
            verify_draft(self.example, self.draft, self.evidence, llm)
        self.assertEqual(len(llm.calls), 1)

    def test_claim_cannot_cite_another_claims_evidence(self):
        claims = [AtomicClaim("c1", "Ada was born in London."),
                  AtomicClaim("c2", "London is in England.")]
        llm = ScriptedLLM(verification_response(
            verdict("c1", "SUPPORTED", ["London::0"]),
        ))
        with self.assertRaisesRegex(VerificationError, "Citation is not in this target"):
            verify_claims(self.example, self.draft, claims,
                          {"c1": self.evidence[:1], "c2": self.evidence[1:]}, llm)

    def test_supported_and_contradicted_require_citations(self):
        for label in ("SUPPORTED", "CONTRADICTED"):
            with self.subTest(label=label):
                llm = ScriptedLLM(verification_response(
                    verdict(self.draft.draft_id, label, [])
                ))
                with self.assertRaises(VerificationError):
                    verify_draft(self.example, self.draft, self.evidence, llm)

    def test_wrong_or_repeated_target_is_rejected(self):
        c1 = verdict("claim-1")
        for items in ([], [c1, deepcopy(c1)], [verdict("another-claim")]):
            with self.subTest(items=items):
                llm = ScriptedLLM(verification_response(*items))
                with self.assertRaises(VerificationError):
                    verify_claims(self.example, self.draft, self.claims,
                                  {"claim-1": self.evidence}, llm)

    def test_nonempty_rationale_and_valid_label_required(self):
        for key, value in (("rationale", " "), ("rationale", None),
                           ("label", "TRUE"), ("label", None),
                           ("evidence_ids", "Ada Lovelace::0")):
            with self.subTest(key=key, value=value):
                item = verdict(self.draft.draft_id)
                item[key] = value
                llm = ScriptedLLM(verification_response(item))
                with self.assertRaises(VerificationError):
                    verify_draft(self.example, self.draft, self.evidence, llm)

    def test_evidence_provenance_must_match_original_context(self):
        for changes in ({"text": "Ada Lovelace was born in Paris."},
                        {"sentence_id": 99}, {"doc_title": "Unknown"}):
            with self.subTest(changes=changes):
                evidence = deepcopy(self.evidence)
                for name, value in changes.items():
                    setattr(evidence[0], name, value)
                llm = ScriptedLLM()
                with self.assertRaises(VerificationError):
                    verify_draft(self.example, self.draft, evidence, llm)
                self.assertEqual(llm.calls, [])

    def test_titles_containing_delimiter_keep_exact_provenance(self):
        title = "Topic::Part"
        sentence = Sentence(title, 7, "This sentence is the supplied evidence.")
        example = QAInput("example-delimiter", "What does the passage say?", [sentence])
        evidence = [Evidence(title, 7, sentence.text, 0.8, "query", 1)]
        llm = ScriptedLLM(verification_response(
            verdict(self.draft.draft_id, "SUPPORTED", ["Topic::Part::7"])
        ))
        result = verify_draft(example, self.draft, evidence, llm)
        self.assertEqual(result.items[0].evidence_ids, ["Topic::Part::7"])

    def test_verification_json_errors_are_not_empty_success(self):
        for response in ("not JSON", "[]", '{"items":"wrong","overall_comment":"x"}'):
            with self.subTest(response=response):
                llm = ScriptedLLM(response)
                with self.assertRaises(VerificationError):
                    verify_draft(self.example, self.draft, self.evidence, llm)
                self.assertEqual(len(llm.calls), 1)

    def test_api_error_is_propagated_unchanged_without_local_retry(self):
        for stage in ("draft", "claims", "revision"):
            with self.subTest(stage=stage):
                failure = RuntimeError("Synthetic provider failure")
                llm = ScriptedLLM(failure)
                try:
                    if stage == "draft":
                        verify_draft(self.example, self.draft, self.evidence, llm)
                    elif stage == "claims":
                        verify_claims(self.example, self.draft, self.claims,
                                      {"claim-1": self.evidence}, llm)
                    else:
                        revise_answer(self.example, self.draft, self.feedback(), llm)
                except RuntimeError as caught:
                    self.assertIs(caught, failure)
                else:
                    self.fail("The provider exception should propagate")
                self.assertEqual(len(llm.calls), 1)

    def test_empty_supported_and_unactionable_unsupported_feedback_preserve_draft(self):
        cases = [
            VerificationFeedback("feedback-empty", [], "No actionable criticism."),
            self.feedback("SUPPORTED", correction=None),
            self.feedback("UNSUPPORTED", correction=None),
        ]
        for feedback in cases:
            with self.subTest(feedback=feedback):
                llm = ScriptedLLM()
                result = revise_answer(self.example, self.draft, feedback, llm)
                self.assert_unchanged(result, feedback)
                self.assertEqual(llm.calls, [])

    def test_m3_nullable_label_is_accepted_and_scripted_correction_keeps_ids(self):
        feedback = self.feedback(label=None)
        llm = ScriptedLLM(corrected_response(self.draft.draft_id))
        result = revise_answer(self.example, self.draft, feedback, llm)
        self.assertIsInstance(result, RevisionResult)
        self.assertEqual(result.final_answer, "England")
        self.assertEqual(result.draft_id, self.draft.draft_id)
        self.assertEqual(result.feedback_id, feedback.feedback_id)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["stage"], "revise_answer")

    def test_revision_prompt_and_settings_are_shared_across_m3_m4_m5(self):
        calls = []
        for label, scope in ((None, "draft"), ("CONTRADICTED", "draft"),
                             ("CONTRADICTED", "claim")):
            feedback = self.feedback(label=label, scope=scope)
            llm = ScriptedLLM(corrected_response(feedback.items[0].target_id))
            revise_answer(self.example, self.draft, feedback, llm)
            self.assertEqual(len(llm.calls), 1)
            calls.append(llm.calls[0])
        for key in ("stage", "model", "temperature", "max_tokens"):
            self.assertEqual([call[key] for call in calls], [calls[0][key]] * 3)
        self.assertEqual([call["messages"][0] for call in calls],
                         [calls[0]["messages"][0]] * 3)

    def test_scripted_revision_can_reject_wrong_feedback(self):
        """A scripted no-op verifies plumbing, not real LLM judgment quality."""
        self.draft = AnswerDraft(self.draft.draft_id, "England",
                                 "Ada Lovelace was born in London, England.")
        feedback = self.feedback(None, correction="Replace England with France.")
        llm = ScriptedLLM({"final_answer": self.draft.short_answer,
                           "final_explanation": self.draft.explanation, "edits": []})
        result = revise_answer(self.example, self.draft, feedback, llm)
        self.assert_unchanged(result, feedback)
        self.assertEqual(len(llm.calls), 1)

    def test_unsupported_with_justified_qualification_uses_shared_revision(self):
        feedback = self.feedback("UNSUPPORTED", correction="Qualify the unsupported location.")
        response = {
            "final_answer": self.draft.short_answer,
            "final_explanation": "The supplied evidence does not establish this location.",
            "edits": [{"field": "final_explanation",
                       "feedback_target_ids": [self.draft.draft_id],
                       "reason": "Qualify the unsupported location."}],
        }
        llm = ScriptedLLM(response)
        result = revise_answer(self.example, self.draft, feedback, llm)
        self.assertEqual(result.final_answer, self.draft.short_answer)
        self.assertEqual(result.final_explanation, response["final_explanation"])
        self.assertEqual(len(llm.calls), 1)

    def test_missing_model_notes_use_computed_edit_provenance(self):
        for edits in ([], [{"field": "final_answer", "feedback_target_ids": [self.draft.draft_id],
                           "reason": "Only one of two changed fields is explained."}]):
            with self.subTest(edits=edits):
                response = corrected_response(self.draft.draft_id)
                response["edits"] = edits
                llm = ScriptedLLM(response)
                with self.assertLogs("src.agents.revision", level="INFO") as logged:
                    result = revise_answer(self.example, self.draft, self.feedback(), llm)
                self.assertEqual(result.final_answer, "England")
                message = next(record.getMessage() for record in logged.records
                               if record.getMessage().startswith("D revision audit: "))
                audit = json.loads(message.removeprefix("D revision audit: "))
                self.assertEqual({item["field"] for item in audit["changes"]},
                                 {"final_answer", "final_explanation"})
                self.assertTrue(all(item["source"] == "computed_diff" for item in audit["changes"]))
                self.assertEqual(len(llm.calls), 1)

    def test_invalid_optional_edit_trace_is_ignored_with_warning(self):
        response = corrected_response(self.draft.draft_id)
        response["edits"][0]["feedback_target_ids"] = ["invented-target"]
        llm = ScriptedLLM(response)
        with self.assertLogs("src.agents.revision", level="INFO") as logged:
            result = revise_answer(self.example, self.draft, self.feedback(), llm)
        self.assertEqual(result.final_answer, "England")
        self.assertTrue(any(record.levelname == "WARNING" for record in logged.records))
        message = next(record.getMessage() for record in logged.records
                       if record.getMessage().startswith("D revision audit: "))
        self.assertNotIn("invented-target", message)

    def test_revision_bad_json_or_output_shape_raises_without_retry(self):
        responses = ["not JSON", {"final_answer": "England", "edits": []},
                     {"final_answer": 123, "final_explanation": "Text", "edits": []}]
        for response in responses:
            with self.subTest(response=response):
                llm = ScriptedLLM(response)
                with self.assertRaises(RevisionError):
                    revise_answer(self.example, self.draft, self.feedback(), llm)
                self.assertEqual(len(llm.calls), 1)

    def test_gold_fields_are_never_forwarded_to_llm(self):
        @dataclass
        class ContaminatedInput(QAInput):
            gold_answer: str = "DO_NOT_SEND_GOLD_7391"
            gold_supporting_facts: list = field(default_factory=lambda: ["SECRET_GOLD_FACT_7391"])

        example = ContaminatedInput(self.example.example_id, self.example.question,
                                    self.example.context)
        clients = [
            ScriptedLLM(verification_response(verdict(self.draft.draft_id))),
            ScriptedLLM(verification_response(verdict("claim-1"))),
            ScriptedLLM(corrected_response(self.draft.draft_id)),
        ]
        verify_draft(example, self.draft, self.evidence, clients[0])
        verify_claims(example, self.draft, self.claims, {"claim-1": self.evidence}, clients[1])
        revise_answer(example, self.draft, self.feedback(), clients[2])
        for client in clients:
            payload = json.dumps(client.calls)
            self.assertNotIn("DO_NOT_SEND_GOLD_7391", payload)
            self.assertNotIn("SECRET_GOLD_FACT_7391", payload)
            self.assertNotIn("gold_answer", payload)
            self.assertNotIn("gold_supporting_facts", payload)

    def test_runtime_metadata_is_available_for_runner_logging(self):
        metadata = runtime_metadata()
        self.assertIsInstance(metadata, dict)
        text = json.dumps(metadata)
        self.assertIn("mock-model", text)

    def test_missing_shared_contracts_fails_before_a_model_call(self):
        for operation in ("verification", "revision"):
            with self.subTest(operation=operation):
                llm = ScriptedLLM()
                with patch.dict(sys.modules, {"contracts.schemas": None}):
                    with self.assertRaises(ModuleNotFoundError):
                        if operation == "verification":
                            verify_draft(self.example, self.draft, self.evidence, llm)
                        else:
                            revise_answer(self.example, self.draft, self.feedback(), llm)
                self.assertEqual(llm.calls, [])

    def test_feedback_item_class_name_is_not_assumed(self):
        team_item = make_dataclass("TeamVerdict", [
            ("scope", str), ("target_id", str), ("rationale", str),
            ("label", str | None), ("evidence_ids", list[str]),
            ("suggested_correction", str | None),
        ])
        team_feedback = make_dataclass("TeamFeedback", [
            ("feedback_id", str), ("items", list[team_item]), ("overall_comment", str),
        ])
        with patch.object(sys.modules["contracts.schemas"], "VerificationFeedback", team_feedback):
            llm = ScriptedLLM(verification_response(verdict(self.draft.draft_id)))
            result = verify_draft(self.example, self.draft, self.evidence, llm)
        self.assertIsInstance(result, team_feedback)
        self.assertIsInstance(result.items[0], team_item)

    def test_missing_or_extra_evidence_mapping_is_an_integration_error(self):
        for mapping in ({}, {"claim-1": self.evidence, "unknown": []}):
            with self.subTest(mapping=mapping):
                llm = ScriptedLLM()
                with self.assertRaises(VerificationError):
                    verify_claims(self.example, self.draft, self.claims, mapping, llm)
                self.assertEqual(llm.calls, [])

    def test_mixed_evidence_skips_model_for_unsubstantiated_claim(self):
        claims = self.claims + [AtomicClaim("claim-2", "A separate claim without evidence.")]
        good = verdict("claim-1")
        llm = ScriptedLLM(verification_response(good))
        feedback = verify_claims(self.example, self.draft, claims,
                                  {"claim-1": self.evidence, "claim-2": []}, llm)
        self.assertEqual(feedback.items[1].label, "UNSUPPORTED")
        self.assertEqual(feedback.items[1].evidence_ids, [])
        self.assertIsNone(feedback.items[1].suggested_correction)
        self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
