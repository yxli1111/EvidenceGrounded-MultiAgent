"""Regression coverage for permissive edit audit and isolated M5 citations.

Scripted replies verify protocol behavior, not a model's factuality or compliance.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from src.agents.revision import revise_answer
from src.verification import configure, verify_claims, verify_draft
from src.verification.errors import RevisionError, VerificationError
from tests_d.support import (
    AtomicClaim, FeedbackItem, ScriptedLLM, VerificationFeedback,
    mock_contracts, sample_inputs,
)


def verdict(target_id, citations, label="SUPPORTED"):
    return {
        "items": [{"target_id": target_id, "label": label,
                   "rationale": "Checked this target's supplied evidence.",
                   "evidence_ids": citations, "suggested_correction": None}],
        "overall_comment": "Checked one target.",
    }


class RegressionV2Tests(unittest.TestCase):
    def setUp(self):
        manager = mock_contracts()
        manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        configure(model="mock-model")
        self.example, self.draft, self.claims, self.evidence = sample_inputs()
        self.feedback = VerificationFeedback(
            "feedback-v2",
            [FeedbackItem("draft", self.draft.draft_id,
                          "Check the country against the supplied context.",
                          None, [], "Use the correct country.")],
            "Independent criticism.",
        )

    def revised(self, **updates):
        result = {
            "final_answer": "England",
            "final_explanation": "Ada Lovelace was born in London, which is in England.",
        }
        result.update(updates)
        return result

    def note(self, field="final_answer", **updates):
        result = {
            "field": field,
            "feedback_target_ids": [self.draft.draft_id],
            "reason": "The context supports this correction.",
        }
        result.update(updates)
        return result

    def call_revision(self, response):
        client = ScriptedLLM(response)
        with self.assertLogs("src.agents.revision", level="INFO") as captured:
            result = revise_answer(self.example, self.draft, self.feedback, client)
        audits = [json.loads(record.getMessage().split("D revision audit: ", 1)[1])
                  for record in captured.records
                  if record.getMessage().startswith("D revision audit: ")]
        self.assertEqual(len(audits), 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result.feedback_id, self.feedback.feedback_id)
        return result, audits[0], captured.records, client

    def test_missing_edit_notes_use_diff_and_do_not_invent_reasons(self):
        for values in ({}, {"edits": None}, {"edits": []}):
            with self.subTest(values=values):
                result, audit, logs, _ = self.call_revision(self.revised(**values))
                self.assertEqual(result.final_answer, "England")
                self.assertEqual(audit["draft_id"], self.draft.draft_id)
                self.assertEqual(audit["feedback_id"], self.feedback.feedback_id)
                changes = {change["field"]: change for change in audit["changes"]}
                self.assertEqual(set(changes), {"final_answer", "final_explanation"})
                self.assertEqual(changes["final_answer"]["before"], "France")
                self.assertEqual(changes["final_answer"]["after"], "England")
                for change in changes.values():
                    self.assertEqual(change["source"], "computed_diff")
                    self.assertEqual(change["model_notes"], [])
                self.assertTrue(any(record.levelname == "WARNING" for record in logs))

    def test_repeated_notes_for_same_field_are_retained(self):
        notes = [self.note(), self.note(reason="A second model explanation."),
                 self.note("final_explanation")]
        _, audit, logs, _ = self.call_revision(self.revised(edits=notes))
        changes = {change["field"]: change for change in audit["changes"]}
        self.assertEqual(len(changes["final_answer"]["model_notes"]), 2)
        self.assertFalse(any(record.levelname == "WARNING" for record in logs))

    def test_field_aliases_are_accepted_for_audit_notes(self):
        _, audit, logs, _ = self.call_revision(self.revised(edits=[
            self.note("short_answer"), self.note("explanation"),
        ]))
        self.assertEqual({change["field"] for change in audit["changes"]},
                         {"final_answer", "final_explanation"})
        self.assertTrue(all(change["model_notes"] for change in audit["changes"]))
        self.assertFalse(any(record.levelname == "WARNING" for record in logs))

    def test_note_for_unchanged_field_does_not_fail_valid_answer(self):
        result, audit, logs, _ = self.call_revision(self.revised(
            final_explanation=self.draft.explanation,
            edits=[self.note(), self.note("final_explanation")],
        ))
        self.assertEqual(result.final_answer, "England")
        self.assertEqual([change["field"] for change in audit["changes"]], ["final_answer"])
        self.assertTrue(any("did not change" in issue for issue in audit["issues"]))
        self.assertTrue(any(record.levelname == "WARNING" for record in logs))

    def test_bad_optional_metadata_warns_but_keeps_required_output(self):
        variants = [
            "not a list", [None], [self.note(field="unknown")],
            [self.note(feedback_target_ids=["not-in-feedback"])],
            [self.note(reason="")], [self.note(feedback_target_ids=123)],
        ]
        for edits in variants:
            with self.subTest(edits=edits):
                result, audit, logs, _ = self.call_revision(self.revised(edits=edits))
                self.assertEqual(result.final_answer, "England")
                self.assertTrue(audit["issues"])
                self.assertTrue(any(record.levelname == "WARNING" for record in logs))
                self.assertTrue(all(change["model_notes"] == [] for change in audit["changes"]))

    def test_required_output_types_and_client_failure_still_fail_once(self):
        for response in (self.revised(final_answer=42),
                         {"final_answer": "England"},
                         self.revised(final_explanation=["not a string"])):
            with self.subTest(response=response):
                client = ScriptedLLM(response)
                with self.assertRaises(RevisionError):
                    revise_answer(self.example, self.draft, self.feedback, client)
                self.assertEqual(len(client.calls), 1)
        failure = RuntimeError("scripted transport failure")
        client = ScriptedLLM(failure)
        with self.assertRaises(RuntimeError) as caught:
            revise_answer(self.example, self.draft, self.feedback, client)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(client.calls), 1)

    def test_short_answer_prompt_and_complete_multiword_name(self):
        name = "University of New South Wales"
        result, _, _, client = self.call_revision(self.revised(final_answer=name))
        self.assertEqual(result.final_answer, name)
        prompt = client.calls[0]["messages"][0]["content"].lower()
        for requirement in ("entity", "number", '"yes"', '"no"',
                            "full sentence", "final_explanation", "multiword"):
            self.assertIn(requirement, prompt)

    def isolated_claim_inputs(self):
        claims = [AtomicClaim("claim-one", "Ada Lovelace was born in London."),
                  AtomicClaim("claim-two", "London is in England.")]
        return claims, {"claim-one": self.evidence[:1], "claim-two": self.evidence[1:]}

    def test_two_claims_have_separate_requests_and_unique_resolved_aliases(self):
        claims, evidence = self.isolated_claim_inputs()
        client = ScriptedLLM(verdict("claim-one", ["C1E1"]),
                             verdict("claim-two", ["C2E1"]))
        feedback = verify_claims(self.example, self.draft, claims, evidence, client)
        self.assertEqual([item.evidence_ids for item in feedback.items],
                         [["Ada Lovelace::0"], ["London::0"]])
        self.assertEqual(len(client.calls), 2)
        for index, call in enumerate(client.calls):
            payload = json.loads(call["messages"][1]["content"])
            self.assertEqual(len(payload["targets"]), 1)
            target = payload["targets"][0]
            self.assertEqual(target["target_id"], claims[index].claim_id)
            self.assertEqual(target["allowed_evidence_ids"], [f"C{index + 1}E1"])
            self.assertEqual(len(target["evidence"]), 1)
            other = self.evidence[1 - index]
            other_canonical = f"{other.doc_title}::{other.sentence_id}"
            self.assertNotIn(other_canonical, json.dumps(payload))

    def test_canonical_and_alias_citations_are_deduplicated(self):
        client = ScriptedLLM(verdict("claim-1", [
            "C1E1", "Ada Lovelace::0", "C1E2", "London::0", "C1E1",
        ]))
        feedback = verify_claims(self.example, self.draft, self.claims,
                                 {"claim-1": self.evidence}, client)
        self.assertEqual(feedback.items[0].evidence_ids, ["Ada Lovelace::0", "London::0"])

    def test_cross_claim_alias_and_canonical_citation_remain_invalid(self):
        claims, evidence = self.isolated_claim_inputs()
        for incorrect in ("C2E1", "London::0"):
            with self.subTest(incorrect=incorrect):
                client = ScriptedLLM(verdict("claim-one", [incorrect]))
                with self.assertRaisesRegex(VerificationError, "claim-one"):
                    verify_claims(self.example, self.draft, claims, evidence, client)
                self.assertEqual(len(client.calls), 1)
        client = ScriptedLLM(verdict("claim-one", ["C1E1"]),
                             verdict("claim-two", ["C1E1"]))
        with self.assertRaisesRegex(VerificationError, "claim-two"):
            verify_claims(self.example, self.draft, claims, evidence, client)
        self.assertEqual(len(client.calls), 2)

    def test_supported_or_contradicted_without_citation_remains_invalid(self):
        for label in ("SUPPORTED", "CONTRADICTED"):
            with self.subTest(label=label):
                client = ScriptedLLM(verdict("claim-1", [], label))
                with self.assertRaisesRegex(VerificationError, "require evidence citations"):
                    verify_claims(self.example, self.draft, self.claims,
                                  {"claim-1": self.evidence}, client)
                self.assertEqual(len(client.calls), 1)

    def test_empty_evidence_claim_skips_call_and_preserves_alias_position(self):
        claims, evidence = self.isolated_claim_inputs()
        evidence["claim-one"] = []
        client = ScriptedLLM(verdict("claim-two", ["C2E1"]))
        feedback = verify_claims(self.example, self.draft, claims, evidence, client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(feedback.items[0].label, "UNSUPPORTED")
        self.assertEqual(feedback.items[0].evidence_ids, [])
        self.assertEqual(feedback.items[1].evidence_ids, ["London::0"])

    def test_m4_short_citation_alias_resolves_to_original_sentence(self):
        client = ScriptedLLM(verdict(self.draft.draft_id, ["D1E1", "D1E2"]))
        feedback = verify_draft(self.example, self.draft, self.evidence, client)
        self.assertEqual(feedback.items[0].evidence_ids, ["Ada Lovelace::0", "London::0"])

    def test_real_team_contracts_without_mock_schema_in_child_process(self):
        contracts_root = Path(os.environ.get(
            "D_TEAM_ROOT",
            r"C:\Users\duradura\Downloads\EvidenceGrounded-MultiAgent-0a2ab15aaecd0c71d949815743e25895a8f896f5",
        ))
        if not (contracts_root / "contracts" / "schemas.py").is_file():
            contracts_root = Path(__file__).resolve().parents[1]
        if not (contracts_root / "contracts" / "schemas.py").is_file():
            self.skipTest("The user-provided real contracts snapshot is not available here.")
        project_root = Path(__file__).resolve().parents[1]
        script = textwrap.dedent('''
            import json
            import sys
            sys.dont_write_bytecode = True
            sys.path[:0] = [sys.argv[1], sys.argv[2]]
            from contracts.schemas import (QAInput, Sentence, AnswerDraft, AtomicClaim,
                Evidence, VerificationFeedback, FeedbackItem, RevisionResult,
                LLMResponse, LLMUsage)
            from src.verification import configure, verify_claims
            from src.agents.revision import revise_answer
            class Client:
                def __init__(self, *responses): self.responses = list(responses); self.calls = []
                def generate(self, **request):
                    self.calls.append(request)
                    return LLMResponse(json.dumps(self.responses.pop(0)),
                        LLMUsage(stage=request['stage'], model=request['model']))
            configure(model='real-contracts-scripted-mock')
            example = QAInput('e', 'Where?', [Sentence('Place', 0, 'The city is London.')])
            draft = AnswerDraft('d', 'Paris', 'The city is Paris.')
            claim = AtomicClaim('c', 'The city is Paris.')
            evidence = Evidence('Place', 0, example.context[0].text, 0.9, 'Where?', 1)
            client = Client({'items':[{'target_id':'c','label':'CONTRADICTED',
                'rationale':'The sentence says London.','evidence_ids':['C1E1'],
                'suggested_correction':'Use London.'}],'overall_comment':'Correct city.'},
                {'final_answer':'London','final_explanation':'The city is London.'})
            feedback = verify_claims(example, draft, [claim], {'c':[evidence]}, client)
            assert isinstance(feedback, VerificationFeedback)
            assert isinstance(feedback.items[0], FeedbackItem)
            assert feedback.items[0].evidence_ids == ['Place::0']
            result = revise_answer(example, draft, feedback, client)
            assert isinstance(result, RevisionResult)
            assert result.final_answer == 'London'
            assert len(client.calls) == 2
            assert 'tests_d.support' not in sys.modules
            print('Real contracts integration passed without mock schemas.')
        ''')
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script, str(project_root), str(contracts_root)],
            cwd=project_root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Real contracts integration passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
