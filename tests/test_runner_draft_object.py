"""Tests for draft cache returning AnswerDraft objects on resume (item 5).

When a run is resumed and the draft is loaded from drafts.jsonl, the runner
must convert the cached dict to an AnswerDraft object (contracts.schemas or
_DraftFallback) so that real ports can do draft.draft_id, draft.explanation, etc.
"""

import json
import pytest


# ---------------------------------------------------------------------------
# Helper: a port that REQUIRES attribute access on draft
# ---------------------------------------------------------------------------

class _AttributeRequiringDraftPort:
    """Self-reflect port that will AttributeError if draft is a dict."""

    def __init__(self):
        self.generate_draft_calls = 0
        self.self_reflect_calls = 0

    def generate_draft(self, example, llm):
        from src.runner.mocks import MockAnswerDraft
        self.generate_draft_calls += 1
        eid = example.example_id if hasattr(example, "example_id") else example["example_id"]
        llm.generate(stage="generate_draft", messages=[], model="mock", temperature=0.0, max_tokens=512)
        return MockAnswerDraft(
            draft_id=f"{eid}:draft",
            short_answer="Draft answer.",
            explanation="Draft explanation text.",
        )

    def self_reflect(self, example, draft, llm):
        """Access draft via attribute — fails if draft is a dict."""
        from src.runner.mocks import MockRevisionResult
        self.self_reflect_calls += 1
        # This line raises AttributeError if draft is a plain dict
        draft_id = draft.draft_id
        explanation = draft.explanation
        llm.generate(stage="self_reflect", messages=[], model="mock", temperature=0.0, max_tokens=512)
        return MockRevisionResult(
            draft_id=draft_id,
            feedback_id=f"{draft_id}:reflect",
            final_answer="Reflected answer.",
            final_explanation=f"Reflected on: {explanation}",
        )

    def independent_critique(self, example, draft, llm):
        from src.runner.mocks import MockVerificationFeedback, MockFeedbackItem
        _ = draft.draft_id  # attribute access
        llm.generate(stage="independent_critique", messages=[], model="mock", temperature=0.0, max_tokens=512)
        return MockVerificationFeedback(
            feedback_id=f"{draft.draft_id}:critique",
            items=[MockFeedbackItem("draft", draft.draft_id, "ok", None, [])],
            overall_comment="fine",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDraftCacheReturnsObjects:

    def _run_partial(self, tmp_path):
        """Run once, then remove M2's record so resume is triggered."""
        from src.runner.mocks import MockClaimsPort, MockLLMClient, MockVerificationPort, make_synthetic_example
        from src.runner.records import load_records
        from src.runner.runner import ExperimentRunner

        draft_port = _AttributeRequiringDraftPort()
        runner = ExperimentRunner(
            draft_port=draft_port,
            retrieve_port=_PassthroughRetrievePort(),
            claims_port=MockClaimsPort(),
            verification_port=MockVerificationPort(),
            llm=MockLLMClient(),
            top_k=2,
        )
        out = str(tmp_path / "out.jsonl")
        runner.run([make_synthetic_example("e1")], out)
        return out, draft_port

    def test_resumed_draft_uses_attribute_access_not_error(self, tmp_path):
        """After partial run, resume must NOT error on draft attribute access."""
        from src.runner.mocks import MockClaimsPort, MockLLMClient, MockVerificationPort, make_synthetic_example
        from src.runner.records import load_records
        from src.runner.runner import ExperimentRunner

        # First full run
        out = str(tmp_path / "out.jsonl")
        draft_port_1 = _AttributeRequiringDraftPort()
        runner1 = ExperimentRunner(
            draft_port=draft_port_1,
            retrieve_port=_PassthroughRetrievePort(),
            claims_port=MockClaimsPort(),
            verification_port=MockVerificationPort(),
            llm=MockLLMClient(),
            top_k=2,
        )
        runner1.run([make_synthetic_example("e1")], out)

        # Delete M2's record to simulate partial run
        records = load_records(out)
        non_m2 = [r for r in records if r.method_id != "M2"]
        with open(out, "w", encoding="utf-8") as f:
            from src.runner.records import to_jsonl_line
            for r in non_m2:
                f.write(to_jsonl_line(r) + "\n")

        # Resume: draft_port must not be called for generate_draft again,
        # and self_reflect must succeed using attribute access on draft
        draft_port_2 = _AttributeRequiringDraftPort()
        runner2 = ExperimentRunner(
            draft_port=draft_port_2,
            retrieve_port=_PassthroughRetrievePort(),
            claims_port=MockClaimsPort(),
            verification_port=MockVerificationPort(),
            llm=MockLLMClient(),
            top_k=2,
        )
        new_records = runner2.run([make_synthetic_example("e1")], out)

        # generate_draft was NOT called again (draft came from cache)
        assert draft_port_2.generate_draft_calls == 0, (
            "generate_draft must not be called again on resume"
        )
        # M2 was generated and succeeded (not an error from AttributeError)
        all_records = load_records(out)
        m2 = next(r for r in all_records if r.method_id == "M2")
        assert m2.status == "ok", (
            f"M2 should be ok after resume, got status={m2.status!r}, error={m2.error!r}"
        )

    def test_draft_object_has_correct_fields(self, tmp_path):
        """The draft object loaded from cache must have draft_id, short_answer,
        explanation matching the cached values."""
        from src.runner.mocks import MockClaimsPort, MockLLMClient, MockVerificationPort, make_synthetic_example
        from src.runner.records import load_records
        from src.runner.runner import ExperimentRunner

        # First run
        out = str(tmp_path / "out.jsonl")
        runner = ExperimentRunner(
            draft_port=_AttributeRequiringDraftPort(),
            retrieve_port=_PassthroughRetrievePort(),
            claims_port=MockClaimsPort(),
            verification_port=MockVerificationPort(),
            llm=MockLLMClient(),
            top_k=2,
        )
        runner.run([make_synthetic_example("e1")], out)
        records_before = {r.method_id: r for r in load_records(out)}
        original_draft_id = (records_before["M1"].draft or {}).get("draft_id")

        # Delete M3 to force resume
        with open(out, "w", encoding="utf-8") as f:
            from src.runner.records import to_jsonl_line
            for r in records_before.values():
                if r.method_id != "M3":
                    f.write(to_jsonl_line(r) + "\n")

        # Resume: M3 involves independent_critique which does draft.draft_id
        runner2 = ExperimentRunner(
            draft_port=_AttributeRequiringDraftPort(),
            retrieve_port=_PassthroughRetrievePort(),
            claims_port=MockClaimsPort(),
            verification_port=MockVerificationPort(),
            llm=MockLLMClient(),
            top_k=2,
        )
        runner2.run([make_synthetic_example("e1")], out)
        all_records = {r.method_id: r for r in load_records(out)}

        assert all_records["M3"].status == "ok", (
            f"M3 failed on resume: {all_records['M3'].error}"
        )
        # draft_id must match original (same cached draft)
        assert (all_records["M3"].draft or {}).get("draft_id") == original_draft_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _PassthroughRetrievePort:
    """Returns first top_k context sentences as mock evidence objects."""
    def retrieve(self, query, context, top_k):
        from src.runner.mocks import MockEvidence
        results = []
        for i, sent in enumerate(list(context)[:top_k], start=1):
            results.append(MockEvidence(
                doc_title=sent.title if hasattr(sent, "title") else sent["title"],
                sentence_id=sent.sentence_id if hasattr(sent, "sentence_id") else sent["sentence_id"],
                text=sent.text if hasattr(sent, "text") else sent["text"],
                score=round(1.0 / i, 4),
                retrieval_query=query,
                rank=i,
            ))
        return results
