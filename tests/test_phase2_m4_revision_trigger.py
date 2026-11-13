
from src.runner.mocks import (
    MockDraftPort,
    MockRetrievePort,
    MockClaimsPort,
    MockVerificationPort,
    MockLLMClient,
    MockFeedbackItem,
    MockVerificationFeedback,
    make_synthetic_example,
)
from src.runner.runner import ExperimentRunner


class ContradictedVerifier(MockVerificationPort):
    def __init__(self):
        self.revision_called = False
        self.feedback_received = None

    def verify_draft(self, example, draft, evidence, llm):
        return MockVerificationFeedback(
            feedback_id=f"{draft.draft_id}:test",
            items=[
                MockFeedbackItem(
                    scope="draft",
                    target_id=draft.draft_id,
                    rationale="Synthetic contradiction for integration testing.",
                    label="CONTRADICTED",
                    evidence_ids=[],
                    suggested_correction="Correct the factual error.",
                )
            ],
            overall_comment="Test contradiction.",
        )

    def revise_answer(self, example, draft, feedback, llm):
        self.revision_called = True
        self.feedback_received = feedback
        return super().revise_answer(
            example, draft, feedback, llm
        )


def test_m4_contradicted_feedback_reaches_revision(tmp_path):
    verifier = ContradictedVerifier()

    runner = ExperimentRunner(
        draft_port=MockDraftPort(),
        retrieve_port=MockRetrievePort(),
        claims_port=MockClaimsPort(),
        verification_port=verifier,
        llm=MockLLMClient(),
        m4_top_k=5,
    )

    records = runner.run(
        examples=[make_synthetic_example()],
        output_path=str(tmp_path / "records.jsonl"),
        methods=["M4"],
    )

    assert len(records) == 1
    assert records[0].status == "ok"

    assert verifier.revision_called
    assert verifier.feedback_received.items[0].label == "CONTRADICTED"

    assert records[0].final is not None
    assert records[0].feedback["items"][0]["label"] == "CONTRADICTED"
