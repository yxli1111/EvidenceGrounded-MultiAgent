"""End-to-end tests for ExperimentRunner using only mock ports.

No real API calls, no HotpotQA data.  All inputs are synthetic.

Coverage
--------
1. Happy path: 5 conditions × 3 examples = 15 records.
2. Shared draft: generate_draft called exactly once per example, not per method.
3. JSONL round-trip: records written to file can be read back unchanged.
4. Checkpoint/resume: re-running the same output file adds zero new records.
5. Partial resume: pre-existing records for some pairs are skipped; remaining run.
6. Error handling: one method failure produces status="error" while siblings succeed.
7. Draft failure: all 5 methods fail with status="error" when generate_draft raises.
8. Record fields: correct method_id, status, final_answer populated for each method.
9. Evidence stored: M4 has evidence list; M5 has evidence and evidence_by_claim.
10. Usage tracking: non-empty usage lists for M2–M5; draft_usage on every record.
"""

import json
import os
import tempfile

import pytest

from src.runner.mocks import (
    MockClaimsPort,
    MockDraftPort,
    MockLLMClient,
    MockRetrievePort,
    MockVerificationPort,
    make_synthetic_example,
    make_synthetic_gold,
)
from src.runner.records import (
    ExperimentRecord,
    from_jsonl_line,
    load_completed,
    load_records,
    to_jsonl_line,
)
from src.runner.runner import ExperimentRunner

METHOD_IDS = ["M1", "M2", "M3", "M4", "M5"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ports():
    return {
        "draft_port": MockDraftPort(),
        "retrieve_port": MockRetrievePort(),
        "claims_port": MockClaimsPort(),
        "verification_port": MockVerificationPort(),
        "llm": MockLLMClient(),
    }


@pytest.fixture()
def runner(ports):
    return ExperimentRunner(**ports, top_k=3)


@pytest.fixture()
def three_examples():
    return [
        make_synthetic_example(f"e{i:02d}", f"Question {i}?")
        for i in range(3)
    ]


@pytest.fixture()
def output_file():
    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", mode="w", delete=False, encoding="utf-8"
    ) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# 1. Happy path: 15 records for 3 examples × 5 methods
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_record_count(self, runner, three_examples, output_file):
        records = runner.run(three_examples, output_file)
        assert len(records) == 15

    def test_all_method_ids_present(self, runner, three_examples, output_file):
        records = runner.run(three_examples, output_file)
        by_example = {}
        for r in records:
            by_example.setdefault(r.example_id, set()).add(r.method_id)
        for eid, methods in by_example.items():
            assert methods == set(METHOD_IDS), f"Missing methods for {eid}: {methods}"

    def test_all_status_ok(self, runner, three_examples, output_file):
        records = runner.run(three_examples, output_file)
        for rec in records:
            assert rec.status == "ok", f"{rec.method_id} / {rec.example_id}: {rec.error}"

    def test_example_ids_match(self, runner, three_examples, output_file):
        records = runner.run(three_examples, output_file)
        produced = {r.example_id for r in records}
        expected = {ex.example_id for ex in three_examples}
        assert produced == expected


# ---------------------------------------------------------------------------
# 2. Shared draft: generate_draft called once per example
# ---------------------------------------------------------------------------

class TestSharedDraft:
    def test_draft_generated_once_per_example(self, ports, three_examples, output_file):
        draft_port = ports["draft_port"]
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run(three_examples, output_file)
        assert draft_port.generate_draft_call_count == len(three_examples)

    def test_same_draft_id_across_methods(self, runner, output_file):
        example = make_synthetic_example("e99")
        runner.run([example], output_file)
        records = load_records(output_file)
        draft_ids = {r.draft_id for r in records}
        assert len(draft_ids) == 1, f"Expected one shared draft_id, got: {draft_ids}"


# ---------------------------------------------------------------------------
# 3. JSONL round-trip
# ---------------------------------------------------------------------------

class TestJsonlRoundTrip:
    def test_records_survive_round_trip(self, runner, three_examples, output_file):
        original = runner.run(three_examples, output_file)
        loaded = load_records(output_file)
        assert len(loaded) == len(original)
        for orig, load in zip(
            sorted(original, key=lambda r: (r.example_id, r.method_id)),
            sorted(loaded, key=lambda r: (r.example_id, r.method_id)),
        ):
            assert orig.example_id == load.example_id
            assert orig.method_id == load.method_id
            assert orig.status == load.status
            assert orig.final_answer == load.final_answer

    def test_every_line_is_valid_json(self, runner, three_examples, output_file):
        runner.run(three_examples, output_file)
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)  # must not raise


# ---------------------------------------------------------------------------
# 4. Checkpoint / resume: no duplicates on second run
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    def test_no_new_records_on_rerun(self, runner, three_examples, output_file):
        runner.run(three_examples, output_file)
        new = runner.run(three_examples, output_file)
        assert new == []

    def test_total_records_unchanged_after_rerun(self, runner, three_examples, output_file):
        runner.run(three_examples, output_file)
        runner.run(three_examples, output_file)
        records = load_records(output_file)
        assert len(records) == 15


# ---------------------------------------------------------------------------
# 5. Partial resume
# ---------------------------------------------------------------------------

class TestPartialResume:
    def test_only_missing_pairs_are_run(self, ports, output_file):
        example = make_synthetic_example("e_partial")
        runner = ExperimentRunner(**ports, top_k=3)

        # First pass: run only M1 and M2 by pre-populating the file.
        first_run = ExperimentRunner(**ports, top_k=3)
        first_run.run([example], output_file)
        all_records = load_records(output_file)
        assert len(all_records) == 5

        # Remove M3/M4/M5 records from the file to simulate partial run.
        remaining = [r for r in all_records if r.method_id in ("M1", "M2")]
        with open(output_file, "w", encoding="utf-8") as f:
            for r in remaining:
                f.write(to_jsonl_line(r) + "\n")

        # Second pass should only add M3, M4, M5.
        new_records = runner.run([example], output_file)
        assert {r.method_id for r in new_records} == {"M3", "M4", "M5"}
        assert len(load_records(output_file)) == 5


# ---------------------------------------------------------------------------
# 6. Error handling: one method fails, siblings still run
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_error_record_created_on_method_failure(self, ports, output_file):
        example = make_synthetic_example("e_err")

        class BrokenVerificationPort(MockVerificationPort):
            def verify_draft(self, *args, **kwargs):
                raise RuntimeError("M4 mock failure")

        ports["verification_port"] = BrokenVerificationPort()
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)

        records = {r.method_id: r for r in load_records(output_file)}
        assert records["M4"].status == "error"
        assert "M4 mock failure" in records["M4"].error
        for mid in ["M1", "M2", "M3", "M5"]:
            assert records[mid].status == "ok"

    def test_error_record_has_example_id_and_method_id(self, ports, output_file):
        example = make_synthetic_example("e_err2")

        class BrokenClaims(MockClaimsPort):
            def extract_claims(self, *args, **kwargs):
                raise ValueError("claims broke")

        ports["claims_port"] = BrokenClaims()
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)

        records = load_records(output_file)
        error_recs = [r for r in records if r.status == "error"]
        assert len(error_recs) == 1
        assert error_recs[0].method_id == "M5"
        assert error_recs[0].example_id == "e_err2"


# ---------------------------------------------------------------------------
# 7. Draft failure: all 5 methods get status="error"
# ---------------------------------------------------------------------------

class TestDraftFailure:
    def test_all_methods_error_when_draft_fails(self, ports, output_file):
        example = make_synthetic_example("e_draftfail")

        class BrokenDraftPort(MockDraftPort):
            def generate_draft(self, *args, **kwargs):
                raise RuntimeError("draft generation failed")

        ports["draft_port"] = BrokenDraftPort()
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)

        records = load_records(output_file)
        assert len(records) == 5
        for rec in records:
            assert rec.status == "error"
            assert "draft generation failed" in rec.error


# ---------------------------------------------------------------------------
# 8. Record field correctness per method
# ---------------------------------------------------------------------------

class TestRecordFields:
    @pytest.fixture(autouse=True)
    def _run(self, runner, output_file):
        example = make_synthetic_example("e_fields")
        runner.run([example], output_file)
        self.records = {r.method_id: r for r in load_records(output_file)}

    def test_m1_final_equals_draft(self):
        r = self.records["M1"]
        assert r.final_answer == r.short_answer
        assert r.final_explanation == r.explanation

    def test_m2_has_revised_answer(self):
        r = self.records["M2"]
        assert r.final_answer is not None
        assert r.feedback is not None
        assert r.feedback["feedback_id"].endswith(":self-reflect")

    def test_m3_has_feedback_with_null_labels(self):
        r = self.records["M3"]
        assert r.feedback is not None
        for item in r.feedback["items"]:
            assert item["label"] is None          # M3 critic must not invent verdicts

    def test_m4_has_evidence_list(self):
        r = self.records["M4"]
        assert r.evidence is not None
        assert len(r.evidence) > 0
        for ev in r.evidence:
            assert "doc_title" in ev
            assert "sentence_id" in ev

    def test_m5_has_claims_and_evidence_by_claim(self):
        r = self.records["M5"]
        assert r.claims is not None and len(r.claims) > 0
        assert r.evidence_by_claim is not None
        claim_ids = {c["claim_id"] for c in r.claims}
        assert set(r.evidence_by_claim.keys()) == claim_ids

    def test_all_records_have_draft_id(self):
        for mid, r in self.records.items():
            assert r.draft_id is not None, f"{mid} missing draft_id"

    def test_all_records_have_question(self):
        for mid, r in self.records.items():
            assert r.question == "What is the capital of France?", f"{mid} has wrong question"


# ---------------------------------------------------------------------------
# 9. Usage tracking
# ---------------------------------------------------------------------------

class TestUsageTracking:
    @pytest.fixture(autouse=True)
    def _run(self, runner, output_file):
        example = make_synthetic_example("e_usage")
        runner.run([example], output_file)
        self.records = {r.method_id: r for r in load_records(output_file)}

    def test_draft_usage_on_all_records(self):
        for mid, r in self.records.items():
            assert len(r.draft_usage) > 0, f"{mid} missing draft_usage"
            assert r.draft_usage[0]["stage"] == "generate_draft"

    def test_m1_has_no_method_specific_usage(self):
        # M1 makes no additional LLM calls beyond generate_draft
        assert self.records["M1"].usage == []

    def test_m2_usage_has_self_reflect_stage(self):
        stages = [u["stage"] for u in self.records["M2"].usage]
        assert "self_reflect" in stages

    def test_m3_usage_has_critique_and_revise_stages(self):
        stages = [u["stage"] for u in self.records["M3"].usage]
        assert "independent_critique" in stages
        assert "revise_answer" in stages

    def test_m4_usage_has_verify_and_revise_stages(self):
        stages = [u["stage"] for u in self.records["M4"].usage]
        assert "verify_draft" in stages
        assert "revise_answer" in stages

    def test_m5_usage_includes_extract_claims_and_verify(self):
        stages = [u["stage"] for u in self.records["M5"].usage]
        assert "extract_claims" in stages
        assert "verify_claims" in stages
        assert "revise_answer" in stages

    def test_usage_entries_have_required_keys(self):
        required = {"stage", "model", "input_tokens", "output_tokens",
                    "cost_usd", "latency_ms", "cache_hit"}
        for r in self.records.values():
            for entry in r.usage + r.draft_usage:
                assert required.issubset(entry.keys()), (
                    f"Usage entry missing keys: {required - entry.keys()}"
                )
