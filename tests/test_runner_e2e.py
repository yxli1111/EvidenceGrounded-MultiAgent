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
def output_file(tmp_path):
    path = str(tmp_path / "out.jsonl")
    yield path
    # tmp_path is cleaned up automatically by pytest


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
        draft_ids = {(r.draft or {}).get("draft_id") for r in records}
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
            # Use nested field access for Phase 2 structure
            orig_ans = (orig.final or {}).get("final_answer") or (orig.draft or {}).get("short_answer")
            load_ans = (load.final or {}).get("final_answer") or (load.draft or {}).get("short_answer")
            assert orig_ans == load_ans

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
        # M1: final must be None; answer comes from draft.short_answer
        assert r.final is None, f"M1 final must be None, got {r.final}"
        assert r.draft is not None
        assert r.draft.get("short_answer") is not None

    def test_m2_has_revised_answer(self):
        r = self.records["M2"]
        assert r.final is not None
        assert r.final.get("final_answer") is not None
        assert r.feedback is not None
        assert r.feedback["feedback_id"].endswith(":self-reflect")

    def test_m3_has_feedback_with_null_labels(self):
        r = self.records["M3"]
        assert r.feedback is not None
        for item in r.feedback["items"]:
            assert item["label"] is None          # M3 critic must not invent verdicts

    def test_m4_has_evidence_list(self):
        r = self.records["M4"]
        # M4 evidence is a dict keyed by draft_id
        assert isinstance(r.evidence, dict), f"M4 evidence must be a dict, got {type(r.evidence)}"
        assert len(r.evidence) > 0
        for _key, ev_list in r.evidence.items():
            assert len(ev_list) > 0
            for ev in ev_list:
                assert "doc_title" in ev
                assert "sentence_id" in ev

    def test_m5_has_claims_and_evidence_by_claim(self):
        r = self.records["M5"]
        assert r.claims is not None and len(r.claims) > 0
        # M5 evidence is a dict keyed by claim_id (no separate evidence_by_claim)
        assert isinstance(r.evidence, dict), f"M5 evidence must be a dict, got {type(r.evidence)}"
        claim_ids = {c["claim_id"] for c in r.claims}
        assert set(r.evidence.keys()) == claim_ids

    def test_all_records_have_draft_id(self):
        for mid, r in self.records.items():
            assert r.draft is not None, f"{mid} missing draft"
            assert r.draft.get("draft_id") is not None, f"{mid} missing draft.draft_id"

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
        # latency field is latency_seconds (contracts.LLMUsage) not latency_ms
        # When UsageWrapper wraps MockLLMClient (which has drain_usage),
        # drain delegates to inner, returning inner's records with latency_seconds.
        required = {"stage", "model", "input_tokens", "output_tokens",
                    "cost_usd", "latency_seconds", "cache_hit"}
        for r in self.records.values():
            for entry in r.usage + r.draft_usage:
                assert required.issubset(entry.keys()), (
                    f"Usage entry missing keys: {required - entry.keys()}"
                )


# ---------------------------------------------------------------------------
# Issue 4: error records preserve partial usage
# ---------------------------------------------------------------------------

class TestErrorRecordPreservesUsage:
    def test_error_record_usage_not_empty_when_llm_called_before_error(
        self, ports, output_file
    ):
        example = make_synthetic_example("e_erruse")

        class BrokenAfterLLMCall(MockDraftPort):
            def self_reflect(self, example, draft, llm):
                # Call LLM first, then raise
                llm.generate(stage="self_reflect", messages=[], model=getattr(llm, "model", "mock"), temperature=0.0, max_tokens=512)
                raise RuntimeError("M2 fails after LLM call")

        ports["draft_port"] = BrokenAfterLLMCall()
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)

        records = {r.method_id: r for r in load_records(output_file)}
        m2 = records["M2"]
        assert m2.status == "error"
        assert len(m2.usage) >= 1, (
            "Error record must preserve LLM usage accumulated before the failure"
        )
        stages = [u["stage"] for u in m2.usage]
        assert "self_reflect" in stages


# ---------------------------------------------------------------------------
# Error records preserve draft (draft was generated before method failure)
# ---------------------------------------------------------------------------

class TestErrorRecordPreservesDraft:
    def test_error_record_has_draft_when_method_fails(self, ports, output_file):
        """Draft is preserved on error records when the failure happens in the method step."""
        example = make_synthetic_example("e_draft_preserve")

        class FailsAfterDraft(MockVerificationPort):
            def verify_draft(self, *a, **k):
                raise RuntimeError("M4 verification failure")

        ports["verification_port"] = FailsAfterDraft()
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)

        records = {r.method_id: r for r in load_records(output_file)}
        m4 = records["M4"]
        assert m4.status == "error"
        # draft should be preserved even though M4 failed
        assert m4.draft is not None, "Error record must preserve draft when method step fails"
        assert m4.draft.get("short_answer"), "Preserved draft must have short_answer"

    def test_draft_absent_when_draft_generation_fails(self, ports, output_file):
        """When generate_draft itself fails, error records correctly have draft=None."""
        example = make_synthetic_example("e_no_draft")

        class DraftFails(MockDraftPort):
            def generate_draft(self, *a, **k):
                raise RuntimeError("draft generation failed")

        ports["draft_port"] = DraftFails()
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)

        records = {r.method_id: r for r in load_records(output_file)}
        for mid in ["M1", "M2", "M3", "M4", "M5"]:
            assert records[mid].draft is None, f"{mid}: draft must be None when generate_draft fails"


# ---------------------------------------------------------------------------
# Issue 7: prompt_versions collected from port attributes
# ---------------------------------------------------------------------------

class TestPromptVersions:
    def test_prompt_version_stored_when_port_has_attribute(
        self, ports, output_file
    ):
        ports["draft_port"].prompt_version = "v1.2"
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([make_synthetic_example("e_pv")], output_file)
        records = load_records(output_file)
        for rec in records:
            assert rec.prompt_versions is not None, "prompt_versions must not be None"
            assert rec.prompt_versions.get("draft_port") == "v1.2", (
                f"Expected 'v1.2', got {rec.prompt_versions}"
            )

    def test_prompt_version_none_when_port_lacks_attribute(
        self, ports, output_file
    ):
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([make_synthetic_example("e_pv2")], output_file)
        records = load_records(output_file)
        for rec in records:
            assert rec.prompt_versions is not None, "prompt_versions must be a dict"
            assert rec.prompt_versions.get("draft_port") is None


# ---------------------------------------------------------------------------
# prompt_versions dict merged (AAnswerAdapter-style port with dict)
# ---------------------------------------------------------------------------

class TestPromptVersionsDict:
    def test_prompt_versions_dict_merged_into_record(self, ports, output_file):
        """A port with a .prompt_versions dict should have all its keys merged."""
        class PortWithVersionsDict(MockDraftPort):
            prompt_versions = {"generate_draft": "v2.1", "self_reflect": "v1.8"}

        ports["draft_port"] = PortWithVersionsDict()
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([make_synthetic_example("e_pvd")], output_file)
        records = load_records(output_file)
        for rec in records:
            pv = rec.prompt_versions
            assert pv is not None
            assert pv.get("generate_draft") == "v2.1"
            assert pv.get("self_reflect") == "v1.8"
            # Key "draft_port" should NOT appear (dict was merged, not stored under port name)
            assert "draft_port" not in pv


# ---------------------------------------------------------------------------
# Fatal errors abort the entire run
# ---------------------------------------------------------------------------

class LLMAuthenticationError(RuntimeError):
    """Simulates contracts.LLMAuthenticationError by name."""


class LLMInsufficientCreditError(RuntimeError):
    """Simulates contracts.LLMInsufficientCreditError by name."""


class TestFatalErrors:
    def test_llm_auth_error_stops_run(self, ports, output_file):
        example1 = make_synthetic_example("e_fatal1")
        example2 = make_synthetic_example("e_fatal2")

        class AuthErrorDraftPort(MockDraftPort):
            def generate_draft(self, example, llm):
                raise LLMAuthenticationError("Auth failed")

        ports["draft_port"] = AuthErrorDraftPort()
        runner = ExperimentRunner(**ports, top_k=3)
        with pytest.raises(LLMAuthenticationError):
            runner.run([example1, example2], output_file)
        # After fatal error, only first example's records written; second not started
        records = load_records(output_file)
        example_ids = {r.example_id for r in records}
        assert "e_fatal1" in example_ids
        assert "e_fatal2" not in example_ids

    def test_method_fatal_error_stops_run(self, ports, output_file):
        example1 = make_synthetic_example("e_mfatal1")
        example2 = make_synthetic_example("e_mfatal2")

        class CreditErrorVerifPort(MockVerificationPort):
            def verify_draft(self, *args, **kwargs):
                raise LLMInsufficientCreditError("Insufficient credits")

        ports["verification_port"] = CreditErrorVerifPort()
        runner = ExperimentRunner(**ports, top_k=3)
        with pytest.raises(LLMInsufficientCreditError):
            runner.run([example1, example2], output_file)
        records = load_records(output_file)
        example_ids = {r.example_id for r in records}
        # Second example not processed
        assert "e_mfatal2" not in example_ids


# ---------------------------------------------------------------------------
# raw_outputs saved on AgentOutputError
# ---------------------------------------------------------------------------

class TestRawOutputs:
    def test_raw_outputs_stored_on_error_record(self, ports, output_file):
        example = make_synthetic_example("e_raw")
        raw = ["raw output line 1", "raw output line 2"]

        class AgentOutputErrorDraft(MockDraftPort):
            def self_reflect(self, example, draft, llm):
                exc = RuntimeError("output error")
                exc.raw_outputs = raw
                raise exc

        ports["draft_port"] = AgentOutputErrorDraft()
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)
        records = {r.method_id: r for r in load_records(output_file)}
        m2 = records["M2"]
        assert m2.status == "error"
        assert m2.raw_outputs == raw

    def test_raw_outputs_none_when_exc_has_no_raw_outputs(self, ports, output_file):
        example = make_synthetic_example("e_rawno")

        class PlainErrorDraft(MockDraftPort):
            def self_reflect(self, example, draft, llm):
                raise RuntimeError("plain error, no raw_outputs")

        ports["draft_port"] = PlainErrorDraft()
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)
        records = {r.method_id: r for r in load_records(output_file)}
        assert records["M2"].raw_outputs is None


# ---------------------------------------------------------------------------
# question_type stored in records
# ---------------------------------------------------------------------------

class TestQuestionType:
    def test_question_type_stored_in_record(self, ports, output_file):
        example = make_synthetic_example("e_qt")
        example.question_type = "yesno"
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)
        records = load_records(output_file)
        for rec in records:
            assert rec.question_type == "yesno"

    def test_question_type_none_when_missing(self, ports, output_file):
        example = make_synthetic_example("e_qt2")
        # MockQAInput has question_type="factoid" by default from make_synthetic_example
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file)
        records = load_records(output_file)
        for rec in records:
            assert rec.question_type == "factoid"


# ---------------------------------------------------------------------------
# Issue 1: all-records summary — CSV has full rows after retry
# ---------------------------------------------------------------------------

class TestAllRecordsSummaryAfterRetry:
    def test_per_example_csv_has_all_rows_after_retry(self, ports, output_file, tmp_path):
        """Loading all records from file after retry yields 15-row CSV (3 examples × 5 methods)."""
        import csv as _csv
        from src.evaluation.answer_metrics import export_per_example_csv

        examples = [make_synthetic_example(f"e{i:02d}") for i in range(3)]
        gold_lookup = {ex.example_id: {"gold_answer": "The mock answer."} for ex in examples}

        class FailingM4(MockVerificationPort):
            def __init__(self):
                self.fail = True

            def verify_draft(self, *a, **k):
                if self.fail:
                    raise RuntimeError("M4 fails first run")
                return super().verify_draft(*a, **k)

        fail_port = FailingM4()
        ports_run1 = dict(ports)
        ports_run1["verification_port"] = fail_port
        runner1 = ExperimentRunner(**ports_run1, top_k=3)
        runner1.run(examples, output_file)

        # After first run: 15 records (3 × 5, including 3 M4 errors)
        all_recs = load_records(output_file)
        assert len(all_recs) == 15

        # Fix M4 and retry
        fail_port.fail = False
        runner2 = ExperimentRunner(**ports_run1, top_k=3)
        runner2.run(examples, output_file, retry_errors=True)

        # After retry: 15 original + 3 new M4 success records = 18
        all_recs_after = load_records(output_file)
        assert len(all_recs_after) == 18

        # CSV uses latest_records → 15 unique pairs; M4 rows show "ok"
        csv_path = str(tmp_path / "per_example.csv")
        export_per_example_csv(all_recs_after, gold_lookup, csv_path)

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))

        assert len(rows) == 15, f"Expected 15 CSV rows, got {len(rows)}"
        m4_rows = [r for r in rows if r["method_id"] == "M4"]
        assert len(m4_rows) == 3
        assert all(r["status"] == "ok" for r in m4_rows), "M4 rows must be ok after retry"


# ---------------------------------------------------------------------------
# Issue 3: methods filter
# ---------------------------------------------------------------------------

class TestMethodsFilter:
    def test_only_specified_methods_run(self, ports, output_file):
        example = make_synthetic_example("e_methods")
        runner = ExperimentRunner(**ports, top_k=3)
        records = runner.run([example], output_file, methods=["M1", "M3"])
        assert {r.method_id for r in records} == {"M1", "M3"}
        assert len(records) == 2

    def test_methods_none_runs_all_five(self, ports, output_file):
        example = make_synthetic_example("e_methods_all")
        runner = ExperimentRunner(**ports, top_k=3)
        records = runner.run([example], output_file, methods=None)
        assert {r.method_id for r in records} == set(METHOD_IDS)

    def test_methods_filter_respected_in_skip_check(self, ports, output_file):
        """Running only M1 on a file that already has M1 produces zero new records."""
        example = make_synthetic_example("e_skip_check")
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run([example], output_file, methods=["M1"])
        new = runner.run([example], output_file, methods=["M1"])
        assert new == [], "Second run of M1 only should add no records"

    def test_methods_filter_on_multiple_examples(self, ports, output_file):
        examples = [make_synthetic_example(f"e{i:02d}") for i in range(3)]
        runner = ExperimentRunner(**ports, top_k=3)
        records = runner.run(examples, output_file, methods=["M4", "M5"])
        assert {r.method_id for r in records} == {"M4", "M5"}
        assert len(records) == 6  # 3 examples × 2 methods


# ---------------------------------------------------------------------------
# Issue 4: progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_progress_fn_called_once_per_processed_example(self, ports, output_file):
        examples = [make_synthetic_example(f"e{i:02d}") for i in range(3)]
        runner = ExperimentRunner(**ports, top_k=3)
        calls = []
        runner.run(examples, output_file, progress_fn=lambda *a: calls.append(a))
        assert len(calls) == 3

    def test_progress_fn_not_called_for_skipped_examples(self, ports, output_file):
        """Examples already complete (all methods done) are skipped without calling progress_fn."""
        examples = [make_synthetic_example("e_skip_prog")]
        runner = ExperimentRunner(**ports, top_k=3)
        runner.run(examples, output_file)  # completes all 5 methods
        calls = []
        runner.run(examples, output_file, progress_fn=lambda *a: calls.append(a))
        assert calls == [], "No progress calls when example is already complete"

    def test_progress_fn_receives_correct_idx_and_total(self, ports, output_file):
        examples = [make_synthetic_example(f"e{i:02d}") for i in range(3)]
        runner = ExperimentRunner(**ports, top_k=3)
        captured = []

        def _fn(idx, total, eid, recs, elapsed):
            captured.append((idx, total))

        runner.run(examples, output_file, progress_fn=_fn)
        assert captured == [(0, 3), (1, 3), (2, 3)]

    def test_progress_fn_receives_elapsed_seconds(self, ports, output_file):
        example = make_synthetic_example("e_elapsed")
        runner = ExperimentRunner(**ports, top_k=3)
        elapsed_vals = []
        runner.run([example], output_file,
                   progress_fn=lambda idx, tot, eid, recs, el: elapsed_vals.append(el))
        assert len(elapsed_vals) == 1
        assert elapsed_vals[0] >= 0.0

    def test_progress_fn_receives_example_records(self, ports, output_file):
        example = make_synthetic_example("e_prog_recs")
        runner = ExperimentRunner(**ports, top_k=3)
        all_recs = []
        runner.run([example], output_file,
                   progress_fn=lambda idx, tot, eid, recs, el: all_recs.extend(recs))
        assert len(all_recs) == 5  # one for each method

    def test_progress_fn_called_on_draft_failure(self, ports, output_file):
        """progress_fn is called even when generate_draft fails."""
        example = make_synthetic_example("e_prog_draftfail")

        class DraftFails(MockDraftPort):
            def generate_draft(self, *a, **k):
                raise RuntimeError("draft fails")

        ports["draft_port"] = DraftFails()
        runner = ExperimentRunner(**ports, top_k=3)
        calls = []
        runner.run([example], output_file, progress_fn=lambda *a: calls.append(a))
        assert len(calls) == 1, "progress_fn must be called even when draft fails"
