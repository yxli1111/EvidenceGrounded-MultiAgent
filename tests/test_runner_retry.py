"""Tests for retry_errors, attempt field, and latest_records (item 6).

Covers:
- attempt field starts at 1 for fresh records
- retry_errors=True re-runs (example_id, method_id) pairs whose latest attempt is "error"
- New attempt appended (not overwritten); old error records stay in the file
- latest_records() returns only the highest-attempt record per pair
- EM/F1 metrics use latest_records (no double-counting)
- Cost metrics include ALL attempts (retry waste visible)
- Fatal-error runs can be resumed with retry_errors=True after root cause fixed
"""

import json
import pytest

from src.runner.mocks import (
    MockClaimsPort,
    MockDraftPort,
    MockLLMClient,
    MockRetrievePort,
    MockVerificationPort,
    make_synthetic_example,
)
from src.runner.records import load_records
from src.runner.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(tmp_path, draft_port=None, verification_port=None, **kwargs):
    ports = dict(
        draft_port=draft_port or MockDraftPort(),
        retrieve_port=MockRetrievePort(),
        claims_port=MockClaimsPort(),
        verification_port=verification_port or MockVerificationPort(),
        llm=MockLLMClient(),
        top_k=2,
    )
    ports.update(kwargs)
    runner = ExperimentRunner(**ports)
    out = str(tmp_path / "out.jsonl")
    return runner, out


class _FailOnceThenSucceedVerificationPort(MockVerificationPort):
    """verify_claims raises on the first call, then succeeds."""

    def __init__(self):
        self._call_count = 0

    def verify_claims(self, example, draft, claims, evidence_by_claim, llm):
        self._call_count += 1
        if self._call_count == 1:
            raise RuntimeError("Transient M5 verification failure")
        return super().verify_claims(example, draft, claims, evidence_by_claim, llm)


class _AlwaysFailDraftPort(MockDraftPort):
    """generate_draft always raises on the first call, succeeds after."""
    def __init__(self):
        super().__init__()
        self._call_count = 0

    def generate_draft(self, example, llm):
        self._call_count += 1
        if self._call_count == 1:
            raise RuntimeError("Draft generation failed")
        return super().generate_draft(example, llm)


# ---------------------------------------------------------------------------
# attempt field
# ---------------------------------------------------------------------------

class TestAttemptField:

    def test_fresh_records_have_attempt_1(self, tmp_path):
        runner, out = _make_runner(tmp_path)
        runner.run([make_synthetic_example("e1")], out)
        for rec in load_records(out):
            assert rec.attempt == 1, f"{rec.method_id} has attempt={rec.attempt}"

    def test_attempt_field_survives_jsonl_roundtrip(self, tmp_path):
        runner, out = _make_runner(tmp_path)
        runner.run([make_synthetic_example("e1")], out)
        raw = [json.loads(line) for line in open(out) if line.strip()]
        for row in raw:
            assert "attempt" in row, "attempt field missing from JSONL"

    def test_old_records_without_attempt_load_as_1(self, tmp_path):
        """Records written before the attempt field was added default to 1."""
        from src.runner.records import from_jsonl_line, ExperimentRecord
        # Simulate old-format line without attempt field
        old_line = json.dumps({
            "example_id": "e1",
            "method_id": "M1",
            "status": "ok",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "question": "",
        })
        rec = from_jsonl_line(old_line)
        assert rec.attempt == 1


# ---------------------------------------------------------------------------
# latest_records
# ---------------------------------------------------------------------------

class TestLatestRecords:

    def test_returns_one_record_per_pair_when_no_retries(self, tmp_path):
        from src.runner.records import latest_records

        runner, out = _make_runner(tmp_path)
        runner.run([make_synthetic_example("e1")], out)
        all_recs = load_records(out)
        latest = latest_records(all_recs)
        assert len(latest) == len(all_recs)

    def test_returns_highest_attempt(self, tmp_path):
        from src.runner.records import ExperimentRecord, latest_records, _now_iso

        a1 = ExperimentRecord("e1", "M1", "error", _now_iso(), attempt=1)
        a2 = ExperimentRecord("e1", "M1", "ok",    _now_iso(), attempt=2)
        a3 = ExperimentRecord("e1", "M2", "ok",    _now_iso(), attempt=1)

        result = {(r.example_id, r.method_id): r for r in latest_records([a1, a2, a3])}
        assert result[("e1", "M1")].attempt == 2
        assert result[("e1", "M1")].status == "ok"
        assert result[("e1", "M2")].attempt == 1

    def test_latest_records_no_duplicates(self, tmp_path):
        from src.runner.records import ExperimentRecord, latest_records, _now_iso

        records = [
            ExperimentRecord("e1", "M1", "error", _now_iso(), attempt=1),
            ExperimentRecord("e1", "M1", "ok",    _now_iso(), attempt=2),
            ExperimentRecord("e1", "M1", "ok",    _now_iso(), attempt=3),
        ]
        result = latest_records(records)
        assert len(result) == 1
        assert result[0].attempt == 3


# ---------------------------------------------------------------------------
# retry_errors — basic behaviour
# ---------------------------------------------------------------------------

class TestRetryErrors:

    def test_error_record_is_retried(self, tmp_path):
        """M5 errors on first run; retry_errors=True re-runs it successfully."""
        port = _FailOnceThenSucceedVerificationPort()
        runner, out = _make_runner(tmp_path, verification_port=port)

        # First run: M5 errors
        runner.run([make_synthetic_example("e1")], out)
        records_first = {r.method_id: r for r in load_records(out)}
        assert records_first["M5"].status == "error"

        # Second run with retry_errors=True
        port2 = MockVerificationPort()
        runner2, _ = _make_runner(tmp_path, verification_port=port2)
        runner2.run([make_synthetic_example("e1")], out, retry_errors=True)

        all_recs = load_records(out)
        from src.runner.records import latest_records
        latest = {r.method_id: r for r in latest_records(all_recs)}
        assert latest["M5"].status == "ok", f"Expected M5 ok, got {latest['M5'].status}"

    def test_new_record_appended_not_overwritten(self, tmp_path):
        """Old error records are preserved; new attempt is appended."""
        port = _FailOnceThenSucceedVerificationPort()
        runner, out = _make_runner(tmp_path, verification_port=port)
        runner.run([make_synthetic_example("e1")], out)

        runner2, _ = _make_runner(tmp_path)
        runner2.run([make_synthetic_example("e1")], out, retry_errors=True)

        all_recs = load_records(out)
        m5_recs = [r for r in all_recs if r.method_id == "M5"]
        # Both the error record (attempt=1) and the ok record (attempt=2) must exist
        assert len(m5_recs) == 2, f"Expected 2 M5 records, got {len(m5_recs)}"
        attempts = sorted(r.attempt for r in m5_recs)
        assert attempts == [1, 2]

    def test_retry_attempt_number_incremented(self, tmp_path):
        """Retried record has attempt = previous max + 1."""
        port = _FailOnceThenSucceedVerificationPort()
        runner, out = _make_runner(tmp_path, verification_port=port)
        runner.run([make_synthetic_example("e1")], out)

        runner2, _ = _make_runner(tmp_path)
        runner2.run([make_synthetic_example("e1")], out, retry_errors=True)

        all_recs = load_records(out)
        from src.runner.records import latest_records
        latest = {r.method_id: r for r in latest_records(all_recs)}
        assert latest["M5"].attempt == 2

    def test_ok_record_not_retried(self, tmp_path):
        """Records with status=ok must not be re-run."""
        runner, out = _make_runner(tmp_path)
        runner.run([make_synthetic_example("e1")], out)

        draft_port = MockDraftPort()
        runner2, _ = _make_runner(tmp_path, draft_port=draft_port)
        runner2.run([make_synthetic_example("e1")], out, retry_errors=True)

        # Nothing was re-run: total records still 5 (one per method)
        assert draft_port.generate_draft_call_count == 0
        all_recs = load_records(out)
        assert len(all_recs) == 5

    def test_draft_failure_retried(self, tmp_path):
        """When draft generation fails, retry_errors=True re-generates the draft."""
        port = _AlwaysFailDraftPort()
        runner, out = _make_runner(tmp_path, draft_port=port)
        runner.run([make_synthetic_example("e1")], out)

        # All 5 methods should be errors
        records_first = load_records(out)
        assert all(r.status == "error" for r in records_first)

        # Retry: draft_port will now succeed (call_count > 1)
        runner2, _ = _make_runner(tmp_path, draft_port=port)
        runner2.run([make_synthetic_example("e1")], out, retry_errors=True)

        from src.runner.records import latest_records
        latest = latest_records(load_records(out))
        assert all(r.status == "ok" for r in latest), (
            f"Expected all ok after retry, got {[(r.method_id, r.status) for r in latest]}"
        )


# ---------------------------------------------------------------------------
# Metrics use latest_records; costs include all attempts
# ---------------------------------------------------------------------------

class TestMetricsWithRetry:

    def test_em_f1_on_latest_records_no_double_count(self, tmp_path):
        """EM/F1 computed on latest_records counts each example once."""
        from src.evaluation.answer_metrics import compute_em_f1_batch
        from src.runner.records import latest_records

        port = _FailOnceThenSucceedVerificationPort()
        runner, out = _make_runner(tmp_path, verification_port=port)
        runner.run([make_synthetic_example("e1")], out)

        runner2, _ = _make_runner(tmp_path)
        runner2.run([make_synthetic_example("e1")], out, retry_errors=True)

        all_recs = load_records(out)
        latest = latest_records(all_recs)

        # Only one M5 record in latest_records
        m5_latest = [r for r in latest if r.method_id == "M5"]
        assert len(m5_latest) == 1

        gold = {"e1": {"gold_answer": "The mock answer."}}
        result = compute_em_f1_batch([r.__dict__ for r in latest], gold)
        # M5 should appear once in result with n=1
        assert result["M5"]["n"] == 1

    def test_cost_includes_all_attempts(self, tmp_path):
        """aggregate_usage_by_method on ALL records includes retry cost."""
        from src.evaluation.cost_metrics import aggregate_usage_by_method

        port = _FailOnceThenSucceedVerificationPort()
        runner, out = _make_runner(tmp_path, verification_port=port)
        runner.run([make_synthetic_example("e1")], out)

        runner2, _ = _make_runner(tmp_path)
        runner2.run([make_synthetic_example("e1")], out, retry_errors=True)

        all_recs = load_records(out)

        # ALL records (including failed attempt=1) → higher total cost
        cost_all = aggregate_usage_by_method(all_recs)

        from src.runner.records import latest_records
        latest = latest_records(all_recs)
        cost_latest = aggregate_usage_by_method(latest)

        # M5 was run twice (attempt 1 errored + attempt 2 ok)
        # Total calls for M5 must be >= latest-only calls
        m5_all_calls = cost_all["M5"]["llm_calls"]
        m5_latest_calls = cost_latest["M5"]["llm_calls"]
        assert m5_all_calls >= m5_latest_calls, (
            "All-records cost must include retry waste"
        )


# ---------------------------------------------------------------------------
# Fatal error can be resumed with retry_errors=True
# ---------------------------------------------------------------------------

class TestFatalErrorRetry:

    def test_fatal_error_run_resumed_after_fix(self, tmp_path):
        """After a fatal-error interruption, retry_errors=True can resume."""
        class LLMInsufficientCreditError(RuntimeError):
            pass

        call_count = {"n": 0}

        class FailAfterTwoCallsLLM(MockLLMClient):
            def generate(self, **kwargs):
                call_count["n"] += 1
                if call_count["n"] > 3:
                    raise LLMInsufficientCreditError("out of credit")
                return super().generate(**kwargs)

        runner, out = _make_runner(tmp_path, llm=FailAfterTwoCallsLLM())
        with pytest.raises(LLMInsufficientCreditError):
            runner.run([make_synthetic_example("e1")], out)

        # Some records written before the crash
        records_before = load_records(out)
        assert len(records_before) > 0

        # After "refill" — fresh LLM that doesn't crash
        runner2, _ = _make_runner(tmp_path)
        runner2.run([make_synthetic_example("e1")], out, retry_errors=True)

        from src.runner.records import latest_records
        latest = latest_records(load_records(out))
        assert len(latest) == 5
        ok_count = sum(1 for r in latest if r.status == "ok")
        assert ok_count == 5, (
            f"Expected all 5 ok after retry, got {[(r.method_id, r.status) for r in latest]}"
        )
