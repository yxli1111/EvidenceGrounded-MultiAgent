"""Test that UsageWrapper writes raw_calls.jsonl with example_id and method_id."""

import json
import pathlib
import pytest

from src.runner.mocks import (
    MockClaimsPort,
    MockDraftPort,
    MockLLMClient,
    MockRetrievePort,
    MockVerificationPort,
    make_synthetic_example,
)
from src.runner.runner import ExperimentRunner


def _make_runner(tmp_path):
    runner = ExperimentRunner(
        draft_port=MockDraftPort(),
        retrieve_port=MockRetrievePort(),
        claims_port=MockClaimsPort(),
        verification_port=MockVerificationPort(),
        llm=MockLLMClient(),
        top_k=2,
    )
    out = str(tmp_path / "run.jsonl")
    return runner, out


class TestRawCallsLog:

    def test_raw_calls_file_created(self, tmp_path):
        runner, out = _make_runner(tmp_path)
        runner.run([make_synthetic_example("e1")], out)
        raw_path = tmp_path / "raw_calls.jsonl"
        assert raw_path.exists(), "raw_calls.jsonl must be created during a run"

    def test_m4_verify_draft_and_revise_answer_in_raw_calls(self, tmp_path):
        """M4 run must write verify_draft and revise_answer entries to raw_calls.jsonl."""
        runner, out = _make_runner(tmp_path)
        runner.run([make_synthetic_example("e1")], out)
        raw_path = tmp_path / "raw_calls.jsonl"
        entries = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        m4_entries = [e for e in entries if e.get("method_id") == "M4" and e.get("example_id") == "e1"]
        stages = {e["stage"] for e in m4_entries}
        assert "verify_draft" in stages, f"verify_draft not found in M4 raw calls; stages={stages}"
        assert "revise_answer" in stages, f"revise_answer not found in M4 raw calls; stages={stages}"

    def test_raw_calls_have_example_id_and_method_id(self, tmp_path):
        runner, out = _make_runner(tmp_path)
        runner.run([make_synthetic_example("e1")], out)
        raw_path = tmp_path / "raw_calls.jsonl"
        entries = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for e in entries:
            assert "example_id" in e, "every raw call entry must have example_id"
            assert "method_id" in e, "every raw call entry must have method_id"
            assert "stage" in e
            assert "messages" in e
            assert "response_text" in e

    def test_raw_calls_path_in_experiment_record(self, tmp_path):
        from src.runner.records import load_records
        runner, out = _make_runner(tmp_path)
        runner.run([make_synthetic_example("e1")], out)
        records = load_records(out)
        for rec in records:
            assert rec.raw_calls_path is not None, "raw_calls_path must be set in ExperimentRecord"

    def test_multiple_examples_in_raw_calls(self, tmp_path):
        """Multiple examples should all be written to the same raw_calls.jsonl."""
        runner, out = _make_runner(tmp_path)
        examples = [make_synthetic_example(f"e{i}") for i in range(3)]
        runner.run(examples, out)
        raw_path = tmp_path / "raw_calls.jsonl"
        entries = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        example_ids = {e["example_id"] for e in entries}
        assert example_ids == {"e0", "e1", "e2"}, f"Expected all 3 examples in raw_calls, got {example_ids}"

    def test_draft_stage_in_raw_calls(self, tmp_path):
        """generate_draft call should also appear in raw_calls.jsonl."""
        runner, out = _make_runner(tmp_path)
        runner.run([make_synthetic_example("e1")], out)
        raw_path = tmp_path / "raw_calls.jsonl"
        entries = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        draft_entries = [e for e in entries if e.get("stage") == "generate_draft"]
        assert len(draft_entries) >= 1, "generate_draft must appear in raw_calls.jsonl"
