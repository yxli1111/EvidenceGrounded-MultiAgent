"""Tests for draft-caching behaviour in ExperimentRunner (Issue 1).

Rules under test:
1. If ALL 5 methods for an example are already complete, generate_draft must not
   be called again (whole example is skipped).
2. After a successful draft generation, the draft is persisted to drafts.jsonl in
   the same directory as the output JSONL.
3. On resume, the cached draft is loaded and reused; generate_draft is NOT called.
4. All methods (M1–M5) produced in the resume run share the same draft_id as the
   original M1 record.
"""

import dataclasses
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


def _make_runner(draft_port=None, **overrides):
    defaults = dict(
        draft_port=draft_port or MockDraftPort(),
        retrieve_port=MockRetrievePort(),
        claims_port=MockClaimsPort(),
        verification_port=MockVerificationPort(),
        llm=MockLLMClient(),
    )
    defaults.update(overrides)
    return ExperimentRunner(**defaults)


def _keep_only(jsonl_path: str, method_ids: set) -> None:
    """Rewrite JSONL keeping only records whose method_id is in method_ids."""
    records = load_records(jsonl_path)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            if r.method_id in method_ids:
                f.write(json.dumps(dataclasses.asdict(r)) + "\n")


# ---------------------------------------------------------------------------
# 1. Skip complete example
# ---------------------------------------------------------------------------

class TestSkipCompleteExample:
    def test_all_complete_does_not_call_generate_draft(self, tmp_path):
        """After a full run, re-running the same examples must not call generate_draft."""
        jsonl = str(tmp_path / "out.jsonl")
        draft_port = MockDraftPort()
        runner = _make_runner(draft_port=draft_port)
        runner.run([make_synthetic_example("e_skip")], jsonl)
        first_count = draft_port.generate_draft_call_count

        # Second run: all 5 methods complete → must skip entirely
        runner.run([make_synthetic_example("e_skip")], jsonl)
        assert draft_port.generate_draft_call_count == first_count, (
            "generate_draft was called even though all 5 methods are complete"
        )

    def test_partial_complete_still_calls_generate_draft(self, tmp_path):
        """If any method is incomplete, the runner must (re)use the draft."""
        jsonl = str(tmp_path / "out.jsonl")
        runner = _make_runner()
        runner.run([make_synthetic_example("e_part")], jsonl)

        _keep_only(jsonl, {"M1"})          # simulate partial run

        draft_port2 = MockDraftPort()
        runner2 = _make_runner(draft_port=draft_port2)
        runner2.run([make_synthetic_example("e_part")], jsonl)
        # Either cached draft is found (0 calls) or regenerated (≥1 call) — either way
        # M2-M5 must be present in the output.
        records = {r.method_id for r in load_records(jsonl)}
        assert records == {"M1", "M2", "M3", "M4", "M5"}


# ---------------------------------------------------------------------------
# 2. Draft persistence
# ---------------------------------------------------------------------------

class TestDraftPersistence:
    def test_drafts_jsonl_created(self, tmp_path):
        """drafts.jsonl must exist in the same directory as out.jsonl after a run."""
        jsonl = str(tmp_path / "out.jsonl")
        _make_runner().run([make_synthetic_example("e_persist")], jsonl)
        assert (tmp_path / "drafts.jsonl").exists(), (
            "drafts.jsonl was not created in the output directory"
        )

    def test_drafts_jsonl_has_required_fields(self, tmp_path):
        """Each entry in drafts.jsonl must contain the required keys."""
        jsonl = str(tmp_path / "out.jsonl")
        _make_runner().run([make_synthetic_example("e_fields")], jsonl)
        lines = [l for l in (tmp_path / "drafts.jsonl").read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        d = json.loads(lines[0])
        for key in ("example_id", "draft_id", "short_answer", "explanation", "draft_usage"):
            assert key in d, f"drafts.jsonl entry missing key '{key}'"

    def test_draft_usage_persisted(self, tmp_path):
        """draft_usage must be a non-empty list in drafts.jsonl."""
        jsonl = str(tmp_path / "out.jsonl")
        _make_runner().run([make_synthetic_example("e_du")], jsonl)
        d = json.loads((tmp_path / "drafts.jsonl").read_text("utf-8").strip())
        assert isinstance(d["draft_usage"], list)
        assert len(d["draft_usage"]) >= 1

    def test_one_draft_entry_per_example(self, tmp_path):
        """drafts.jsonl must have exactly one entry per unique example_id."""
        jsonl = str(tmp_path / "out.jsonl")
        examples = [make_synthetic_example(f"ex{i}") for i in range(3)]
        _make_runner().run(examples, jsonl)
        lines = [l for l in (tmp_path / "drafts.jsonl").read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# 3. Draft cache reuse on resume
# ---------------------------------------------------------------------------

class TestDraftCacheResume:
    def _setup_partial(self, tmp_path, eid="e_resume"):
        """Full run then trim to M1 only, keeping drafts.jsonl intact."""
        jsonl = str(tmp_path / "out.jsonl")
        _make_runner().run([make_synthetic_example(eid)], jsonl)
        _keep_only(jsonl, {"M1"})
        return jsonl

    def test_resume_does_not_call_generate_draft(self, tmp_path):
        """After a partial run with cached draft, generate_draft must NOT be called."""
        jsonl = self._setup_partial(tmp_path)
        draft_port = MockDraftPort()
        runner = _make_runner(draft_port=draft_port)
        runner.run([make_synthetic_example("e_resume")], jsonl)
        assert draft_port.generate_draft_call_count == 0, (
            "generate_draft was called despite a valid cached draft"
        )

    def test_all_methods_share_same_draft_id_after_resume(self, tmp_path):
        """M1 (from first run) and M2–M5 (from resume) must share the same draft_id."""
        jsonl = self._setup_partial(tmp_path, eid="e_shared")
        _make_runner().run([make_synthetic_example("e_shared")], jsonl)
        records = load_records(jsonl)
        # Phase 2: draft_id is nested in rec.draft dict
        draft_ids = {(r.draft or {}).get("draft_id") for r in records}
        assert len(draft_ids) == 1, (
            f"Multiple draft_ids after resume: {draft_ids}"
        )

    def test_resumed_records_have_draft_usage_from_cache(self, tmp_path):
        """Records created during resume must have the cached draft_usage, not empty."""
        jsonl = self._setup_partial(tmp_path, eid="e_du_resume")
        _make_runner().run([make_synthetic_example("e_du_resume")], jsonl)
        records = load_records(jsonl)
        for rec in records:
            assert len(rec.draft_usage) >= 1, (
                f"{rec.method_id} record has empty draft_usage after resume"
            )
