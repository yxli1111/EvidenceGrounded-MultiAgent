
from types import SimpleNamespace

import pytest

from scripts.analyze_claim_distribution import (
    run_batch_journal,
    reconcile_journal_csv,
)
from src.core.run_journal import (
    RunJournal,
    JournalUncertainState,
)


class FakeExtractor:
    max_repair_attempts = 1

    def extract_claims(self, draft, llm):
        llm.generate(
            stage="claim_extraction",
            messages=[],
            model="fake",
            temperature=0,
            max_tokens=50,
        )
        return ["claim"]


class FakeLLM:
    def __init__(self, cost=0.001, fail=False):
        self.cost = cost
        self.fail = fail
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1

        if self.fail:
            raise RuntimeError("network error")

        return SimpleNamespace(
            text='{"claims": ["claim"]}',
            usage=SimpleNamespace(
                cost_usd=self.cost,
                cache_hit=False,
            ),
        )


def setup_case(tmp_path):
    output = tmp_path / "counts.csv"

    journal = RunJournal(
        tmp_path / "counts.journal.json",
        "fingerprint",
        6,
        0.02,
    )
    journal.create()

    draft = SimpleNamespace(
        draft_id="d1",
        explanation="A fact.",
        short_answer="A",
    )

    return output, journal, [draft]


def test_success_and_resume_do_not_repeat_calls(tmp_path):
    output, journal, drafts = setup_case(tmp_path)
    client = FakeLLM()

    rows = run_batch_journal(
        drafts, FakeExtractor(), client,
        output, 1, [], journal, 0.005,
    )

    assert rows[0]["claim_count"] == 1
    assert journal.summary()["calls"] == 1
    assert journal.summary()["spent_usd"] == pytest.approx(0.001)

    journal2 = RunJournal(
        journal.path, "fingerprint", 6, 0.02
    )
    journal2.load()

    rows2 = run_batch_journal(
        drafts, FakeExtractor(), client,
        output, 1, rows, journal2, 0.005,
    )

    assert rows2 == rows
    assert client.calls == 1


def test_crash_between_journal_and_csv_is_recoverable(tmp_path):
    output, journal, drafts = setup_case(tmp_path)

    run_batch_journal(
        drafts, FakeExtractor(), FakeLLM(),
        output, 1, [], journal, 0.005,
    )

    output.unlink()

    recovered = reconcile_journal_csv(
        journal, [], output
    )

    assert recovered[0]["draft_id"] == "d1"
    assert output.exists()


def test_unknown_cost_fails_closed(tmp_path):
    output, journal, drafts = setup_case(tmp_path)

    with pytest.raises(JournalUncertainState):
        run_batch_journal(
            drafts, FakeExtractor(),
            FakeLLM(cost=None),
            output, 1, [], journal, 0.005,
        )

    assert journal.get_status("d1") == "in_progress"

    with pytest.raises(JournalUncertainState):
        journal.assert_safe_to_resume()


def test_exception_fails_closed(tmp_path):
    output, journal, drafts = setup_case(tmp_path)

    with pytest.raises(RuntimeError):
        run_batch_journal(
            drafts, FakeExtractor(),
            FakeLLM(fail=True),
            output, 1, [], journal, 0.005,
        )

    assert journal.get_status("d1") == "in_progress"


def test_cost_over_reservation_fails_closed(tmp_path):
    output, journal, drafts = setup_case(tmp_path)

    with pytest.raises(JournalUncertainState):
        run_batch_journal(
            drafts, FakeExtractor(),
            FakeLLM(cost=0.01),
            output, 1, [], journal, 0.005,
        )

    assert journal.get_status("d1") == "in_progress"


def test_prepare_run_allows_missing_csv_for_journal(
    tmp_path,
):
    from scripts.analyze_claim_distribution import (
        prepare_run,
        metadata_path,
        save_metadata,
    )

    output = tmp_path / "claims.csv"
    fingerprint = "test-fingerprint"

    save_metadata(
        metadata_path(output),
        {
            "fingerprint": fingerprint,
            "format_version": 1,
        },
    )

    # Real-mode recovery can proceed without CSV.
    rows = prepare_run(
        output,
        fingerprint,
        resume=True,
        allow_missing_csv=True,
    )

    assert rows == []

    # Existing mock/CSV-only behavior remains strict.
    with pytest.raises(ValueError):
        prepare_run(
            output,
            fingerprint,
            resume=True,
        )
