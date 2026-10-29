"""Offline tests for persistent run journal."""

import pytest

from src.core.run_journal import (
    RunJournal,
    JournalError,
    JournalBudgetExceeded,
    JournalUncertainState,
)


def make_journal(tmp_path):
    return RunJournal(
        path=tmp_path / "claims.csv.run.json",
        fingerprint="test-fingerprint",
        max_calls=6,
        max_cost_usd=0.02,
    )


def test_journal_creation(tmp_path):
    journal = make_journal(tmp_path)
    journal.create()

    assert journal.path.exists()
    assert journal.remaining_calls == 6
    assert journal.remaining_cost_usd == pytest.approx(0.02)


def test_journal_persists_in_progress(tmp_path):
    journal = make_journal(tmp_path)
    journal.create()
    journal.start("d1", reserved_calls=2, reserved_usd=0.005)

    restored = make_journal(tmp_path)
    restored.load()

    assert restored.get_status("d1") == "in_progress"

    with pytest.raises(JournalUncertainState):
        restored.assert_safe_to_resume()


def test_journal_successful_settlement(tmp_path):
    journal = make_journal(tmp_path)
    journal.create()
    journal.start("d1", reserved_calls=2, reserved_usd=0.005)

    journal.finish(
        draft_id="d1",
        status="success",
        claim_count=3,
        actual_calls=1,
        actual_cost_usd=0.001,
    )

    assert journal.get_status("d1") == "success"
    assert journal.data["calls"] == 1
    assert journal.data["spent_usd"] == pytest.approx(0.001)
    assert journal.data["uncertain_usd"] == pytest.approx(0.0)


def test_journal_budget_limit(tmp_path):
    journal = make_journal(tmp_path)
    journal.create()

    with pytest.raises(JournalBudgetExceeded):
        journal.start(
            "d1",
            reserved_calls=7,
            reserved_usd=0.005,
        )

    with pytest.raises(JournalBudgetExceeded):
        journal.start(
            "d1",
            reserved_calls=2,
            reserved_usd=0.03,
        )


def test_journal_rejects_duplicate_draft(tmp_path):
    journal = make_journal(tmp_path)
    journal.create()

    journal.start("d1", reserved_calls=1, reserved_usd=0.002)

    with pytest.raises(JournalError):
        journal.start("d1", reserved_calls=1, reserved_usd=0.002)


def test_journal_rejects_changed_fingerprint(tmp_path):
    journal = make_journal(tmp_path)
    journal.create()

    other = RunJournal(
        path=journal.path,
        fingerprint="changed",
        max_calls=6,
        max_cost_usd=0.02,
    )

    with pytest.raises(JournalError):
        other.load()


def test_journal_unknown_cost_stays_uncertain(tmp_path):
    journal = make_journal(tmp_path)
    journal.create()
    journal.start("d1", reserved_calls=1, reserved_usd=0.003)

    with pytest.raises(JournalUncertainState):
        journal.finish(
            draft_id="d1",
            status="error",
            claim_count=None,
            actual_calls=1,
            actual_cost_usd=None,
            error_type="NetworkError",
        )

    assert journal.get_status("d1") == "in_progress"
    assert journal.data["uncertain_usd"] == pytest.approx(0.003)


def test_journal_budget_persists_across_restart(tmp_path):
    journal = make_journal(tmp_path)
    journal.create()

    journal.start("d1", reserved_calls=2, reserved_usd=0.005)
    journal.finish(
        "d1",
        status="success",
        claim_count=2,
        actual_calls=2,
        actual_cost_usd=0.004,
    )

    restored = make_journal(tmp_path)
    restored.load()

    assert restored.remaining_calls == 4
    assert restored.remaining_cost_usd == pytest.approx(0.016)


def test_journal_rejects_corrupt_file(tmp_path):
    journal = make_journal(tmp_path)
    journal.path.write_text("{broken", encoding="utf-8")

    with pytest.raises(JournalError):
        journal.load()


def test_journal_over_budget_settlement_stays_uncertain(tmp_path):
    """Do not settle a request exceeding its remaining budget."""
    journal = make_journal(tmp_path)
    journal.create()

    journal.start(
        "d1",
        reserved_calls=1,
        reserved_usd=0.019,
    )

    with pytest.raises(JournalUncertainState):
        journal.finish(
            draft_id="d1",
            status="success",
            claim_count=2,
            actual_calls=1,
            actual_cost_usd=0.025,
        )

    restored = make_journal(tmp_path)
    restored.load()

    assert restored.get_status("d1") == "in_progress"

    with pytest.raises(JournalUncertainState):
        restored.assert_safe_to_resume()


def test_journal_rejects_invalid_actual_calls(tmp_path):
    """Actual call count cannot exceed reserved calls."""
    journal = make_journal(tmp_path)
    journal.create()

    journal.start(
        "d1",
        reserved_calls=1,
        reserved_usd=0.005,
    )

    with pytest.raises(ValueError, match="Invalid actual call count"):
        journal.finish(
            draft_id="d1",
            status="success",
            claim_count=1,
            actual_calls=2,
            actual_cost_usd=0.001,
        )

    assert journal.get_status("d1") == "in_progress"
