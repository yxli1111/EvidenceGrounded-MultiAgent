
import pytest
from types import SimpleNamespace

from contracts.schemas import AnswerDraft
from src.claims.extractor import AtomicClaimExtractor
from src.core.budget import BudgetExceededError


class BudgetFailureLLM:
    def __init__(self, fail_on_call=1):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def generate(self, *, stage, messages, model,
                 temperature, max_tokens):
        self.calls += 1

        if self.calls == self.fail_on_call:
            raise BudgetExceededError("Budget exhausted")

        return SimpleNamespace(text="not valid json")


def make_draft():
    return AnswerDraft(
        draft_id="budget-test",
        short_answer="Paris",
        explanation="Paris is the capital of France."
    )


def test_budget_error_initial_extraction():
    extractor = AtomicClaimExtractor()
    llm = BudgetFailureLLM(fail_on_call=1)

    with pytest.raises(BudgetExceededError):
        extractor.extract_claims(make_draft(), llm)

    assert llm.calls == 1


def test_budget_error_during_json_repair():
    extractor = AtomicClaimExtractor(
        max_repair_attempts=1
    )
    llm = BudgetFailureLLM(fail_on_call=2)

    with pytest.raises(BudgetExceededError):
        extractor.extract_claims(make_draft(), llm)

    assert llm.calls == 2
