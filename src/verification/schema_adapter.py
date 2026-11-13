"""Construct team-owned dataclasses without defining a competing contracts package.

Only this boundary knows how feedback items are constructed. The Markdown contract
does not name their class, so use the annotation on VerificationFeedback.items.
Mock schemas are installed explicitly by tests/demo, never by production code.
"""

from dataclasses import is_dataclass
from typing import get_args, get_type_hints

from .errors import ConfigurationError


def feedback_schema():
    """Fail before a model call if the team's shared types are unavailable."""
    from contracts.schemas import VerificationFeedback

    try:
        item_args = get_args(get_type_hints(VerificationFeedback)["items"])
        item_type = item_args[0]
    except (KeyError, IndexError, NameError, TypeError) as exc:
        raise ConfigurationError(
            "VerificationFeedback.items must annotate a sequence of feedback-item "
            "dataclasses; align src/verification/schema_adapter.py with team schemas."
        ) from exc
    if not isinstance(item_type, type) or not is_dataclass(item_type):
        raise ConfigurationError("The shared feedback-item type must be a dataclass.")
    return VerificationFeedback, item_type


def make_feedback(feedback_id: str, items: list[dict], overall_comment: str):
    feedback_type, item_type = feedback_schema()
    item_objects = [item_type(**item) for item in items]
    return feedback_type(
        feedback_id=feedback_id, items=item_objects, overall_comment=overall_comment
    )


def make_revision(draft_id: str, feedback_id: str, final_answer: str, final_explanation: str):
    from contracts.schemas import RevisionResult

    return RevisionResult(
        draft_id=draft_id,
        feedback_id=feedback_id,
        final_answer=final_answer,
        final_explanation=final_explanation,
    )
