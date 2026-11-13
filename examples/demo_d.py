"""Run D's real functions with explicit mocks for A/B/C and shared schemas.

From the package root: python -m examples.demo_d --output demo_result.json
No network, model credentials, retrieval model or third-party libraries are used.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from src.agents.revision import revise_answer
from src.verification import configure, runtime_metadata, verify_claims, verify_draft
from tests_d.support import (
    FeedbackItem, ScriptedLLM, VerificationFeedback, mock_contracts, sample_inputs,
)


def _verdict(target_id):
    return {
        "items": [{
            "target_id": target_id,
            "label": "CONTRADICTED",
            "rationale": "Ada was born in London; the second sentence locates London in England.",
            "evidence_ids": ["Ada Lovelace::0", "London::0"],
            "suggested_correction": "Replace France with England and Paris with London.",
        }],
        "overall_comment": "The retrieved sentences contradict the draft's birthplace.",
    }


def _revision(target_id):
    return {
        "final_answer": "England",
        "final_explanation": "Ada Lovelace was born in London, which is in England.",
        "edits": [{
            "field": field,
            "feedback_target_ids": [target_id],
            "reason": "Use the supported London-to-England relationship instead of Paris/France.",
        } for field in ("final_answer", "final_explanation")],
    }


def run_demo() -> dict:
    with mock_contracts():
        configure(model="offline-scripted-mock")
        # These fixed fixtures stand in for A's draft, C's claims and B's evidence.
        example, draft, claims, evidence = sample_inputs()
        records = []
        for method in ("M3", "M4", "M5"):
            target_id = claims[0].claim_id if method == "M5" else draft.draft_id
            replies = [_revision(target_id)]
            if method != "M3":
                replies.insert(0, _verdict(target_id))
            llm = ScriptedLLM(*replies)
            if method == "M3":
                feedback = VerificationFeedback(
                    "example-1:mock-independent-critic",
                    [FeedbackItem(
                        scope="draft", target_id=draft.draft_id,
                        rationale="Check whether the context says London rather than Paris.",
                        label=None, evidence_ids=[],
                        suggested_correction="Use the country established by the context.",
                    )],
                    "Mock A independent criticism; no retrieved evidence verdicts.",
                )
            elif method == "M4":
                feedback = verify_draft(example, draft, evidence, llm)
            else:
                feedback = verify_claims(example, draft, claims, {claims[0].claim_id: evidence}, llm)
            result = revise_answer(example, draft, feedback, llm)
            records.append({
                "method_id": method,
                "draft": asdict(draft),
                "feedback": asdict(feedback),
                "final": asdict(result),
                "llm_requests": llm.calls,
                "scripted_llm_responses": replies,
            })
        return {
            "mode": "offline_scripted_mock",
            "notice": "Fixed synthetic example and scripted replies. These are not research results or model-quality measurements.",
            "configuration": runtime_metadata(),
            "example": asdict(example),
            "records": records,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for the synthetic demo JSON")
    args = parser.parse_args()
    result = run_demo()
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Saved offline mock demonstration to {args.output}")
        for record in result["records"]:
            print(f"{record['method_id']}: {record['draft']['short_answer']} -> {record['final']['final_answer']}; "
                  f"{len(record['llm_requests'])} scripted calls")
    else:
        print(output)


if __name__ == "__main__":
    main()
