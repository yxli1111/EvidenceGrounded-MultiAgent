# Phase 2 V3 Revision Safety (opt-in experiment)

This is a separate experimental revision path. It does NOT change V2 `revise_answer`, shared contracts, or the default Runner.

API: `from src.agents.revision_v3 import revise_with_sufficiency_v3`

Call with `(example, draft, diagnostic, evidence, llm)` where `diagnostic` is a `DualDimensionResult` produced for the same example and draft and `evidence` is the SAME evidence supplied to the sufficiency diagnostic.

Decision: `V3RevisionDecision(revision, attempted, accepted, reason, candidate_citations, post_sufficiency, post_support)`. Save `decision.to_dict()` separately in JSONL. The `revision` field is the shared `RevisionResult` and uses the V2 feedback ID for compatibility.

Protocol: only INCOMPLETE with cited evidence can trigger a single candidate revision call. Candidate citations are checked against actual supplied evidence. Recheck the candidate using V3 sufficiency and V2 draft-level support. Accept only COMPLETE + SUPPORTED; otherwise return unchanged draft. Invalid JSON, citations or API failures propagate as errors, never silently become successes.

**Important**: these are LLM self-checks, NOT independent ground-truth verification. A model can approve its own incorrect answer. Real-case manual adjudication and EM/F1 comparison remain essential. The path may add up to THREE calls per attempted revision, plus the earlier diagnostic calls. E must log usage/cost/latency and treat these as a separate ablation condition, not mix into Evidence Budget runs.

Run: `python -m unittest tests_d.test_phase2_revision_safety -v`, then full tests.

Implementation note: the V3 revision prompt lives under `prompts/verification/` because the existing `_common.generate_json()` routes every stage other than `revise_answer` there. This avoids modifying shared V2 helper code; the stage name remains `revise_sufficiency_v3` for usage tracking.
