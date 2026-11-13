# Phase 2 V3 diagnostic patch (no changes to existing V2)

Copy the two Python files and prompt into the corresponding paths in the project.

Run:

    python -m unittest tests_d.test_phase2_dual_dimension -v
    python -m unittest discover -s tests_d -p 'test*.py' -v

The new functions are `verify_answer_sufficiency`, `verify_dual_dimension`, and
`verify_draft_dual_dimension` in `src.verification.dual_dimension`.
Call `src.verification.configure(model=...)` as before.

`verify_dual_dimension` takes the same example, draft, claims, evidence mapping,
and LLM client as V2 `verify_claims`. It makes existing V2 per-claim calls PLUS
one additional question-level sufficiency call. `verify_draft_dual_dimension`
likewise adds a question-level call after V2 draft verification.

Store `result.to_dict()` as a **separate diagnostic JSONL**. Do not feed it to
V2 `revise_answer` or pretend it is the shared VerificationFeedback dataclass.
Do not change `contracts/schemas.py` or `src/runner/serialization.py` for this
isolated diagnostic. E must explicitly wire a V3 run path and record additional
LLM usage; no Runner integration is included in this patch.

For a fair experiment, use fixed draft, question, and evidence for V2/V3.
Review the predicted labels against human adjudication. Scripted tests only
check protocol, not model accuracy.

Caveat: if a claim has no evidence, V2 skips its LLM call. The separate
question-level sufficiency call uses the union of all claim evidence. If this
union is empty, it returns UNCERTAIN without an API call.
