# Phase 2: V3 Evidence-Gated Revision Safety

## Scope

`src/agents/revision_v3.py` is an **experimental, separate** revision path for V3 answer-sufficiency diagnostics. It does not replace `src/agents/revision.py` or automatically alter M3/M4/M5 runs. Its prompt is `prompts/verification/revise_sufficiency_v3.txt`.

## Intended decision flow

1. Receive the original question/draft, retrieved evidence, and V3 sufficiency diagnostic.
2. Preserve the original draft for `COMPLETE` or `UNCERTAIN`, and when there is no usable cited evidence for a proposed correction.
3. For actionable `INCOMPLETE` cases, request a candidate answer grounded in supplied evidence; reject invalid citations.
4. Re-check candidate answer sufficiency and draft-level support.
5. Accept a changed candidate only if post-checks satisfy the configured safety conditions; otherwise retain the original draft.

This is an **evidence-gated policy**, not a guarantee of correctness. LLM generation and LLM post-checks can share mistakes. In particular, an `INCOMPLETE` label does not itself justify inventing missing facts. Preserve the original answer, candidate, citations, checks, accept/reject decision, and reason for audit.

## Tests

```powershell
python -m unittest tests_d.test_phase2_revision_safety -v
python -m unittest discover -s tests_d -p "test*.py" -v
python -m pytest tests tests_d -q
```

Offline tests cover skip/rollback paths, invalid citations, and acceptance when simulated post-checks pass. Passing tests verifies interface and policy behavior under scripted outputs, **not** real-model factual improvement.

## Runner integration (E)

The existing runner does not automatically call V3 revision. Integrate it explicitly behind a version/configuration switch, leaving V2 as the default for existing experiments. First run a verifier-only comparison on identical questions, drafts, and retrieved evidence. Only then run a revision-enabled comparison with the same inputs and model settings.

Log: input question/draft/evidence, V2 support feedback, V3 sufficiency and missing-information fields, candidate and final answer, citations, post-check outcomes, acceptance/rollback, EM/F1, wrong-to-right and right-to-wrong transitions, all LLM calls/tokens/cost, latency, prompt hash, and code commit. Include additional V3 calls in budget and latency totals.

Do not interpret the 100-question verifier-only diagnostic as evidence of improved final EM/F1; the real revision-enabled ablation has to be run and evaluated separately. Keep prompt versions and caches isolated, and retain API failures rather than silently treating them as correct outcomes.
