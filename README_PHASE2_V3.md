# Phase 2: V3 Dual-Dimension Verification

## Purpose

The original V2 verifier evaluates whether statements in a draft are supported by retrieved evidence (`SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED`). This is distinct from whether the short answer supplies the information requested by the question. V3 adds an independent answer-sufficiency diagnostic without replacing V2 claim-support feedback.

## Components

- `src/verification/dual_dimension.py`: V3 answer-sufficiency diagnostic and combined verification interface.
- `prompts/verification/verify_sufficiency.txt`: V3 sufficiency instructions.
- `src/agents/revision_v3.py`: separate evidence-gated revision experiment (see `README_V3_REVISION.md`).
- `research/run_v3_diagnostic.py`: saved-M4 verifier-only diagnostic runner.
- `tests_d/test_phase2_dual_dimension.py` and `tests_d/test_phase2_sufficiency_prompt_contract.py`: offline checks.

V2 and V3 are separate experimental paths. Importing the V3 module does **not** automatically enable it in the existing experiment runner.

## Meaning of labels

- `COMPLETE`: the short answer supplies the information requested by the question. This does **not** establish factual correctness or evidence support.
- `INCOMPLETE`: the short answer omits an element needed to answer the question.
- `UNCERTAIN`: available evidence or the answer format does not permit a reliable completeness decision.

The revised prompt distinguishes the short answer from explanation grounding. An incorrect explanatory detail alone should not cause the short answer to be labeled incomplete. V2 support labels must not be replaced with V3 sufficiency labels.

## Running the diagnostic

Run commands from the repository root with project dependencies and an `OPENROUTER_API_KEY` configured locally. Do not commit credentials.

```powershell
python -m research.run_v3_diagnostic --dry-run
python -m research.run_v3_diagnostic --limit 100
```

The script uses saved M4 records and their existing drafts/retrieved evidence; it restores original HotpotQA validation context for provenance checks. It does not re-run retrieval or pass gold answers into the model. Confirm the exact CLI options with `python -m research.run_v3_diagnostic --help` before changing paths.

Outputs include diagnostic records (JSONL), a per-example CSV, and a summary JSON. The diagnostic is *verifier-only*; it does not demonstrate a change in final-answer EM/F1.

**Reproducibility warning:** the current resume/reuse logic does not reliably isolate predictions by prompt content/version. Separate output directories alone are not sufficient if prior predictions are imported for reuse. For a strict prompt comparison, disable reuse of earlier-version predictions and record the exact prompt hash, model, temperature, code commit, evidence, and draft. Keep API errors as failures, not as predicted labels.

## Observed exploratory results (not accuracy)

On 100 saved `main_v2` M4 cases, the initial V3 diagnostic returned 100/100 successful responses: 92 `COMPLETE`, 8 `INCOMPLETE`. A subsequent run after revising the sufficiency prompt reported 98 successful responses (85 `COMPLETE`, 13 `INCOMPLETE`) and 2 API connection failures; 10 of the 98 successful responses were reused from the earlier prompt run. These are **model-predicted label distributions**, not human-verified accuracy, and the two runs are **not a clean paired prompt-version comparison**.

A small AI-assisted case review suggested possible confusion between short-answer completeness, answer correctness, and explanation grounding. Any accuracy claim requires independent human review with a separately defined sufficiency annotation protocol.

## Tests and handoff

```powershell
python -m unittest discover -s tests_d -p "test*.py" -v
python -m pytest tests tests_d -q
```

The D module compatibility and V3 diagnostic tests are offline tests. Real API quality and end-to-end runner behavior require separate evaluation. E should explicitly integrate the V3 diagnostic as a versioned option, record added LLM calls/tokens/cost/latency, and keep the evidence-budget experiment on unchanged V2 verification.
