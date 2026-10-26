# Contributor E — Evaluation & Runner Handoff

**DSA4213 Group 15 · Contributor E**

---

## What was built

| Path | Description |
|------|-------------|
| `src/evaluation/answer_metrics.py` | EM / token-F1 (HotpotQA normalization) |
| `src/evaluation/retrieval_metrics.py` | Recall@k with question- / claim-conditioned denominator |
| `src/evaluation/cost_metrics.py` | LLM usage aggregation, budget estimation |
| `src/evaluation/statistical.py` | Paired bootstrap confidence intervals |
| `src/runner/protocols.py` | `typing.Protocol` interfaces for A/B/C/D ports |
| `src/runner/records.py` | `ExperimentRecord` dataclass + JSONL serialization |
| `src/runner/runner.py` | `ExperimentRunner` orchestrating M1–M5 |
| `src/runner/mocks.py` | Drop-in mocks for offline testing |
| `configs/model_pricing.json` | Unit prices (USD/1k tokens) per model |
| `research/E_tradeoff.ipynb` | RQ3 end-to-end analysis notebook |
| `tests/test_evaluation_*.py` | 76 unit tests for evaluation modules |
| `tests/test_runner_e2e.py` | 28 end-to-end runner tests |

---

## Quick start

```bash
# Run all tests
python -m pytest tests/ -v

# Execute the RQ3 notebook
jupyter nbconvert --to notebook --execute --inplace research/E_tradeoff.ipynb
```

---

## Plugging in real components (for A / C / D)

`ExperimentRunner` accepts four port objects via its constructor.  Each port
must satisfy the corresponding `typing.Protocol` defined in
`src/runner/protocols.py`.

```python
from src.runner.runner import ExperimentRunner

runner = ExperimentRunner(
    draft_port=YourDraftPort(),          # implements DraftPort
    retrieve_port=YourRetrievePort(),    # implements RetrievePort
    claims_port=YourClaimsPort(),        # implements ClaimsPort
    verification_port=YourVerifPort(),   # implements VerificationPort
    llm=YourLLMClient(),                 # implements LLMClient
    top_k=5,
)

records = runner.run(
    examples,          # list of objects with .example_id, .question, .context
    "output.jsonl",    # checkpoint path — safe to resume if interrupted
    model_settings={"model": "openai/gpt-4o", "temperature": 0.0},
)
```

The runner writes one JSONL line per `(example_id, method_id)` pair.  If the
output file already exists, completed pairs are skipped (idempotent resume).

---

## Cache hit cost (Contributor C must confirm)

`cost_metrics.py` separates `cached_cost_sum` from `cost_usd_sum` so that
`per_example_cost_usd` and budget estimates reflect only **paid inference**.
This separation relies on C's LLM client marking cache replays with
`cache_hit=True` in the usage dict.

Per the OpenRouter contract, cache replays are not billed as new inference —
only the first call that populates the cache is charged.  If C's client marks
cache hits correctly, cost analysis will be accurate automatically.  If
`cache_hit` is absent or always `None`, all calls are counted as paid (the
safe over-estimate).

---

## LLMClient interface (Contributor C must confirm)

`ExperimentRunner` calls `llm.generate(stage, messages, model, temperature,
max_tokens)` for every LLM call.  After each method finishes, it calls
`llm.drain_usage()` which must return a list of usage dicts with these keys:

```python
{
    "stage":         str,    # e.g. "self_reflect", "verify_draft"
    "model":         str,    # model id used
    "input_tokens":  int,
    "output_tokens": int,
    "cost_usd":      float | None,
    "latency_ms":    float | None,
    "cache_hit":     bool | None,
}
```

If `drain_usage` is absent on the client, usage tracking is silently skipped
(no crash).  If `cost_usd` is `None`, the cost module will attempt to
re-derive it from `configs/model_pricing.json`; if the model is not in the
pricing table the cost stays `None` (never fabricated).

**GAP-4**: Please confirm the exact field names and types in `LLMResponse`
so that `MockLLMUsage` in `src/runner/mocks.py` can be updated to match.

---

## Data format requirements (Contributor B must provide)

### Dev/eval split manifest (GAP-3)

`ExperimentRunner.run()` accepts any iterable of objects with `.example_id`,
`.question`, and `.context` attributes (or dict equivalents).  B must
provide a split file that can be loaded into this format.  Suggested schema:

```json
{"example_id": "hotpot_dev_0", "question": "...", "context": [...]}
```

### Gold supporting facts key (GAP-1 — already handled)

B's data uses `"title"` for gold supporting facts (`gold_supporting_facts`
list).  E's retrieval metrics read `item["title"]` for gold and
`item["doc_title"]` for retrieved evidence.  This asymmetry is intentional
and documented in `evaluation/E_CONTRACT_PROPOSAL.md` (GAP-1).

---

## Claim-conditioned Recall@k (GAP-7)

`RecallDenominator.CLAIM_CONDITIONED` is implemented but requires a mapping
from each generated claim to its subset of gold supporting facts.  This
mapping is not available until B and C confirm claim generation schemas.
For now, all Recall@k results use `QUESTION_CONDITIONED`.

---

## Running on real data

1. Replace the mock setup in `research/E_tradeoff.ipynb` Section 1 with:
   ```python
   records = load_records("path/to/real_records.jsonl")
   ```
2. Build `GOLD_LOOKUP` from B's gold file:
   ```python
   GOLD_LOOKUP = {item["id"]: {"gold_answer": item["answer"],
                                "gold_supporting_facts": item["supporting_facts"]}
                  for item in gold_data}
   ```
3. Update `configs/model_pricing.json` with actual provider prices.
4. Re-run all cells — the analysis code is unchanged.

---

## Known gaps and owner table

| Gap | Description | Owner | Priority |
|-----|-------------|-------|----------|
| GAP-2 | `contracts/schemas.py` not yet created | Team | Critical |
| GAP-3 | Dev/eval split manifest missing | B | High |
| GAP-4 | `LLMResponse.usage` field names unconfirmed | C | High |
| GAP-5 | A's `DraftPort` method signatures unconfirmed | A | Medium |
| GAP-6 | D's `configure()` must be called once before experiments | D | Medium |
| GAP-7 | Claim→gold mapping needed for claim-conditioned Recall@k | B+C+E | Low |

---

## Commit history (this branch)

```
P1  src/evaluation/  — EM/F1 + Recall@k
P2  src/runner/      — Protocol, ExperimentRecord, M1–M5 runner
P3  cost_metrics.py  — usage aggregation, budget estimation
P4  research/E_tradeoff.ipynb — RQ3 analysis notebook
P5  statistical.py   — paired bootstrap CI
```
