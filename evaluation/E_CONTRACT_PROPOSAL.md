# E Contract Proposals — Pending Team Sign-off

Contributor E's interface needs that are not yet covered by `PROJECT_CONTRACT.md` v0.1
or that have ambiguities discovered during implementation.
Each item requires the named owner(s) to confirm before E wires real code.

---

## GAP-1: `title` vs `doc_title` key name asymmetry (B)

**Status:** Known, handled in E's code; no change needed from B.

B's `context` list items use key `"title"`.  
B's `Evidence` dicts use key `"doc_title"`.  
B's own `recall_at_k` in `research/B_retrieval_diagnostics.py` already handles
this asymmetry explicitly.

E's `retrieval_metrics.py` does the same:
- gold `_gold_set()` reads `item["title"]`
- retrieved `_predicted_set()` reads `item["doc_title"]`

**No code change needed from B.**  Documented here so the asymmetry is not
accidentally "fixed" by a future refactor.

---

## GAP-2: `contracts/schemas.py` does not exist yet (Team)

**Blocking:** D's `schema_adapter.py` does `from contracts.schemas import VerificationFeedback`
at runtime.  D's `verifier.py` also imports `QAInput`, `AnswerDraft`, `AtomicClaim`,
`Evidence`, `RevisionResult`.

**Request:** Someone (team-owned) must create `contracts/schemas.py` with these
dataclasses before E can wire real D calls.  Minimum required fields per
`PROJECT_CONTRACT.md` Section 2:

```python
@dataclass
class Sentence:
    title: str
    sentence_id: int
    text: str

@dataclass
class QAInput:
    example_id: str
    question: str
    context: list[Sentence]
    question_type: str = ""

@dataclass
class QAGold:
    example_id: str
    gold_answer: str
    gold_supporting_facts: list[dict]   # {title, sentence_id}

@dataclass
class AnswerDraft:
    draft_id: str
    short_answer: str
    explanation: str

@dataclass
class AtomicClaim:
    claim_id: str
    claim_text: str
    source_span: tuple[int, int] | None = None

@dataclass
class Evidence:
    doc_title: str
    sentence_id: int
    text: str
    score: float
    retrieval_query: str
    rank: int

@dataclass
class FeedbackItem:
    scope: str
    target_id: str
    rationale: str
    label: str | None
    evidence_ids: list[str]
    suggested_correction: str | None = None

@dataclass
class VerificationFeedback:
    feedback_id: str
    items: list[FeedbackItem]
    overall_comment: str

@dataclass
class RevisionResult:
    draft_id: str
    feedback_id: str
    final_answer: str
    final_explanation: str
```

Until `contracts/schemas.py` exists, E's runner uses a mock version in
`src/runner/mocks.py`.

---

## GAP-3: Dev/eval split manifest (B)

**Status:** Partially resolved.  Format confirmed; file pending from B.

`PROJECT_CONTRACT.md` Section 2: "B generates fixed-seed disjoint split manifests."

**Confirmed format:** `research/B_dev_50_ids.json` — a JSON `list[str]` of the 50
HotpotQA dev example IDs used during development (standard HotpotQA ID strings,
e.g. `"5a8b57f25542995d1e6f1371"`).

**Action required from B:** Commit `research/B_dev_50_ids.json` to the repo.

**E's enforcement:** `src/evaluation/manifest.py` provides:
```python
dev_ids = load_id_manifest("research/B_dev_50_ids.json")  # returns set[str]
assert_disjoint(dev_ids, eval_ids)                          # raises if overlap
runner.run(examples, out, exclude_ids=dev_ids)              # raises if overlap
```
`run()` raises `ValueError` listing all overlapping IDs — it does NOT skip
silently.  The caller must filter the example list before calling `run()`.

**M5 evidence mapping contract:**  `evidence_by_claim` passed to D's
`verify_claims()` is `dict[str, list]` keyed by `claim_id`.  Every claim_id
is present even when `m5_total_evidence_budget` limits evidence to 0 for some
claims (empty list, never missing key).  D must not KeyError on absent claim_ids.

**Retrieval parameter defaults (pending team decision):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 5 | Global fallback |
| `m4_top_k` | None (= top_k) | M4-specific override |
| `m5_top_k_per_claim` | None (= top_k) | M5 per-claim override |
| `m5_total_evidence_budget` | None (no limit) | Round-robin cap across M5 claims |
| `query_include_question` | False | Prepend question + newline to M4/M5 queries |

All retrieval params are logged in every `ExperimentRecord.model_settings["_retrieval"]`
for full reproducibility.  Team to confirm defaults before formal evaluation run.

---

## GAP-4: `LLMResponse` and `LLMUsage` schema (C) — CONFIRMED

**Status:** Resolved from `feat/contracts-v0.1`.

C's actual schema (confirmed from contracts branch):

```python
@dataclass
class LLMUsage:
    stage: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_seconds: float | None   # ← seconds, not milliseconds
    cache_hit: bool | None

@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage | None
    stage: str
    model: str
```

**Key difference from E's original proposal:** `latency_seconds` (not `latency_ms`).
E's `UsageWrapper` converts `latency_seconds → latency_ms` internally.
E's `cost_metrics._accumulate()` accepts both `latency_ms` and `latency_seconds`
(with `latency_seconds * 1000` fallback).
E's `ExperimentRecord.usage` entries now carry `latency_seconds` (matching C).

`usage` fields use `None` (not zero) when the provider does not report them.
Cache replay must not be counted as new paid usage.

---

## ExperimentRecord schema divergence — E's flat record vs contracts AnswerAgent output

E's `ExperimentRecord` is a flat JSONL-serialisable dict optimised for offline
metric computation.  `contracts.ExperimentRecord` (if defined) may use nested
dataclass objects.  **E's flat format is deliberately kept separate.**

| Field | E's ExperimentRecord | contracts / other branches |
|-------|---------------------|---------------------------|
| `question` | `str` (snapshot for debugging) | not in contracts |
| `draft_id` | `str \| None` | part of `AnswerDraft` |
| `claims` | `list[dict]` (serialised) | `list[AtomicClaim]` objects |
| `evidence` | `list[dict]` (serialised) | `list[Evidence]` objects |
| `feedback` | `dict` (serialised) | `VerificationFeedback` object |
| `usage` | `list[dict]` with `latency_seconds` | `list[LLMUsage]` objects |
| `question_type` | `str \| None` (P8 addition) | in `QAInput.question_type` |
| `raw_outputs` | `list[str] \| None` (P8 addition) | `AgentOutputError.raw_outputs` |

**Rationale for keeping E's format:** `json.dumps(dataclasses.asdict(rec))`
round-trips cleanly; nested dataclasses would require custom encoders.
Metric functions (`compute_em_f1_batch`, `mean_recall_at_k`, `aggregate_usage_by_method`)
operate on plain dicts — no imports of A/B/C/D packages needed at evaluation time.

---

## GAP-5: A's interface signatures (A)

**Status:** Needed for P2 M1/M2/M3 wiring.

None of A's functions (`generate_draft`, `self_reflect`, `independent_critique`)
exist in any branch yet.

**Proposed Protocol (A to confirm):**

```python
class DraftPort(Protocol):
    def generate_draft(self, example: QAInput, llm: LLMClient) -> AnswerDraft: ...
    def self_reflect(self, example: QAInput, draft: AnswerDraft,
                     llm: LLMClient) -> RevisionResult: ...
    def independent_critique(self, example: QAInput, draft: AnswerDraft,
                             llm: LLMClient) -> VerificationFeedback: ...
```

Confirmed from `PROJECT_CONTRACT.md` Section 2:
M3 independent critique must return `label=None, evidence_ids=[]` per item —
A must not invent evidence verdicts or citations.

---

## D's two-module structure (confirmed from feat/d-verification-revision)

D's public API is split across two Python modules:

| Module | Functions |
|--------|-----------|
| `src.verification` | `configure(**kwargs)`, `runtime_metadata()`, `verify_draft(...)`, `verify_claims(...)` |
| `src.agents.revision` | `revise_answer(example, draft, feedback, llm)` |

`DVerificationAdapter` in `src/runner/adapters.py` wraps both.  The constructor
takes `verification_module` and `revision_module` as separate arguments, calls
`configure()` once at construction, and exposes all four methods through a single
interface.  **Do not pass a single merged module** — the adapter specifically
routes `revise_answer` to `src.agents.revision`.

---

## GAP-6: D's `configure()` must be called once per experiment run (D → E)

**Status:** E will handle this; documenting for clarity.

`src/verification/config.py` uses a global singleton `_settings`.
`src.verification.configure(model=..., temperature=0.0, ...)` must be called
once before any `verify_draft` / `verify_claims` / `revise_answer` call.
Must not be called again between M3/M4/M5 within the same run.

E's runner will call `configure()` at experiment start and store the config
snapshot in every `ExperimentRecord`.

---

## GAP-7: Claim-conditioned Recall@k requires claim→gold mapping (B + C + E)

**Status:** Future work; not blocking P1/P2.

Claim-conditioned Recall@k (denominator = gold facts for one specific claim)
requires knowing which gold supporting facts are relevant to each `AtomicClaim`.
No such mapping exists in the current data format.

**Options:**
1. Skip claim-conditioned Recall@k for now; report only question-conditioned.
2. B/C/E jointly define a `claim_gold_mapping` field.

E will report question-conditioned Recall@k for the slides prototype and
note this limitation explicitly.
