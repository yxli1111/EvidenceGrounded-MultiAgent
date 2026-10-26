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

## GAP-3: Dev/eval split manifest does not exist (B)

**Status:** Blocking for evaluation fairness.

`PROJECT_CONTRACT.md` Section 2: "B generates fixed-seed disjoint split manifests."
`PROJECT_CONTRACT.md` Section 3: `DatasetPort.load_split(split_name)`.

**Current state:** B's `load_hotpotqa(split, limit)` uses HotpotQA's built-in
splits only; no custom manifest file exists; `DatasetPort` Protocol not implemented.

**Request from B:**
1. Generate a fixed-seed split manifest (e.g. `data/splits/eval_ids.txt`) that
   separates dev examples (tuning allowed) from eval examples (tuning forbidden).
2. Implement `load_split(split_name: str)` that returns
   `(list[QAInput], dict[str, QAGold])` using the manifest.

**E's interim plan:** E's runner accepts a plain list of examples passed in
from outside; the split source is injected at call time.

---

## GAP-4: `LLMResponse` and `LLMUsage` schema (C)

**Status:** Needed for P3 cost tracking.

`PROJECT_CONTRACT.md` Section 2 describes usage fields but gives no Python class.

**Proposed schema (C to confirm or revise):**

```python
@dataclass
class LLMUsage:
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: float | None
    cache_hit: bool | None

@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage | None
    stage: str
    model: str
```

`usage` fields use `None` (not zero) when the provider does not report them.
Cache replay must not be counted as new paid usage.

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
