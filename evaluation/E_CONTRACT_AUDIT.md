# E Code Audit — `PROJECT_CONTRACT.md` v0.1 Compliance
**Auditor:** Contributor E  
**Date:** 2026-10-28  
**Scope:** `src/runner/`, `src/evaluation/`, `src/runner/protocols.py`  
**References:** `PROJECT_CONTRACT.md` §2/§3/§4/§5/§6.2/§8/§9.3;  
`contracts/schemas.py` (origin/feat/contracts-v0.1);  
`contracts/interfaces.py` (origin/feat/contracts-v0.1)

Format per §8: **Critical / Important / Optional**, section ref, minimal fix.

---

## Critical Issues

### C-1 `ExperimentRecord.final` must be `None` for M1 (§2, §4)

**Location:** `src/runner/runner.py:_run_m1` (lines 400–401)

**Problem:** Contract (`contracts/schemas.py`) defines `final: RevisionResult | None = None` and
§4 says "M1: final output = draft; **no revision**." §2 says "M1 has **no RevisionResult**."
E currently sets `final_answer=dd["short_answer"]` and `final_explanation=dd["explanation"]`
for M1, which populates `final` with draft content. Downstream code that reads
`record.final is None` to detect M1 will misidentify M1 records.

**Minimal fix:** In `_run_m1`, set `final_answer=None`, `final_explanation=None`.
Metrics (`compute_em_f1_batch`, `export_per_example_csv`) must fall back to
`short_answer` when `final_answer is None` — add `final_answer or short_answer` logic there.

---

### C-2 `ExperimentRecord.evidence` structure mismatch (§2)

**Location:** `src/runner/records.py:ExperimentRecord` fields; `runner.py:_run_m4`, `_run_m5`

**Problem:** Contract's `ExperimentRecord.evidence` is `dict[str, list[Evidence]]` keyed by
query/claim ID (both M4 and M5 use the same field).
E uses two separate flat fields:
- `evidence: list[dict]` — flat deduplicated evidence list (M4 and M5 combined)  
- `evidence_by_claim: dict | None` — M5-only per-claim breakdown

Consumers that parse E's JSONL expecting `evidence[claim_id]` access will get `list`, not `dict`.

**Minimal fix (E-own-format option):** Keep E's flat format as-is but rename
`evidence_by_claim` → drop M4/M5 inconsistency by storing M4 evidence as
`evidence_by_claim = {"draft": [...]}` (keyed by `"draft"`) so the field is uniformly
`dict[str, list[dict]]` in every method.  Document in `E_CONTRACT_PROPOSAL.md` that
E uses this divergence deliberately for metric-code simplicity; update `ExperimentRecord`
docstring to explain the keying convention.  
*(Full alignment to contract nested type is Phase 2 item A.)*

---

### C-3 `ExperimentRecord.draft` stored as flat fields, not `AnswerDraft` object (§2)

**Location:** `src/runner/records.py:ExperimentRecord` (`draft_id`, `short_answer`, `explanation`)

**Problem:** Contract has `draft: AnswerDraft | None`; E stores three flat fields.
Any code that accesses `record.draft.draft_id` will fail; only `record.draft_id` works.

**Minimal fix (E-own-format option):** Acceptable for JSONL serialisation — `dataclasses.asdict`
of a nested `AnswerDraft` would produce the same three keys.  Explicitly document in
`E_CONTRACT_PROPOSAL.md` that E's flat fields are intentional (avoids custom encoder,
round-trips via `asdict` to the same JSON keys as nested structure).
Runner already re-constructs `AnswerDraft` objects via `_make_draft_obj` when passing to ports.
**No code change required** if documented; add a comment in `records.py` field declaration.

---

### C-4 `model_settings` key name conflicts with contract's `model_config` (§2)

**Location:** `src/runner/records.py:ExperimentRecord.model_settings`

**Problem:** Contract uses `model_config: dict`; E uses `model_settings: dict`.
Any code reading `record.model_config` from E's JSONL files will get `None` (key absent).

**Minimal fix:** Rename field `model_settings` → `model_config` in `ExperimentRecord` and
update all `records.py`, `runner.py`, test references. Add backward-compat
`data.setdefault("model_config", data.pop("model_settings", None))` in `from_jsonl_line`.

---

### C-5 No shared public serializer/deserializer (§2)

**Location:** `src/runner/runner.py` private helpers `_ser_draft`, `_ser_revision`, etc.

**Problem:** §2 states "implement one **shared** serializer/loader in E's runner (not five
independent variants)."  Serialisation helpers are private functions in `runner.py`, not
exported from a public module.  A/B/C/D cannot import them for integration testing or
replay without importing the whole runner.

**Minimal fix:** Move `_ser_draft`, `_ser_revision`, `_ser_feedback`, `_ser_evidence_list`,
`_ser_claims` into `src/runner/serialization.py` as public functions.  Import them in
`runner.py`.  No algorithm change needed.

---

## Important Issues

### I-1 `LLMClient.generate()` return type not used — `drain_usage()` not in contract (§2, §3)

**Location:** `src/runner/runner.py:_drain_usage`; `src/runner/protocols.py:LLMClient`

**Problem:** `contracts/interfaces.py` defines:
```python
class LLMClient(Protocol):
    def generate(self, *, stage, messages, model, temperature, max_tokens) -> LLMResponse: ...
```
No `drain_usage()` method is defined in the contract.  E's runner relies entirely on
`llm.drain_usage()` for usage collection and silently returns `[]` when the method is absent.
C's real client returns usage inside `LLMResponse.usage`, not via a separate drain call.

**Minimal fix:** Keep `drain_usage()` as E's optional extension (for `UsageWrapper`) but add
a secondary path: if `llm.drain_usage` is absent, runner should attempt to extract usage from
`LLMResponse` objects returned by each port call (ports could expose last-call usage, or runner
could ask C to add `drain_usage` as an agreed extension).
Document in `E_CONTRACT_PROPOSAL.md` that `drain_usage()` is a contract extension that C must
either implement or E must adapt.  This is currently a GAP — add as **GAP-8**.

---

### I-2 `LLMUsage.cache_hit` default is `False`, not `None` (§2)

**Location:** `src/evaluation/cost_metrics.py:_accumulate` (line 118: `if cache is True`)

**Problem:** Contract defines `cache_hit: bool = False` (always a bool, never None).
E's `_accumulate` already handles `cache is True` correctly; the issue is that mock/real
usage dicts may set `cache_hit=None` (not `False`), which E treats as a non-cached call.
This is the safe direction (over-counting paid calls) but creates a discrepancy with the
contract guarantee that `cache_hit` is always a bool.

**Minimal fix:** In `UsageWrapper` and `MockLLMClient.drain_usage`, ensure `cache_hit` is
always emitted as `False` (not `None`).  `_accumulate` already handles `None` safely; the
fix is upstream.

---

### I-3 `latency_ms` in stored records vs contract's `latency_seconds` (§2, GAP-4)

**Location:** `src/runner/usage_wrapper.py`; `src/runner/records.py:ExperimentRecord.usage`

**Problem:** Contract's `LLMUsage` field is `latency_seconds: float | None`.
E converts to `latency_ms` before storing in `ExperimentRecord.usage`.  Existing JSONL files
contain `latency_ms` keys; if another tool reads them expecting `latency_seconds` it will fail.

**Current status:** `cost_metrics._accumulate` has a `latency_seconds` fallback (reads either
key), which is correct.  `UsageWrapper` documents the conversion.  Documented in GAP-4 as
resolved.

**Minimal fix (documentation only):** Add explicit note in `E_CONTRACT_PROPOSAL.md` that
E's stored records carry `latency_ms` (not `latency_seconds`) by design; cross-branch tools
must handle both.  No code change needed; fallback already in place.

---

### I-4 `QAGold` isolation not enforced in `runner.run()` (§2)

**Location:** `src/runner/runner.py:run`

**Problem:** §2 says "**Never pass QAGold into A, C, D or B's retrieval inference.**"
`run()` accepts `exclude_ids` to enforce the dev/eval boundary but does not check that the
example objects passed to it contain no gold data (e.g. a QAInput accidentally merged with
a QAGold).  Contract requires E to enforce this boundary.

**Minimal fix:** Add a guard: if `examples` items have a `.gold_answer` attribute, raise
`ValueError` with a message explaining that QAGold must not be passed to `run()`.  Check
only `hasattr(example, "gold_answer")` — a one-line check per example.

---

### I-5 Evidence stored as flat deduplicated list in M5 — contract expects `dict` (§2, §4)

**Location:** `src/runner/runner.py:_run_m5` (lines 550–567)

**Problem:** M5 stores `evidence=all_ev_dicts` (flat deduplicated list) and
`evidence_by_claim=evidence_by_claim_dicts` (per-claim dict).  Contract uses a single
`evidence: dict[str, list[Evidence]]` keyed by claim_id.  When E's JSONL is read by other
branches expecting `record["evidence"]["claim_id"]`, they will get a `list` index error.

**Minimal fix:** Keep E's extra `evidence_by_claim` field for its own metric code.
For the contract-aligned `evidence` field, store as `evidence_by_claim_dicts` in M5
(dict keyed by claim_id) and as `{"draft": ev_dicts}` in M4 (single key).
Update `ExperimentRecord.evidence` type annotation to `dict | list | None` to reflect
the interim state.

---

### I-6 `_run_m5`: empty `claims` must still call `verify_claims` (§9.3)

**Location:** `src/runner/runner.py:_run_m5` (lines 515–546)

**Problem:** §9.3 table says `[]` from `extract_claims` means "no factual claims" and is a
valid success, **not** a reason to skip verification.  E's current code does call
`verify_claims(example, draft, [], {}, llm)` when `claims=[]`, which is correct.

**However:** the runner's outer `try/except` around `method_fn(...)` catches all exceptions
from `extract_claims` and turns them into `status="error"` records.  If C raises
`ClaimExtractionError` (defined in §9.3), this is correctly recorded as an error, not silently
swallowed.  **No code change needed here** — behaviour is contract-compliant.

**Action:** Add a comment in `_run_m5` explicitly noting that `claims=[]` is valid and the
call to `verify_claims` still runs.

---

### I-7 `results/` directory not the default output path (§1)

**Location:** `src/runner/runner.py:run` parameter `output_path`

**Problem:** §1 lists `results/` as the canonical location for "generated JSONL and analysis,
never hand-edit raw logs."  E's `run()` accepts any path; callers commonly use `output.jsonl`
in the working directory.

**Minimal fix:** Update `HANDOFF.md` quick-start example to use `results/run.jsonl`.
Optionally, assert `output_path` is inside `results/` or warn if not.

---

### I-8 Preliminary run labeling missing from exports (§5)

**Location:** `src/evaluation/answer_metrics.py:export_per_example_csv`

**Problem:** §5: "If only a small run is completed for slides, label it **preliminary** and
state sample size and omitted metrics explicitly."  E's CSV export and `E_tradeoff.ipynb`
notebook have no `preliminary` flag, sample size annotation, or omitted-metric list.

**Minimal fix:** Add a `preliminary: bool = True` parameter to `export_per_example_csv`
that writes a `# PRELIMINARY — n=<count> — omitted: <list>` header line to the CSV.
Update `E_tradeoff.ipynb` Section header cells to include the preliminary caveat.

---

## Optional Issues

### O-1 `status` field is `str` not a typed `Status` literal/enum (§2)

**Location:** `src/runner/records.py:ExperimentRecord.status`

Contract implies a `Status = Literal["ok","error","skipped"]` or enum type. E uses raw `str`.
No runtime impact (values match); static type checkers will not catch invalid status strings.
**Fix:** annotate as `Literal["ok","error","skipped"]`.

---

### O-2 `requirements.txt` not yet created; E owns coordination (§6.2)

**Location:** (missing file at repo root)

§6.2: "E coordinates one root `requirements.txt`."  No such file exists yet.
**Fix:** Create a draft `requirements.txt` with versions for all packages E imports
(`pytest`, `numpy`, `pandas`, etc.) and send to team for C (API deps) and B (embedding deps)
review.

---

### O-3 `configs/experiment.yaml` not consumed (§1, §7)

**Location:** `src/runner/runner.py`; `research/E_tradeoff.ipynb`

§1 names `configs/experiment.yaml` as the frozen model/sampling/top-k config (team-owned).
E currently takes `model_settings` as a plain dict passed at call time.  Once the YAML is
created (team action), E should load it and use it as the default `model_settings`.

**Fix:** After team freezes `configs/experiment.yaml`, add a
`load_experiment_config(path="configs/experiment.yaml") -> dict` helper in `src/runner/`.

---

### O-4 E's Protocol interfaces use `Any` throughout (§3)

**Location:** `src/runner/protocols.py`

All return types are `Any`.  `contracts/interfaces.py` uses concrete types from
`contracts/schemas.py`.  Once `contracts/schemas.py` is importable, replace `Any` with
`QAInput`, `AnswerDraft`, etc. for static-analysis benefit.

---

## Summary Table

| ID | Severity | Section | File(s) | Status |
|----|----------|---------|---------|--------|
| C-1 | **Critical** | §2, §4 | `runner.py:_run_m1` | M1 `final` must be None |
| C-2 | **Critical** | §2 | `records.py`, `runner.py` | `evidence` dict vs list |
| C-3 | **Critical** | §2 | `records.py` | flat draft fields — document as intentional |
| C-4 | **Critical** | §2 | `records.py` | `model_settings` → `model_config` rename |
| C-5 | **Critical** | §2 | `runner.py` | move serializers to `serialization.py` |
| I-1 | Important | §2, §3 | `runner.py`, `protocols.py` | `drain_usage` not in contract → GAP-8 |
| I-2 | Important | §2 | `cost_metrics.py`, `usage_wrapper.py` | `cache_hit=False` not `None` |
| I-3 | Important | §2 | `usage_wrapper.py` | latency field name — doc only |
| I-4 | Important | §2 | `runner.py:run` | QAGold isolation guard |
| I-5 | Important | §2, §4 | `runner.py:_run_m5` | flat evidence vs dict |
| I-6 | Important | §9.3 | `runner.py:_run_m5` | empty claims comment only |
| I-7 | Important | §1 | `runner.py` | default output to `results/` |
| I-8 | Important | §5 | `answer_metrics.py`, notebook | preliminary labeling |
| O-1 | Optional | §2 | `records.py` | `status` type annotation |
| O-2 | Optional | §6.2 | (missing) | `requirements.txt` |
| O-3 | Optional | §1 | `runner.py` | `configs/experiment.yaml` |
| O-4 | Optional | §3 | `protocols.py` | `Any` → concrete types |

---

## Items Already Compliant

- **GAP-1** (`title` vs `doc_title` asymmetry) — handled in `retrieval_metrics.py`; documented.
- **GAP-4** (`LLMUsage.latency_seconds`) — `UsageWrapper` stores `latency_seconds`; `_accumulate` prefers it.
- **§4 M3 common revision** — `_run_m3` calls `verification_port.revise_answer`; same code path as M4/M5. ✓
- **§2 draft reuse** — one draft generated per example, reused for M1–M5 via draft cache. ✓
- **§2 failed method not silently dropped** — error records written with `status="error"`, included in denominators. ✓
- **§5 same QA examples and shared draft** — enforced by runner design. ✓
- **§2 dev/eval ID isolation** — `manifest.py` provides `load_id_manifest` + `assert_disjoint` + `filter_out_ids` ✓
- **§2 QAGold object isolation** — `runner.run()` guards against QAInput+gold mix (Item C) ✓
- **§9.3 `[]` is valid for M5** — `_run_m5` calls `verify_claims` even with empty claims. ✓
- **§5 Recall@k denominator** — uses `(title,sentence_id)` tuples; aggregation method stated. ✓
- **GAP-3 B's manifest** — `load_id_manifest` supports both list and object formats. ✓
- **Retry/resume** — `attempt` field, `retry_errors`, `latest_records` all in place. ✓
