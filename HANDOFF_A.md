# HANDOFF — Contributor A (baselines M1/M2 + independent critic M3)

Built against **contract v0.1** (`PROJECT_CONTRACT.md`, `contracts/schemas.py`, `contracts/interfaces.py`, all unchanged).
Status: S1 (mock-tested). **No real LLM run has been done and no empirical results exist.**

## Owned files in this PR

| Path | Purpose |
| --- | --- |
| `src/agents/answer.py` | `AgentConfig`, `generate_draft`, `AnswerAgent` (implements `AnswerPort`), prompt loading/versioning, context rendering, JSON parsing |
| `src/agents/baselines.py` | M1 `run_m1`; M2 `self_check` + `self_revise` (`self_reflect`, `run_m2`); M3 helper `run_m3` (critic → D `revise_answer`) |
| `src/agents/independent_critic.py` | M3 `independent_critique` returning feedback only (`VerificationFeedback`) |
| `src/agents/__init__.py` | empty package marker (kept import-free so D's `revision.py` stays independent) |
| `prompts/answer/draft_v1.md` | draft prompt |
| `prompts/baselines/{self_check,self_revise,independent_critic}_v1.md` | M2/M3 prompts |
| `research/A_baselines.ipynb` | offline mock demo + M1–M3 analysis that only runs once E's results exist |
| `tests/agents/test_A_agents.py`, `tests/agents/mocks_A.py` | acceptance tests + test-only mocks (`MockLLMClient`, `MockRevisionPort`, mock QA examples) |

Package markers `src/__init__.py`, `tests/__init__.py`, `tests/agents/__init__.py` are empty. If the repo already has them, keep the existing ones.

## How to run

```bash
# repo root, Python 3.11, standard library only (no API key, no network)
python -m unittest discover -s tests -t .      # or: python -m pytest tests/agents
```

## Minimal usage (what E's runner calls)

```python
from src.agents.answer import AnswerAgent, AgentConfig
agent = AnswerAgent(AgentConfig(model=cfg['model'], temperature=cfg['temperature'], max_tokens=cfg['max_tokens']))
draft = agent.generate_draft(example, llm)          # ONE shared draft per example, reuse for M1–M5
# M1: final output = draft (no RevisionResult)
m2_final = agent.self_reflect(example, draft, llm)  # RevisionResult, feedback_id = '<draft_id>:M2:self_reflection'
m3_fb = agent.independent_critique(example, draft, llm)   # VerificationFeedback, feedback_id = '<draft_id>:M3:critic'
m3_final = D.revise_answer(example, draft, m3_fb, llm)    # D's shared revision, called by E
record.prompt_versions.update(agent.prompt_versions)
```

To keep M2's self-check feedback in `ExperimentRecord.feedback`, E can call `baselines.run_m2(example, draft, llm, config)`. It returns `draft`, `feedback` and `final` together.

## Prompt v2 (2026-10-29), driven by the pilot run main_n100 (v1)
Pilot (n=100, gpt-4o-mini, prompt set v1). Scored with E's functions via `research/A_baselines_analysis.py`:
- EM M1 0.63 / M2 0.46 / M3 0.34 (errors = 0). Paired bootstrap CIs exclude 0 for M2-M1, M3-M1 and M3-M2.
- M2 correct→wrong 19 (12 format: final contains the gold answer but as a sentence; 3 replaced by "unknown"; 4 other). M3 correct→wrong 18 (10 format), plus 19 errors from D's v1 edit-provenance validation.
- Critics raise issues about equally often on correct and wrong drafts (M2 70% vs 70%, M3 67% vs 78%), so they barely discriminate.

Prompt set v2 (default):
- `self_revise_v2`: copy `short_answer` unless the review shows it is wrong; span-only format with examples; never replace an answer with "unknown".
- `self_check_v2` / `independent_critic_v2` (changed identically): only real factual or reasoning problems, no style; answer corrections must be the span only; check both paragraphs before claiming information is missing; never suggest "unknown".
- `answer.draft` stays **v1** (`draft_v1@1dd344355e`) in both sets, so cached shared drafts remain valid.
- `A_PROMPT_SET=v1` reproduces the pilot prompts exactly (same hashes) for a same-question dev comparison. The active versions are always logged in `prompt_versions`.
- Because the pilot informed v2, the 100 pilot questions should be reported as a pilot only; the frozen evaluation must use unseen questions.
- Windows test fix: the CRLF test double-converted files that were already CRLF. The production hash was always identical on Windows and Linux.
- All A entry points (`AnswerAgent` methods and module-level `run_m1/run_m2/run_m3/self_check/self_revise/independent_critique`) accept the contract dataclasses or the equivalent dicts.

## Analysis tooling
- `python research/A_baselines_analysis.py --run results/runs/<run>` writes `results/analysis/A_<run>/{summary.json, per_example_A.csv, cases.md}`.
- `research/A_baselines.ipynb` shows the same tables for several runs side by side.

## Integration additions (v1.1)

- `AnswerAgent` methods accept the contract dataclasses **or** the equivalent plain dicts. Today B's loader yields dict examples, and E's runner re-reads cached drafts as dicts. Use `as_qa_input` / `as_draft` in `src/agents/answer.py`.
- `AnswerAgent.prompt_version` (alias of `prompt_versions`) is the attribute E's runner reads.
- `AnswerAgent.self_reflect_with_feedback(example, draft, llm) -> (feedback, revision)`: M2 with the self-check kept (the same two calls). E can use it to fill `ExperimentRecord.feedback` for M2.
- `research/A_generate_dev_drafts.py`: dev-only draft generation (B loader + C client, budget-capped, resumable). The first run (validation[:50], openai/gpt-4o-mini, draft_v1) gave 50/50 ok for USD 0.0155.

### Offline integration rehearsal (E runner + real A + real D, scripted LLM)
- A: draft, M2 and M3 critique all work, including on resume with E's cached dict drafts.
- **E:** on resume the runner passes the cached draft to D's `revise_answer` as a dict, and D raises `'dict' object has no attribute 'draft_id'`. The runner should rebuild `AnswerDraft(**cached)` before calling ports. The runner's M2 record also hard-codes `items=[]`.
- **B / E:** with B-style dict examples, D raises `'dict' object has no attribute 'context'`. Either B's loader returns `QAInput`, or E converts before calling ports.
- **E / C:** E's mocks call `llm.generate(...)` positionally, but C's client and the contract are keyword-only. Real ports must call it with keywords.

## Behaviour guarantees (tested)

- `draft_id = f"{example_id}:draft"` (reproducible). Only `QAInput` is accepted, and no function takes `QAGold`.
- Context is rendered grouped by title with the original `(sentence_id)`, in loader order.
- **M1**: 0 extra calls, `final=None`.
- **M2**: exactly 2 calls (`A.M2.self_check`, `A.M2.self_revise`), exactly one self-revision, and the shared draft is not regenerated.
- **M3**: exactly 1 A call (`A.M3.independent_critic`) plus exactly 1 `revise_answer` call to D. A has no M3 revision prompt. Every feedback item has `scope='draft'`, `target_id=draft_id`, `label=None` and `evidence_ids=[]`, even if the model tries to emit labels or citations. `items=[]` means no actionable criticism.
- Output parsing tolerates ``` fences and surrounding prose, and maps yes/no booleans and numbers to strings. If parsing fails, A makes at most `max_parse_repairs` (default 1) repair calls on stage `<stage>.repair`. The repair request includes the bad reply, so it never collides with C's exact-request cache. If repair also fails, A raises `AgentOutputError(raw_outputs=[...])`. **E should record that as `status='error'` and keep `raw_outputs`.**
- Prompt versions look like `draft_v1@<sha256[:10]>` and are computed on LF-normalized text, so Windows CRLF checkouts produce the same hash.

## Contract review notes / change proposals (not applied — need team decision)

1. **Important — generation settings are not in the ports.** `LLMClient.generate` requires `model/temperature/max_tokens`, but the `AnswerPort` and `VerificationPort` methods don't carry them. A's adapter takes them in its constructor (`AgentConfig`), and E should build it from `configs/experiment.yaml`. Proposal: every owner does the same, so all conditions share one config source.
2. **Important — M2 feedback is not returned by `AnswerPort.self_reflect`.** The port returns only a `RevisionResult`, so `ExperimentRecord.feedback` for M2 would be empty. Proposal: E uses `baselines.run_m2`, or the team adds an optional `self_check` port method. A already exposes both.
3. **Important — who calls D for M3.** Plan v2 §4 says "for M3 invoke D's revision API", while the contract §4 says E orchestrates. The current code follows the contract: the port returns feedback only and E calls D. `run_m3` exists only for A's own tests and notebook, and it goes through the `VerificationPort` contract.
4. **Optional — stage names / cost.** A uses `A.draft`, `A.M2.self_check`, `A.M2.self_revise`, `A.M3.independent_critic`, plus `.repair` suffixes. Repair calls are real paid calls and should count in RQ3 cost. Please confirm the naming convention with C/E.
5. **Optional — draft cost attribution.** The shared draft is generated once and reused by M1–M5. E should decide whether the `A.draft` cost is added to every method's cost (recommended, so per-method totals are complete) and state that in the report.

## Known limitations

- Prompts are v1 and untuned. Tune them only on the dev split, then freeze them before S3.
- M2 uses A's own revision prompt, while M3 uses D's shared one. M2 vs M3 therefore changes both the critic and the reviser, and the report must state this.
- The critic gets the same full context as the drafter. M3 vs M4 differs in retrieved evidence **and** in the critic/verifier prompt (contract §5).
- The notebook's analysis section assumes E exports `results/evaluation/per_example.csv` with `example_id, method_id, status, em, f1` (+ `draft_em`, `question_type`). That format still needs to be confirmed with E.
