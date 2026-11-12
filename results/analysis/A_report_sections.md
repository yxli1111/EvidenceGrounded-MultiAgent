# Final report: Contributor A sections (draft)

Numbers in **[brackets]** are filled after the Experiment A runs. Everything else uses the frozen main_v2 results.

---

## Method: Answer agent and non-evidence baselines (M1–M3)

**Shared draft.** For every question, one answer agent (gpt-4o-mini, temperature 0) reads the question and the full HotpotQA distractor context and returns a JSON object with a minimal-span `short_answer` and a 1–4 sentence `explanation`. The draft is generated once and reused by M1–M5, so every condition starts from an identical answer and differences can be attributed to the checking step.

**M1 Direct answer.** The shared draft is the final answer; no further call is made.

**M2 Self-reflection.** The same agent reviews its own draft against the context (one call) and revises it once (one call), using its own revision prompt.

**M3 Independent critic.** A separate critic reviews the draft against the same context (one call) and returns feedback only. The critic may not assign verification labels or cite evidence; the code clears both even if the model emits them. The revision is performed by the shared revision function used by M4 and M5, with at most one round. M2 vs M3 therefore differs in both the critic and the reviser; M3 vs M4 differs mainly in retrieved evidence.

**Prompt revision (v1 → v2).** The pilot run showed that revisions often rewrote a correct span as a sentence (e.g. *A41* → *"A41, which is a major truck road"*), which scores zero on EM. Prompt set v2 requires the short answer to stay a minimal span unless it is shown to be wrong, restricts critics to factual or reasoning problems, and forbids replacing an answer with "unknown". The draft prompt was unchanged. On the 50 development questions with identical drafts, sentence-form answers fell from 10% to 0% (M2) and from 6% to 2% (M3), and the rate at which critics flagged a correct draft fell from 67% to 41% (M2) and from 89% to 63% (M3). All prompts are versioned by content hash and logged with every record.

## Results: RQ1, M1–M3 (main_v2, n = 100)

| Method | EM | F1 | ΔEM vs M1 [95% CI] | Correct→wrong | Wrong→correct |
|---|---|---|---|---|---|
| M1 Direct answer | 0.63 | 0.72 | — | — | — |
| M2 Self-reflection | 0.51 | 0.64 | −0.12 [−0.19, −0.05] | 14 | 2 |
| M3 Independent critic | 0.55 | 0.65 | −0.08 [−0.17, 0.00] | 13 | 5 |

Neither baseline improved on the unchanged draft. Self-reflection reduced EM by 12 points (the interval excludes zero); the independent critic reduced it by 8 points (the interval reaches zero). Revisions broke more correct answers than they fixed. After the v2 fix, most breaks were content changes (10 of 14 for M2, 10 of 13 for M3): the critic misread the question, reversed a comparison, or relied on outside knowledge. The critics barely discriminated between drafts that needed fixing and drafts that did not, raising an issue on 48% of correct vs 54% of wrong drafts (M2) and 57% vs 62% (M3). M3 was 0.04 EM above M2 [−0.03, +0.11], not a significant difference. Criticism that is not grounded in explicit evidence was therefore not a reliable error detector in this setting.

## Phase 2, Experiment A: answer–explanation consistency check (M1C)

**Motivation.** In main_v2, 15 of the 37 incorrect drafts contained the gold answer string in their own explanation (evaluation-only diagnostic). The typical pattern was an answer that stopped one hop early. For example, the question asks for a city's population and the explanation states "276,170 inhabitants", but the short answer is the city name. Unlike M2 and M3, which judged the answer against the context, this error can be detected from the draft alone.

**Design.** A checker reads only the question, the short answer and the explanation; it sees no context, no retrieved evidence and no gold answer. It returns *consistent / inconsistent*, a one-sentence rationale, and optionally a proposed short answer with the explanation sentence that states it (one call). Only the short answer may change; the explanation and the original draft are kept. An edit is applied only if a deterministic safety gate passes:

1. the checker flags an inconsistency and proposes a different answer (≤ 8 words);
2. the quoted sentence occurs verbatim in the explanation; and
3. the proposed answer occurs verbatim inside that sentence. For yes/no questions, the edit must flip yes ↔ no.

Every check is logged with its gate outcome, including flags that were not applied. M1 and M1C are compared on identical example IDs and drafts. Prompts were developed on the 50 development questions and frozen before evaluation on **[the fresh evaluation set, n = …]**. Uncertainty is reported as paired bootstrap 95% intervals (5,000 resamples over question IDs), and failures remain in the denominator.

**Results.** **[Table: M1 vs M1C EM/F1 with ΔEM CI; detection rate; edit rate; wrong→correct; correct→wrong; gate outcomes; flag rate on correct vs wrong drafts; manual precision of flags and edits (n reviewed); cost and latency per question.]**

**[Two cases: one fixed intermediate-hop answer, and one flag blocked by the gate or one harmful edit.]**

**Interpretation.** **[One paragraph: whether a narrow, explanation-grounded check improves accuracy without harmful revisions, compared with the broad critics of M2/M3.]**

## Limitations (A)

- The formal comparison uses 100 questions, fewer than the planned 300–500, so EM differences below about 0.08 are not resolvable.
- main_v2 reuses the pilot questions that informed prompt v2, so it is not a fresh held-out test for M2/M3. The development-set comparison with identical drafts points the same way.
- M2 and M3 differ in both critic and reviser.
- All results use one backbone (gpt-4o-mini); stronger critics may behave differently.
- Experiment A's headroom is bounded by how often the explanation states the right answer. It cannot fix drafts whose reasoning is wrong.
