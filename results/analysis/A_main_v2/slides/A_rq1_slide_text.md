# Slide 8a: RQ1 (part A). Do self-reflection and an independent critic help?

**Source:** run `main_v2` (E): 100 HotpotQA distractor questions, gpt-4o-mini, temperature 0, shared draft, 0 failures. EM/F1 are the official HotpotQA scores. CIs are paired bootstrap 95% (E's numbers). A prompt set v2.

## Headline table

| Method | EM | F1 | ΔEM vs M1 [95% CI] | Correct→wrong | Wrong→correct | Cost / question (method only) |
|---|---|---|---|---|---|---|
| M1 Direct answer | 0.63 | 0.72 | n/a | n/a | n/a | $0 |
| M2 Self-reflection | 0.51 | 0.64 | −0.12 [−0.19, −0.05] | 14 | 2 | $0.00064 |
| M3 Independent critic (+ shared revision) | 0.55 | 0.65 | −0.08 [−0.17, 0.00] | 13 | 5 | $0.00062 |

Charts: `A_rq1_delta_em.png` (ΔEM with CI) and `A_rq1_flips.png` (correct→wrong vs wrong→correct).

## Key observations (slide bullets)
- **Neither baseline improves on the draft.** Self-reflection lowers EM by 12 points (CI excludes 0). The independent critic lowers it by 8 (CI reaches 0).
- **Revisions mostly break correct answers.** M2 turns 14 correct answers into wrong ones and fixes only 2. M3 breaks 13 and fixes 5. Most of the breaks (10 of 14 for M2, 10 of 13 for M3) replace the answer with a different one; the rest are format changes.
- **Critics cannot tell which drafts need fixing.** They raise an issue on 48% of correct drafts vs 54% of wrong ones (M2), and 57% vs 62% (M3).
- **Independence helps only marginally.** M3 vs M2: +0.04 EM [−0.03, +0.11], not significant. M3 fixes more wrong drafts (5 vs 2).
- Takeaway for RQ1: criticism that is not grounded in explicit evidence is not a reliable error detector. This motivates M4/M5 (B and C slides).

## Example cases (pick one or two)
- **Critic misreads the question (correct → wrong).** "Who died first, George Archainbaud or Ralph Murphy?" Draft: *George Archainbaud* (correct). The critic claims the answer "should identify the person who died first, not George Archainbaud himself" and suggests *Ralph Murphy*. The revision follows it.
- **Critic catches a wrong bridge entity (wrong → correct).** "Alice David is the voice of Lara Croft in a video game developed by which company?" Draft: *Square Enix* (the publisher). The critic points to *Crystal Dynamics* (the developer), and the revision fixes it.

## Footnote / caveats (small text)
- n = 100 (proposal: 300–500), so differences smaller than about 0.08 EM are not resolvable.
- These are the same 100 questions as the pilot run. A's v2 prompts were revised after inspecting pilot failures, so M2/M3 here are not fully held-out. The dev-set check (50 unseen questions, same drafts) showed the same direction: v2 removed format errors, and EM stayed within noise of M1.
- M2 uses A's own revision prompt; M3 uses the shared revision (D), as M4/M5 do. So M2 vs M3 differs in both the critic and the reviser.

## Speaker notes (≈40 s)
"Our first question is whether checking the draft helps at all when no evidence is retrieved. Both baselines make things worse. Self-reflection drops exact match from 0.63 to 0.51, and an independent critic drops it to 0.55. Looking at individual questions, the revisions break about three times as many correct answers as they fix, because the critics flag correct and wrong drafts at almost the same rate. They cannot tell which drafts need fixing. This is the gap evidence grounding is meant to close, which B and C cover next."

---

# Material for slide 13 (revision error analysis): A's prompt v1 → v2 fix

Same 50 dev questions, identical shared drafts, only A's prompts differ (dev_A_v1 vs dev_A_v2):

| | M2 v1 → v2 | M3 v1 → v2 |
|---|---|---|
| Final answer written as a sentence | 10% → 0% | 6% → 2% |
| Format-type correct→wrong | 3 → 1 | 2 → 0 |
| Issue rate on correct drafts | 67% → 41% | 89% → 63% |
| Cost / question | $0.00067 → $0.00054 | $0.00080 → $0.00059 |

Example of the v1 failure: Q "What major truck road is located in Backford Cross?" Draft *A41* (correct) was revised to *"A41, which is a major truck road"*, which scores 0 on EM. v2 keeps the span unless the answer itself is wrong.
