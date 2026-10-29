# Slide 14: Human annotation pilot

**Chart:** `A_annotation_confusion.png` (rows: adjudicated human label; columns: verifier prediction)

## Slide bullets
- **Sample:** 10 atomic claims from M5 on the dev split (run dev_v2, prompt set v2). Stratified with seed 42: 5 claims the verifier labelled non-SUPPORTED and 5 it labelled SUPPORTED, drawn from 204 claims that had evidence (only 12 were non-SUPPORTED). Order shuffled.
- **Labels (judged only against the evidence sentences shown):**
  - SUPPORTED: the evidence establishes everything the claim states.
  - CONTRADICTED: the evidence clearly conflicts with the claim.
  - UNSUPPORTED: insufficient evidence, including partial support.
- **Protocol:** two annotators (A, B) labelled independently and blind to the verifier output. The 2 disagreements were resolved by discussion. Rule clarified: the source title shown with an evidence sentence counts as evidence.
- **Inter-annotator agreement:** 8/10 raw; Cohen's κ = 0.59 (moderate).
- **Verifier vs adjudicated labels:** accuracy 7/10 (Cohen's κ 0.43). Per class F1: SUPPORTED 0.83, UNSUPPORTED 0.57. The verifier is **too conservative**: 2 of 7 supported claims were labelled UNSUPPORTED, and one UNSUPPORTED claim was over-called as CONTRADICTED.
- **Error patterns:**
  - P02: judged whether the *question* could be answered, not whether the *claim* is supported. "Annie Morton was born on October 8, 1970" is stated verbatim in E1 but was labelled UNSUPPORTED because Terry Richardson's birth date was missing.
  - P07: ignored the source title (`[I"s]`) that links the sentence to the series.
  - P09: entity confusion (Kenton Terry Richardson ≠ Terry Richardson). The label is CONTRADICTED but the verifier's own rationale says the evidence is insufficient.
- **Limitations:** n = 10, so all numbers are indicative only. The sample is stratified (50% non-SUPPORTED vs about 6% overall), so accuracy does not estimate the verifier's overall accuracy. No CONTRADICTED gold item was sampled, so 3-class macro-F1 (0.47) is not meaningful; macro-F1 over the observed classes is 0.70. Dev split, one model.

## Speaker notes (≈30 s)
"To check the verifier we labelled a small stratified sample of ten claims. The two annotators agreed on eight, a kappa of 0.59, and resolved the rest by discussion. Against these labels the verifier was right seven times out of ten. Its errors lean conservative: it called supported claims unsupported, once because it judged the whole question instead of the single claim. With only ten items this is a sanity check, not an accuracy estimate."

---

# Slide 13: material from A (revision behaviour, prompt v1 → v2)

**Chart:** `A_v1_v2_revision_fix.png` (dev set, 50 questions, identical shared drafts; only A's prompts differ)

## Bullets
- **Format failure in v1:** revisions rewrote a correct span answer as a sentence, which scores 0 on EM.
  - Example: "What major truck road is located in Backford Cross?" Draft *A41* (correct) became *"A41, which is a major truck road"*.
- **Fix (v2):** keep the short answer unless it is shown wrong; span-only corrections; no "unknown".
  - Sentence-form answers: M2 10% → 0%, M3 6% → 2%.
  - The critic flags a correct draft less often: M2 67% → 41%, M3 89% → 63%.
  - Cost per question 20–26% lower.
- **What remains:** content errors. The critic misreads the question or a comparison, for example "Who died first…?" flipped to the wrong person. Format fixes do not make ungrounded criticism reliable.
