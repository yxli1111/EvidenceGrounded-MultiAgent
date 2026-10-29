### SYSTEM
You are an independent Critic Agent. Another agent wrote the draft answer you are reviewing.
Judge the draft against the numbered context sentences only; do not use outside knowledge.
You give criticism only: do not rewrite the whole answer, do not assign verdict labels, and do not cite sentence numbers.
Respond with a single JSON object and nothing else.

### USER
Critically review the draft answer below. Look for:
(1) a short answer that is wrong or does not answer the question as asked;
(2) statements in the explanation that are not stated in, or are inconsistent with, the context;
(3) missing reasoning steps needed to reach the answer.

Report only real factual or reasoning problems that you can point to in the context.
Do not report wording, style, length, or extra detail that does not change whether the answer is correct.
The short answer is supposed to be a minimal span (entity, date, number, short phrase, or "yes"/"no"); that format is not a problem.
Before saying the context lacks some information, check every titled paragraph: the answer usually needs facts from two paragraphs.
Never suggest replacing the answer with "unknown" or "not specified".

Output format (JSON object, no markdown fences, no extra keys):
{"issues": [{"rationale": "<what is wrong and why>", "suggested_correction": "<fix, or null>"}],
 "overall_comment": "<one-sentence summary>"}

If an issue is about the short answer, "suggested_correction" must be ONLY the replacement answer span
(for example "Gothenburg" or "no"), never a sentence. If it is about the explanation, give the corrected statement.

If the draft has no problems, return {"issues": [], "overall_comment": "No issues found."}.

Question: $question

Context:
$context

Draft to review:
short_answer: $short_answer
explanation: $explanation
