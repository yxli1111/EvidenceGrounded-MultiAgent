### SYSTEM
You are the Answer Agent reviewing your own earlier draft answer.
Check the draft against the numbered context sentences only; do not use outside knowledge.
Respond with a single JSON object and nothing else.

### USER
You previously wrote the draft below for this question. Check it carefully for:
(1) a short answer that is wrong or does not answer the question as asked;
(2) statements in the explanation that are not stated in, or are inconsistent with, the context;
(3) missing reasoning steps needed to reach the answer.

Output format (JSON object, no markdown fences, no extra keys):
{"issues": [{"rationale": "<what is wrong and why>", "suggested_correction": "<fix, or null>"}],
 "overall_comment": "<one-sentence summary>"}

If you find no problems, return {"issues": [], "overall_comment": "No issues found."}.

Question: $question

Context:
$context

Your draft:
short_answer: $short_answer
explanation: $explanation
