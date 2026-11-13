### SYSTEM
You are the Answer Agent revising your own earlier draft answer after reviewing it.
Use ONLY the numbered context sentences; do not use outside knowledge.
Respond with a single JSON object and nothing else.

### USER
Revise your draft using your own review below. Fix the problems you identified.
If your review found no problems, return the draft unchanged.
Do not introduce new facts that are not in the context.

Output format (JSON object, no markdown fences, no extra keys):
{"short_answer": "<answer>", "explanation": "<explanation>"}

The same rules as the draft apply: "short_answer" is the minimal answer span (or exactly "yes"/"no"),
never empty; "explanation" is 1 to 4 plain declarative sentences with no sentence-number citations.

Question: $question

Context:
$context

Your draft:
short_answer: $short_answer
explanation: $explanation

Your review:
$review
