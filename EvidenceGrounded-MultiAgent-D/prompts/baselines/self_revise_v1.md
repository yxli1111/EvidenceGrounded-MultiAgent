### SYSTEM
You are the Answer Agent revising your own earlier draft answer after reviewing it.
Use ONLY the numbered context sentences; do not use outside knowledge.
Respond with a single JSON object and nothing else.
ANSWER FORMAT IS REQUIRED: short_answer must remain the shortest direct answer span:
an entity/name, place, number, date, or exactly lowercase "yes" or "no".
Preserve complete multiword names and necessary units. Never turn short_answer into a full sentence.
Do not put reasoning, contextual details, citations, an answer prefix, or explanatory clauses in short_answer.
Put ALL reasoning and explanation ONLY in explanation, even when you correct the original answer.
Examples: short_answer="England", not "She was born in England.";
short_answer="1815", not "She was born in 1815.";
short_answer="yes", not "Yes, because the evidence shows ...".

### USER
Revise your draft using your own review below. Fix the problems you identified.
If your review found no factual problems, preserve the answer's factual content and still obey the required short-answer format.
Do not introduce new facts that are not in the context.

Output format (JSON object, no markdown fences, no extra keys):
{"short_answer": "<answer>", "explanation": "<explanation>"}

The same rules as the draft apply: "short_answer" is the minimal answer span (or exactly "yes"/"no"),
never empty; "explanation" is 1 to 4 plain declarative sentences with no sentence-number citations.
Keep these fields separate: {"short_answer": "England", "explanation": "The context places London in England."}.

Question: $question

Context:
$context

Your draft:
short_answer: $short_answer
explanation: $explanation

Your review:
$review
