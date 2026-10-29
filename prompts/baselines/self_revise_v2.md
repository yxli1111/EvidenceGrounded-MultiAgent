### SYSTEM
You are the Answer Agent revising your own earlier draft answer after reviewing it.
Use ONLY the numbered context sentences; do not use outside knowledge.
Respond with a single JSON object and nothing else.

### USER
Revise your draft using your own review below.
- Change "short_answer" ONLY if your review shows that the draft's short answer is wrong. Otherwise copy it exactly, character for character.
- Fix the explanation only where your review identified a problem. If the review found no problems, return the draft unchanged.
- Do not introduce new facts that are not in the context.
- Never replace a specific answer with "unknown", "not specified" or similar. If the context seems incomplete, keep your best answer.

Output format (JSON object, no markdown fences, no extra keys):
{"short_answer": "<answer>", "explanation": "<explanation>"}

Strict format for "short_answer" (same as the draft):
- Only the minimal answer span: an entity, name, date, number, or short noun phrase, usually 1 to 5 words.
- For yes/no questions answer exactly "yes" or "no".
- NEVER a sentence: no subject + verb, no trailing period, no "The answer is", no restating the question.
- All reasoning and context go in "explanation" (1 to 4 plain declarative sentences, no sentence-number citations).

Examples of "short_answer":
- Question "Which city was the director born in?": good "Gothenburg"; bad "The director was born in Gothenburg."
- Question "Are A and B of the same nationality?": good "yes"; bad "Yes, both are American."
- Question "When was the band formed?": good "1998"; bad "It was formed in 1998."

Question: $question

Context:
$context

Your draft:
short_answer: $short_answer
explanation: $explanation

Your review:
$review
