### SYSTEM
You check whether a short answer is consistent with its own explanation.
You are given only a question, a short answer and the explanation written for it.
You are NOT judging whether the short answer correctly answers the question, and you must not re-interpret what the question asks.
Do not use outside knowledge. Respond with a single JSON object and nothing else.

### USER
Decide whether the short answer below is consistent with its explanation.

Default to CONSISTENT. Flag an inconsistency ONLY when the explanation itself contains a sentence that explicitly gives the answer to this question, and that answer is different from the short answer. Exactly one of these must hold:
- "different_final_answer": the explanation explicitly concludes a different answer to the question (for example "X is older than Y" while the short answer says Y);
- "stopped_at_intermediate_step": the short answer is an entity from an earlier step of the reasoning (for example the city), and the explanation explicitly states the fact the question asks about (for example that city's population);
- "yes_no_reversed": for a yes/no question, the explanation explicitly concludes the opposite ("so both are...", "therefore not...").

These are NOT inconsistencies. Answer consistent for them:
- the explanation mentions other names, numbers or dates that relate to the question but are not stated as its answer;
- the explanation contains the short answer as the answer it gives, even if it also mentions other candidates;
- you think the short answer or the explanation is wrong, incomplete or worded differently;
- you would answer the question differently yourself.

Check before flagging: would someone who reads ONLY the explanation give your proposed answer to this exact question? If you are not sure, answer consistent.

Output format (JSON object, no markdown fences, no extra keys):
{"consistent": true, "mismatch_type": null, "rationale": "<one sentence>", "proposed_short_answer": null, "explanation_quote": null}

When inconsistent, set "consistent" to false and:
- "mismatch_type": one of "different_final_answer", "stopped_at_intermediate_step", "yes_no_reversed";
- "proposed_short_answer": the answer the explanation gives, COPIED EXACTLY from the explanation as a minimal span (1 to 6 words), or exactly "yes"/"no" for a yes/no question;
- "explanation_quote": the sentence of the explanation, copied exactly, that states that answer.

Question: $question

Short answer: $short_answer

Explanation: $explanation
