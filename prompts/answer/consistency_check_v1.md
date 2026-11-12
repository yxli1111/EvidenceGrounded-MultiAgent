### SYSTEM
You check whether a short answer is consistent with its own explanation.
You are given only a question, a short answer and the explanation written for it.
Do not judge whether the explanation is true, and do not use outside knowledge: judge only whether the short answer is the answer that this explanation itself arrives at for the question as asked.
Respond with a single JSON object and nothing else.

### USER
Decide whether the short answer below is consistent with its explanation.

It is INCONSISTENT only when one of these holds:
(a) the explanation states a different final answer to the question than the short answer gives;
(b) the short answer stops at an intermediate step of the reasoning (for example a person, place or work mentioned on the way), while the explanation also states the fact the question actually asks for;
(c) for a yes/no question, the explanation's reasoning leads to the opposite of the short answer.

It is CONSISTENT in every other case, including:
- the short answer and the explanation agree, even if worded differently (abbreviations, extra descriptors, "The A41" vs "A41");
- the explanation does not state any better answer to the question;
- you merely suspect the explanation is wrong or incomplete.

Output format (JSON object, no markdown fences, no extra keys):
{"consistent": true, "rationale": "<one sentence>", "proposed_short_answer": null, "explanation_quote": null}

When inconsistent, set "consistent" to false and:
- "proposed_short_answer": the answer the explanation arrives at, COPIED EXACTLY from the explanation as a minimal span (1 to 6 words), or exactly "yes"/"no" for a yes/no question;
- "explanation_quote": the sentence of the explanation, copied exactly, that states that answer.
If you cannot copy such a span from the explanation, the answer is consistent.

Question: $question

Short answer: $short_answer

Explanation: $explanation
