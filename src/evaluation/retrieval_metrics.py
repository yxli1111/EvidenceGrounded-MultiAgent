"""Retrieval evaluation metrics: Recall@k.

Two distinct denominators are supported and must not be mixed:

  QUESTION_CONDITIONED:
      Denominator = all gold supporting facts for the question.
      Use this when the retrieval query is derived from the question
      (e.g. B's M4 whole-draft retrieval benchmarked against question-level gold).

  CLAIM_CONDITIONED:
      Denominator = gold supporting facts for one specific claim.
      Use this when each claim is retrieved independently (M5) and a
      claim-to-gold mapping is available.
      Without that mapping this mode cannot be computed; callers must
      supply it explicitly.

Key names (from B's actual data format):
  gold_supporting_facts items use key  "title"
  retrieved Evidence items use key     "doc_title"
Both use "sentence_id".  The asymmetry is intentional in B's design
and must not be papered over silently.
"""

from __future__ import annotations

from enum import Enum


def _gold_get(gold, key, default=None):
    """Get a field from a gold entry that may be a dict or a QAGold object."""
    if isinstance(gold, dict):
        return gold.get(key, default)
    return getattr(gold, key, default)


class RecallDenominator(Enum):
    QUESTION_CONDITIONED = "question_conditioned"
    CLAIM_CONDITIONED = "claim_conditioned"


def _gold_set(gold_supporting_facts: list) -> set[tuple]:
    """Convert gold facts to (title, sentence_id) tuples.

    Accepts both dict format {"title": str, "sentence_id": int} (B's format)
    and tuple format (str, int) (contracts.QAGold format).
    """
    result: set[tuple] = set()
    for f in gold_supporting_facts:
        if isinstance(f, tuple):
            result.add((f[0], f[1]))
        else:
            result.add((f["title"], f["sentence_id"]))
    return result


def _predicted_set(retrieved: list[dict], k: int) -> set[tuple]:
    """Convert retrieved evidence (key 'doc_title') to (title, sentence_id) tuples."""
    return {(e["doc_title"], e["sentence_id"]) for e in retrieved[:k]}


def recall_at_k(
    retrieved: list[dict],
    gold_supporting_facts: list[dict],
    k: int,
    denominator: RecallDenominator = RecallDenominator.QUESTION_CONDITIONED,
) -> float:
    """Recall@k for a single example or a single claim.

    Args:
        retrieved: list of Evidence dicts from B's retriever
                   (key 'doc_title', not 'title').
        gold_supporting_facts: list of gold fact dicts
                   (key 'title', not 'doc_title').
        k: cutoff rank (uses retrieved[:k]).
        denominator: which denominator mode to use.
                     The caller is responsible for passing the correct
                     gold set for the chosen mode.

    Returns:
        float in [0.0, 1.0].  Returns 0.0 if gold set is empty
        (failure counted in caller's aggregation).
    """
    gold = _gold_set(gold_supporting_facts)
    if not gold:
        return 0.0

    predicted = _predicted_set(retrieved, k)
    return len(gold & predicted) / len(gold)


def full_coverage_at_k(
    retrieved: list[dict],
    gold_supporting_facts: list[dict],
    k: int,
) -> bool:
    """True iff all gold facts are covered within top-k retrieved evidence."""
    gold = _gold_set(gold_supporting_facts)
    if not gold:
        return False
    predicted = _predicted_set(retrieved, k)
    return gold.issubset(predicted)


def mean_recall_at_k(
    records: list[dict],
    gold_lookup: dict[str, dict],
    k: int,
    method_id: str,
    denominator: RecallDenominator = RecallDenominator.QUESTION_CONDITIONED,
) -> dict:
    """Aggregate Recall@k across all records for one method.

    Args:
        records: list of ExperimentRecord dicts or objects.  Records with
                 status != "ok" contribute 0 to the numerator but
                 are still counted in the denominator.
        gold_lookup: maps example_id -> QAGold dict.
        k: rank cutoff.
        method_id: only records with this method_id are processed.
                   Passing method_id="M5" raises NotImplementedError because
                   M5 evidence is keyed by claim_id and question-conditioned
                   Recall@k does not have a valid semantic for it.
        denominator: must be QUESTION_CONDITIONED.
                     CLAIM_CONDITIONED raises NotImplementedError because
                     a claim→gold mapping is not yet available (GAP-7).

    Returns:
        {"mean_recall": float, "n": int, "n_ok": int, "n_error": int,
         "denominator": str, "k": int}
    """
    if denominator == RecallDenominator.CLAIM_CONDITIONED:
        raise NotImplementedError(
            "CLAIM_CONDITIONED requires a per-example claim→gold supporting-facts "
            "mapping. M5 evidence is a merge of per-claim retrievals and must NOT "
            "be compared directly against question-level gold facts. "
            "See GAP-7 in evaluation/E_CONTRACT_PROPOSAL.md."
        )

    if method_id == "M5":
        raise NotImplementedError(
            "QUESTION_CONDITIONED Recall@k is not defined for M5. "
            "M5 evidence is per-claim (dict keyed by claim_id); merging "
            "per-claim retrievals and comparing against question-level gold "
            "is not semantically valid. Use per-claim Recall@k (GAP-7) instead."
        )

    def _f(rec, key, default=None):
        return rec.get(key, default) if isinstance(rec, dict) else getattr(rec, key, default)

    # Dedup: keep highest attempt per (example_id, method_id)
    seen: dict = {}
    for r in records:
        r_mid = _f(r, "method_id", "")
        if r_mid != method_id:
            continue
        r_eid = _f(r, "example_id", "")
        r_attempt = _f(r, "attempt", 1) or 1
        key = (r_eid, r_mid)
        existing = seen.get(key)
        if existing is None or r_attempt > (_f(existing, "attempt", 1) or 1):
            seen[key] = r

    subset = list(seen.values())
    total = len(subset)
    n_ok = 0
    n_error = 0
    recall_sum = 0.0

    for rec in subset:
        eid = _f(rec, "example_id", "")
        if _f(rec, "status", "error") != "ok":
            n_error += 1
            continue

        n_ok += 1
        gold_entry = gold_lookup.get(eid)
        gold = _gold_get(gold_entry, "gold_supporting_facts", []) if gold_entry is not None else []

        # Extract evidence list: new dict format {key: [list]} or old list format
        raw_ev = _f(rec, "evidence", [])
        if isinstance(raw_ev, dict):
            ev_list = [e for evlist in raw_ev.values() for e in evlist]
        else:
            ev_list = raw_ev or []

        recall_sum += recall_at_k(ev_list, gold, k, denominator)

    return {
        "mean_recall": recall_sum / total if total > 0 else 0.0,
        "n": total,
        "n_ok": n_ok,
        "n_error": n_error,
        "denominator": denominator.value,
        "k": k,
    }
