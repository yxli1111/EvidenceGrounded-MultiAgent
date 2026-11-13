"""Phase 2 Experiment A analysis: M1 (unchanged shared draft) vs M1C (consistency check).

Paired on identical example ids and drafts. Scores with E's official functions
(src.evaluation.answer_metrics.exact_match / token_f1 / normalize_answer and
src.evaluation.statistical.paired_bootstrap_ci with 5,000 resamples over question ids).
Gold is used here for evaluation only.

Outputs in results/analysis/A_<run name>/:
    summary.json      metrics, flips, gate outcomes, flag rates by draft correctness, headroom, cost
    per_example.csv   one row per example (M1 vs M1C)
    case_table.csv    every flagged case + every missed headroom case, with empty review columns
                      (review_flag_valid / review_edit_valid: Y or N, review_notes) for the manual
                      precision audit; pass the filled file back with --reviewed to score it.

Usage:
    python research/A_consistency_analysis.py --source-run results/runs/dev_v2 --run results/runs/A_m1c_dev
    python research/A_consistency_analysis.py ... --reviewed results/analysis/A_A_m1c_dev/case_table_reviewed.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_RESAMPLES = 5000
SEED = 42
REVIEW_COLS = ['review_flag_valid', 'review_edit_valid', 'review_notes']


def _latest(path: Path, method: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get('method_id') == method:
                    out[r['example_id']] = r
    return out


def _r(x, k=4):
    return round(x, k) if isinstance(x, float) else x


def analyse(source_run: Path, run_dir: Path, out_dir: Path, reviewed: Path | None = None) -> dict:
    from src.evaluation.answer_metrics import exact_match, normalize_answer, token_f1  # E
    from src.evaluation.statistical import paired_bootstrap_ci                       # E

    m1 = _latest(source_run / 'records.jsonl', 'M1')
    m1c = _latest(run_dir / 'records.jsonl', 'M1C')
    gold = json.loads((source_run / 'gold_lookup.json').read_text(encoding='utf-8'))
    meta_p = run_dir / 'metadata.json'
    meta = json.loads(meta_p.read_text(encoding='utf-8')) if meta_p.exists() else {}
    ids = [e for e in m1 if e in m1c]                      # paired set, source order
    rows, cases = [], []
    for eid in ids:
        g = gold[eid]['gold_answer']
        a, c = m1[eid], m1c[eid]
        d_ans = a['draft']['short_answer']
        expl = a['draft']['explanation']
        ok = c['status'] == 'ok'
        dec = c.get('decision') or {}
        f_ans = ((c.get('final') or {}).get('final_answer') or d_ans) if ok else ''
        em1, f11 = exact_match(d_ans, g), token_f1(d_ans, g)
        em2, f12 = (exact_match(f_ans, g), token_f1(f_ans, g)) if ok else (0.0, 0.0)  # error = 0
        ng = normalize_answer(g)
        headroom = (not em1) and bool(ng) and ng not in ('yes', 'no') and ng in normalize_answer(expl)
        usage = c.get('usage') or []
        row = {'example_id': eid, 'question_type': c.get('question_type'), 'status': c['status'],
               'gold': g, 'draft_answer': d_ans, 'final_answer': f_ans,
               'm1_em': em1, 'm1c_em': em2, 'm1_f1': _r(f11), 'm1c_f1': _r(f12),
               'flagged': int(bool(dec.get('flagged'))), 'edited': int(bool(dec.get('edited'))),
               'gate': dec.get('gate') or ('error' if not ok else ''),
               'mismatch_type': dec.get('mismatch_type') or '',
               'transition': ('correct' if em1 else 'wrong') + '->' + ('correct' if em2 else 'wrong') if ok
               else ('correct' if em1 else 'wrong') + '->error',
               'headroom_gold_in_explanation': int(headroom),
               'cost_usd': round(sum(u.get('cost_usd') or 0 for u in usage), 8),
               'latency_s': round(sum(u.get('latency_seconds') or 0 for u in usage), 3),
               'input_tokens': sum(u.get('input_tokens') or 0 for u in usage),
               'output_tokens': sum(u.get('output_tokens') or 0 for u in usage)}
        rows.append(row)
        if row['flagged'] or headroom:
            cases.append({'example_id': eid, 'question': c.get('question'), 'gold': g,
                          'draft_answer': d_ans, 'final_answer': f_ans, 'flagged': row['flagged'],
                          'edited': row['edited'], 'gate': row['gate'], 'transition': row['transition'],
                          'headroom_gold_in_explanation': int(headroom),
                          'mismatch_type': dec.get('mismatch_type'), 'checker_rationale': dec.get('rationale'),
                          'proposed_short_answer': dec.get('proposed_short_answer'),
                          'explanation_quote': dec.get('explanation_quote'), 'explanation': expl,
                          **{k: '' for k in REVIEW_COLS}})

    n = len(rows)
    ok_rows = [r for r in rows if r['status'] == 'ok']
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    boot = lambda k: {kk: _r(v) for kk, v in paired_bootstrap_ci(
        [r[f'm1c_{k}'] for r in rows], [r[f'm1_{k}'] for r in rows],
        n_resamples=N_RESAMPLES, seed=SEED).items()}
    by_draft = {}
    for label, cond in (('draft_correct', lambda r: r['m1_em'] == 1), ('draft_wrong', lambda r: r['m1_em'] == 0)):
        sub = [r for r in ok_rows if cond(r)]
        by_draft[label] = {'n': len(sub), 'flag_rate': mean([r['flagged'] for r in sub]),
                           'edit_rate': mean([r['edited'] for r in sub])}
    hr = [r for r in rows if r['headroom_gold_in_explanation']]
    summary = {
        'run_dir': str(run_dir), 'source_run': str(source_run), 'n_paired': n,
        'n_ok': len(ok_rows), 'n_error': n - len(ok_rows),
        'prompt_versions': meta.get('prompt_versions'), 'agent_config': meta.get('agent_config'),
        'metrics': {'M1': {'em': mean([r['m1_em'] for r in rows]), 'f1': mean([r['m1_f1'] for r in rows])},
                    'M1C': {'em': mean([r['m1c_em'] for r in rows]), 'f1': mean([r['m1c_f1'] for r in rows])}},
        'paired_bootstrap_M1C_minus_M1': {'em': boot('em'), 'f1': boot('f1')},
        'detection_rate': mean([r['flagged'] for r in rows]), 'edit_rate': mean([r['edited'] for r in rows]),
        'gate_outcomes': dict(collections.Counter(r['gate'] for r in rows)),
        'mismatch_types_flagged': dict(collections.Counter(r['mismatch_type'] for r in rows if r['flagged'])),
        'transitions': dict(collections.Counter(r['transition'] for r in rows)),
        'flag_and_edit_rate_by_draft_correctness': by_draft,
        'headroom': {'wrong_drafts_with_gold_in_explanation': len(hr),
                     'flagged': sum(r['flagged'] for r in hr), 'edited': sum(r['edited'] for r in hr),
                     'fixed': sum(1 for r in hr if r['m1c_em'] == 1)},
        'cost': {'total_usd': round(sum(r['cost_usd'] for r in rows), 6),
                 'per_question_usd': round(sum(r['cost_usd'] for r in rows) / n, 7) if n else None,
                 'mean_latency_s': mean([r['latency_s'] for r in rows]),
                 'mean_input_tokens': mean([r['input_tokens'] for r in rows]),
                 'mean_output_tokens': mean([r['output_tokens'] for r in rows])},
        'notes': ['errors count as EM/F1 = 0 for M1C (protocol); M1 is the unchanged shared draft',
                  'headroom = wrong drafts whose gold answer string appears in their own explanation '
                  '(evaluation-only diagnostic, gold never used by the checker)',
                  f'paired bootstrap: {N_RESAMPLES} resamples over question ids, seed {SEED}; '
                  'a CI that includes 0 is not a significant difference'],
    }
    if reviewed is not None:
        summary['manual_review'] = score_review(reviewed)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    _write_csv(out_dir / 'per_example.csv', rows)
    if cases:
        _write_csv(out_dir / 'case_table.csv', cases)
    return summary


def score_review(path: Path) -> dict:
    """Precision of flags / edits from a filled case table (Y/N per row; blanks ignored)."""
    with path.open(encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    yn = lambda v: (v or '').strip().upper()[:1]
    flag = [yn(r['review_flag_valid']) for r in rows if r.get('flagged') in ('1', 1) and yn(r['review_flag_valid']) in 'YN' and yn(r['review_flag_valid'])]
    edit = [yn(r['review_edit_valid']) for r in rows if r.get('edited') in ('1', 1) and yn(r['review_edit_valid']) in 'YN' and yn(r['review_edit_valid'])]
    p = lambda xs: round(xs.count('Y') / len(xs), 4) if xs else None
    return {'flags_reviewed': len(flag), 'flag_precision': p(flag),
            'edits_reviewed': len(edit), 'edit_precision': p(edit)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as fh:   # BOM: opens cleanly in Excel
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Phase 2 Experiment A analysis (M1 vs M1C)')
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--reviewed', type=Path, default=None)
    a = p.parse_args(argv)
    out = a.out or ROOT / 'results' / 'analysis' / f'A_{a.run.name}'
    s = analyse(a.source_run, a.run, out, a.reviewed)
    print(json.dumps({k: s[k] for k in ('n_paired', 'n_error', 'metrics', 'paired_bootstrap_M1C_minus_M1',
                                         'detection_rate', 'edit_rate', 'gate_outcomes', 'transitions',
                                         'flag_and_edit_rate_by_draft_correctness', 'headroom', 'cost')},
                     indent=1, ensure_ascii=False))
    print(f'written to {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
