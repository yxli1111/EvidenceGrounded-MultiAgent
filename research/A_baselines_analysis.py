"""Contributor A analysis: M1 / M2 / M3 (part of RQ1) from one runner output directory.

Reads E's run directory (records.jsonl + gold_lookup.json) and scores answers with
E's official functions (src.evaluation.answer_metrics.exact_match / token_f1 and
src.evaluation.statistical.paired_bootstrap_ci). A does not implement its own scoring.

Outputs go to results/analysis/A_<run_name>/:
    summary.json            headline metrics, transitions, feedback behaviour, cost, errors
    per_example_A.csv       one row per (example, method in M1-M3) with A's diagnostic columns
    cases.md                readable correct->wrong / wrong->correct / error cases for inspection

Usage (repo root, on a branch that contains E's src/evaluation):
    python research/A_baselines_analysis.py --run results/runs/main_n100
Gold is used only for evaluation, never for prompting.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ('M1', 'M2', 'M3')


# --------------------------------------------------------------------------- loading

def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    """Return (latest record per (eid, method), gold lookup, metadata)."""
    latest: dict[tuple[str, str], dict] = {}
    with (run_dir / 'records.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                latest[(r['example_id'], r['method_id'])] = r   # later lines win (retries)
    gold = json.loads((run_dir / 'gold_lookup.json').read_text(encoding='utf-8'))
    meta_p = run_dir / 'metadata.json'
    meta = json.loads(meta_p.read_text(encoding='utf-8')) if meta_p.exists() else {}
    return latest, gold, meta


def final_of(rec: dict) -> tuple[str, str]:
    """(final short answer, final explanation); M1 / missing final -> the draft."""
    fin = rec.get('final') or {}
    d = rec.get('draft') or {}
    if fin:
        return fin.get('final_answer') or '', fin.get('final_explanation') or ''
    return d.get('short_answer') or '', d.get('explanation') or ''


# --------------------------------------------------------------------------- diagnostics

_SENTENCE_RE = re.compile(r'\b(is|was|are|were|has|had|the answer)\b', re.I)


def looks_like_sentence(ans: str) -> bool:
    """Heuristic flag for 'short answer became a sentence' (format failure)."""
    a = ans.strip()
    return len(a.split()) > 6 or a.endswith('.') or bool(_SENTENCE_RE.search(a) and len(a.split()) > 3)


def classify_transition(draft_ok: bool, final_ok: bool, status: str) -> str:
    if status != 'ok':
        return ('correct' if draft_ok else 'wrong') + '->error'
    return ('correct' if draft_ok else 'wrong') + '->' + ('correct' if final_ok else 'wrong')


def classify_c2w(draft_ans: str, final_ans: str, gold_ans: str, normalize) -> str:
    """Why a correct draft became wrong."""
    nd, nf, ng = normalize(draft_ans), normalize(final_ans), normalize(gold_ans)
    if ng and ng in nf and nf != ng:
        return 'format: final contains gold but is longer (sentence / extra words)'
    if nd and nd in nf:
        return 'format: final contains the draft answer plus extra text'
    return 'content: answer replaced by a different answer'


# --------------------------------------------------------------------------- main analysis

def analyse(run_dir: Path, out_dir: Path) -> dict:
    from src.evaluation.answer_metrics import exact_match, normalize_answer, token_f1  # E
    from src.evaluation.statistical import paired_bootstrap_ci                       # E

    latest, gold, meta = load_run(run_dir)
    eids = sorted({e for e, m in latest if m in METHODS})
    rows, cases = [], collections.defaultdict(list)

    for eid in eids:
        g = gold[eid]['gold_answer']
        m1 = latest.get((eid, 'M1'))
        draft = (m1 or {}).get('draft') or next(
            (latest[(eid, m)].get('draft') for m in METHODS if (eid, m) in latest
             and latest[(eid, m)].get('draft')), {}) or {}
        d_ans = draft.get('short_answer', '')
        d_em = exact_match(d_ans, g)
        for m in METHODS:
            rec = latest.get((eid, m))
            if rec is None:
                continue
            status = rec['status']
            f_ans, f_expl = final_of(rec) if status == 'ok' else ('', '')
            em = exact_match(f_ans, g) if status == 'ok' else 0.0       # errors score 0 (E's rule)
            f1 = token_f1(f_ans, g) if status == 'ok' else 0.0
            items = ((rec.get('feedback') or {}).get('items') or [])
            usage = rec.get('usage') or []
            row = {
                'example_id': eid, 'method_id': m, 'status': status,
                'question_type': rec.get('question_type'),
                'em': em, 'f1': round(f1, 4), 'draft_em': d_em,
                'transition': classify_transition(bool(d_em), bool(em), status) if m != 'M1' else '',
                'answer_changed': int(status == 'ok' and normalize_answer(f_ans) != normalize_answer(d_ans)) if m != 'M1' else '',
                'final_is_sentence': int(status == 'ok' and looks_like_sentence(f_ans)),
                'n_feedback_items': len(items) if m != 'M1' else '',
                'method_cost_usd': round(sum(u.get('cost_usd') or 0 for u in usage), 6),
                'method_calls': sum(0 if u.get('cache_hit') else 1 for u in usage),
                'latency_s': round(sum(u.get('latency_seconds') or 0 for u in usage), 3),
                'error': (rec.get('error') or '')[:200],
            }
            rows.append(row)
            if m == 'M1':
                continue
            case = {'eid': eid, 'q': rec.get('question', ''), 'gold': g, 'draft': d_ans,
                    'final': f_ans, 'items': items, 'error': rec.get('error'), 'qtype': rec.get('question_type')}
            if row['transition'] == 'correct->wrong':
                case['why'] = classify_c2w(d_ans, f_ans, g, normalize_answer)
                cases[(m, 'correct->wrong')].append(case)
            elif row['transition'] == 'wrong->correct':
                cases[(m, 'wrong->correct')].append(case)
            elif row['transition'].endswith('->error'):
                cases[(m, 'error')].append(case)

    # ---- headline metrics (all examples, errors = 0) and paired CIs
    by_m = {m: {r['example_id']: r for r in rows if r['method_id'] == m} for m in METHODS}
    common = [e for e in eids if all(e in by_m[m] for m in METHODS)]
    headline = {}
    for m in METHODS:
        rs = [by_m[m][e] for e in common]
        ok = [r for r in rs if r['status'] == 'ok']
        headline[m] = {'n': len(rs), 'n_ok': len(ok), 'n_error': len(rs) - len(ok),
                       'em': round(sum(r['em'] for r in rs) / len(rs), 4),
                       'f1': round(sum(r['f1'] for r in rs) / len(rs), 4),
                       'em_on_ok_only': round(sum(r['em'] for r in ok) / len(ok), 4) if ok else None,
                       'final_is_sentence_rate': round(sum(r['final_is_sentence'] for r in ok) / len(ok), 4) if ok else None,
                       'mean_cost_usd': round(sum(r['method_cost_usd'] for r in rs) / len(rs), 6),
                       'mean_calls': round(sum(r['method_calls'] for r in rs) / len(rs), 3),
                       'mean_latency_s': round(sum(r['latency_s'] for r in rs) / len(rs), 3)}
    paired = {}
    for a, b in (('M2', 'M1'), ('M3', 'M1'), ('M3', 'M2')):
        for metric in ('em', 'f1'):
            ci = paired_bootstrap_ci([by_m[a][e][metric] for e in common],
                                     [by_m[b][e][metric] for e in common], seed=42)
            paired[f'{a}-{b}:{metric}'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ci.items()}
    # M3 restricted to examples where D's revision succeeded (separates critic from D failures)
    ok3 = [e for e in common if by_m['M3'][e]['status'] == 'ok']
    if ok3:
        ci = paired_bootstrap_ci([by_m['M3'][e]['em'] for e in ok3], [by_m['M1'][e]['em'] for e in ok3], seed=42)
        paired['M3-M1:em (M3 ok subset)'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ci.items()}

    # ---- transitions, c->w reasons, feedback behaviour
    transitions = {m: dict(collections.Counter(by_m[m][e]['transition'] for e in common)) for m in ('M2', 'M3')}
    c2w_reasons = {m: dict(collections.Counter(c['why'] for c in cases[(m, 'correct->wrong')])) for m in ('M2', 'M3')}
    feedback = {}
    for m in ('M2', 'M3'):
        rs = [by_m[m][e] for e in common]
        corr = [r for r in rs if r['draft_em'] == 1]
        wrong = [r for r in rs if r['draft_em'] == 0]
        feedback[m] = {
            'feedback_recorded': any(r['n_feedback_items'] not in ('', 0) for r in rs),
            'issue_raised_rate_when_draft_correct': round(sum(1 for r in corr if r['n_feedback_items']) / len(corr), 4) if corr else None,
            'issue_raised_rate_when_draft_wrong': round(sum(1 for r in wrong if r['n_feedback_items']) / len(wrong), 4) if wrong else None,
            'mean_items': round(sum(r['n_feedback_items'] or 0 for r in rs) / len(rs), 3),
            'answer_changed_rate_ok': round(sum(r['answer_changed'] for r in rs if r['status'] == 'ok') /
                                            max(1, sum(1 for r in rs if r['status'] == 'ok')), 4),
        }
    errors = {m: dict(collections.Counter(re.sub(r'\s+', ' ', (by_m[m][e]['error'] or ''))[:90]
                                          for e in common if by_m[m][e]['status'] != 'ok')) for m in METHODS}
    by_type = {}
    for m in METHODS:
        for t in sorted({by_m[m][e]['question_type'] for e in common}):
            rs = [by_m[m][e] for e in common if by_m[m][e]['question_type'] == t]
            by_type[f'{m}:{t}'] = {'n': len(rs), 'em': round(sum(r['em'] for r in rs) / len(rs), 4)}

    summary = {'run_dir': str(run_dir), 'run_metadata': meta, 'n_examples': len(common),
               'prompt_versions': next((latest[k].get('prompt_versions') for k in latest if k[1] == 'M2'), None),
               'headline': headline, 'paired_bootstrap': paired, 'transitions': transitions,
               'correct_to_wrong_reasons': c2w_reasons, 'feedback_behaviour': feedback,
               'errors': errors, 'by_question_type': by_type,
               'notes': ['errors count as EM/F1 = 0 (E rule); draft correctness taken from the shared draft, '
                         'not from error rows', 'final_is_sentence and c->w reasons are heuristics for triage, '
                         'not official metrics', 'PRELIMINARY: pilot run, not the frozen evaluation']}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    with (out_dir / 'per_example_A.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_cases(out_dir / 'cases.md', cases)
    return summary


def write_cases(path: Path, cases: dict) -> None:
    lines = ['# A: M2/M3 case list (PRELIMINARY pilot)', '']
    for key in sorted(cases):
        m, kind = key
        lines += [f'## {m} {kind} ({len(cases[key])})', '']
        for c in cases[key]:
            lines.append(f"- **{c['eid']}** ({c['qtype']}) Q: {c['q']}")
            lines.append(f"  - gold: `{c['gold']}` | draft: `{c['draft']}` | final: `{c['final']}`")
            if c.get('why'):
                lines.append(f"  - reason: {c['why']}")
            if c.get('error'):
                lines.append(f"  - error: {c['error'][:200]}")
            for it in c['items'][:3]:
                corr = f" | fix: {it.get('suggested_correction')}" if it.get('suggested_correction') else ''
                lines.append(f"  - feedback: {it.get('rationale', '')[:220]}{corr[:160]}")
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Contributor A M1-M3 analysis')
    p.add_argument('--run', type=Path, required=True, help='runner output dir, e.g. results/runs/main_n100')
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args(argv)
    out = a.out or ROOT / 'results' / 'analysis' / f'A_{a.run.name}'
    s = analyse(a.run, out)
    print(json.dumps({k: s[k] for k in ('n_examples', 'headline', 'transitions', 'correct_to_wrong_reasons',
                                         'feedback_behaviour')}, indent=1, ensure_ascii=False))
    print(f'written to {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
