"""Phase 2 Experiment A: run the answer-explanation consistency check (M1C) on an existing run.

Reads the shared drafts (M1 records) of a finished runner output directory, runs ONE
consistency check per draft with C's OpenRouterLLMClient, and writes M1C records in the
same layout as E's runner (``records.jsonl``, ``gold_lookup.json`` copied for evaluation
only, ``metadata.json``). Drafts are never regenerated, so M1 vs M1C is a paired comparison
on identical drafts. Gold answers are never read by the checker.

Usage (repo root; key from the OPENROUTER_API_KEY environment variable, never on disk):
    python research/A_consistency_experiment.py --source-run results/runs/dev_v2 --run-name A_m1c_dev --dry-run
    python research/A_consistency_experiment.py --source-run results/runs/dev_v2 --run-name A_m1c_dev
    python research/A_consistency_analysis.py --source-run results/runs/dev_v2 --run results/runs/A_m1c_dev
Re-running resumes: examples already recorded with status ok are skipped.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.schemas import AnswerDraft, QAInput  # noqa: E402
from src.agents.answer import PROMPT_VERSIONS, AgentConfig, AgentOutputError, load_prompt  # noqa: E402
from src.agents.consistency import METHOD_ID, check_consistency  # noqa: E402

STOP_ERRORS = {'BudgetExceededError', 'LLMInsufficientCreditError', 'LLMAuthenticationError'}


class UsageRecorder:
    def __init__(self, llm):
        self.llm, self.usages = llm, []

    def generate(self, **kw):
        resp = self.llm.generate(**kw)
        u = resp.usage
        self.usages.append(dataclasses.asdict(u) if dataclasses.is_dataclass(u) else dict(vars(u)) if u else {})
        return resp

    def take(self):
        out, self.usages = self.usages, []
        return out


def load_source(source_run: Path, limit: int | None, exclude: set[str]) -> list[dict]:
    """M1 records of the source run, in file order (latest line per example wins)."""
    latest: dict[str, dict] = {}
    with (source_run / 'records.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get('method_id') == 'M1' and r.get('status') == 'ok' and r.get('draft'):
                    latest[r['example_id']] = r
    rows = [r for eid, r in latest.items() if eid not in exclude]
    return rows[:limit] if limit else rows


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)['example_id'] for l in path.read_text(encoding='utf-8').splitlines()
            if l.strip() and json.loads(l).get('status') == 'ok'}


def run(rows: list[dict], llm, config: AgentConfig, out_dir: Path, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / 'records.jsonl'
    skip = done_ids(rec_path)
    rec_llm = UsageRecorder(llm)
    stats = {'ok': 0, 'error': 0, 'skipped': 0, 'flagged': 0, 'edited': 0}
    with rec_path.open('a', encoding='utf-8') as fh:
        for i, src in enumerate(rows, 1):
            eid = src['example_id']
            if eid in skip:
                stats['skipped'] += 1
                continue
            example = QAInput(example_id=eid, question=src['question'], context=[],
                              question_type=src.get('question_type'))
            draft = AnswerDraft(**{k: src['draft'][k] for k in ('draft_id', 'short_answer', 'explanation')})
            rec = {'example_id': eid, 'method_id': METHOD_ID, 'status': 'ok',
                   'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'question': src['question'], 'question_type': src.get('question_type'),
                   'draft': dataclasses.asdict(draft), 'final': None, 'feedback': None, 'decision': None,
                   'model_config': config.as_log_dict(), 'prompt_versions': dict(PROMPT_VERSIONS),
                   'error': None, 'raw_outputs': []}
            stop = False
            try:
                fb, rev, dec = check_consistency(example, draft, rec_llm, config)
                rec.update(feedback=dataclasses.asdict(fb), decision=dec.to_dict(),
                           final=dataclasses.asdict(rev) if rev else None)
                stats['flagged'] += dec.flagged
                stats['edited'] += dec.edited
            except AgentOutputError as err:
                rec.update(status='error', error=f'AgentOutputError: {err}', raw_outputs=err.raw_outputs)
            except Exception as err:  # transport / budget errors from C's client
                rec.update(status='error', error=f'{type(err).__name__}: {err}')
                stop = type(err).__name__ in STOP_ERRORS
            rec['usage'] = rec_llm.take()
            stats[rec['status']] += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fh.flush()
            d = rec['decision'] or {}
            log(f"[{i}/{len(rows)}] {eid}: {rec['status']} gate={d.get('gate')} "
                f"{d.get('original_short_answer')!r} -> {d.get('final_short_answer')!r}")
            if stop:
                log('STOP: budget/credit/auth error')
                break
    return stats


def main(argv=None, *, llm=None) -> int:
    p = argparse.ArgumentParser(description='Phase 2 Experiment A: M1C consistency check')
    p.add_argument('--source-run', type=Path, required=True, help='run dir with M1 records (shared drafts)')
    p.add_argument('--run-name', required=True, help='output dir name under results/runs/')
    p.add_argument('--model', default=None, help='default: the source run model')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=int, default=300)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--exclude-ids-file', type=Path, default=None, help='JSON list of example ids to skip')
    p.add_argument('--exclude-run', type=Path, action='append', default=[],
                   help='skip every example id that appears in this run dir (repeatable), e.g. results/runs/main_v2')
    p.add_argument('--max-cost-usd', type=float, default=0.50)
    p.add_argument('--max-calls', type=int, default=None)
    p.add_argument('--min-key-remaining-usd', type=float, default=1.0)
    p.add_argument('--results-root', type=Path, default=ROOT / 'results' / 'runs')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)

    exclude = set(json.loads(a.exclude_ids_file.read_text(encoding='utf-8'))) if a.exclude_ids_file else set()
    for run_dir in a.exclude_run:
        with (run_dir / 'records.jsonl').open(encoding='utf-8') as fh:
            exclude |= {json.loads(l)['example_id'] for l in fh if l.strip()}
    rows = load_source(a.source_run, a.limit, exclude)
    if not rows:
        print(f'no M1 records found in {a.source_run}')
        return 2
    model = a.model or (rows[0].get('model_config') or {}).get('model')
    if not model:
        print('model unknown: pass --model')
        return 2
    config = AgentConfig(model=model, temperature=a.temperature, max_tokens=a.max_tokens)
    out_dir = a.results_root / a.run_name
    meta = {'experiment': 'Phase 2 Experiment A: answer-explanation consistency check (M1C)',
            'method_id': METHOD_ID, 'source_run': str(a.source_run), 'n_examples': len(rows),
            'example_ids': [r['example_id'] for r in rows], 'agent_config': config.as_log_dict(),
            'excluded_runs': [str(x) for x in a.exclude_run], 'n_excluded_ids': len(exclude),
            'prompt_versions': dict(PROMPT_VERSIONS), 'edit_policy': 'short_answer only; explanation and draft preserved; '
            'edit only if proposal is copied verbatim from a quoted explanation sentence',
            'gold_used_in_prompts': False,
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if a.dry_run:
        r0 = rows[0]
        msgs = load_prompt('answer.consistency_check').render(
            question=r0['question'], short_answer=r0['draft']['short_answer'],
            explanation=r0['draft']['explanation'])
        print(json.dumps({k: v for k, v in meta.items() if k != 'example_ids'}, indent=1))
        for m in msgs:
            print(f"\n[{m['role']}]\n{m['content']}")
        print(f'\nwould write to {out_dir}')
        return 0
    if llm is None:
        from src.core.budget import BudgetConfig, BudgetGuard   # C
        from src.core.cache import SQLiteCache                  # C
        from src.core.llm_client import OpenRouterLLMClient     # C
        llm = OpenRouterLLMClient(
            budget_guard=BudgetGuard(BudgetConfig(max_calls=a.max_calls or 2 * len(rows) + 10,
                                                  max_cost_usd=a.max_cost_usd,
                                                  min_key_remaining_usd=a.min_key_remaining_usd)),
            cache=SQLiteCache(ttl_seconds=60 * 60 * 24 * 365))
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = a.source_run / 'gold_lookup.json'
    if gold.exists():
        shutil.copyfile(gold, out_dir / 'gold_lookup.json')   # evaluation only
    stats = run(rows, llm, config, out_dir)
    meta['run_stats'] = stats
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(stats))
    return 0 if stats['error'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
