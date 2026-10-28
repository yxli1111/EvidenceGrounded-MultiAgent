"""Generate shared AnswerDrafts for the DEV split only (owner: Contributor A).

Purpose: give C a fixed set of dev drafts for claim-count statistics, and let A
inspect real draft quality while tuning prompts. These drafts are DEV artefacts:
they must not be reported as experimental results, and the held-out evaluation
drafts are produced later by E's runner with the frozen settings.

Integration (run on a LOCAL scratch branch that merges A + B + C; never push it):
    B  src.data.hotpot_loader.load_hotpotqa   -> QA inputs (gold is discarded here)
    C  src.core.llm_client.OpenRouterLLMClient -> all LLM calls (cache, usage, budget)
    A  src.agents.answer.generate_draft        -> AnswerDraft

The API key is read by C's client from the OPENROUTER_API_KEY environment variable
(or a git-ignored .env). It is never passed on the command line, printed or saved.

Usage (repo root):
    python research/A_generate_dev_drafts.py --model openai/gpt-4o-mini --dry-run
    python research/A_generate_dev_drafts.py --model openai/gpt-4o-mini --limit 50
Re-running resumes: examples already in the output file with status "ok" are skipped.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.schemas import LLMResponse, QAInput, Sentence  # noqa: E402
from src.agents.answer import (PROMPT_VERSIONS, AgentConfig, AgentOutputError,  # noqa: E402
                               generate_draft, load_prompt, render_context)

DEFAULT_OUT_DIR = ROOT / 'results' / 'dev_drafts'


# --------------------------------------------------------------------------- adapters

def to_qa_input(item: Any) -> QAInput:
    """Accept B's loader output (dict today, QAInput later) and return a contract QAInput.

    This is a format adapter only; loading/normalisation logic stays in B's loader.
    """
    if isinstance(item, QAInput):
        return item
    ctx = [s if isinstance(s, Sentence) else
           Sentence(title=s['title'], sentence_id=int(s['sentence_id']), text=s['text'])
           for s in item['context']]
    return QAInput(example_id=str(item['example_id']), question=item['question'],
                   context=ctx, question_type=item.get('question_type'))


def read_id_manifest(path: Path) -> list[str]:
    """Read an id manifest: a JSON list of ids, a list of {"example_id": ..}, or a dict
    holding such a list under "ids"/"example_ids"/"dev"."""
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        for key in ('ids', 'example_ids', 'dev', 'dev_ids'):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError(f'{path}: cannot find an id list (keys: {list(data)})')
    ids = [d['example_id'] if isinstance(d, dict) else d for d in data]
    return [str(i) for i in ids]


class UsageRecorder:
    """Wraps an LLMClient and keeps the LLMUsage of every call (for per-example logs)."""

    def __init__(self, llm):
        self.llm = llm
        self.usages: list[dict] = []

    def generate(self, **kwargs) -> LLMResponse:
        resp = self.llm.generate(**kwargs)
        self.usages.append(dataclasses.asdict(resp.usage))
        return resp

    def take(self) -> list[dict]:
        out, self.usages = self.usages, []
        return out


# --------------------------------------------------------------------------- core loop

def load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get('status') == 'ok':
                    done.add(rec['example_id'])
    return done


def generate_all(examples: Iterable[QAInput], llm, config: AgentConfig, out_path: Path,
                 log: Callable[[str], None] = print) -> dict:
    """Generate one draft per example, appending one JSONL record per example."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    rec_llm = UsageRecorder(llm)
    stats = {'ok': 0, 'error': 0, 'skipped': 0, 'calls': 0, 'cost_usd': 0.0, 'cost_unknown_calls': 0}
    with out_path.open('a', encoding='utf-8') as fh:
        for i, ex in enumerate(examples, 1):
            if ex.example_id in done:
                stats['skipped'] += 1
                continue
            rec: dict[str, Any] = {'example_id': ex.example_id, 'question_type': ex.question_type,
                                   'question': ex.question, 'status': 'ok', 'draft': None,
                                   'error': None, 'raw_outputs': []}
            t0 = time.perf_counter()
            try:
                rec['draft'] = dataclasses.asdict(generate_draft(ex, rec_llm, config))
            except AgentOutputError as err:          # unparseable model output
                rec.update(status='error', error=f'AgentOutputError: {err}', raw_outputs=err.raw_outputs)
            except Exception as err:                 # C's LLMClientError, budget stop, network...
                rec.update(status='error', error=f'{type(err).__name__}: {err}')
                if type(err).__name__ in {'BudgetExceededError', 'LLMInsufficientCreditError',
                                          'LLMAuthenticationError'}:
                    rec['usage'] = rec_llm.take()
                    fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    stats['error'] += 1
                    log(f'[{i}] {ex.example_id}: STOP ({rec["error"]})')
                    break
            rec['usage'] = rec_llm.take()
            rec['wall_seconds'] = round(time.perf_counter() - t0, 3)
            for u in rec['usage']:
                stats['calls'] += 0 if u.get('cache_hit') else 1
                if u.get('cost_usd') is None:
                    stats['cost_unknown_calls'] += 1
                else:
                    stats['cost_usd'] += u['cost_usd']
            stats[rec['status']] += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fh.flush()
            short = rec['draft']['short_answer'] if rec['draft'] else rec['error']
            log(f'[{i}] {ex.example_id} ({ex.question_type}): {short}')
    stats['cost_usd'] = round(stats['cost_usd'], 6)
    return stats


def summarize(out_path: Path) -> dict:
    """Descriptive draft statistics for the hand-off to C (no gold, no scoring)."""
    recs = [json.loads(l) for l in out_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    ok = [r for r in recs if r['status'] == 'ok']
    n_sent = [max(1, r['draft']['explanation'].count('. ') + 1) if r['draft']['explanation'] else 0
              for r in ok]
    n_chars = [len(r['draft']['explanation']) for r in ok]
    return {'records': len(recs), 'ok': len(ok), 'error': len(recs) - len(ok),
            'empty_explanations': sum(1 for c in n_chars if c == 0),
            'mean_explanation_chars': round(sum(n_chars) / len(ok), 1) if ok else None,
            'approx_mean_sentences': round(sum(n_sent) / len(ok), 2) if ok else None}


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--model', required=True, help='OpenRouter model id, e.g. openai/gpt-4o-mini')
    p.add_argument('--split', default='validation')
    p.add_argument('--limit', type=int, default=50, help='first N examples of the split (B dev = 50)')
    p.add_argument('--ids-file', type=Path, default=None,
                   help="optional id manifest (e.g. research/B_dev_50_ids.json); loaded examples must match it")
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--max-cost-usd', type=float, default=0.30, help='budget cap for this run (C BudgetGuard)')
    p.add_argument('--max-calls', type=int, default=150)
    p.add_argument('--min-key-remaining-usd', type=float, default=1.0)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--dry-run', action='store_true', help='load data and print the first prompt; no API call')
    return p


def main(argv: list[str] | None = None, *, loader=None, llm=None) -> int:
    args = build_parser().parse_args(argv)
    config = AgentConfig(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)

    if loader is None:
        from src.data.hotpot_loader import load_hotpotqa as loader  # B
    raw_inputs, _gold_discarded = loader(split=args.split, limit=args.limit)
    del _gold_discarded  # QAGold must never reach A
    examples = [to_qa_input(x) for x in raw_inputs]

    if args.ids_file is not None:
        want = read_id_manifest(args.ids_file)
        got = [e.example_id for e in examples]
        if got != want:
            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            print(f'ERROR: loaded examples do not match {args.ids_file} '
                  f'(missing {len(missing)}, extra {len(extra)}, order_equal={sorted(got) == sorted(want)})')
            return 2
        print(f'id manifest check passed: {len(want)} ids match {args.ids_file}')

    tag = f"{args.split}{args.limit}_{args.model.replace('/', '_')}_{PROMPT_VERSIONS['answer.draft'].replace('@', '-')}"
    out_path = args.out or DEFAULT_OUT_DIR / f'A_dev_drafts_{tag}.jsonl'
    meta = {'purpose': 'DEV drafts for claim-count statistics / prompt inspection; not experimental results',
            'split': args.split, 'limit': args.limit, 'n_examples': len(examples),
            'example_ids': [e.example_id for e in examples],
            'question_types': {t: sum(e.question_type == t for e in examples)
                               for t in sorted({str(e.question_type) for e in examples})},
            'agent_config': config.as_log_dict(), 'prompt_versions': dict(PROMPT_VERSIONS),
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}

    if args.dry_run:
        msgs = load_prompt('answer.draft').render(question=examples[0].question.strip(),
                                                  context=render_context(examples[0].context))
        chars = sum(len(m['content']) for m in msgs)
        print(json.dumps({k: v for k, v in meta.items() if k != 'example_ids'}, indent=1, ensure_ascii=False))
        print(f'\n--- first prompt ({chars} chars ~ {chars // 4} tokens) ---')
        for m in msgs:
            print(f"[{m['role']}]\n{m['content']}\n")
        print(f'output would go to: {out_path}')
        return 0

    if llm is None:
        from src.core.budget import BudgetConfig, BudgetGuard   # C
        from src.core.cache import SQLiteCache                  # C
        from src.core.llm_client import OpenRouterLLMClient     # C
        llm = OpenRouterLLMClient(
            budget_guard=BudgetGuard(BudgetConfig(max_calls=args.max_calls,
                                                  max_cost_usd=args.max_cost_usd,
                                                  min_key_remaining_usd=args.min_key_remaining_usd)),
            cache=SQLiteCache(ttl_seconds=60 * 60 * 24 * 365))

    print(f'writing to {out_path}')
    stats = generate_all(examples, llm, config, out_path)
    meta.update(run_stats=stats, summary=summarize(out_path))
    meta_path = out_path.with_suffix('.meta.json')
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'run_stats': stats, 'summary': meta['summary']}, indent=1))
    print(f'meta written to {meta_path}')
    return 0 if stats['error'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
