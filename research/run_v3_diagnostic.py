"""V3 sufficiency-only diagnostics for saved M4 main_v2 records.

Run from repo root:
 python -m research.run_v3_diagnostic --dry-run
 python -m research.run_v3_diagnostic --limit 100

No retrieval, revision, or gold labels are supplied to the LLM.
"""
import argparse
import csv
import inspect
import json
import os
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

MODEL = 'openai/gpt-4o-mini'
CSV_COLUMNS = ['example_id', 'status', 'question', 'draft_short_answer', 'v2_label',
               'v3_label', 'missing_information', 'v3_rationale', 'v3_evidence_ids',
               'retrieved_evidence_count', 'api_calls', 'input_tokens', 'output_tokens',
               'cost_usd', 'elapsed_seconds', 'error', 'reused_previous',
               'v2_feedback_differs_from_previous']


def to_dict(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, 'model_dump'):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    if hasattr(value, '__dict__'):
        return vars(value)
    return {'repr': repr(value)}


def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def write_views(out, rows, total_requested, skipped):
    output = out / 'v3_diagnostic_results.csv'
    with output.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in rows:
            result = item.get('v3_result') or {}
            calls = item.get('calls') or []
            usages = [c.get('response', {}).get('usage') or {} for c in calls if isinstance(c.get('response'), dict)]
            label_items = (item.get('v2_feedback') or {}).get('items') or []
            writer.writerow({
                'example_id': item['example_id'], 'status': item['status'],
                'question': item.get('question'), 'draft_short_answer': item.get('draft_short_answer'),
                'v2_label': '|'.join(str(x.get('label')) for x in label_items),
                'v3_label': result.get('answer_sufficiency'),
                'missing_information': json.dumps(result.get('missing_information'), ensure_ascii=False),
                'v3_rationale': result.get('rationale'),
                'v3_evidence_ids': json.dumps(result.get('evidence_ids'), ensure_ascii=False),
                'retrieved_evidence_count': item.get('retrieved_evidence_count'),
                'api_calls': len(calls),
                'input_tokens': sum(int(u.get('input_tokens') or 0) for u in usages),
                'output_tokens': sum(int(u.get('output_tokens') or 0) for u in usages),
                'cost_usd': round(sum(float(u.get('cost_usd') or 0) for u in usages), 9),
                'elapsed_seconds': item.get('elapsed_seconds'), 'error': item.get('error'),
                'reused_previous': item.get('reused_previous', False),
                'v2_feedback_differs_from_previous': item.get('v2_feedback_differs_from_previous', False),
            })
    summary = {
        'requested_m4_records': total_requested,
        'successful_diagnostics': sum(x.get('status') == 'ok' for x in rows),
        'failed_diagnostics': sum(x.get('status') == 'error' for x in rows),
        'unavailable_m4_records': skipped,
        'labels': dict(Counter((x.get('v3_result') or {}).get('answer_sufficiency') for x in rows if x.get('status') == 'ok')),
        'reused_previous': sum(bool(x.get('reused_previous')) for x in rows),
        'v2_feedback_mismatches': sum(bool(x.get('v2_feedback_differs_from_previous')) for x in rows),
        'total_cost_usd_including_reused': round(sum(sum(float((c.get('response') or {}).get('usage', {}).get('cost_usd') or 0) for c in (x.get('calls') or [])) for x in rows), 8),
        'notes': 'V3 labels are predictions, not human-verified accuracy. Costs include reused calls and exclude failed calls without usage data. M4 error records are excluded from API calls.',
    }
    atomic_json(out / 'v3_diagnostic_summary.json', summary)
    return summary


class ObservedLLM:
    def __init__(self, client):
        self.client, self.calls = client, []

    def generate(self, **kwargs):
        start = time.perf_counter()
        entry = {'stage': kwargs.get('stage'), 'model': kwargs.get('model')}
        try:
            response = self.client.generate(**kwargs)
            entry['response'] = to_dict(response)
            return response
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            entry['elapsed_seconds'] = round(time.perf_counter() - start, 4)
            self.calls.append(entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', default='results/runs/main_v2/records.jsonl')
    parser.add_argument('--out', default='data/phase2_diagnostic/v3_results_100')
    parser.add_argument('--previous', default='data/phase2_diagnostic/v3_results/v3_diagnostic_records.jsonl')
    parser.add_argument('--previous-sample', default='data/phase2_diagnostic/m4_diagnostic_sample10.csv')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be >= 1')
    raw = load_jsonl(args.records)
    m4 = [r for r in raw if r.get('method_id') == 'M4']
    if not m4:
        raise RuntimeError(f'No M4 records in {args.records}')
    ids = [str(r['example_id']) for r in m4]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Duplicate M4 example IDs in records; resolve before running')
    chosen = m4[:args.limit]
    valid, skipped = [], []
    for record in chosen:
        draft = record.get('draft') or {}
        evidence = (record.get('evidence') or {}).get(draft.get('draft_id'))
        if record.get('status') != 'ok' or not draft or not isinstance(evidence, list):
            skipped.append({'example_id': record['example_id'], 'status': record.get('status'), 'error': record.get('error')})
        else:
            valid.append((record, evidence))

    # Previously run sample must be matched to the same question/draft/evidence.
    prior_samples = {}
    if Path(args.previous_sample).exists():
        with open(args.previous_sample, newline='', encoding='utf-8-sig') as stream:
            prior_samples = {r['example_id']: r for r in csv.DictReader(stream)}
    previous = {str(x['example_id']): x for x in load_jsonl(args.previous) if x.get('status') == 'ok'}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / 'v3_diagnostic_records.jsonl'
    existing = {str(x['example_id']): x for x in load_jsonl(dest)}
    reuse = {}
    for record, evidence in valid:
        eid = str(record['example_id'])
        if eid in existing:
            reuse[eid] = existing[eid]
            continue
        prev, sample = previous.get(eid), prior_samples.get(eid)
        if prev and sample:
            same = (prev.get('question') == record.get('question') == sample.get('question')
                    and prev.get('draft_short_answer') == (record.get('draft') or {}).get('short_answer') == sample.get('draft_short_answer')
                    and (record.get('draft') or {}).get('explanation') == sample.get('draft_explanation')
                    and json.loads(sample['evidence_json']) == evidence
                    and all(c.get('model') in (None, MODEL) and (c.get('response') or {}).get('usage', {}).get('model', MODEL) == MODEL for c in prev.get('calls', [])))
            if same:
                entry = dict(prev)
                entry['v2_feedback_differs_from_previous'] = prev.get('v2_feedback') != record.get('feedback')
                entry['v2_feedback'] = record.get('feedback')
                entry['retrieved_evidence_count'] = len(evidence)
                entry['reused_previous'] = True
                reuse[eid] = entry
    pending = [r for r, _ in valid if str(r['example_id']) not in reuse or (args.retry_errors and reuse[str(r['example_id'])].get('status') == 'error')]
    print(f'M4 selected={len(chosen)} valid={len(valid)} unavailable={len(skipped)} reuse={len(reuse)} pending={len(pending)}')
    if args.dry_run:
        print('Dry run only. No API calls. Existing results are unchanged.')
        if skipped:
            print('Unavailable IDs:', [x['example_id'] for x in skipped])
        return

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if pending and not (os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')):
        raise RuntimeError('API key not loaded')
    from contracts.schemas import QAInput, AnswerDraft, Evidence, Sentence
    from src.core.llm_client import OpenRouterLLMClient
    from src.data.hotpot_loader import load_hotpotqa
    from src.verification.dual_dimension import verify_answer_sufficiency
    from src.verification import configure
    configure(model=MODEL, temperature=0.0, verification_max_tokens=2048, revision_max_tokens=2048)
    examples, _ = load_hotpotqa(split='validation')
    lookup = {str(e['example_id']): e for e in examples}
    missing = [r['example_id'] for r in pending if str(r['example_id']) not in lookup]
    if missing:
        raise RuntimeError(f'Original validation examples missing: {missing}')
    client = ObservedLLM(OpenRouterLLMClient()) if pending else None
    # Write reusable results first so interruption does not lose prior successful work.
    if not dest.exists() and reuse:
        with dest.open('w', encoding='utf-8') as stream:
            for r, _ in valid:
                if str(r['example_id']) in reuse:
                    stream.write(json.dumps(reuse[str(r['example_id'])], ensure_ascii=False, default=str) + '\n')
    elif dest.exists():
        with dest.open('a', encoding='utf-8') as stream:
            for r, _ in valid:
                eid = str(r['example_id'])
                if eid in reuse and eid not in existing:
                    stream.write(json.dumps(reuse[eid], ensure_ascii=False, default=str) + '\n')
    for i, record in enumerate(pending, 1):
        eid = str(record['example_id'])
        source = lookup[eid]
        draft_data = record['draft']
        evidence_raw = record['evidence'][draft_data['draft_id']]
        if source['question'] != record['question']:
            raise RuntimeError(f'{eid}: source question mismatch')
        sentence_params = inspect.signature(Sentence).parameters
        context = [Sentence(**{('doc_title' if 'doc_title' in sentence_params else 'title'): s['title'],
                               'sentence_id': int(s['sentence_id']), 'text': s['text']}) for s in source['context']]
        qa = QAInput(example_id=eid, question=source['question'], context=context,
                     question_type=source.get('question_type'))
        draft = AnswerDraft(**draft_data)
        evidence = [Evidence(**item) for item in evidence_raw]
        client.calls.clear()
        started = time.perf_counter()
        try:
            result = to_dict(verify_answer_sufficiency(qa, draft, evidence, client))
            status, error = 'ok', None
        except Exception as exc:
            result, status, error = None, 'error', f'{type(exc).__name__}: {exc}'
        entry = {'example_id': eid, 'status': status, 'question': source['question'],
                 'draft_short_answer': draft.short_answer, 'v2_feedback': record.get('feedback'),
                 'v3_result': result, 'error': error,
                 'elapsed_seconds': round(time.perf_counter() - started, 4),
                 'calls': list(client.calls), 'retrieved_evidence_count': len(evidence),
                 'reused_previous': False, 'v2_feedback_differs_from_previous': False}
        with dest.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')
        reuse[eid] = entry
        print(f'[{i}/{len(pending)}] {eid}: {status} label={(result or {}).get("answer_sufficiency")} elapsed={entry["elapsed_seconds"]}s', flush=True)
    # If a failed record was retried, retain only its latest result in the exported views.
    latest = {str(x['example_id']): x for x in load_jsonl(dest)}
    ordered = [latest[str(r['example_id'])] for r, _ in valid if str(r['example_id']) in latest]
    summary = write_views(out, ordered, len(chosen), skipped)
    atomic_json(out / 'm4_unavailable_records.json', skipped)
    print('Saved:', dest, out / 'v3_diagnostic_results.csv', out / 'v3_diagnostic_summary.json')
    print('Summary:', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
