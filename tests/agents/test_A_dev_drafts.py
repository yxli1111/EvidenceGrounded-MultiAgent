"""Tests for research/A_generate_dev_drafts.py (offline: fake loader + MockLLMClient)."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.agents import answer as A
from tests.agents.mocks_A import MockLLMClient, j

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('A_gen', ROOT / 'research' / 'A_generate_dev_drafts.py')
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

GOLD_SENTINEL = 'GOLD-ANSWER-MUST-NOT-LEAK'


def fake_loader(split='validation', limit=None):
    """Mimics B's load_hotpotqa: dict inputs + separate gold lookup."""
    inputs, gold = [], {}
    for k in range(3):
        eid = f'id{k}'
        inputs.append({'example_id': eid, 'question': f'Question {k}?',
                       'question_type': 'bridge' if k % 2 == 0 else 'comparison',
                       'context': [{'title': 'T', 'sentence_id': 0, 'text': f'Fact {k}.'},
                                   {'title': 'U', 'sentence_id': 0, 'text': 'Other.'}]})
        gold[eid] = {'example_id': eid, 'gold_answer': GOLD_SENTINEL, 'gold_supporting_facts': []}
    return inputs[:limit], gold


def draft_reply(call):
    q = call.messages[1]['content'].split('Question: ')[1].split('\n')[0]
    return j({'short_answer': f'ans for {q}', 'explanation': 'A fact. Another fact.'})


def run(argv, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = G.main(argv, **kw)
    return code, buf.getvalue()


class TestDevDrafts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'drafts.jsonl'

    def tearDown(self):
        self.tmp.cleanup()

    def records(self):
        return [json.loads(l) for l in self.out.read_text(encoding='utf-8').splitlines()]

    def test_generates_one_draft_per_example_without_gold(self):
        llm = MockLLMClient({A.STAGE_DRAFT: draft_reply})
        code, _ = run(['--model', 'm', '--limit', '3', '--out', str(self.out)], loader=fake_loader, llm=llm)
        self.assertEqual(code, 0)
        recs = self.records()
        self.assertEqual([r['example_id'] for r in recs], ['id0', 'id1', 'id2'])
        self.assertTrue(all(r['status'] == 'ok' for r in recs))
        self.assertEqual(recs[0]['draft']['draft_id'], 'id0:draft')
        self.assertEqual(len(recs[0]['usage']), 1)
        everything_sent = json.dumps([c.messages for c in llm.calls])
        self.assertNotIn(GOLD_SENTINEL, everything_sent)
        self.assertNotIn(GOLD_SENTINEL, self.out.read_text(encoding='utf-8'))
        meta = json.loads(self.out.with_suffix('.meta.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['agent_config']['model'], 'm')
        self.assertEqual(meta['prompt_versions'], A.PROMPT_VERSIONS)
        self.assertEqual(meta['summary']['ok'], 3)

    def test_resume_skips_done_examples(self):
        run(['--model', 'm', '--limit', '2', '--out', str(self.out)], loader=fake_loader,
            llm=MockLLMClient({A.STAGE_DRAFT: draft_reply}))
        llm2 = MockLLMClient({A.STAGE_DRAFT: draft_reply})
        run(['--model', 'm', '--limit', '3', '--out', str(self.out)], loader=fake_loader, llm=llm2)
        self.assertEqual(len(llm2.calls), 1)              # only id2 is new
        self.assertEqual([r['example_id'] for r in self.records()], ['id0', 'id1', 'id2'])

    def test_parse_failure_recorded_not_dropped(self):
        llm = MockLLMClient({A.STAGE_DRAFT: 'garbage', A.STAGE_DRAFT + '.repair': 'garbage'})
        code, _ = run(['--model', 'm', '--limit', '1', '--out', str(self.out)], loader=fake_loader, llm=llm)
        self.assertEqual(code, 1)
        rec = self.records()[0]
        self.assertEqual(rec['status'], 'error')
        self.assertEqual(rec['raw_outputs'], ['garbage', 'garbage'])

    def test_budget_stop_halts_run(self):
        class BudgetExceededError(Exception):
            pass

        class StopLLM(MockLLMClient):
            def generate(self, **kw):
                if self.calls:
                    raise BudgetExceededError('cap reached')
                return super().generate(**kw)
        llm = StopLLM({A.STAGE_DRAFT: draft_reply})
        code, out = run(['--model', 'm', '--limit', '3', '--out', str(self.out)], loader=fake_loader, llm=llm)
        recs = self.records()
        self.assertEqual([r['status'] for r in recs], ['ok', 'error'])   # stopped, id2 not attempted
        self.assertIn('STOP', out)
        self.assertEqual(code, 1)

    def test_id_manifest_check(self):
        good = Path(self.tmp.name) / 'ids.json'
        good.write_text(json.dumps(['id0', 'id1']), encoding='utf-8')
        code, out = run(['--model', 'm', '--limit', '2', '--ids-file', str(good), '--dry-run'],
                        loader=fake_loader)
        self.assertEqual(code, 0)
        self.assertIn('manifest check passed', out)
        bad = Path(self.tmp.name) / 'bad.json'
        bad.write_text(json.dumps({'ids': [{'example_id': 'id0'}, {'example_id': 'zzz'}]}), encoding='utf-8')
        code, out = run(['--model', 'm', '--limit', '2', '--ids-file', str(bad), '--dry-run'],
                        loader=fake_loader)
        self.assertEqual(code, 2)

    def test_dry_run_makes_no_calls(self):
        code, out = run(['--model', 'm', '--limit', '2', '--dry-run'], loader=fake_loader,
                        llm=MockLLMClient({}))
        self.assertEqual(code, 0)
        self.assertIn('[user]', out)
        self.assertIn('Question 0?', out)


if __name__ == '__main__':
    unittest.main()
