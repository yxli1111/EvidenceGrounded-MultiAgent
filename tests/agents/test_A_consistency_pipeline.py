"""End-to-end test of research/A_consistency_experiment.py + A_consistency_analysis.py (mock LLM).

The analysis half needs E's src.evaluation package and is skipped where it is absent.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.agents import consistency as K
from tests.agents.mocks_A import MockLLMClient, j

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'research' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXP = _load('A_consistency_experiment')
try:
    import src.evaluation.answer_metrics  # noqa: F401  (E)
    HAVE_E = True
except ImportError:
    HAVE_E = False

DRAFTS = {
    'e1': ('What is the population of the city where X was formed?', 'Strasbourg',
           'X was formed in Strasbourg. In 2014, the city had 276,170 inhabitants.', '276,170 inhabitants'),
    'e2': ('Which city hosted it?', 'Rome', 'It took place in Rome.', 'Rome'),
    'e3': ('Who directed it?', 'Mara Lind', 'It was directed by Mara Lind.', 'Mara Lind'),
    'e4': ('Which series?', 'Raven', "She voices Raven in 'Teen Titans Go!'.", 'Teen Titans Go!'),
}


def make_source(tmp: Path) -> Path:
    src = tmp / 'src_run'
    src.mkdir()
    with (src / 'records.jsonl').open('w', encoding='utf-8') as fh:
        for eid, (q, a, e, _) in DRAFTS.items():
            fh.write(json.dumps({'example_id': eid, 'method_id': 'M1', 'status': 'ok', 'question': q,
                                 'question_type': 'bridge', 'model_config': {'model': 'mock/model'},
                                 'draft': {'draft_id': f'{eid}:draft', 'short_answer': a, 'explanation': e}}) + '\n')
            fh.write(json.dumps({'example_id': eid, 'method_id': 'M2', 'status': 'ok'}) + '\n')
    (src / 'gold_lookup.json').write_text(json.dumps(
        {eid: {'gold_answer': g, 'gold_supporting_facts': []} for eid, (*_, g) in DRAFTS.items()}), encoding='utf-8')
    return src


def checker(call):
    user = call.messages[1]['content']
    if 'Strasbourg' in user:
        return j({'consistent': False, 'rationale': 'asks population', 'proposed_short_answer': '276,170 inhabitants',
                  'explanation_quote': 'In 2014, the city had 276,170 inhabitants.'})
    if 'Mara Lind' in user:   # a harmful flag that the gate must block (proposal not in explanation)
        return j({'consistent': False, 'rationale': 'x', 'proposed_short_answer': 'Someone Else',
                  'explanation_quote': 'It was directed by Mara Lind.'})
    if 'Teen Titans' in user:
        return 'not json'
    return j({'consistent': True, 'rationale': 'ok', 'proposed_short_answer': None, 'explanation_quote': None})


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.src = make_source(self.t)
        self.runs = self.t / 'runs'

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, llm):
        return quiet(EXP.main, ['--source-run', str(self.src), '--run-name', 'm1c', '--results-root', str(self.runs)],
                     llm=llm)

    def test_experiment_records_paired_drafts_and_errors(self):
        llm = MockLLMClient({K.STAGE_CONSISTENCY: checker,
                             K.STAGE_CONSISTENCY + '.repair': ['still not json']})
        code = self._run(llm)
        self.assertEqual(code, 1)                                         # one error recorded
        recs = [json.loads(l) for l in (self.runs / 'm1c' / 'records.jsonl').read_text().splitlines()]
        self.assertEqual([r['example_id'] for r in recs], ['e1', 'e2', 'e3', 'e4'])
        by = {r['example_id']: r for r in recs}
        self.assertEqual(by['e1']['final']['final_answer'], '276,170 inhabitants')
        self.assertEqual(by['e1']['final']['final_explanation'], DRAFTS['e1'][2])
        self.assertIsNone(by['e3']['final'])                              # gate blocked the bad edit
        self.assertEqual(by['e3']['decision']['gate'], K.GATE_SPAN_NOT_IN_QUOTE)
        self.assertEqual(by['e4']['status'], 'error')
        self.assertEqual(by['e4']['raw_outputs'], ['not json', 'still not json'])
        sent = json.dumps([c.messages for c in llm.calls])
        self.assertNotIn('276,170 inhabitants"', sent.split('Strasbourg')[0])  # nothing before the draft
        for eid, (*_, g) in DRAFTS.items():
            if g not in DRAFTS[eid][2]:
                self.assertNotIn(g, sent)                                 # gold never sent
        meta = json.loads((self.runs / 'm1c' / 'metadata.json').read_text())
        self.assertFalse(meta['gold_used_in_prompts'])
        self.assertIn('answer.consistency_check', meta['prompt_versions'])

    def test_exclude_run_skips_previously_used_ids(self):
        other = self.t / 'other_run'
        other.mkdir()
        (other / 'records.jsonl').write_text(json.dumps({'example_id': 'e1', 'method_id': 'M1'}) + '\n' +
                                             json.dumps({'example_id': 'e2', 'method_id': 'M2'}) + '\n')
        llm = MockLLMClient({K.STAGE_CONSISTENCY: checker, K.STAGE_CONSISTENCY + '.repair': ['x']})
        quiet(EXP.main, ['--source-run', str(self.src), '--run-name', 'fresh', '--results-root', str(self.runs),
                         '--exclude-run', str(other)], llm=llm)
        recs = [json.loads(l)['example_id'] for l in (self.runs / 'fresh' / 'records.jsonl').read_text().splitlines()]
        self.assertEqual(recs, ['e3', 'e4'])
        meta = json.loads((self.runs / 'fresh' / 'metadata.json').read_text())
        self.assertEqual(meta['n_excluded_ids'], 2)

    def test_resume_skips_ok_examples(self):
        self._run(MockLLMClient({K.STAGE_CONSISTENCY: checker, K.STAGE_CONSISTENCY + '.repair': ['x']}))
        llm2 = MockLLMClient({K.STAGE_CONSISTENCY: checker, K.STAGE_CONSISTENCY + '.repair': ['x']})
        self._run(llm2)
        self.assertEqual(len([c for c in llm2.calls if c.stage == K.STAGE_CONSISTENCY]), 1)   # only e4 retried

    @unittest.skipUnless(HAVE_E, "E's src.evaluation not on this branch")
    def test_analysis_metrics_and_case_table(self):
        self._run(MockLLMClient({K.STAGE_CONSISTENCY: checker, K.STAGE_CONSISTENCY + '.repair': ['x']}))
        AN = _load('A_consistency_analysis')
        out = self.t / 'an'
        s = quiet(AN.analyse, self.src, self.runs / 'm1c', out)
        self.assertEqual(s['n_paired'], 4)
        self.assertEqual(s['n_error'], 1)
        self.assertEqual(s['metrics']['M1']['em'], 0.5)                   # e2, e3 correct
        self.assertEqual(s['metrics']['M1C']['em'], 0.75)                 # + e1 fixed
        self.assertEqual(s['transitions'], {'wrong->correct': 1, 'correct->correct': 2, 'wrong->error': 1})
        self.assertEqual(s['headroom']['wrong_drafts_with_gold_in_explanation'], 2)
        self.assertEqual(s['headroom']['fixed'], 1)
        self.assertEqual(s['paired_bootstrap_M1C_minus_M1']['em']['n_resamples'], 5000)
        # strict variant: e1's draft 'Strasbourg' is in its explanation -> edit dropped
        self.assertEqual(s['secondary_strict_gate']['metrics']['em'], 0.5)
        self.assertEqual(s['secondary_strict_gate']['edit_rate'], 0.0)
        with (out / 'case_table.csv').open(encoding='utf-8-sig') as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({r['example_id'] for r in rows}, {'e1', 'e3', 'e4'})
        for col in AN.REVIEW_COLS:
            self.assertIn(col, rows[0])
        # manual review scoring
        for r in rows:
            r['review_flag_valid'] = 'Y' if r['example_id'] == 'e1' else ('N' if r['flagged'] == '1' else '')
            r['review_edit_valid'] = 'Y' if r['edited'] == '1' else ''
        rv = out / 'reviewed.csv'
        with rv.open('w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        self.assertEqual(AN.score_review(rv), {'flags_reviewed': 2, 'flag_precision': 0.5,
                                               'edits_reviewed': 1, 'edit_precision': 1.0})


if __name__ == '__main__':
    unittest.main()
