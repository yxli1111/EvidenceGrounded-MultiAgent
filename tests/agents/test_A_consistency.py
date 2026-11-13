"""Tests for Phase 2 Experiment A: answer-explanation consistency check (offline, mock LLM)."""
from __future__ import annotations

import dataclasses
import inspect
import json
import unittest

from contracts.schemas import AnswerDraft, QAInput, RevisionResult, VerificationFeedback
from src.agents import answer as A
from src.agents import consistency as K
from tests.agents.mocks_A import MockLLMClient, j

CFG = A.AgentConfig(model='mock/model')

# real draft shapes from main_v2 (question, draft) -- gold is never passed to the checker
STRAS = QAInput('ex-stras', 'What is the population of the city where the 122nd SS-Standarte was formed?', [])
STRAS_DRAFT = AnswerDraft('ex-stras:draft', 'Strasbourg',
                          'The 122nd SS-Standarte was formed in the city of Strasbourg. Strasbourg is the '
                          'capital and largest city of the Grand Est region of France. In 2014, the city had '
                          '276,170 inhabitants.')
YN = QAInput('ex-yn', 'Are Cypress and Ajuga both genera?', [])
YN_DRAFT = AnswerDraft('ex-yn:draft', 'yes', 'Cypress is a genus of conifers. Ajuga is a genus of flowering plants.')
OK = QAInput('ex-ok', 'Which city hosted the event?', [])
OK_DRAFT = AnswerDraft('ex-ok:draft', 'Rome', 'The event took place in Rome, Italy.')

QUOTE = 'In 2014, the city had 276,170 inhabitants.'


def reply(consistent, proposed=None, quote=None, rationale='r', mtype=None):
    return j({'consistent': consistent, 'mismatch_type': mtype, 'rationale': rationale,
              'proposed_short_answer': proposed, 'explanation_quote': quote})


def run(example, draft, text):
    llm = MockLLMClient({K.STAGE_CONSISTENCY: text})
    return (*K.check_consistency(example, draft, llm, CFG), llm)


class TestPrompt(unittest.TestCase):
    def test_prompt_registered_and_v2_by_default(self):
        self.assertEqual(A.PROMPT_FILES['answer.consistency_check'], 'answer/consistency_check_v2.md')
        self.assertRegex(A.PROMPT_VERSIONS['answer.consistency_check'], r'^consistency_check_v2@[0-9a-f]{10}$')
        self.assertTrue((A.PROMPTS_DIR / 'answer' / 'consistency_check_v1.md').exists())   # kept for the dev v1 run

    def test_v1_selectable_by_env_for_reproduction(self):
        import os, subprocess, sys
        code = ('import sys; sys.path.insert(0, "."); from src.agents import answer as A; '
                'print(A.PROMPT_VERSIONS["answer.consistency_check"])')
        out = subprocess.run([sys.executable, '-c', code], cwd=str(A.PROMPTS_DIR.parent), capture_output=True,
                             text=True, env=dict(os.environ, A_CONSISTENCY_PROMPT='v1'), check=True).stdout
        self.assertIn('consistency_check_v1@45cbd00fba', out)

    def test_existing_prompt_versions_unchanged(self):
        self.assertEqual(A.PROMPT_VERSIONS['answer.draft'], 'draft_v1@1dd344355e')
        self.assertEqual(A.PROMPT_VERSIONS['baselines.self_revise'], 'self_revise_v2@6679f79380')

    def test_checker_sees_only_question_answer_explanation(self):
        ex = QAInput('ex', 'Q?', context=[__import__('contracts.schemas', fromlist=['Sentence']).Sentence(
            'T', 0, 'SECRET CONTEXT SENTENCE')])
        *_, llm = run(ex, OK_DRAFT, reply(True))
        sent = json.dumps(llm.calls[0].messages)
        self.assertIn('Q?', sent)
        self.assertIn('The event took place in Rome', sent)
        self.assertNotIn('SECRET CONTEXT SENTENCE', sent)       # no context
        self.assertEqual(llm.stages(), [K.STAGE_CONSISTENCY])   # exactly one call

    def test_no_gold_parameter_anywhere(self):
        for fn in (K.check_consistency, K.apply_gate, A.AnswerAgent.consistency_check):
            self.assertNotIn('gold', ' '.join(inspect.signature(fn).parameters))


class TestGate(unittest.TestCase):
    def test_consistent_keeps_draft(self):
        fb, rev, dec, _ = run(OK, OK_DRAFT, reply(True))
        self.assertIsNone(rev)
        self.assertEqual(fb.items, [])
        self.assertEqual((dec.flagged, dec.edited, dec.gate), (False, False, K.GATE_CONSISTENT))
        self.assertEqual(dec.final_short_answer, 'Rome')

    def test_intermediate_hop_is_edited_from_explanation(self):
        fb, rev, dec, _ = run(STRAS, STRAS_DRAFT, reply(False, '276,170 inhabitants', QUOTE,
                                                        'Question asks population; explanation states it.'))
        self.assertTrue(dec.edited)
        self.assertEqual(dec.gate, K.GATE_EDITED)
        self.assertIsInstance(rev, RevisionResult)
        self.assertEqual(rev.final_answer, '276,170 inhabitants')
        self.assertEqual(rev.final_explanation, STRAS_DRAFT.explanation)      # explanation preserved
        self.assertEqual(rev.feedback_id, fb.feedback_id)
        self.assertEqual(fb.feedback_id, 'ex-stras:draft:M1C:consistency')
        item = fb.items[0]
        self.assertEqual((item.scope, item.target_id, item.label, item.evidence_ids),
                         ('draft', 'ex-stras:draft', None, []))
        self.assertEqual(STRAS_DRAFT.short_answer, 'Strasbourg')              # original draft untouched

    def test_quote_with_curly_quotes_and_spacing_still_matches(self):
        d = AnswerDraft('x:draft', 'Raven', "Tara Strong voices Raven in the spin-off ‘Teen Titans Go!’.")
        _, rev, dec, _ = run(QAInput('x', 'Which series?', []), d,
                             reply(False, 'Teen Titans Go!', "Tara Strong voices Raven in the spin-off 'Teen Titans Go!'."))
        self.assertTrue(dec.edited, dec.gate)
        self.assertEqual(rev.final_answer, 'Teen Titans Go!')

    def test_proposal_not_in_explanation_is_rejected(self):
        _, rev, dec, _ = run(STRAS, STRAS_DRAFT, reply(False, '280,000', QUOTE))
        self.assertEqual(dec.gate, K.GATE_SPAN_NOT_IN_QUOTE)
        self.assertTrue(dec.flagged)
        self.assertFalse(dec.edited)
        self.assertIsNone(rev)

    def test_invented_quote_is_rejected(self):
        _, rev, dec, _ = run(STRAS, STRAS_DRAFT, reply(False, '276,170', 'The city has 276,170 people.'))
        self.assertEqual(dec.gate, K.GATE_QUOTE_NOT_FOUND)
        self.assertIsNone(rev)

    def test_missing_quote_or_proposal_is_rejected(self):
        self.assertEqual(run(STRAS, STRAS_DRAFT, reply(False, '276,170 inhabitants', None))[2].gate,
                         K.GATE_QUOTE_NOT_FOUND)
        self.assertEqual(run(STRAS, STRAS_DRAFT, reply(False, None, QUOTE))[2].gate, K.GATE_NO_PROPOSAL)

    def test_same_answer_is_not_an_edit(self):
        _, rev, dec, _ = run(OK, OK_DRAFT, reply(False, 'the Rome', 'The event took place in Rome, Italy.'))
        self.assertEqual(dec.gate, K.GATE_SAME_ANSWER)
        self.assertIsNone(rev)

    def test_long_proposal_is_rejected(self):
        d = AnswerDraft('x:draft', 'A', 'one two three four five six seven eight nine ten words here.')
        _, rev, dec, _ = run(QAInput('x', 'Q?', []), d,
                             reply(False, 'one two three four five six seven eight nine', d.explanation))
        self.assertEqual(dec.gate, K.GATE_TOO_LONG)
        self.assertIsNone(rev)

    def test_yes_no_flip_requires_yes_no_both_sides_and_real_quote(self):
        q = 'Ajuga is a genus of flowering plants.'
        _, rev, dec, _ = run(YN, YN_DRAFT, reply(False, 'no', q))
        self.assertTrue(dec.edited)
        self.assertEqual(rev.final_answer, 'no')
        self.assertEqual(run(YN, YN_DRAFT, reply(False, 'Ajuga', q))[2].gate, K.GATE_YESNO_INVALID)
        self.assertEqual(run(YN, YN_DRAFT, reply(False, 'no', 'Ajuga is not a genus.'))[2].gate,
                         K.GATE_QUOTE_NOT_FOUND)

    def test_v2_mismatch_type_recorded_and_validated(self):
        _, rev, dec, _ = run(STRAS, STRAS_DRAFT, reply(False, '276,170 inhabitants', QUOTE,
                                                      mtype='stopped_at_intermediate_step'))
        self.assertTrue(dec.edited)
        self.assertEqual(dec.mismatch_type, 'stopped_at_intermediate_step')
        _, rev, dec, _ = run(STRAS, STRAS_DRAFT, reply(False, '276,170 inhabitants', QUOTE, mtype='other'))
        self.assertEqual(dec.gate, K.GATE_BAD_TYPE)
        self.assertIsNone(rev)

    def test_yes_no_reversed_type_requires_yes_no_answers(self):
        _, rev, dec, _ = run(STRAS, STRAS_DRAFT, reply(False, '276,170 inhabitants', QUOTE, mtype='yes_no_reversed'))
        self.assertEqual(dec.gate, K.GATE_YESNO_INVALID)
        _, rev, dec, _ = run(YN, YN_DRAFT, reply(False, 'no', 'Ajuga is a genus of flowering plants.',
                                                 mtype='different_final_answer'))
        self.assertEqual(dec.gate, K.GATE_YESNO_INVALID)

    def test_consistent_reply_has_no_mismatch_type(self):
        _, _, dec, _ = run(OK, OK_DRAFT, reply(True, mtype='different_final_answer'))
        self.assertIsNone(dec.mismatch_type)

    def test_string_booleans_and_null_strings(self):
        _, rev, dec, _ = run(OK, OK_DRAFT, j({'consistent': 'true', 'rationale': 'ok',
                                              'proposed_short_answer': 'null', 'explanation_quote': 'none'}))
        self.assertEqual(dec.gate, K.GATE_CONSISTENT)

    def test_unparseable_output_raises_after_one_repair(self):
        llm = MockLLMClient({K.STAGE_CONSISTENCY: 'nope', K.STAGE_CONSISTENCY + '.repair': ['still nope']})
        with self.assertRaises(A.AgentOutputError) as ctx:
            K.check_consistency(OK, OK_DRAFT, llm, CFG)
        self.assertEqual(ctx.exception.raw_outputs, ['nope', 'still nope'])

    def test_decision_is_json_serialisable_and_complete(self):
        _, _, dec, _ = run(STRAS, STRAS_DRAFT, reply(False, '276,170 inhabitants', QUOTE))
        d = json.loads(json.dumps(dec.to_dict()))
        self.assertEqual(set(d), {'flagged', 'mismatch_type', 'edited', 'gate', 'rationale', 'proposed_short_answer',
                                  'explanation_quote', 'original_short_answer', 'final_short_answer'})


class TestStrictVariant(unittest.TestCase):
    def test_strict_blocks_edit_when_explanation_also_contains_original(self):
        d = AnswerDraft('v:draft', 'North Atlantic Conference',
                        'The America East Conference was known as the North Atlantic Conference from 1988 to 1996.')
        _, _, dec, _ = run(QAInput('v', 'Formerly known as what?', []), d,
                           reply(False, 'America East Conference', d.explanation, mtype='different_final_answer'))
        self.assertTrue(dec.edited)                       # standard gate edits
        self.assertFalse(K.strict_gate_allows(d, dec))    # strict variant keeps the draft

    def test_strict_keeps_intermediate_hop_fix_and_yes_no(self):
        _, _, dec, _ = run(STRAS, STRAS_DRAFT, reply(False, '276,170 inhabitants', QUOTE))
        self.assertFalse(K.strict_gate_allows(STRAS_DRAFT, dec))   # 'Strasbourg' is in the explanation
        d = AnswerDraft('h:draft', 'Raven', "Her best-known role is in 'Teen Titans Go!'.")
        _, _, dec, _ = run(QAInput('h', 'Which series?', []), d,
                           reply(False, 'Teen Titans Go!', d.explanation))
        self.assertTrue(K.strict_gate_allows(d, dec))
        _, _, dec, _ = run(YN, YN_DRAFT, reply(False, 'no', 'Ajuga is a genus of flowering plants.'))
        self.assertTrue(K.strict_gate_allows(YN_DRAFT, dec.to_dict()))

    def test_strict_never_adds_edits(self):
        _, _, dec, _ = run(OK, OK_DRAFT, reply(True))
        self.assertFalse(K.strict_gate_allows(OK_DRAFT, dec))


class TestAgentEntryPoint(unittest.TestCase):
    def test_agent_method_accepts_dicts_like_runner_cache(self):
        agent = A.AnswerAgent(CFG)
        llm = MockLLMClient({K.STAGE_CONSISTENCY: reply(False, '276,170 inhabitants', QUOTE)})
        ex = {'example_id': 'ex-stras', 'question': STRAS.question, 'context': []}
        fb, rev, dec = agent.consistency_check(ex, dataclasses.asdict(STRAS_DRAFT), llm)
        self.assertIsInstance(fb, VerificationFeedback)
        self.assertEqual(rev.final_answer, '276,170 inhabitants')

    def test_pure_gate_never_calls_llm(self):
        self.assertEqual(K.apply_gate(OK_DRAFT, True, None, None), (K.GATE_CONSISTENT, None))


if __name__ == '__main__':
    unittest.main()
