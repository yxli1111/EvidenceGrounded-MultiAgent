"""Acceptance tests for Contributor A (plan v2 section 4).

Run from the repository root with either:
    python -m unittest discover -s tests -t .
    python -m pytest tests/agents
Standard library only; no network, no API key.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import unittest

from contracts import interfaces
from contracts.schemas import AnswerDraft, RevisionResult, VerificationFeedback
from src.agents import answer as A
from src.agents import baselines as B
from src.agents import independent_critic as C
from tests.agents.mocks_A import (BRIDGE_EXAMPLE, COMPARISON_EXAMPLE, MockLLMClient,
                                  MockRevisionPort, j)

CFG = A.AgentConfig(model='mock/model', temperature=0.0, max_tokens=256)

DRAFT_JSON = j({'short_answer': 'Stockholm',
                'explanation': 'Silver Harbor was directed by Mara Lind. Lind was born in Stockholm.'})
CRITIC_JSON = j({'issues': [{'rationale': 'The context says Lind was born in Gothenburg, not Stockholm.',
                             'suggested_correction': 'Gothenburg'}],
                 'overall_comment': 'Birthplace is wrong.'})
NO_ISSUES_JSON = j({'issues': [], 'overall_comment': 'No issues found.'})
REVISED_JSON = j({'short_answer': 'Gothenburg',
                  'explanation': 'Silver Harbor was directed by Mara Lind. Lind was born in Gothenburg.'})


def shared_draft() -> AnswerDraft:
    return AnswerDraft(draft_id='mock-bridge-1:draft', short_answer='Stockholm',
                       explanation='Silver Harbor was directed by Mara Lind. Lind was born in Stockholm.')


class TestPromptsAndRendering(unittest.TestCase):
    def test_all_prompt_files_load_with_versions(self):
        for pid in A.PROMPT_FILES:
            p = A.load_prompt(pid)
            self.assertRegex(p.version, r'^[a-z_]+_v\d+@[0-9a-f]{10}$')
        self.assertEqual(set(A.PROMPT_VERSIONS), set(A.PROMPT_FILES))

    def test_context_render_preserves_title_and_sentence_ids(self):
        text = A.render_context(BRIDGE_EXAMPLE.context)
        self.assertIn('[Mara Lind]', text)
        self.assertIn('(1) Lind was born in Gothenburg and studied in Stockholm.', text)
        self.assertLess(text.index('[Silver Harbor]'), text.index('[Mara Lind]'))

    def test_prompt_version_independent_of_crlf_checkout(self):
        import shutil, tempfile
        from pathlib import Path
        orig = A.PROMPTS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(orig, Path(tmp) / 'prompts')
            for f in (Path(tmp) / 'prompts').rglob('*.md'):
                f.write_bytes(f.read_bytes().replace(b'\n', b'\r\n'))
            try:
                A.PROMPTS_DIR = Path(tmp) / 'prompts'
                crlf = A.prompt_versions()
                crlf_msgs = A.load_prompt('answer.draft').render(question='q', context='c')
            finally:
                A.PROMPTS_DIR = orig
        self.assertEqual(crlf, A.prompt_versions())
        self.assertEqual(crlf_msgs, A.load_prompt('answer.draft').render(question='q', context='c'))

    def test_empty_context(self):
        self.assertEqual(A.render_context([]), '(no context provided)')

    def test_template_tolerates_braces_and_dollars_in_values(self):
        msgs = A.load_prompt('answer.draft').render(question='Cost {x} in $USD?', context='c')
        self.assertIn('Cost {x} in $USD?', msgs[1]['content'])


class TestJsonParsing(unittest.TestCase):
    def test_plain_fenced_and_wrapped(self):
        for raw in (DRAFT_JSON, f'```json\n{DRAFT_JSON}\n```', f'Sure! {DRAFT_JSON} Hope it helps.'):
            self.assertEqual(A.parse_answer_payload(A.parse_json_object(raw))[0], 'Stockholm')

    def test_yes_no_bool_and_numbers_coerced(self):
        self.assertEqual(A.parse_answer_payload({'short_answer': False})[0], 'no')
        self.assertEqual(A.parse_answer_payload({'short_answer': 1998})[0], '1998')

    def test_rejects_empty_answer(self):
        with self.assertRaises(ValueError):
            A.parse_answer_payload({'short_answer': '  ', 'explanation': 'x'})

    def test_issue_null_correction_normalized(self):
        issues, _ = A.parse_issues_payload({'issues': [{'rationale': 'r', 'suggested_correction': 'null'},
                                                       {'rationale': ''}]})
        self.assertEqual(issues, [('r', None)])


class TestDraft(unittest.TestCase):
    def test_generate_draft(self):
        llm = MockLLMClient({A.STAGE_DRAFT: DRAFT_JSON})
        d = A.generate_draft(BRIDGE_EXAMPLE, llm, CFG)
        self.assertEqual(d, shared_draft())
        self.assertEqual(llm.stages(), [A.STAGE_DRAFT])
        call = llm.calls[0]
        self.assertEqual((call.model, call.temperature, call.max_tokens), ('mock/model', 0.0, 256))
        self.assertEqual([m['role'] for m in call.messages], ['system', 'user'])
        self.assertIn(BRIDGE_EXAMPLE.question, call.messages[1]['content'])
        self.assertIn('(0) Mara Lind (born 1961)', call.messages[1]['content'])

    def test_draft_id_is_reproducible(self):
        d1 = A.generate_draft(BRIDGE_EXAMPLE, MockLLMClient({A.STAGE_DRAFT: DRAFT_JSON}), CFG)
        d2 = A.generate_draft(BRIDGE_EXAMPLE, MockLLMClient({A.STAGE_DRAFT: DRAFT_JSON}), CFG)
        self.assertEqual(d1, d2)
        self.assertEqual(d1.draft_id, 'mock-bridge-1:draft')

    def test_one_format_repair_then_success(self):
        llm = MockLLMClient({A.STAGE_DRAFT: 'not json', A.STAGE_DRAFT + '.repair': DRAFT_JSON})
        d = A.generate_draft(BRIDGE_EXAMPLE, llm, CFG)
        self.assertEqual(d.short_answer, 'Stockholm')
        self.assertEqual(llm.stages(), [A.STAGE_DRAFT, A.STAGE_DRAFT + '.repair'])
        # repair request differs from original (cache identity) and carries the bad reply
        self.assertNotEqual(llm.calls[0].messages, llm.calls[1].messages)
        self.assertEqual(llm.calls[1].messages[2], {'role': 'assistant', 'content': 'not json'})

    def test_unrecoverable_output_raises_with_raw_outputs(self):
        llm = MockLLMClient({A.STAGE_DRAFT: 'bad', A.STAGE_DRAFT + '.repair': ['still bad']})
        with self.assertRaises(A.AgentOutputError) as ctx:
            A.generate_draft(BRIDGE_EXAMPLE, llm, CFG)
        self.assertEqual(ctx.exception.raw_outputs, ['bad', 'still bad'])
        self.assertEqual(len(llm.calls), 2)  # bounded

    def test_no_gold_leak_in_prompt(self):
        llm = MockLLMClient({A.STAGE_DRAFT: j({'short_answer': 'no', 'explanation': ''})})
        A.generate_draft(COMPARISON_EXAMPLE, llm, CFG)
        sig = inspect.signature(A.generate_draft)
        self.assertNotIn('gold', ' '.join(sig.parameters))


class TestM1(unittest.TestCase):
    def test_m1_has_no_revision_and_no_calls(self):
        out = B.run_m1(BRIDGE_EXAMPLE, shared_draft())
        self.assertIsNone(out.final)
        self.assertIsNone(out.feedback)
        self.assertEqual(out.final_answer, 'Stockholm')


class TestM2(unittest.TestCase):
    def test_exactly_one_self_check_and_one_self_revision(self):
        llm = MockLLMClient({A.STAGE_SELF_CHECK: CRITIC_JSON, A.STAGE_SELF_REVISE: REVISED_JSON})
        out = B.run_m2(BRIDGE_EXAMPLE, shared_draft(), llm, CFG)
        self.assertEqual(llm.stages(), [A.STAGE_SELF_CHECK, A.STAGE_SELF_REVISE])
        self.assertNotIn(A.STAGE_DRAFT, llm.stages())  # shared draft reused, not regenerated
        self.assertIsInstance(out.final, RevisionResult)
        self.assertEqual(out.final.final_answer, 'Gothenburg')
        self.assertEqual(out.final.draft_id, 'mock-bridge-1:draft')
        self.assertEqual(out.final.feedback_id, 'mock-bridge-1:draft:M2:self_reflection')
        self.assertEqual(out.feedback.feedback_id, out.final.feedback_id)
        revise_prompt = llm.calls[1].messages[1]['content']
        self.assertIn('Gothenburg, not Stockholm', revise_prompt)  # review is passed to revision

    def test_self_reflect_port_returns_revision_result(self):
        llm = MockLLMClient({A.STAGE_SELF_CHECK: NO_ISSUES_JSON, A.STAGE_SELF_REVISE: DRAFT_JSON})
        agent = A.AnswerAgent(CFG)
        res = agent.self_reflect(BRIDGE_EXAMPLE, shared_draft(), llm)
        self.assertIsInstance(res, RevisionResult)
        self.assertEqual(res.final_answer, 'Stockholm')
        self.assertEqual(len(llm.calls), 2)
        self.assertIn('No issues found.', llm.calls[1].messages[1]['content'])


class TestM3(unittest.TestCase):
    def test_critic_feedback_schema_compliance(self):
        tricky = j({'issues': [{'rationale': 'Birthplace wrong.', 'suggested_correction': 'Gothenburg',
                                'label': 'CONTRADICTED', 'evidence_ids': ['Mara Lind::1']}],
                    'overall_comment': 'x'})
        llm = MockLLMClient({A.STAGE_CRITIC: tricky})
        fb = C.independent_critique(BRIDGE_EXAMPLE, shared_draft(), llm, CFG)
        self.assertIsInstance(fb, VerificationFeedback)
        self.assertEqual(fb.feedback_id, 'mock-bridge-1:draft:M3:critic')
        self.assertEqual(len(fb.items), 1)
        item = fb.items[0]
        self.assertEqual(item.scope, 'draft')
        self.assertEqual(item.target_id, 'mock-bridge-1:draft')
        self.assertIsNone(item.label)          # critic may not invent verdicts
        self.assertEqual(item.evidence_ids, [])  # ... or citations
        self.assertEqual(item.suggested_correction, 'Gothenburg')
        json.dumps(dataclasses.asdict(fb))     # JSON-serializable via asdict

    def test_no_issues_gives_empty_items(self):
        fb = C.independent_critique(BRIDGE_EXAMPLE, shared_draft(),
                                    MockLLMClient({A.STAGE_CRITIC: NO_ISSUES_JSON}), CFG)
        self.assertEqual(fb.items, [])

    def test_critic_prompt_has_context_and_draft_but_is_separate_agent(self):
        llm = MockLLMClient({A.STAGE_CRITIC: NO_ISSUES_JSON})
        C.independent_critique(BRIDGE_EXAMPLE, shared_draft(), llm, CFG)
        sys_msg, user_msg = (m['content'] for m in llm.calls[0].messages)
        self.assertIn('independent Critic Agent', sys_msg)
        self.assertIn('[Mara Lind]', user_msg)
        self.assertIn('short_answer: Stockholm', user_msg)

    def test_m3_uses_exactly_one_d_mediated_revision(self):
        llm = MockLLMClient({A.STAGE_CRITIC: CRITIC_JSON})
        port = MockRevisionPort()
        draft = shared_draft()
        out = B.run_m3(BRIDGE_EXAMPLE, draft, llm, CFG, port)
        self.assertEqual(llm.stages(), [A.STAGE_CRITIC])     # A makes no revision call itself
        self.assertEqual(len(port.calls), 1)                 # exactly one D revision
        ex, d, fb = port.calls[0]
        self.assertIs(ex, BRIDGE_EXAMPLE)
        self.assertIs(d, draft)
        self.assertIs(fb, out.feedback)
        self.assertEqual(out.final.feedback_id, 'mock-bridge-1:draft:M3:critic')

    def test_m3_rejects_non_contract_revision(self):
        class Bad(MockRevisionPort):
            def revise_answer(self, *a, **k):
                return {'final_answer': 'x'}
        with self.assertRaises(TypeError):
            B.run_m3(BRIDGE_EXAMPLE, shared_draft(), MockLLMClient({A.STAGE_CRITIC: NO_ISSUES_JSON}),
                     CFG, Bad())


class TestPortConformance(unittest.TestCase):
    def test_answer_agent_matches_answer_port_signatures(self):
        for name in ('generate_draft', 'self_reflect', 'independent_critique'):
            proto = inspect.signature(getattr(interfaces.AnswerPort, name))
            impl = inspect.signature(getattr(A.AnswerAgent, name))
            self.assertEqual(list(proto.parameters), list(impl.parameters), name)

    def test_deterministic_given_same_mock(self):
        def run():
            llm = MockLLMClient({A.STAGE_DRAFT: DRAFT_JSON, A.STAGE_CRITIC: CRITIC_JSON,
                                 A.STAGE_SELF_CHECK: CRITIC_JSON, A.STAGE_SELF_REVISE: REVISED_JSON})
            agent = A.AnswerAgent(CFG)
            d = agent.generate_draft(BRIDGE_EXAMPLE, llm)
            return (d, agent.self_reflect(BRIDGE_EXAMPLE, d, llm),
                    agent.independent_critique(BRIDGE_EXAMPLE, d, llm), llm.stages())
        self.assertEqual(run(), run())

    def test_config_from_dict(self):
        cfg = A.AgentConfig.from_dict({'model': 'm', 'temperature': 0.2, 'unrelated': 1})
        self.assertEqual((cfg.model, cfg.temperature, cfg.max_tokens), ('m', 0.2, 512))


if __name__ == '__main__':
    unittest.main()
