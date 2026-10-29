"""Exercise A's actual self_revise with the delivered prompt and real contracts.

Only model replies are scripted. These tests verify prompt wiring and result shape,
not whether a real model will always obey the answer-format instruction.
Set D_A_ROOT / D_CONTRACTS_ROOT for a checkout outside the original supplied paths.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


_A_DEFAULT = r"C:\Users\duradura\Downloads\EvidenceGrounded-MultiAgent-feat-A-baselines"
_CONTRACTS_DEFAULT = (
    r"C:\Users\duradura\Downloads\EvidenceGrounded-MultiAgent-0a2ab15aaecd0c71d949815743e25895a8f896f5"
)


_CHILD = textwrap.dedent('''
    import hashlib
    import json
    from pathlib import Path
    import shutil
    import sys
    import tempfile
    from unittest.mock import patch

    sys.dont_write_bytecode = True
    project_root, a_root, contracts_root = map(Path, sys.argv[1:4])
    mode = sys.argv[4]
    sys.path[:0] = [str(contracts_root), str(a_root)]
    from contracts.schemas import (QAInput, Sentence, AnswerDraft, FeedbackItem,
        VerificationFeedback, RevisionResult, LLMResponse, LLMUsage)
    from src.agents import answer as A
    from src.agents.baselines import self_revise

    assert Path(A.__file__).resolve().is_relative_to(a_root.resolve())
    assert 'tests_d.support' not in sys.modules
    original_versions = A.prompt_versions()
    original_system = A.load_prompt('baselines.self_revise').system.template
    replacement = project_root / 'prompts' / 'baselines' / 'self_revise_v1.md'
    replacement_raw = replacement.read_text(encoding='utf-8').replace('\\r\\n', '\\n')
    expected_version = replacement.stem + '@' + hashlib.sha256(replacement_raw.encode('utf-8')).hexdigest()[:10]

    # Keep all of A's original prompt files, replacing only the delivered self-revise.
    # The temporary copy stays inside this writable checkout, not inside Downloads.
    with tempfile.TemporaryDirectory(prefix='a-prompt-test-', dir=project_root) as temporary:
        temporary = Path(temporary).resolve()
        assert temporary.is_relative_to(project_root.resolve())
        prompt_dir = temporary / 'prompts'
        shutil.copytree(a_root / 'prompts', prompt_dir)
        shutil.copyfile(replacement, prompt_dir / 'baselines' / 'self_revise_v1.md')
        with patch.object(A, 'PROMPTS_DIR', prompt_dir):
            versions = A.prompt_versions()
            prompt = A.load_prompt('baselines.self_revise')
            if mode == 'version':
                assert versions['baselines.self_revise'] == expected_version
                # Also works after the patch has already been merged into A's checkout.
                if prompt.system.template != original_system:
                    assert versions['baselines.self_revise'] != original_versions['baselines.self_revise']
                for prompt_id in versions:
                    if prompt_id != 'baselines.self_revise':
                        assert versions[prompt_id] == original_versions[prompt_id]
            else:
                class Client:
                    def __init__(self, value, explanation):
                        self.value = value
                        self.explanation = explanation
                        self.calls = []
                    def generate(self, **request):
                        self.calls.append(request)
                        return LLMResponse(
                            json.dumps({'short_answer': self.value, 'explanation': self.explanation}),
                            LLMUsage(stage=request['stage'], model=request['model']))

                explanation = 'Reasoning belongs only in this explanation field.'
                examples = [('England', 'England')] if mode == 'system' else [
                    ('University of New South Wales', 'University of New South Wales'),
                    ('1815', '1815'), ('yes', 'yes'), ('no', 'no'),
                    (42, '42'), (True, 'yes'), (False, 'no'),
                ]
                for value, expected in examples:
                    example = QAInput('example-a', 'What is the answer?',
                        [Sentence('Reference', 0, 'The supplied passage establishes the answer.')])
                    draft = AnswerDraft('example-a:draft', 'previous', 'Previous explanation.')
                    feedback = VerificationFeedback('feedback-a',
                        [FeedbackItem('draft', draft.draft_id, 'Check the answer against context.')])
                    client = Client(value, explanation)
                    result = self_revise(example, draft, feedback, client,
                        A.AgentConfig(model='scripted-mock', max_parse_repairs=0))
                    assert isinstance(result, RevisionResult)
                    assert result.draft_id == draft.draft_id
                    assert result.feedback_id == feedback.feedback_id
                    assert result.final_answer == expected
                    assert result.final_explanation == explanation
                    assert explanation not in result.final_answer
                    assert len(client.calls) == 1
                    call = client.calls[0]
                    assert call['stage'] == 'A.M2.self_revise'
                    assert [message['role'] for message in call['messages']] == ['system', 'user']
                    system = call['messages'][0]['content']
                    assert system == prompt.system.template.strip()
                    for phrase in ('short_answer', 'entity/name', 'number', '"yes"', '"no"',
                                   'multiword names', 'full sentence', 'ONLY in explanation'):
                        assert phrase in system, phrase
                    assert '"short_answer"' in call['messages'][1]['content']
                    assert '"explanation"' in call['messages'][1]['content']
    print('Actual A self_revise regression passed: ' + mode)
''')


class AShortAnswerPromptTests(unittest.TestCase):
    def run_actual_a(self, mode):
        project_root = Path(__file__).resolve().parents[1]
        default_a = project_root if (project_root / 'src/agents/baselines.py').is_file() else _A_DEFAULT
        default_contracts = project_root if (project_root / 'contracts/schemas.py').is_file() else _CONTRACTS_DEFAULT
        a_root = Path(os.environ.get('D_A_ROOT', default_a))
        contracts_root = Path(os.environ.get('D_CONTRACTS_ROOT', os.environ.get('D_TEAM_ROOT', default_contracts)))
        if not (a_root / 'src' / 'agents' / 'baselines.py').is_file():
            self.skipTest('A checkout unavailable; set D_A_ROOT to run the real-A integration test.')
        if not (contracts_root / 'contracts' / 'schemas.py').is_file():
            self.skipTest('Shared contracts unavailable; set D_CONTRACTS_ROOT to run this test.')
        completed = subprocess.run(
            [sys.executable, '-B', '-c', _CHILD,
             str(project_root), str(a_root), str(contracts_root), mode],
            cwd=project_root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('Actual A self_revise regression passed', completed.stdout)

    def test_actual_self_revise_loads_short_answer_rules_in_system_prompt(self):
        self.run_actual_a('system')

    def test_answer_explanation_separation_and_complete_answer_values(self):
        self.run_actual_a('values')

    def test_content_hash_matches_updated_self_revise_prompt(self):
        self.run_actual_a('version')


if __name__ == '__main__':
    unittest.main()
