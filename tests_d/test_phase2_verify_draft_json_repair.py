"""Syntax-only repair regression tests for D's shared JSON helper."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.verification import _common
from src.verification.errors import VerificationError


class ScriptedLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self.outputs)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


class VerifyDraftJsonRepairTests(unittest.TestCase):
    def run_json(self, stage, outputs):
        llm = ScriptedLLM(outputs)
        cfg = SimpleNamespace(model="test-model", temperature=0,
                              verification_max_tokens=256, revision_max_tokens=256)
        with patch.object(_common, "settings", return_value=cfg), patch.object(
            _common.Path, "read_text", return_value="Verifier prompt"
        ):
            result = _common.generate_json(llm, stage, {"question": "Q"}, VerificationError)
        return result, llm

    def test_valid_draft_json_one_call(self):
        result, llm = self.run_json("verify_draft", ['{"items":[],"overall_comment":"ok"}'])
        self.assertEqual(result["overall_comment"], "ok")
        self.assertEqual(len(llm.calls), 1)

    def test_malformed_draft_json_repaired_once(self):
        result, llm = self.run_json("verify_draft", [
            '{"items":[] "overall_comment":"ok"}',
            '{"items":[],"overall_comment":"ok"}',
        ])
        self.assertEqual(result["items"], [])
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[1]["stage"], "verify_draft")
        self.assertIn("Repair only JSON syntax", llm.calls[1]["messages"][0]["content"])

    def test_second_invalid_json_fails_without_third_call(self):
        llm = ScriptedLLM(['{"items":[]', '{"items":[]'])
        cfg = SimpleNamespace(model="test", temperature=0, verification_max_tokens=256, revision_max_tokens=256)
        with patch.object(_common, "settings", return_value=cfg), patch.object(_common.Path, "read_text", return_value="prompt"):
            with self.assertRaisesRegex(VerificationError, "repair failed"):
                _common.generate_json(llm, "verify_draft", {}, VerificationError)
        self.assertEqual(len(llm.calls), 2)

    def test_claim_json_not_retried(self):
        llm = ScriptedLLM(['{"items":[]'])
        cfg = SimpleNamespace(model="test", temperature=0, verification_max_tokens=256, revision_max_tokens=256)
        with patch.object(_common, "settings", return_value=cfg), patch.object(_common.Path, "read_text", return_value="prompt"):
            with self.assertRaises(VerificationError):
                _common.generate_json(llm, "verify_claims", {}, VerificationError)
        self.assertEqual(len(llm.calls), 1)

    def test_api_failure_not_retried(self):
        llm = ScriptedLLM([RuntimeError("API unavailable")])
        cfg = SimpleNamespace(model="test", temperature=0, verification_max_tokens=256, revision_max_tokens=256)
        with patch.object(_common, "settings", return_value=cfg), patch.object(_common.Path, "read_text", return_value="prompt"):
            with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                _common.generate_json(llm, "verify_draft", {}, VerificationError)
        self.assertEqual(len(llm.calls), 1)

    def test_valid_json_wrong_top_level_not_retried(self):
        llm = ScriptedLLM(['[]'])
        cfg = SimpleNamespace(model="test", temperature=0, verification_max_tokens=256, revision_max_tokens=256)
        with patch.object(_common, "settings", return_value=cfg), patch.object(_common.Path, "read_text", return_value="prompt"):
            with self.assertRaisesRegex(VerificationError, "must be a JSON object"):
                _common.generate_json(llm, "verify_draft", {}, VerificationError)
        self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
