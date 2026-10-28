"""Tests for port adapters (src/runner/adapters.py).

All tests that require contracts.schemas or the real A/B/C/D packages are
gated with pytest.importorskip so they are skipped cleanly when those
packages are not installed in the current environment.
"""

import pytest


# ---------------------------------------------------------------------------
# BRetrievalAdapter
# ---------------------------------------------------------------------------

class TestBRetrievalAdapter:
    """BRetrievalAdapter converts Sentence objects to dicts and back."""

    def test_sentences_converted_to_dicts_for_retriever(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import BRetrievalAdapter

        received_context = []

        class FakeRetriever:
            def retrieve(self, query, context, top_k):
                received_context.extend(context)
                return []

        adapter = BRetrievalAdapter(FakeRetriever())
        sentences = _make_sentences([("Doc1", 0, "First sentence."),
                                     ("Doc2", 1, "Second sentence.")])
        adapter.retrieve("test query", sentences, top_k=2)

        assert len(received_context) == 2
        assert received_context[0] == {"title": "Doc1", "sentence_id": 0, "text": "First sentence."}
        assert received_context[1] == {"title": "Doc2", "sentence_id": 1, "text": "Second sentence."}

    def test_dict_context_passed_through_unchanged(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import BRetrievalAdapter

        received_context = []

        class FakeRetriever:
            def retrieve(self, query, context, top_k):
                received_context.extend(context)
                return []

        adapter = BRetrievalAdapter(FakeRetriever())
        dict_context = [{"title": "A", "sentence_id": 0, "text": "hello"}]
        adapter.retrieve("q", dict_context, top_k=1)
        assert received_context == dict_context

    def test_result_dicts_converted_to_evidence(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import BRetrievalAdapter

        class FakeRetriever:
            def retrieve(self, query, context, top_k):
                return [
                    {"doc_title": "D1", "sentence_id": 2, "text": "t", "score": 0.9,
                     "retrieval_query": query, "rank": 1},
                ]

        adapter = BRetrievalAdapter(FakeRetriever())
        results = adapter.retrieve("q", [], top_k=5)
        assert len(results) == 1
        ev = results[0]
        assert ev["doc_title"] == "D1"
        assert ev["sentence_id"] == 2
        assert ev["text"] == "t"

    def test_evidence_text_matches_sentence_text(self):
        """D's evidence_payload validation requires text to match context exactly."""
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import BRetrievalAdapter

        class EchoRetriever:
            """Returns the first context item as a result dict."""
            def retrieve(self, query, context, top_k):
                if not context:
                    return []
                item = context[0]
                return [{"doc_title": item["title"], "sentence_id": item["sentence_id"],
                          "text": item["text"], "score": 1.0, "retrieval_query": query, "rank": 1}]

        adapter = BRetrievalAdapter(EchoRetriever())
        sentences = _make_sentences([("Doc", 0, "Exact text here.")])
        results = adapter.retrieve("q", sentences, top_k=1)
        assert results[0]["text"] == "Exact text here."


# ---------------------------------------------------------------------------
# DVerificationAdapter prompt_versions
# ---------------------------------------------------------------------------

class TestDVerificationAdapterPromptVersions:
    def test_prompt_versions_from_runtime_metadata(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import DVerificationAdapter

        pv = {"verify_draft": "v2.0", "revise_answer": "v1.3"}

        class FakeModule:
            @staticmethod
            def configure(**_kwargs):
                pass

            @staticmethod
            def runtime_metadata():
                return {"prompt_versions": pv}

            @staticmethod
            def verify_draft(*args, **kwargs):
                pass

            @staticmethod
            def verify_claims(*args, **kwargs):
                pass

            @staticmethod
            def revise_answer(*args, **kwargs):
                pass

        adapter = DVerificationAdapter(module=FakeModule())
        assert adapter.prompt_versions == pv

    def test_prompt_versions_empty_when_no_runtime_metadata(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import DVerificationAdapter

        class FakeModule:
            @staticmethod
            def configure(**_kwargs):
                pass

            @staticmethod
            def verify_draft(*args, **kwargs):
                pass

            @staticmethod
            def verify_claims(*args, **kwargs):
                pass

            @staticmethod
            def revise_answer(*args, **kwargs):
                pass

        adapter = DVerificationAdapter(module=FakeModule())
        assert isinstance(adapter.prompt_versions, dict)


# ---------------------------------------------------------------------------
# B data conversion helpers
# ---------------------------------------------------------------------------

class TestBDataConversion:
    def test_gold_dict_converted_to_tuple(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import b_gold_to_contracts

        result = b_gold_to_contracts([{"title": "France", "sentence_id": 0}])
        assert result == [("France", 0)]

    def test_gold_list_of_dicts_all_converted(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import b_gold_to_contracts

        facts = [
            {"title": "France", "sentence_id": 0},
            {"title": "Germany", "sentence_id": 2},
        ]
        result = b_gold_to_contracts(facts)
        assert result == [("France", 0), ("Germany", 2)]

    def test_tuple_input_passed_through(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import b_gold_to_contracts

        facts = [("France", 0), ("Germany", 2)]
        result = b_gold_to_contracts(facts)
        assert result == [("France", 0), ("Germany", 2)]

    def test_mixed_dict_and_tuple_input(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import b_gold_to_contracts

        facts = [{"title": "France", "sentence_id": 0}, ("Germany", 2)]
        result = b_gold_to_contracts(facts)
        assert result == [("France", 0), ("Germany", 2)]


# ---------------------------------------------------------------------------
# AAnswerAdapter
# ---------------------------------------------------------------------------

class TestAAnswerAdapter:
    def test_run_m2_method_exists(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import AAnswerAdapter

        class FakeAgent:
            prompt_versions = {"generate_draft": "v1.0", "self_reflect": "v1.1"}

            def generate_draft(self, *args, **kwargs):
                pass

            def self_reflect(self, *args, **kwargs):
                pass

            def independent_critique(self, *args, **kwargs):
                pass

        adapter = AAnswerAdapter(FakeAgent())
        assert hasattr(adapter, "run_m2")
        assert callable(adapter.run_m2)

    def test_prompt_versions_is_dict(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import AAnswerAdapter

        class FakeAgent:
            prompt_versions = {"draft": "v1.0"}

            def generate_draft(self, *args, **kwargs):
                pass

            def self_reflect(self, *args, **kwargs):
                pass

            def independent_critique(self, *args, **kwargs):
                pass

        adapter = AAnswerAdapter(FakeAgent())
        assert isinstance(adapter.prompt_versions, dict)
        assert adapter.prompt_versions == {"draft": "v1.0"}


# ---------------------------------------------------------------------------
# sentences_to_context_dicts helper
# ---------------------------------------------------------------------------

class TestSentencesToContextDicts:
    def test_converts_sentence_objects_to_dicts(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import sentences_to_context_dicts

        sentences = _make_sentences([("TitleA", 3, "Some text.")])
        result = sentences_to_context_dicts(sentences)
        assert result == [{"title": "TitleA", "sentence_id": 3, "text": "Some text."}]

    def test_dict_items_returned_unchanged(self):
        pytest.importorskip("contracts.schemas")
        from src.runner.adapters import sentences_to_context_dicts

        dicts = [{"title": "X", "sentence_id": 0, "text": "hi"}]
        assert sentences_to_context_dicts(dicts) == dicts


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_sentences(specs: list[tuple]) -> list:
    """Create simple sentence-like objects with title/sentence_id/text attrs."""

    class _Sent:
        def __init__(self, title, sentence_id, text):
            self.title = title
            self.sentence_id = sentence_id
            self.text = text

    return [_Sent(t, s, tx) for t, s, tx in specs]
