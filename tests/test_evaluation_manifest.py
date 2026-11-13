"""Tests for src/evaluation/manifest.py (load_id_manifest, assert_disjoint).

No real API calls or external data needed.
Format assumption: B_dev_50_ids.json is a JSON list of strings.
(B's file does not yet exist in feat/retrieval; format confirmed as list[str]
based on B's standard hotpot ID format; see E_CONTRACT_PROPOSAL.md GAP-3.)
"""

import json
import pytest


# ---------------------------------------------------------------------------
# load_id_manifest
# ---------------------------------------------------------------------------

class TestLoadIdManifest:
    def test_loads_json_list_of_strings(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        ids = ["hotpot_dev_0", "hotpot_dev_1", "hotpot_dev_2"]
        manifest = tmp_path / "ids.json"
        manifest.write_text(json.dumps(ids), encoding="utf-8")

        result = load_id_manifest(str(manifest))
        assert result == set(ids)

    def test_returns_set_not_list(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        manifest = tmp_path / "ids.json"
        manifest.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        result = load_id_manifest(str(manifest))
        assert isinstance(result, set)

    def test_raises_on_non_list_json(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        manifest = tmp_path / "ids.json"
        manifest.write_text(json.dumps({"id": "val"}), encoding="utf-8")
        with pytest.raises(ValueError, match="list"):
            load_id_manifest(str(manifest))

    def test_raises_on_missing_file(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        with pytest.raises(FileNotFoundError):
            load_id_manifest(str(tmp_path / "nonexistent.json"))

    def test_duplicate_ids_deduplicated(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        manifest = tmp_path / "ids.json"
        manifest.write_text(json.dumps(["a", "b", "a"]), encoding="utf-8")
        result = load_id_manifest(str(manifest))
        assert result == {"a", "b"}

    def test_accepts_path_object(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        manifest = tmp_path / "ids.json"
        manifest.write_text(json.dumps(["x"]), encoding="utf-8")
        result = load_id_manifest(manifest)  # Path object, not str
        assert result == {"x"}


# ---------------------------------------------------------------------------
# assert_disjoint
# ---------------------------------------------------------------------------

class TestAssertDisjoint:
    def test_disjoint_sets_pass(self):
        from src.evaluation.manifest import assert_disjoint

        assert_disjoint({"dev_0", "dev_1"}, {"eval_0", "eval_1"})  # no exception

    def test_overlapping_sets_raise(self):
        from src.evaluation.manifest import assert_disjoint

        with pytest.raises(ValueError, match="shared_id"):
            assert_disjoint({"shared_id", "dev_0"}, {"shared_id", "eval_0"})

    def test_error_message_lists_overlap(self):
        from src.evaluation.manifest import assert_disjoint

        with pytest.raises(ValueError) as exc_info:
            assert_disjoint({"a", "b", "c"}, {"b", "c", "d"})
        msg = str(exc_info.value)
        assert "b" in msg and "c" in msg

    def test_empty_sets_pass(self):
        from src.evaluation.manifest import assert_disjoint

        assert_disjoint(set(), set())  # no exception

    def test_one_empty_set_passes(self):
        from src.evaluation.manifest import assert_disjoint

        assert_disjoint({"dev_0"}, set())
        assert_disjoint(set(), {"eval_0"})


# ---------------------------------------------------------------------------
# load_id_manifest — B's object format ({"example_ids": [...], ...})
# ---------------------------------------------------------------------------

class TestLoadIdManifestObjectFormat:
    """B's real B_dev_50_ids.json is an object, not a plain list."""

    def test_object_format_reads_example_ids_field(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        manifest = tmp_path / "ids.json"
        manifest.write_text(
            '{"example_ids": ["hotpot_dev_0", "hotpot_dev_1"], '
            '"dataset": "hotpotqa", "split": "dev", '
            '"selection": "first_50", "purpose": "dev_tuning"}',
            encoding="utf-8",
        )
        result = load_id_manifest(str(manifest))
        assert result == {"hotpot_dev_0", "hotpot_dev_1"}

    def test_object_format_returns_set(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        manifest = tmp_path / "ids.json"
        manifest.write_text('{"example_ids": ["a", "b"]}', encoding="utf-8")
        result = load_id_manifest(str(manifest))
        assert isinstance(result, set)

    def test_object_format_deduplicates(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        manifest = tmp_path / "ids.json"
        manifest.write_text('{"example_ids": ["a", "b", "a"]}', encoding="utf-8")
        result = load_id_manifest(str(manifest))
        assert result == {"a", "b"}

    def test_object_format_missing_example_ids_raises(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        manifest = tmp_path / "ids.json"
        manifest.write_text('{"ids": ["a"]}', encoding="utf-8")
        with pytest.raises(ValueError, match="example_ids"):
            load_id_manifest(str(manifest))

    def test_object_format_non_list_example_ids_raises(self, tmp_path):
        from src.evaluation.manifest import load_id_manifest

        manifest = tmp_path / "ids.json"
        manifest.write_text('{"example_ids": "not_a_list"}', encoding="utf-8")
        with pytest.raises(ValueError):
            load_id_manifest(str(manifest))

    def test_b_dev_50_structure(self, tmp_path):
        """Mirrors the structure of B's actual B_dev_50_ids.json."""
        import json
        from src.evaluation.manifest import load_id_manifest

        ids = [f"5a8b57f25542995d1e6f{i:04d}" for i in range(50)]
        manifest_data = {
            "example_ids": ids,
            "dataset": "hotpotqa",
            "split": "validation",
            "selection": "first_50",
            "purpose": "dev_tuning",
        }
        manifest = tmp_path / "B_dev_50_ids.json"
        manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
        result = load_id_manifest(str(manifest))
        assert len(result) == 50
        assert all(isinstance(x, str) for x in result)


# ---------------------------------------------------------------------------
# filter_out_ids
# ---------------------------------------------------------------------------

class TestFilterOutIds:

    def test_removes_matching_dict_examples(self):
        from src.evaluation.manifest import filter_out_ids

        examples = [
            {"example_id": "a"},
            {"example_id": "b"},
            {"example_id": "c"},
        ]
        filtered, n_removed = filter_out_ids(examples, {"a", "c"})
        assert len(filtered) == 1
        assert filtered[0]["example_id"] == "b"
        assert n_removed == 2

    def test_removes_matching_attribute_objects(self):
        from src.evaluation.manifest import filter_out_ids

        class Ex:
            def __init__(self, eid):
                self.example_id = eid

        examples = [Ex("a"), Ex("b"), Ex("c")]
        filtered, n_removed = filter_out_ids(examples, {"b"})
        assert len(filtered) == 2
        assert n_removed == 1

    def test_no_overlap_returns_all(self):
        from src.evaluation.manifest import filter_out_ids

        examples = [{"example_id": "a"}, {"example_id": "b"}]
        filtered, n_removed = filter_out_ids(examples, {"x", "y"})
        assert len(filtered) == 2
        assert n_removed == 0

    def test_all_excluded_returns_empty(self):
        from src.evaluation.manifest import filter_out_ids

        examples = [{"example_id": "a"}, {"example_id": "b"}]
        filtered, n_removed = filter_out_ids(examples, {"a", "b"})
        assert len(filtered) == 0
        assert n_removed == 2

    def test_empty_exclude_ids(self):
        from src.evaluation.manifest import filter_out_ids

        examples = [{"example_id": "a"}]
        filtered, n_removed = filter_out_ids(examples, set())
        assert len(filtered) == 1
        assert n_removed == 0

    def test_returns_tuple_of_list_and_int(self):
        from src.evaluation.manifest import filter_out_ids

        result = filter_out_ids([{"example_id": "x"}], {"y"})
        assert isinstance(result, tuple)
        assert isinstance(result[0], list)
        assert isinstance(result[1], int)

    def test_order_preserved(self):
        from src.evaluation.manifest import filter_out_ids

        examples = [
            {"example_id": "a"},
            {"example_id": "b"},
            {"example_id": "c"},
            {"example_id": "d"},
        ]
        filtered, _ = filter_out_ids(examples, {"b"})
        assert [e["example_id"] for e in filtered] == ["a", "c", "d"]
