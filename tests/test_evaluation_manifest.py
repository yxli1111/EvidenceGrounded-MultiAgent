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
