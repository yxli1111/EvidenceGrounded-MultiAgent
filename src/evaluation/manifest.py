"""ID manifest utilities for dev/eval isolation.

Functions
---------
load_id_manifest(path) -> set[str]
    Load example IDs from a file and return as a set.  Accepts two formats:
    - Plain JSON list:  ["id0", "id1", ...]
    - Object format:    {"example_ids": [...], "dataset": ..., ...}
      (B's actual B_dev_50_ids.json uses this format)

assert_disjoint(dev_ids, eval_ids) -> None
    Raise ValueError if the two ID sets overlap, listing all shared IDs.

filter_out_ids(examples, exclude_ids) -> tuple[list, int]
    Return (filtered_list, n_removed) after dropping examples whose
    example_id is in exclude_ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_id_manifest(path) -> set[str]:
    """Load example IDs from path and return as set[str].

    Accepts two JSON formats:
    - Plain list:   ["id0", "id1", ...]
    - Object:       {"example_ids": [...], "dataset": "hotpotqa", ...}

    The object format is B's actual format for B_dev_50_ids.json.
    Extra metadata fields (dataset, split, selection, purpose) in the object
    format are silently ignored.

    Args:
        path: file path (str or Path) to a JSON file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the JSON cannot be parsed as either accepted format.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ID manifest not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(item) for item in data}
    if isinstance(data, dict):
        if "example_ids" not in data:
            raise ValueError(
                f"ID manifest object is missing 'example_ids' key: {path}"
            )
        ids = data["example_ids"]
        if not isinstance(ids, list):
            raise ValueError(
                f"'example_ids' must be a list, got {type(ids).__name__}: {path}"
            )
        return {str(item) for item in ids}
    raise ValueError(
        f"ID manifest must be a JSON list or object with 'example_ids', "
        f"got {type(data).__name__}: {path}"
    )


def filter_out_ids(examples: list, exclude_ids: set) -> tuple[list, int]:
    """Remove examples whose example_id is in exclude_ids.

    Args:
        examples: list of QAInput objects or dicts with an example_id field.
        exclude_ids: set of example_id strings to remove.

    Returns:
        (filtered_list, n_removed) where n_removed = len(examples) - len(filtered_list).
    """
    if not exclude_ids:
        return list(examples), 0
    filtered = []
    for ex in examples:
        eid = ex.get("example_id") if isinstance(ex, dict) else getattr(ex, "example_id", None)
        if eid not in exclude_ids:
            filtered.append(ex)
    return filtered, len(examples) - len(filtered)


def assert_disjoint(dev_ids: set, eval_ids: set) -> None:
    """Raise ValueError if dev_ids and eval_ids share any IDs.

    Args:
        dev_ids: set of development example IDs.
        eval_ids: set of evaluation example IDs.

    Raises:
        ValueError: listing all shared IDs if any overlap exists.
    """
    overlap = dev_ids & eval_ids
    if overlap:
        raise ValueError(
            f"Dev and eval ID sets must be disjoint. "
            f"Shared IDs: {sorted(overlap)}"
        )
