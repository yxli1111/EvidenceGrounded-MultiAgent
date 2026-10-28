"""ID manifest utilities for dev/eval isolation.

Functions
---------
load_id_manifest(path) -> set[str]
    Load a JSON list of example IDs from a file and return as a set.

assert_disjoint(dev_ids, eval_ids) -> None
    Raise ValueError if the two ID sets overlap, listing all shared IDs.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_id_manifest(path) -> set[str]:
    """Load a JSON list of example IDs from path and return as set[str].

    Args:
        path: file path (str or Path) to a JSON file containing a list of strings.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the JSON root is not a list.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ID manifest not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"ID manifest must be a JSON list, got {type(data).__name__}: {path}"
        )
    return {str(item) for item in data}


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
