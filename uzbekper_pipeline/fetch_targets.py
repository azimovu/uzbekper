"""Load and count reconstruction targets from data/join_*.json files.

Extracted from scripts/fetch_test_audio.py so the counting logic is unit-
testable. The original inline version computed
``need = sum(len(v) for v in targets[src].values())`` — the field count of
each record dict, not the record count — producing wrong progress/summaries.
"""
from __future__ import annotations

import glob
import json
import os

# join filename -> manifest source tag (only this one differs)
_JOIN_SRC_ALIAS = {"podcasts": "podcasts_dialect"}


def load_targets(joins_dir: str) -> dict[str, dict[str, dict]]:
    """Return {source_tag: {key: {'idx': int, ...}}} for MATCHED rows only."""
    targets: dict[str, dict[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(joins_dir, "join_*.json"))):
        src = os.path.basename(path)[5:-5]
        src = _JOIN_SRC_ALIAS.get(src, src)
        d = json.load(open(path))
        matched = {
            k: v for k, v in d.get("rows", {}).items()
            if v.get("matched") and "idx" in v
        }
        if matched:
            # merge rather than overwrite: join files are one per source
            existing = targets.setdefault(src, {})
            existing.update(matched)
    return targets


def count_needed(targets: dict[str, dict[str, dict]]) -> int:
    """Total clips to materialize — counts RECORDS, never dict fields."""
    return sum(len(v) for v in targets.values())
