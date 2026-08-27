"""Prepare-step core for the UzbekPER Modal pipeline (rewrite-plan Task 3).

The provider-neutral body behind the Modal CPU ``prepare_audio`` function:

1. Probe every ready-manifest row with :mod:`audio_inventory` (structural WAV
   validation, never size-based acceptance).
2. Write a content-addressed, atomically-published inventory:
   ``inventory_<dataset-sha8>.jsonl`` — one line per clip + one summary line.
3. Refuse to silently overwrite an existing inventory for the same dataset
   digest (append-only spirit; regenerate only with ``force=True``).
4. Return a receipt-usable summary dict.

Pure arguments (rows / audio_root / out_dir) so local pytest and a Modal CPU
container drive identical code.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from uzbekper_pipeline.audio_inventory import build_inventory


class PrepareError(RuntimeError):
    """Raised when preparation cannot proceed safely."""


def _dataset_digest(rows: list[dict[str, Any]]) -> str:
    """Stable sha256 over sorted clip_ids — identity of the clip set."""
    payload = json.dumps(
        sorted(r.get("clip_id", "") for r in rows),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_prepare(
    rows: list[dict[str, Any]],
    audio_root: str,
    out_dir: str,
    dataset_jsonl: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Probe all rows and write the content-addressed inventory JSONL.

    Args:
        rows: ready-manifest rows (must carry ``clip_id``).
        audio_root: directory under which clip paths resolve.
        out_dir: inventory output directory.
        dataset_jsonl: optional explicit digest source override (file path)
            — when provided, its file sha256 names the inventory instead of
            hashing row clip_ids.
        force: allow regenerating an existing same-digest inventory.
        Returns dict with present/missing/corrupt/total_present_seconds and
        the artifact path. Raises PrepareError on unsafe states.
    """
    if not rows:
        raise PrepareError("run_prepare: no rows to prepare")

    # ---- digest: prefer explicit dataset file hash, else hash the row set --
    if dataset_jsonl:
        h = hashlib.sha256()
        with open(dataset_jsonl, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        digest = h.hexdigest()
    else:
        digest = _dataset_digest(rows)

    os.makedirs(out_dir, exist_ok=True)
    inv_path = os.path.join(out_dir, f"inventory_{digest[:8]}.jsonl")

    if os.path.exists(inv_path) and not force:
        raise PrepareError(
            f"refusing to overwrite existing inventory {inv_path} "
            f"(same dataset digest); pass force=True to regenerate"
        )

    inv = build_inventory(rows, audio_root)
    summary = summarize(inv)
    summary["dataset_sha256"] = digest

    # ---- atomic write: temp then rename -------------------------------------
    tmp = inv_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in inv:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        fh.write(
            json.dumps(
                {"record_type": "summary", **summary},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    os.rename(tmp, inv_path)

    return {
        "present": summary["present"],
        "missing": summary["missing"],
        "corrupt": summary["corrupt"],
        "total_present_seconds": summary["total_present_seconds"],
        "dataset_sha256": digest,
        "inventory_path": inv_path,
    }


def summarize(inv: list[dict[str, Any]]) -> dict[str, Any]:
    """Local copy of counts so callers don't need both modules."""
    from uzbekper_pipeline.audio_inventory import summarize_inventory as _s
    s = _s(inv)
    return {
        "present": s["present"],
        "missing": s["missing"],
        "corrupt": s["corrupt"],
        "total_present_seconds": s["total_present_seconds"],
    }
