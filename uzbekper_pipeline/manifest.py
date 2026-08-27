"""Manifest freezing for the UzbekPER Modal pipeline.

The 2026-08-25 Vast run lost data because resume logic keyed on list positions
while the manifest grew between runs. Everything downstream now keys on
``clip_id == audio_filepath`` (unique in the official manifest), and the exact
evaluation set is frozen to a byte-reproducible JSONL + SHA-256 before any
cloud work starts.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from typing import Any

REQUIRED_FIELDS = ("audio_filepath", "text", "source")


class ManifestError(ValueError):
    """Raised when a manifest violates structural invariants."""


def load_manifest(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    _validate_rows(rows)
    return rows


# join filename -> manifest source tag (only this one differs)
_JOIN_SRC_ALIAS = {"podcasts": "podcasts_dialect"}


def load_matched_keys(joins_dir: str) -> dict[str, set[str]]:
    """Return {source_tag: set(manifest_idx)} of matched clips from join files.

    Join files (data/join_*.json) map either an upstream ID or a normalized
    text key to {'idx': <manifest position>, 'matched': bool, ...}. The idx
    here is only used to find the clip's *path* once; identity afterwards is
    always the path itself.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(glob.glob(os.path.join(joins_dir, "join_*.json"))):
        src = os.path.basename(path)[5:-5]
        src = _JOIN_SRC_ALIAS.get(src, src)
        d = json.load(open(path))
        matched = set()
        for k, v in d.get("rows", {}).items():
            if v.get("matched") and "idx" in v:
                matched.add(v["idx"])
        if matched:
            out[src] = matched
    return out


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    for i, r in enumerate(rows):
        for field in REQUIRED_FIELDS:
            if not r.get(field):
                raise ManifestError(f"row {i}: missing {field!r}")
    ids = [r["audio_filepath"] for r in rows]
    seen = set()
    dupes = set()
    for cid in ids:
        if cid in seen:
            dupes.add(cid)
        seen.add(cid)
    if dupes:
        raise ManifestError(f"duplicate audio_filepath values: {sorted(dupes)[:5]}")


def load_ready_manifest(path: str) -> list[dict[str, Any]]:
    """Load a FROZEN ready manifest (the file freeze_manifest writes).

    Ready manifests are keyed by ``clip_id`` (== audio_filepath at freeze
    time); they do NOT carry the raw-manifest ``audio_filepath`` field, so
    :func:`load_manifest` rejects them. Validates clip_id presence and
    uniqueness — identity downstream is always clip_id.
    """
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("clip_id"):
                raise ManifestError(f"row {i}: missing 'clip_id'")
            rows.append(r)
    ids = [r["clip_id"] for r in rows]
    dupes = {c for c in ids if ids.count(c) > 1}
    if dupes:
        raise ManifestError(f"duplicate clip_id values: {sorted(dupes)[:5]}")
    return rows


def build_ready_manifest(
    rows: list[dict[str, Any]], joins_dir: str
) -> list[dict[str, Any]]:
    """Freeze the evaluation set to exactly the clips the join files marked.

    Rows carry a stable ``clip_id`` (= audio_filepath). Ordering follows the
    upstream join-file keys per source; consumers must never rely on order.
    """
    _validate_rows(rows)
    by_pos = {pos: r for pos, r in enumerate(rows)}
    matched = load_matched_keys(joins_dir)
    if not matched:
        raise ManifestError(f"no matched clips found under {joins_dir}")
    ready = []
    seen = set()
    for src, positions in sorted(matched.items()):
        for pos in sorted(positions):
            r = by_pos.get(pos)
            if r is None:
                raise ManifestError(f"join file references unknown row {pos}")
            if r["source"] != src:
                # source tag disagreement between join and manifest would
                # silently mislabel per-source scores -- refuse loudly.
                raise ManifestError(
                    f"source mismatch at row {pos}: join={src} manifest={r['source']}"
                )
            cid = r["audio_filepath"]
            if cid in seen:
                raise ManifestError(f"duplicate ready clip_id: {cid}")
            seen.add(cid)
            ready.append(
                {
                    "clip_id": cid,
                    "text": r["text"],
                    "source": src,
                    "duration": r.get("duration"),
                }
            )
    return ready


def freeze_manifest(ready: list[dict[str, Any]], out_path: str) -> str:
    """Write ready rows as one sorted JSONL; returns sha256 of the bytes."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ordered = sorted(ready, key=lambda r: r["clip_id"])
    with open(out_path, "w") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with open(out_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_run_spec(
    spec_path: str,
    dataset_sha: str,
    models: dict[str, dict[str, str]],
    decode: dict[str, Any],
    code_version: str,
    n_clips: int,
) -> dict[str, Any]:
    if not models:
        raise ManifestError("run spec must pin at least one model")
    for name, m in models.items():
        if m.get("revision") in (None, "", "FIXED_BY_PREFLIGHT"):
            raise ManifestError(f"model {name!r}: revision must be a pinned SHA")
    spec = {
        "dataset_sha256": dataset_sha,
        "n_clips": int(n_clips),
        "models": {
            name: {"repo_id": m["repo_id"], "revision": m["revision"]}
            for name, m in models.items()
        },
        "decode": decode,
        "code_version": code_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    return spec
