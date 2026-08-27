"""Crash-safe transcript artifacts for the UzbekPER Modal pipeline.

Design invariants (born from the 2026-08-25 data loss):

1. Chunks are append-only: names carry an ascending sequence + UUID; a flush
   never rewrites an existing file.
2. Resume is keyed by ``clip_id`` only.
3. ``finalize()`` reads the prior canonical artifact FIRST, merges chunks +
   new records deduped by clip_id (last-writer-wins within a single store
   instance), validates the merged document, writes it to a temp file and
   atomically renames it over the canonical path.
4. Any corrupt chunk line aborts publication with ArtifactError — never a
   silent partial merge.
"""
from __future__ import annotations

import glob
import json
import os
import time
import uuid
from pathlib import Path


class ArtifactError(RuntimeError):
    """Raised when artifacts are inconsistent; refuse to publish."""


def _safe(s: str) -> str:
    return s.replace("/", "_")


class ArtifactStore:
    def __init__(
        self,
        root: str,
        model: str,
        revision: str,
        chunk_every: int = 200,
        decode: dict | None = None,
    ):
        self.root = os.path.abspath(root)
        self.model = model
        self.revision = revision
        self.chunk_every = max(1, int(chunk_every))
        self.decode = decode or {}
        self.model_dir = os.path.join(self.root, _safe(model))
        self.chunk_dir = Path(os.path.join(self.model_dir, "chunks"))
        self.canonical_path = os.path.join(
            self.model_dir, f"transcripts_{_safe(model)}.json"
        )
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        self._since_flush = 0

    # ------------------------------------------------------------------ write

    def append(self, record: dict) -> None:
        rec = dict(record)
        rec.setdefault("model", self.model)
        rec.setdefault("revision", self.revision)
        rec.setdefault("decode", self.decode)
        rec.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._records.append(rec)
        self._since_flush += 1
        if self._since_flush >= self.chunk_every:
            self.flush_chunk()

    def flush_chunk(self) -> str | None:
        if not self._records:
            return None
        name = (
            f"chunk_{int(time.time() * 1000):013d}_{uuid.uuid4().hex[:8]}.jsonl"
        )
        path = os.path.join(self.chunk_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for r in self._records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.rename(tmp, path)                     # atomic chunk appearance
        self._records.clear()
        self._since_flush = 0
        return path

    # ------------------------------------------------------------------- read

    @staticmethod
    def _read_chunk(path: str) -> list[dict]:
        """Reads one JSONL chunk; torn/corrupt lines raise ArtifactError."""
        out = []
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ArtifactError(
                        f"corrupt chunk {path}:{lineno}: {e}"
                    ) from e
        return out

    def _iter_all_records(self):
        """Prior canonical first, then chunks oldest-first, then unflushed."""
        if os.path.exists(self.canonical_path):
            try:
                prior = json.load(open(self.canonical_path))
            except json.JSONDecodeError as e:
                raise ArtifactError(
                    f"prior canonical artifact unreadable: {e}"
                ) from e
            yield from prior
        for path in sorted(glob.glob(os.path.join(self.chunk_dir, "*.jsonl"))):
            yield from self._read_chunk(path)
        yield from self._records

    def done_clip_ids(self) -> set[str]:
        done = set()
        for rec in self._iter_all_records():
            cid = rec.get("clip_id")
            if cid is None:
                raise ArtifactError(f"record without clip_id: {rec!r:.120}")
            done.add(cid)
        return done

    # ---------------------------------------------------------------- publish

    def finalize(self) -> int:
        merged: dict[str, dict] = {}
        conflicts: list[str] = []
        for rec in self._iter_all_records():
            cid = rec.get("clip_id")
            if cid is None:
                raise ArtifactError("record without clip_id")
            prev = merged.get(cid)
            if (
                prev is not None
                and prev.get("revision") == rec.get("revision")
                and prev.get("hyp") != rec.get("hyp")
            ):
                # Same pinned revision producing different text for one clip
                # => decode was not deterministic or inputs changed. An
                # operator must look; silently keeping one side corrupts
                # the benchmark. (A NEW revision legitimately replaces.)
                conflicts.append(cid)
            merged[cid] = rec             # last writer wins (only reached
                                          # for genuinely new clips)
        if conflicts:
            raise ArtifactError(
                f"{len(conflicts)} conflicting re-transcriptions "
                f"(same model, different hyp): {sorted(conflicts)[:5]}"
            )
        rows = sorted(merged.values(), key=lambda r: str(r.get("clip_id")))
        # validate serializable before touching the canonical file
        blob = json.dumps(rows, ensure_ascii=False)
        tmp = self.canonical_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(blob)
        os.replace(tmp, self.canonical_path)      # atomic publish
        self._records.clear()
        self._since_flush = 0
        return len(rows)
