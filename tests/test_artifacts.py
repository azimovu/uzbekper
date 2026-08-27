"""Tests for uzbekper_pipeline.artifacts — append-only chunks + crash-safe merge.

The 2026-08-25 Vast post-mortem: the old Recorder started empty on resume and
the finalize step overwrote the merged transcript BEFORE reading it for the
merge, destroying prior records. These tests pin the corrected behavior.
"""
import json
import os

import pytest

from uzbekper_pipeline.artifacts import ArtifactStore, ArtifactError


@pytest.fixture()
def store(tmp_path):
    return ArtifactStore(root=str(tmp_path), model="openai/whisper-large-v3",
                         revision="abc123")


def _rec(clip_id, hyp="salom"):
    return {"clip_id": clip_id, "hyp": hyp}


def test_chunks_are_append_only_and_never_rewritten(store):
    store.append(_rec("c1"))
    store.flush_chunk()
    first = sorted(os.listdir(store.chunk_dir))
    store.append(_rec("c2"))
    store.flush_chunk()
    second = sorted(os.listdir(store.chunk_dir))
    assert first[0] in second            # old chunk untouched, not rewritten
    assert len(second) == len(first) + 1
    for name in second:
        assert (store.chunk_dir / name).stat().st_size > 0


def test_resume_preserves_prior_canonical_records(tmp_path):
    """Reproduces the catastrophic 2026-08-25 loss: resumed run must NOT wipe."""
    s1 = ArtifactStore(root=str(tmp_path), model="m", revision="r1")
    s1.append(_rec("c1", "first"))
    s1.finalize()
    canonical_before = json.load(open(s1.canonical_path))
    assert len(canonical_before) == 1

    # new process resumes: fresh store pointing at same root/model
    s2 = ArtifactStore(root=str(tmp_path), model="m", revision="r1")
    s2.append(_rec("c2", "second"))
    s2.finalize()

    merged = json.load(open(s2.canonical_path))
    ids = {r["clip_id"] for r in merged}
    assert ids == {"c1", "c2"}, "resume must merge with prior canonical file"
    by_id = {r["clip_id"]: r for r in merged}
    assert by_id["c1"]["hyp"] == "first"


def test_chunk_and_canonical_duplicate_is_deduped_by_clip_id(store):
    store.append(_rec("c1", "v1"))
    store.flush_chunk()
    store.append(_rec("c1", "v1"))       # idempotent retry after partial failure
    store.append(_rec("c2"))
    store.finalize()
    merged = json.load(open(store.canonical_path))
    ids = [r["clip_id"] for r in merged]
    assert len(ids) == len(set(ids)) == 2
    assert {r["clip_id"]: r["hyp"] for r in merged}["c1"] == "v1"


def test_corrupt_chunk_line_does_not_block_merge_or_silently_pass(store):
    store.append(_rec("c1"))
    store.flush_chunk()
    # simulate a torn write: truncate last chunk file mid-record
    chunk = sorted((store.chunk_dir).glob("*.jsonl"))[-1]
    raw = chunk.read_bytes()[:-10]
    chunk.write_bytes(raw)
    store.append(_rec("c2"))
    with pytest.raises(ArtifactError):
        store.finalize()                 # refuse to publish over corruption


def test_conflicting_hash_records_abort_publication(store):
    """Same model, different hyp for one clip => refuse to publish."""
    store.append({"clip_id": "c1", "hyp": "bir"})
    store.flush_chunk()
    store.append({"clip_id": "c1", "hyp": "iki"})
    with pytest.raises(ArtifactError):
        store.finalize()
    # canonical file must not exist / remain untouched by the failed publish
    assert not os.path.exists(store.canonical_path)


def test_retranscription_by_different_revision_is_accepted(tmp_path):
    """A pinned re-run with a NEW revision legitimately replaces old output."""
    s1 = ArtifactStore(root=str(tmp_path), model="m", revision="r1")
    s1.append({"clip_id": "c1", "hyp": "old decode"})
    s1.finalize()
    s2 = ArtifactStore(root=str(tmp_path), model="m", revision="r2")
    s2.append({"clip_id": "c1", "hyp": "better decode"})
    s2.finalize()
    merged = json.load(open(s2.canonical_path))
    assert len(merged) == 1
    assert merged[0]["revision"] == "r2"
    assert merged[0]["hyp"] == "better decode"


def test_flush_interval_automatic(store):
    store = ArtifactStore(root=store.root, model="m2", revision="r",
                          chunk_every=3)
    for i in range(7):
        store.append(_rec(f"c{i}"))
    n_chunks = len(list(store.chunk_dir.glob('*.jsonl')))
    assert n_chunks >= 2                 # automatic flush at every 3 records
    store.finalize()
    assert len(json.load(open(store.canonical_path))) == 7


def test_done_clip_ids_roundtrip(store):
    store.append(_rec("c1"))
    store.append(_rec("c2"))
    store.flush_chunk()
    fresh = ArtifactStore(root=store.root, model=store.model,
                          revision=store.revision)
    done = fresh.done_clip_ids()
    assert done == {"c1", "c2"}


def test_atomic_canonical_publish(tmp_path):
    """Canonical publish goes through tmp+rename; a crash cannot leave garbage."""
    import glob

    s = ArtifactStore(root=str(tmp_path), model="m3", revision="r")
    s.append(_rec("c1"))
    s.finalize()
    leftovers = [p for p in glob.glob(str(tmp_path / "*.tmp*"))]
    assert leftovers == []
    data = json.load(open(s.canonical_path))
    assert data[0]["model"] == "m3"      # provenance stamped automatically
