"""Tests for uzbekper_pipeline.manifest — stable-clip-ID manifest freezing."""
import hashlib
import json
import os

import pytest

from uzbekper_pipeline.manifest import (
    ManifestError,
    build_ready_manifest,
    freeze_manifest,
    load_manifest,
)

REPO = "/home/ubuntu/uzbekper"


def _join_counts():
    """Matched counts per source straight from the join files (ground truth)."""
    import glob

    counts = {}
    for p in glob.glob(os.path.join(REPO, "data", "join_*.json")):
        src = os.path.basename(p)[5:-5]
        if src == "podcasts":
            src = "podcasts_dialect"  # join filename differs from manifest tag
        d = json.load(open(p))
        counts[src] = sum(1 for v in d["rows"].values() if v.get("matched"))
    return counts


def test_load_real_manifest_rows_and_sources():
    rows = load_manifest(os.path.join(REPO, "data", "test_manifest.json"))
    assert len(rows) == 10151
    assert all({"audio_filepath", "text", "source"} <= set(r) for r in rows)
    assert len({r["audio_filepath"] for r in rows}) == len(rows)


def test_ready_manifest_matches_join_ground_truth():
    """Ready set must equal exactly the matched clips across join files (9320)."""
    rows = load_manifest(os.path.join(REPO, "data", "test_manifest.json"))
    ready = build_ready_manifest(rows, os.path.join(REPO, "data"))
    truth = _join_counts()
    assert sum(truth.values()) == 9320
    assert len(ready) == 9320
    got = {}
    for r in ready:
        got[r["source"]] = got.get(r["source"], 0) + 1
    assert got == truth


def test_clip_id_is_stable_and_deterministic():
    """Two independent builds from the canonical manifest agree exactly."""
    rows_a = load_manifest(os.path.join(REPO, "data", "test_manifest.json"))
    rows_b = load_manifest(os.path.join(REPO, "data", "test_manifest.json"))
    ready_a = build_ready_manifest(rows_a, os.path.join(REPO, "data"))
    ready_b = build_ready_manifest(rows_b, os.path.join(REPO, "data"))
    assert [r["clip_id"] for r in ready_a] == [r["clip_id"] for r in ready_b]
    id_set = {r["clip_id"] for r in ready_a}
    manifest_paths = {r["audio_filepath"] for r in rows_a}
    assert id_set <= manifest_paths               # every clip_id is a real path
    assert len(id_set) == len(ready_a)            # unique


def test_freeze_writes_sorted_jsonl_with_sha256(tmp_path):
    rows = load_manifest(os.path.join(REPO, "data", "test_manifest.json"))
    ready = build_ready_manifest(rows, os.path.join(REPO, "data"))
    out = tmp_path / "ready_manifest.jsonl"
    sha = freeze_manifest(ready, str(out))
    data = out.read_text().splitlines()
    assert len(data) == 9320
    # sorted by clip_id => byte-reproducible
    bodies = [json.loads(l)["clip_id"] for l in data]
    assert bodies == sorted(bodies)
    assert sha == hashlib.sha256(out.read_bytes()).hexdigest()


def test_duplicate_clip_ids_rejected():
    rows = [
        {"audio_filepath": "audios/a.wav", "text": "x", "source": "s"},
        {"audio_filepath": "audios/a.wav", "text": "y", "source": "s"},
    ]
    with pytest.raises(ManifestError):
        build_ready_manifest(rows, "/nonexistent")


def test_missing_fields_rejected():
    with pytest.raises(ManifestError):
        build_ready_manifest([{"text": "no path or source"}], "/nonexistent")


def test_run_spec_pins_models_and_dataset_hash(tmp_path):
    from uzbekper_pipeline.manifest import write_run_spec

    rows = load_manifest(os.path.join(REPO, "data", "test_manifest.json"))
    ready = build_ready_manifest(rows, os.path.join(REPO, "data"))
    out = tmp_path / "ready_manifest.jsonl"
    sha = freeze_manifest(ready, str(out))
    spec_path = tmp_path / "run_spec.json"
    spec = write_run_spec(
        spec_path,
        dataset_sha=sha,
        models={
            "nvidia_fastconformer": {
                "repo_id": "nvidia/stt_uz_fastconformer_hybrid_large_pc",
                "revision": "abc1234567890",
            }
        },
        decode={"decoding": "greedy/default", "language": "uz"},
        code_version="modal-rewrite-r1",
        n_clips=len(ready),
    )
    loaded = json.loads(spec_path.read_text())
    assert loaded == spec
    assert spec["dataset_sha256"] == sha
    assert spec["n_clips"] == 9320
    assert "created_utc" in spec
