"""TDD RED tests for audio probe + inventory (rewrite-plan Task 3, item 3).

The 2026-08-25 run accepted any file with size > 1KB as "fetched". Several such
files turned out to be HTML error pages / truncated parquet garbage, poisoning
inference runs downstream. The contract below makes validation real:

* ``probe_audio`` must parse actual PCM structure and report sample_rate,
  channels, duration_seconds, sha256 content hash — or raise AudioProbeError.
* ``build_inventory`` walks ready-manifest clip_ids, tolerates missing/corrupt
  files per-record (status field), never throws away the whole walk.
* ``summarize_inventory`` yields present/missing/corrupt counts + total
  duration of present clips only.
* Everything must work offline with the Python standard library only.
"""

from __future__ import annotations

import importlib
import math
import struct
import sys
import wave

import pytest

repo = "/home/ubuntu/uzbekper"
if repo not in sys.path:
    sys.path.insert(0, repo)


def _mod():
    sys.modules.pop("uzbekper_pipeline.audio_inventory", None)
    return importlib.import_module("uzbekper_pipeline.audio_inventory")


def _write_wav(path, seconds=0.5, rate=16000, channels=1):
    """Deterministic sine WAV via stdlib."""
    n = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(20000 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(n * channels)
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


@pytest.fixture()
def mod():
    return _mod()


# ----- probe_audio -----------------------------------------------------------

def test_probe_valid_wav_reports_real_properties(tmp_path, mod):
    p = tmp_path / "a.wav"
    _write_wav(p, seconds=0.5, rate=16000, channels=1)
    info = mod.probe_audio(str(p))
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    # sine of exactly 0.5s at 16k -> 8000 samples
    assert abs(info["duration_seconds"] - 0.5) < 0.01
    assert len(info["sha256"]) == 64


def test_probe_hash_is_content_deterministic(tmp_path, mod):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wav(a)
    _write_wav(b)          # identical bytes -> identical hash
    c = tmp_path / "c.wav"
    _write_wav(c, seconds=0.7)  # different content -> different hash
    assert mod.probe_audio(str(a))["sha256"] == mod.probe_audio(str(b))["sha256"]
    assert mod.probe_audio(str(a))["sha256"] != mod.probe_audio(str(c))["sha256"]


def test_probe_garbage_file_raises(tmp_path, mod):
    p = tmp_path / "garbage.wav"
    p.write_bytes(b"<!DOCTYPE html><html><body>403 Forbidden</body></html>" * 30)
    with pytest.raises(mod.AudioProbeError):
        mod.probe_audio(str(p))


def test_probe_empty_file_raises(tmp_path, mod):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(mod.AudioProbeError):
        mod.probe_audio(str(p))


def test_probe_truncated_wav_raises(tmp_path, mod):
    good = tmp_path / "g.wav"
    _write_wav(good)
    raw = good.read_bytes()[:64]  # header-ish but truncated body
    bad = tmp_path / "t.wav"
    bad.write_bytes(raw)
    with pytest.raises(mod.AudioProbeError):
        mod.probe_audio(str(bad))


def test_probe_missing_file_raises_filenotfound(tmp_path, mod):
    with pytest.raises(FileNotFoundError):
        mod.probe_audio(str(tmp_path / "nope.wav"))


# ----- build_inventory ---------------------------------------------------------

def test_build_inventory_all_present(tmp_path, mod):
    rows = []
    for i, cid in enumerate(["audios/x/1.wav", "audios/y/2.wav"]):
        _write_wav(tmp_path / f"u{i}.wav")
        rows.append({"clip_id": cid})
    # map clip_id -> local fixture path via audio_root layout: we stage a real
    # root instead: create audios/x/1.wav etc.
    root = tmp_path / "root"
    for cid in ["x/1.wav", "y/2.wav"]:
        f = root / cid
        f.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(f)
    inv = mod.build_inventory(
        [{"clip_id": "x/1.wav"}, {"clip_id": "y/2.wav"}], str(root),
        relpath_key="clip_id",
    )
    assert [r["status"] for r in inv] == ["present", "present"]
    assert all(r["sample_rate"] == 16000 for r in inv)


def test_build_inventory_marks_missing_without_crashing(tmp_path, mod):
    (tmp_path / "here").mkdir()
    _write_wav(tmp_path / "here" / "ok.wav")
    inv = mod.build_inventory(
        [
            {"clip_id": "here/ok.wav"},
            {"clip_id": "there/never_fetched.wav"},
        ],
        str(tmp_path),
        relpath_key="clip_id",
    )
    by_cid = {r["clip_id"]: r["status"] for r in inv}
    assert by_cid["here/ok.wav"] == "present"
    assert by_cid["there/never_fetched.wav"] == "missing"


def test_build_inventory_marks_corrupt_without_crashing(tmp_path, mod):
    bad = tmp_path / "bad" / "corrupt.wav"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"\xff" * 5000)  # binary junk, right size band
    inv = mod.build_inventory([{"clip_id": "bad/corrupt.wav"}], str(tmp_path))
    assert inv[0]["status"] == "corrupt"
    assert "error_kind" in inv[0]


# ----- summarize_inventory -----------------------------------------------------

def test_summarize_counts_and_duration_only_present(tmp_path, mod):
    root = tmp_path
    _write_wav(root / "ok1.wav", seconds=1.0)
    _write_wav(root / "ok2.wav", seconds=2.0)
    (root / "nope.wav").parent.mkdir(exist_ok=True)
    (root / "junk.wav").write_bytes(b"junk-not-wav")
    inv = mod.build_inventory(
        [
            {"clip_id": "ok1.wav"},
            {"clip_id": "ok2.wav"},
            {"clip_id": "nope.wav"},
            {"clip_id": "junk.wav"},
        ],
        str(root),
    )
    s = mod.summarize_inventory(inv)
    assert s["present"] == 2
    assert s["missing"] == 1
    assert s["corrupt"] == 1
    assert abs(s["total_present_seconds"] - 3.0) < 0.05


def test_inventory_is_sorted_by_clip_id(tmp_path, mod):
    root = tmp_path
    for cid in ["b.wav", "a.wav"]:
        _write_wav(root / cid)
    inv = mod.build_inventory([{"clip_id": "b.wav"}, {"clip_id": "a.wav"}], str(root))
    assert [r["clip_id"] for r in inv] == ["a.wav", "b.wav"]
