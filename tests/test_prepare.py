"""TDD RED tests for the prepare-step core (rewrite-plan Task 3, items 5–6).

``uzbekper_pipeline.prepare.run_prepare`` is the provider-neutral body the
Modal CPU function will call. Contract:

* Probes every ready-manifest row via ``audio_inventory`` (real structural
  WAV validation, NOT size checks).
* Writes an atomic, content-addressed inventory:
  ``inventory_<dataset-sha8>.jsonl`` containing one line per clip plus a
  final summary line.
* NEVER silently overwrites an existing inventory for the same dataset
  digest -- raises PrepareError instead (append-only spirit).
* Returns a summary dict {present, missing, corrupt, total_present_seconds,
  inventory_path} usable for receipts.
* Works with plain arguments only (rows, audio_root, out_dir) so both local
  pytest and a Modal CPU container can drive it identically.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
import sys
import wave
from pathlib import Path

import pytest

repo = "/home/ubuntu/uzbekper"
if repo not in sys.path:
    sys.path.insert(0, repo)


def _mod():
    sys.modules.pop("uzbekper_pipeline.prepare", None)
    return importlib.import_module("uzbekper_pipeline.prepare")


@pytest.fixture()
def mod():
    return _mod()


def _write_wav(path, seconds=0.3, rate=16000, channels=1):
    n = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(18000 * math.sin(2 * math.pi * 330 * i / rate)))
        for i in range(n * channels)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready_rows(tmp_path):
    """Two good clips + one missing, mirroring manifest shape."""
    _write_wav(tmp_path / "audios" / "a" / "1.wav", seconds=1.0)
    _write_wav(tmp_path / "audios" / "b" / "2.wav", seconds=2.0)
    return [
        {"clip_id": "audios/a/1.wav", "text": "salom", "source": "usc"},
        {"clip_id": "audios/b/2.wav", "text": "xayr", "source": "usc"},
        {"clip_id": "audios/c/3.wav", "text": "yo'q", "source": "cv"},
    ]


def test_run_prepare_writes_content_addressed_inventory_and_summary(tmp_path, mod):
    rows = _ready_rows(tmp_path)
    audio_root = str(tmp_path)
    out_dir = tmp_path / "inventory"

    summary = mod.run_prepare(
        rows, audio_root, out_dir=str(out_dir),
        dataset_jsonl=None,
    )

    # Exactly one corrupt-free accounting: 2 present, 1 missing.
    assert summary["present"] == 2
    assert summary["missing"] == 1
    assert summary["corrupt"] == 0
    inv = Path(summary["inventory_path"])
    assert inv.exists()

    # Content-addressed name derived from the ROW SET digest.
    expected_tag = summary["dataset_sha256"][:8]
    assert inv.name == f"inventory_{expected_tag}.jsonl"

    lines = [json.loads(l) for l in inv.read_text().splitlines() if l.strip()]
    # n records + 1 trailing summary marker
    assert len(lines) == len(rows) + 1
    assert lines[-1]["record_type"] == "summary"
    assert lines[-1]["present"] == 2


def test_run_prepare_refuses_to_overwrite_same_digest(tmp_path, mod):
    rows = _ready_rows(tmp_path)
    kwargs = dict(rows=rows, audio_root=str(tmp_path), out_dir=str(tmp_path / "inv"))
    mod.run_prepare(dataset_jsonl=None, **kwargs)

    # Same inputs -> same digest -> must REFUSE rather than clobber.
    with pytest.raises(mod.PrepareError):
        mod.run_prepare(dataset_jsonl=None, **kwargs)


def test_run_prepare_force_allows_regenerate_for_same_digest(tmp_path, mod):
    rows = _ready_rows(tmp_path)
    kwargs = dict(rows=rows, audio_root=str(tmp_path), out_dir=str(tmp_path / "inv"))
    mod.run_prepare(dataset_jsonl=None, **kwargs)
    s2 = mod.run_prepare(dataset_jsonl=None, force=True, **kwargs)
    assert Path(s2["inventory_path"]).exists()


def test_run_prepare_atomic_no_tmp_leftovers(tmp_path, mod):
    rows = _ready_rows(tmp_path)
    out_dir = tmp_path / "inv"
    mod.run_prepare(rows=rows, audio_root=str(tmp_path), out_dir=str(out_dir), dataset_jsonl=None)
    leftovers = [p.name for p in out_dir.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_run_prepare_is_idempotent_other_dataset_gets_own_file(tmp_path, mod):
    rows_a = _ready_rows(tmp_path)
    out_dir = tmp_path / "inv"
    sa = mod.run_prepare(rows=rows_a, audio_root=str(tmp_path), out_dir=str(out_dir), dataset_jsonl=None)

    rows_b = rows_a[:2]  # different set -> different digest
    sb = mod.run_prepare(rows=rows_b, audio_root=str(tmp_path), out_dir=str(out_dir), dataset_jsonl=None)
    assert sa["dataset_sha256"] != sb["dataset_sha256"]
    assert sa["inventory_path"] != sb["inventory_path"]


def test_run_prepare_rejects_empty_rows(tmp_path, mod):
    with pytest.raises(mod.PrepareError):
        mod.run_prepare(rows=[], audio_root=str(tmp_path), out_dir=str(tmp_path / "inv"), dataset_jsonl=None)
