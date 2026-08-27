"""TDD RED tests for real Modal wiring (rewrite-plan Task 4).

Task 7 gave us offline stubs; Task 4 replaces them with REAL Modal objects
when the SDK is available (it is, in this repo's .venv: modal 1.5.4).

Contract added on top of test_modal_config.py:

* When ``import modal`` succeeds, the registries hold REAL modal.Image /
  modal.Volume instances — not stubs. Whisper and NeMo stay distinct images.
* Dependency pins are recorded explicitly (module-level dicts), matching the
  handoff's hard lesson: NeMo and Whisper MUST NOT share one environment.
* The CPU function bodies are plain callables usable WITHOUT Modal (local
  pytest drives them directly); prepare_audio's body must end-to-end produce
  a content-addressed inventory via uzbekper_pipeline.prepare.run_prepare.
* Old contract (stubs when SDK absent) keeps working — tested implicitly by
  test_modal_config.py staying green.
"""

from __future__ import annotations

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
    sys.modules.pop("modal_app", None)
    return importlib.import_module("modal_app")


@pytest.fixture()
def mod():
    return _mod()


def _modal_or_skip():
    modal = pytest.importorskip("modal")
    return modal


# ----- real images -------------------------------------------------------------

def test_images_are_real_modal_instances(mod):
    modal = _modal_or_skip()
    imgs = mod.app._images
    assert isinstance(imgs["whisper"], modal.Image), type(imgs["whisper"])
    assert isinstance(imgs["nemo"], modal.Image), type(imgs["nemo"])
    assert imgs["whisper"] is not imgs["nemo"]


def test_volumes_are_real_modal_instances(mod):
    modal = _modal_or_skip()
    vols = mod.app._volumes
    for name in ("audio", "hf_cache", "artifacts"):
        assert isinstance(vols[name], modal.Volume), type(vols[name])


# ----- pinned dependencies ------------------------------------------------------

def test_dependency_pins_recorded_and_split_per_stack(mod):
    wp = getattr(mod, "WHISPER_PINS", None)
    np_ = getattr(mod, "NEMO_PINS", None)
    assert isinstance(wp, dict) and wp, "WHISPER_PINS missing/empty"
    assert isinstance(np_, dict) and np_, "NEMO_PINS missing/empty"

    # Handoff-mandated split: whisper stays on the known-good pairing,
    # nemo needs a newer torch. They must never be identical envs.
    assert wp.get("transformers") == "4.46.3"
    assert str(wp.get("torch")).startswith("2.4")
    assert str(np_.get("torch")) != str(wp.get("torch"))
    assert "nemo" in " ".join(np_.keys()).lower(), "NeMo stack pin missing"


# ----- plain-callable CPU bodies --------------------------------------------------

def _wav(path: Path, seconds=0.2):
    n = int(seconds * 16000)
    frames = b"".join(
        struct.pack("<h", int(15000 * math.sin(2 * math.pi * 300 * i / 16000)))
        for i in range(n)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(frames)


def test_prepare_audio_body_runs_locally_without_modal(tmp_path, mod):
    bodies = getattr(mod, "FUNCTIONS", None)
    assert isinstance(bodies, dict) and "prepare_audio" in bodies
    fn = bodies["prepare_audio"]
    assert callable(fn)

    # Build a tiny ready-manifest JSONL + two wavs (ready manifests carry
    # clip_id/text/source per freeze_manifest(); no audio_filepath).
    root = tmp_path / "audio_vol"
    _wav(root / "x" / "1.wav")
    _wav(root / "y" / "2.wav")
    rows = [
        {"clip_id": "x/1.wav", "text": "a", "source": "usc"},
        {"clip_id": "y/2.wav", "text": "b", "source": "cv"},
    ]
    man = tmp_path / "ready.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows))

    out = fn(
        ready_manifest=str(man),
        audio_root=str(root),
        artifacts_dir=str(tmp_path / "art"),
    )
    assert out["present"] == 2 and Path(out["inventory_path"]).exists()


def test_run_model_full_still_refuses(mod, tmp_path):
    # Guard must survive rewiring: full runs still need a smoke receipt.
    with pytest.raises(RuntimeError, match="smoke"):
        mod.run_model(model="blue_raccoon_whisper_small_uz",
                      full=True, receipts_dir=str(tmp_path))
