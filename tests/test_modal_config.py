"""TDD RED tests for Modal orchestration (Task 7 of the Modal rewrite plan).

Asserts the contract of ``modal_app.py`` without importing the real Modal
client. The implementations under test must be pure-Python introspection of the
``app`` and its registered functions so they can be verified offline (no GPU,
no network, no Modal token required).

What we lock in for Task 7:

* Two separate Images (whisper, nemo) — never shared deps.
* Three persistent Volumes (audio, hf_cache, artifacts).
* CPU-only functions: prepare_audio, warm_model_cache, verify_inventory,
  export_artifacts.
* GPU functions: one model per invocation, fixed concurrency=1, timeout set,
  no keep-warm / min-container.
* A dry-run function that returns the resolved model SHA, clip count, GPU,
  timeout, and cache state without launching compute.
* A refuse-to-run guard: run-model --full is rejected unless matching prep
  and smoke-pass receipts exist.
* CLI subcommands: prepare, smoke, run-model, export, status.
"""

from __future__ import annotations

import importlib
import sys


def _import_modal_app():
    """Import modal_app from the repo root, regardless of pytest cwd."""
    repo_root = "/home/ubuntu/uzbekper"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # Force a fresh import so each test sees the current on-disk file.
    sys.modules.pop("modal_app", None)
    return importlib.import_module("modal_app")


# ----- Image contract --------------------------------------------------------

def test_two_separate_images_for_whisper_and_nemo():
    app = _import_modal_app()
    images = app._images  # type: ignore[attr-defined]
    assert "whisper" in images, "expected a 'whisper' image"
    assert "nemo" in images, "expected a 'nemo' image"
    assert images["whisper"] is not images["nemo"], "Whisper and NeMo must NOT share an image"


# ----- Volume contract --------------------------------------------------------

def test_three_persistent_volumes():
    app = _import_modal_app()
    volumes = app._volumes  # type: ignore[attr-defined]
    assert "audio" in volumes
    assert "hf_cache" in volumes
    assert "artifacts" in volumes


# ----- Function registry ------------------------------------------------------

def test_cpu_only_preparation_functions_are_registered():
    app = _import_modal_app()
    fns = app._cpu_functions  # type: ignore[attr-defined]
    for name in ("prepare_audio", "warm_model_cache", "verify_inventory", "export_artifacts"):
        assert name in fns, f"missing CPU function: {name}"
        assert fns[name].get("gpu") in (None, False), f"{name} must be CPU-only"


def test_gpu_functions_one_model_per_invocation_no_warm_pool():
    app = _import_modal_app()
    gpu = app._gpu_functions  # type: ignore[attr-defined]
    # At least whisper and nemo runners must exist.
    assert "run_whisper" in gpu
    assert "run_nemo" in gpu
    for name, spec in gpu.items():
        assert spec.get("gpu"), f"{name} must request a GPU"
        assert spec.get("concurrency_limit") == 1, f"{name} concurrency must be 1"
        assert spec.get("keep_warm", 0) == 0, f"{name} must not keep warm"
        assert spec.get("container_idle_timeout", 0) == 0, f"{name} must not idle-warm"
        assert spec.get("timeout"), f"{name} must declare a timeout"


# ----- Dry-run ----------------------------------------------------------------

def test_dry_run_returns_full_resolution_without_launching_compute():
    app = _import_modal_app()
    receipt = app.dry_run(model="blue_raccoon_whisper_small_uz")  # type: ignore[attr-defined]
    for key in ("model_sha", "clip_count", "gpu", "timeout", "cache_state"):
        assert key in receipt, f"dry_run receipt missing key: {key}"
    assert receipt["gpu"] in ("L4", "A10G", "A10", "T4")
    assert receipt["clip_count"] > 0
    assert receipt["cache_state"] in ("cold", "warm", "partial")


# ----- Refuse-to-run guard ----------------------------------------------------

def test_run_model_full_refused_without_smoke_receipt(tmp_path):
    app = _import_modal_app()
    # No receipts on disk -> must refuse.
    try:
        app.run_model(model="blue_raccoon_whisper_small_uz", full=True, receipts_dir=str(tmp_path))
    except RuntimeError as exc:
        msg = str(exc).lower()
        assert "smoke" in msg or "receipt" in msg
    else:
        raise AssertionError("run_model --full should refuse without smoke receipt")


# ----- CLI --------------------------------------------------------------------

def test_cli_subcommands_present():
    app = _import_modal_app()
    cli = app.cli  # type: ignore[attr-defined]
    for cmd in ("prepare", "smoke", "run-model", "export", "status"):
        assert cmd in cli, f"CLI missing subcommand: {cmd}"
