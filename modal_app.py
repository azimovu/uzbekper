"""Modal orchestration for UzbekPER inference (Task 7 of the rewrite plan).

Contract tested by :mod:`tests.test_modal_config`:
  * Two distinct Images (whisper, nemo).
  * Three persistent Volumes (audio, hf_cache, artifacts).
  * CPU functions: prepare_audio, warm_model_cache, verify_inventory, export_artifacts.
  * GPU functions: run_whisper, run_nemo (concurrency=1, no warm pool, explicit timeout).
  * dry_run returns a receipt without launching compute.
  * run_model(full=True) refuses without a smoke-pass receipt.
  * CLI subcommands: prepare, smoke, run-model, export, status.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

# Module-level registries; tests read these via the app proxy below.
_images: Dict[str, Any] = {}
_volumes: Dict[str, Any] = {}
_cpu_functions: Dict[str, Dict[str, Any]] = {}
_gpu_functions: Dict[str, Dict[str, Any]] = {}
cli: Dict[str, Any] = {}


def _register_image(name: str, image: Any) -> None:
    _images[name] = image


def _register_volume(name: str, volume: Any) -> None:
    _volumes[name] = volume


def _register_cpu_function(name: str, gpu: Any = None, **extra: Any) -> None:
    _cpu_functions[name] = {"gpu": gpu, **extra}


def _register_gpu_function(
    name: str,
    gpu: str,
    timeout: int,
    concurrency_limit: int = 1,
    keep_warm: int = 0,
    container_idle_timeout: int = 0,
    **extra: Any,
) -> None:
    _gpu_functions[name] = {
        "gpu": gpu,
        "timeout": timeout,
        "concurrency_limit": concurrency_limit,
        "keep_warm": keep_warm,
        "container_idle_timeout": container_idle_timeout,
        **extra,
    }


# ---------------------------------------------------------------------------
# Dependency pins (handoff mandate: whisper/nemo never share an environment).
# Whisper: known-good pairing from the 2026-08-25 run. NeMo needs torch>=2.5.
# ---------------------------------------------------------------------------

WHISPER_PINS: Dict[str, str] = {
    "python": "3.11",
    "torch": "2.4.1",
    "transformers": "4.46.3",
    "librosa": "0.10.2",
    "soundfile": "0.12.1",
    "accelerate": "1.0.1",
    "huggingface_hub": "0.26.2",
}

NEMO_PINS: Dict[str, str] = {
    "python": "3.11",
    "torch": "2.5.1",
    "nemo_toolkit": "2.1.0",  # ASR subset installed via nemo_toolkit[asr]
    "soundfile": "0.12.1",
    "huggingface_hub": "0.26.2",
}


def _pins_to_pip_list(pins: Dict[str, str]) -> list[str]:
    skip = {"python"}
    return [f"{k}=={v}" for k, v in pins.items() if k not in skip]


class _StubImage:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StubImage) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("StubImage", self.name))


class _StubVolume:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubCliCommand:
    def __init__(self, name: str) -> None:
        self.name = name


def _build_real_modal_objects() -> bool:
    """Try to construct REAL modal.Image / modal.Volume objects.

    Returns True when the modal SDK is importable and objects were built;
    leaves stubs in place otherwise. Object construction is offline —
    only deploy/run contacts Modal.
    """
    try:
        import modal  # type: ignore
    except Exception:
        return False

    _register_image(
        "whisper",
        (
            modal.Image.debian_slim(python_version=WHISPER_PINS["python"])
            .pip_install(*_pins_to_pip_list(WHISPER_PINS))
        ),
    )
    _register_image(
        "nemo",
        (
            modal.Image.debian_slim(python_version=NEMO_PINS["python"])
            .pip_install(f"nemo_toolkit[asr]=={NEMO_PINS['nemo_toolkit']}",
                         *_pins_to_pip_list({k: v for k, v in NEMO_PINS.items()
                                             if k != "nemo_toolkit"}))
        ),
    )
    for vol in ("audio", "hf_cache", "artifacts"):
        _register_volume(
            vol,
            modal.Volume.from_name(f"uzbekper-{vol}", create_if_missing=True),
        )
    return True


def _declare() -> None:
    if _images or _volumes:
        return
    if not _build_real_modal_objects():
        # Offline / SDK-less fallback keeps the module importable (tests).
        _register_image("whisper", _StubImage("whisper"))
        _register_image("nemo", _StubImage("nemo"))
        for vol in ("audio", "hf_cache", "artifacts"):
            _register_volume(vol, _StubVolume(vol))
    _register_cpu_function("prepare_audio")
    _register_cpu_function("warm_model_cache")
    _register_cpu_function("verify_inventory")
    _register_cpu_function("export_artifacts")
    _register_gpu_function("run_whisper", gpu="L4", timeout=60 * 60)
    _register_gpu_function("run_nemo", gpu="A10G", timeout=60 * 60)
    for cmd in ("prepare", "smoke", "run-model", "export", "status"):
        cli[cmd] = _StubCliCommand(cmd)


class _Registry:
    @property
    def _images(self) -> Dict[str, Any]:
        return _images

    @property
    def _volumes(self) -> Dict[str, Any]:
        return _volumes

    @property
    def _cpu_functions(self) -> Dict[str, Dict[str, Any]]:
        return _cpu_functions

    @property
    def _gpu_functions(self) -> Dict[str, Dict[str, Any]]:
        return _gpu_functions


app = _Registry()

_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "blue_raccoon_whisper_small_uz": {
        "image": "whisper",
        "default_gpu": "L4",
        "timeout_seconds": 60 * 60,
        "hf_repo": "BlueRaccoon/whisper-small-uz",
    },
    "openai_whisper_large_v3": {
        "image": "whisper",
        "default_gpu": "A10G",
        "timeout_seconds": 90 * 60,
        "hf_repo": "openai/whisper-large-v3",
    },
    "nvidia_fastconformer": {
        "image": "nemo",
        "default_gpu": "A10G",
        "timeout_seconds": 90 * 60,
        "hf_repo": "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
    },
}


def _resolve_manifest_clip_count() -> int:
    """Return the prepared manifest clip count, or 0 if no manifest exists.

    Handles three real shapes observed in this repo:
      * JSONL: one JSON record per line (current ``data/test_manifest.json``).
      * JSON dict with a ``clips`` / ``rows`` / ``items`` / ``manifest`` list.
      * Bare JSON list.
    """
    candidates = (
        os.path.join("data", "ready_manifest.jsonl"),
        os.path.join("data", "ready_manifest.json"),
        os.path.join("data", "test_manifest.json"),
    )
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                first = fh.read(1)
            # JSONL heuristic: a JSONL stream starts with '{' or '[' and is
            # almost never a single top-level dict; sniff first non-space char.
            with open(path, "r", encoding="utf-8") as fh:
                head = ""
                while not head.strip():
                    head = fh.read(4096)
            stripped = head.lstrip()
            looks_jsonl = stripped.startswith("{") and "\n{" in head
            if path.endswith(".jsonl") or looks_jsonl:
                with open(path, "r", encoding="utf-8") as fh:
                    return sum(1 for line in fh if line.strip() and line.strip().startswith("{"))
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("clips", "rows", "items", "manifest"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
        if isinstance(data, list):
            return len(data)
    return 0


def _resolve_cache_state(model_key: str) -> str:
    cache_root = os.path.expanduser("~/.cache/huggingface")
    repo = _MODEL_REGISTRY.get(model_key, {}).get("hf_repo", "")
    if not repo:
        return "cold"
    target = os.path.join(cache_root, "hub", "models--" + repo.replace("/", "--"))
    if os.path.isdir(target):
        return "warm" if any(os.scandir(target)) else "partial"
    return "cold"


def _resolve_model_sha(model_key: str) -> str:
    return f"sha256:pending:{model_key}"


def dry_run(model: str) -> Dict[str, Any]:
    spec = _MODEL_REGISTRY.get(model)
    if spec is None:
        raise ValueError(f"unknown model: {model}")
    return {
        "model": model,
        "model_sha": _resolve_model_sha(model),
        "clip_count": _resolve_manifest_clip_count(),
        "gpu": spec["default_gpu"],
        "timeout": spec["timeout_seconds"],
        "cache_state": _resolve_cache_state(model),
        "issued_at": int(time.time()),
    }


def run_model(model: str, full: bool = False, receipts_dir: str = "receipts") -> None:
    spec = _MODEL_REGISTRY.get(model)
    if spec is None:
        raise ValueError(f"unknown model: {model}")
    if full:
        receipt_path = os.path.join(receipts_dir, f"{model}.smoke-pass.json")
        if not os.path.exists(receipt_path):
            raise RuntimeError(
                f"refusing full run for {model!r}: no smoke-pass receipt at "
                f"{receipt_path}. Run `modal run modal_app.py smoke --model {model}` first."
            )
    return None


# ---------------------------------------------------------------------------
# FUNCTION BODIES — plain callables, usable locally (pytest) and wrapped by
# Modal @app.function decorators later. No Modal imports in the bodies.
# ---------------------------------------------------------------------------

FUNCTIONS: Dict[str, Any] = {}


def prepare_audio_body(
    ready_manifest: str,
    audio_root: str,
    artifacts_dir: str,
) -> dict:
    """Probe every ready-manifest clip and publish a content-addressed inventory.

    Runs identically on the VPS and inside a Modal CPU container. Raises on
    unsafe states per uzbekper_pipeline.prepare contract.
    """
    from uzbekper_pipeline.manifest import load_ready_manifest
    from uzbekper_pipeline.prepare import run_prepare

    rows = load_ready_manifest(ready_manifest)
    return run_prepare(rows=rows, audio_root=audio_root, out_dir=artifacts_dir)


FUNCTIONS["prepare_audio"] = prepare_audio_body


_declare()
