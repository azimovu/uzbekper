"""Runner contract tests — one interface, two frameworks, no signature drift.

The old benchmark_infer.py crashed NeMo runs by calling
``fn(mid, todo, audio_root, rec, language=..., batch_size=..., verify_pin=...)``
for BOTH frameworks although run_nemo() accepted none of the keyword args.
Both runners now conform to run_transcription(spec, rows, adapter) and the
framework-specific machinery is injected via a small ASRAdapter protocol,
which tests fake cheaply without torch/NeMo installed.
"""
import pytest

from uzbekper_pipeline.runners.whisper import WhisperRunner
from uzbekper_pipeline.runners.nemo import NemoRunner
from uzbekper_pipeline.artifacts import ArtifactStore


class FakeAdapter:
    """Records calls; returns deterministic transcript per wav path."""

    def __init__(self, fail_paths=()):
        self.batches = []
        self.fail_paths = set(fail_paths)

    def transcribe_batch(self, paths):
        self.batches.append(list(paths))
        return [
            f"hyp<{p}>" if p not in self.fail_paths else None
            for p in paths
        ]


def _rows(n=5):
    return [{"clip_id": f"audios/{i}.wav", "text": f"ref {i}",
             "source": "test"} for i in range(n)]


def _wavs(rows, tmp_path):
    for r in rows:
        p = tmp_path / r["clip_id"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFFfake")


@pytest.mark.parametrize("runner_cls", [WhisperRunner, NemoRunner])
def test_contract_uniform_signature(tmp_path, runner_cls):
    rows = _rows()
    _wavs(rows, tmp_path)
    store = ArtifactStore(root=str(tmp_path), model="m", revision="r",
                          chunk_every=2)
    adapter = FakeAdapter()
    runner = runner_cls(adapter)
    summary = runner.run_transcription(
        {"model_id": "m/x", "revision": "rev", "batch_size": 2},
        rows, str(tmp_path), store)
    assert summary["n_transcribed"] == 5
    assert summary["n_failed"] == 0
    assert len(adapter.batches[0]) == 2      # batch_size honored
    merged = __import__("json").load(open(store.canonical_path))
    assert len(merged) == 5
    assert merged[0]["clip_id"] == "audios/0.wav"
    assert merged[0]["model"] == "m/x"
    assert merged[0]["revision"] == "rev"


@pytest.mark.parametrize("runner_cls", [WhisperRunner, NemoRunner])
def test_per_clip_failure_does_not_kill_batch(tmp_path, runner_cls):
    rows = _rows()
    _wavs(rows, tmp_path)
    bad_rows = rows + [{"clip_id": "audios/999.wav", "text": "ghost",
                        "source": "test"}]
    store = ArtifactStore(root=str(tmp_path), model="m", revision="r")
    adapter = FakeAdapter()          # never fails; the missing FILE fails
    runner = runner_cls(adapter)
    summary = runner.run_transcription(
        {"model_id": "m/x", "revision": "rev", "batch_size": 3,
         "max_error_rate": 0.5},
        bad_rows, str(tmp_path), store)
    assert summary["n_failed"] == 1          # recorded as failure row...
    assert summary["n_transcribed"] == 5     # ...without losing good clips
    recs = __import__("json").load(open(store.canonical_path))
    hyps = {r["clip_id"]: r["hyp"] for r in recs}
    assert "<ERROR:AudioUnavailable>" in hyps.values()


def test_error_rate_ceiling_aborts_run(tmp_path):
    from uzbekper_pipeline.runners.base import RunnerError

    rows = _rows(6)
    _wavs(rows, tmp_path)
    # remove half the wavs -> loads will fail
    (tmp_path / "audios/1.wav").unlink()
    (tmp_path / "audios/3.wav").unlink()
    (tmp_path / "audios/5.wav").unlink()
    store = ArtifactStore(root=str(tmp_path), model="m", revision="r")

    class BoomAdapter(FakeAdapter):
        def transcribe_batch(self, paths):
            out = []
            for p in paths:
                if p.endswith(("1.wav", "3.wav", "5.wav")):
                    out.append(None)   # unreadable -> None
                else:
                    out.append(f"hyp<{p}>")
            return out

    runner = WhisperRunner(BoomAdapter())
    with pytest.raises(RunnerError):
        runner.run_transcription(
            {"model_id": "m/x", "revision": "rev", "batch_size": 6,
             "max_error_rate": 0.25},
            rows, str(tmp_path), store)


def test_runner_records_decode_and_timing(tmp_path):
    rows = _rows(2)
    _wavs(rows, tmp_path)
    store = ArtifactStore(root=str(tmp_path), model="m", revision="r")
    runner = WhisperRunner(FakeAdapter())
    runner.run_transcription(
        {"model_id": "m/x", "revision": "rev", "batch_size": 2,
         "decode": {"decoding": "greedy", "language": "uz"}},
        rows, str(tmp_path), store)
    recs = __import__("json").load(open(store.canonical_path))
    assert recs[0]["decode"] == {"decoding": "greedy", "language": "uz"}
    assert isinstance(recs[0]["runtime_s"], (int, float))
