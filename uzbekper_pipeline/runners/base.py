"""Shared runner contract for all ASR frameworks."""
from __future__ import annotations

import os
import time


class RunnerError(RuntimeError):
    """Raised when a run violates its contract (e.g. error-rate ceiling)."""


class BaseRunner:
    """Template method: batch, adapt-framework, record — identical for all.

    Subclasses implement _transcribe_batch(adapter, paths) -> list[str|None].
    A None marks that single clip as failed; the batch continues.
    """

    def __init__(self, adapter):
        self.adapter = adapter

    def _load_rows(self, rows: list[dict], audio_root: str):
        """Attach absolute paths; missing files fail per-clip, not fatally."""
        out = []
        for r in rows:
            r = dict(r)
            p = os.path.join(audio_root, r["clip_id"])
            r["_path"] = p
            r["_ok"] = os.path.isfile(p) and os.path.getsize(p) > 0
            out.append(r)
        return out

    def run_transcription(
        self,
        spec: dict,
        rows: list[dict],
        audio_root: str,
        store,
    ) -> dict:
        from uzbekper_pipeline.quality import evaluate_probe_outputs

        max_err = float(spec.get("max_error_rate", 0.05))
        bs = int(spec.get("batch_size", 16))
        decode = spec.get("decode", {})
        prepared = self._load_rows(rows, audio_root)
        n = len(prepared)
        t0 = time.time()
        n_ok = n_fail = 0
        outputs_for_probe = []
        for start in range(0, n, bs):
            batch = prepared[start:start + bs]
            ok_rows = [r for r in batch if r["_ok"]]
            hyps = self._transcribe_batch(
                self.adapter, [r["_path"] for r in ok_rows]
            ) if ok_rows else []
            el = time.time() - t0
            done = min(start + bs, n)
            print(f"{spec['model_id']}: {done}/{n} ({el:.0f}s, "
                  f"{el / done:.2f}s/utt, bs={bs})", flush=True)
            hi = iter(hyps)
            for r in batch:
                if not r["_ok"]:
                    n_fail += 1
                    store.append({"clip_id": r["clip_id"], "ref": r["text"],
                                  "source": r.get("source", ""),
                                  "hyp": "<ERROR:AudioUnavailable>"})
                    continue
                h = next(hi)
                if h is None:
                    n_fail += 1
                    store.append({"clip_id": r["clip_id"], "ref": r["text"],
                                  "source": r.get("source", ""),
                                  "hyp": "<ERROR:TranscriptionFailed>"})
                else:
                    n_ok += 1
                    rec = {"clip_id": r["clip_id"], "ref": r["text"],
                           "source": r.get("source", ""), "hyp": h,
                           "model": spec["model_id"],
                           "revision": spec.get("revision"),
                           "runtime_s": round(el / max(1, done), 4)}
                    if decode:
                        rec["decode"] = decode
                    else:
                        rec.setdefault("decode", {})
                    store.append(rec)
                    outputs_for_probe.append(
                        {"clip_id": r["clip_id"], "ref": r["ref"] if "ref" in r
                         else r["text"], "hyp": h}
                    )
        if n and (n_fail / n) > max_err:
            # flush the store's records into chunks so nothing is lost for
            # inspection, but do NOT publish a canonical file
            store.flush_chunk()
            raise RunnerError(
                f"error rate {n_fail}/{n} exceeds ceiling {max_err}"
            )
        store.finalize()
        # every full run also passes the quality gate over its own outputs
        rep = evaluate_probe_outputs(outputs_for_probe[:50])
        summary = {
            "model_id": spec["model_id"],
            "revision": spec.get("revision"),
            "n_transcribed": n_ok,
            "n_failed": n_fail,
            "wall_s": round(time.time() - t0, 1),
            "probe_passed": rep.passed,
            "probe_flags": {k: v[:5] for k, v in rep.flags.items() if v},
        }
        return summary

    def _transcribe_batch(self, adapter, paths):
        raise NotImplementedError
