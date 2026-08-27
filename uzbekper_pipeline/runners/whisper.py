"""Whisper (transformers) runner — framework logic isolated here.

Production adapter (WhisperHFAdapter) does: processor -> input_features ->
model.generate(forced_decoder_ids=...) -> batch_decode. Heavy imports happen
only when the adapter is constructed, so contract tests stay CPU-only.
"""
from __future__ import annotations

from .base import BaseRunner


class WhisperRunner(BaseRunner):
    def _transcribe_batch(self, adapter, paths):
        return adapter.transcribe_batch(paths)


class WhisperHFAdapter:
    """Real GPU adapter built inside the Modal Whisper image."""

    def __init__(self, model_id: str, revision: str | None = None,
                 language: str = "uz", verify_pin: bool = True):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        kw = {"revision": revision} if revision else {}
        self.processor = WhisperProcessor.from_pretrained(model_id, **kw)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, **kw)
        self.model.eval()
        # transformers 4.46 path; re-verify against the pinned image version
        self.forced_ids = self.processor.get_decoder_prompt_ids(
            language=language, task="transcribe")
        self.language = language
        if verify_pin:
            self._verify_pin()

    def _turkish_hits(self, text: str) -> int:
        import re
        return len(re.findall(r"[çğİıÖöşŞüÇ]", text or ""))

    def _verify_pin(self, n_trials: int = 3) -> None:
        """Generate pure silence and a tone; Uzbek-pinned decoders emit
        *something* Uzbek-ish or empty — never Turkish full sentences."""
        import numpy as np

        for arr in (
            np.zeros(16000, dtype=np.float32),
            np.sin(np.linspace(0, 440 * 2 * 3.14159, 16000)).astype(np.float32)
            * 0.1,
        ):
            feats = self.processor(arr, sampling_rate=16000,
                                   return_tensors="np").input_features[0]
            inp = self.torch.from_numpy(feats).half().cuda().unsqueeze(0)
            with self.torch.no_grad():
                ids = self.model.generate(
                    inp, forced_decoder_ids=self.forced_ids, max_new_tokens=32)
            hyp = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
            if self._turkish_hits(hyp) > 2:
                raise RuntimeError(
                    f"language pin failed on synthetic audio: {hyp!r}")

    def transcribe_batch(self, paths):
        import librosa
        import numpy as np

        out = []
        audios = []
        for p in paths:
            try:
                a, _ = librosa.load(p, sr=16000)
                if len(a) < 1600:
                    a = np.pad(a, (0, 1600 - len(a)))
                audios.append(a)
            except Exception:
                audios.append(None)
        feats = []
        good = []
        for a in audios:
            if a is None:
                continue
            f = self.processor(a, sampling_rate=16000,
                               return_tensors="np").input_features[0]
            feats.append(f)
            good.append(True)
        results: list[str | None] = [None] * len(paths)
        gi = iter(range(len(feats)))
        if feats:
            inp = self.torch.stack(
                [self.torch.from_numpy(f) for f in feats]).half().cuda()
            with self.torch.no_grad():
                ids = self.model.generate(
                    inp, forced_decoder_ids=self.forced_ids, max_new_tokens=256)
            decoded = self.processor.batch_decode(
                ids, skip_special_tokens=True)
            di = iter(decoded)
            for i, a in enumerate(audios):
                if a is None:
                    continue
                results[i] = next(di)
        del good
        return results
