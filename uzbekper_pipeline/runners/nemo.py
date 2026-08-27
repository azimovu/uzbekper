"""NeMo FastConformer runner — batched transcribe, no per-file loop."""
from __future__ import annotations

from .base import BaseRunner


class NemoRunner(BaseRunner):
    def _transcribe_batch(self, adapter, paths):
        return adapter.transcribe_batch(paths)


class NemoASRAdapter:
    """Real GPU adapter built inside the Modal NeMo image."""

    def __init__(self, model_id: str, revision: str | None = None):
        import nemo.collections.asr as nemo_asr

        kw = {"version": revision} if revision else {}
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_id, **kw)

    def transcribe_batch(self, paths):
        """nemo >= 1.20: returns list of transcripts for a file list.

        Older API returned (hypotheses, all_hyps); normalize both.
        """
        try:
            result = self.model.transcribe(paths, batch_size=len(paths))
        except TypeError:
            result = self.model.transcribe(paths)
        if isinstance(result, tuple):
            result = result[0]
        out = []
        for r in result:
            text = getattr(r, "text", None)
            if text is None and isinstance(r, dict):
                text = r.get("text") or r.get("pred_text")
            if text is None:
                text = str(r) if r is not None else None
            out.append(text if text else ("" if r is not None else None))
        return out
