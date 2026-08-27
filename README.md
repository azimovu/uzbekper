# UzbekPER — Uzbek ASR PER Benchmark

A reproducible benchmark measuring **Phoneme Error Rate (PER)** vs **Word Error
Rate (WER)** for Uzbek automatic speech recognition, using a frozen 300-clip
test set and a patched Uzbek grapheme→phoneme (G2P) converter (`uzg2p`).

## Why PER for Uzbek

Uzbek orthography is shallow/transparent, so ASR errors are mostly **orthographic
(spelling/diacritic variants), not phonetic**. WER over-penalizes this: a system
that gets the *phonemes* right but writes them with the *wrong spelling* scores a
high WER while being acoustically near-perfect.

We quantify this with the **WER–PER gap**. On a zero-shot API baseline
(Gemini 3.5 Transcribe, 300 clips):

| Metric | Score |
|--------|-------|
| WER    | 34.55% |
| PER    | 4.40%  |
| Gap    | 30.1 pts |

The 30-point gap is the core finding: errors are orthographic, not phonetic —
**PER is the honest metric for Uzbek ASR.**

## Repository layout

```
data/sample_300.jsonl      # frozen 300-clip benchmark spec (seed 20260827)
                           #   each row: audio_filepath, duration, text(ref), source
scripts/
  sample_300.py            # reproducible stratified sampler
  fetch_sample_300.py      # materialize the 300 clips to ./audios (from HF)
  benchmark_score.py       # WER + PER scorer (uses uzg2p)
  gemini_transcribe_probe.py  # API eval via Modal US egress (geo-block bypass)
  gemini_wer.py            # WER computation helper
uzg2p/                     # patched epitran Uzbek G2P (phoneme converter)
final/transcripts_gemini.json  # Gemini transcripts for the 300 clips
logs/gemini_probe_2026-08-27.md  # full setup + WER/PER analysis
BENCHMARK_PROTOCOL.md      # methodology, systems list, contamination caveats
```

## Quick start

```bash
# 1. Materialize the 300 clips (needs HF_TOKEN for full rate limits)
python scripts/fetch_sample_300.py

# 2. Score a system's transcripts (ref/hyp JSONL) for WER + PER
uv run --with uzg2p python scripts/benchmark_score.py final/transcripts_gemini.json
```

## Reproducing the Gemini baseline

Gemini's free-tier API is geo-blocked outside supported regions. The probe runs
on Modal's US egress (the compute, not the VPS, makes the API call):

```bash
modal secret create google-gemini-staging GEMINI_API_KEY=<key>
modal run scripts/gemini_transcribe_probe.py::main --n 300
```

Results are scored through `uzg2p` so every system is compared on the same
phoneme inventory.

## Status

- [x] Frozen 300-clip test set (`data/sample_300.jsonl`)
- [x] `uzg2p` G2P converter (patched epitran)
- [x] WER + PER scorer
- [x] Gemini 3.5 Transcribe baseline: WER 34.55% / PER 4.40%
- [ ] Additional API systems (OpenAI Whisper, Deepgram)
- [ ] Open-weight models (Whisper-small-uz, whisper-large-uz) via GPU pod

## License

Code: MIT (see LICENSE). The 300-clip audio is **not** stored here — it is
refetched from public upstream HF datasets via `scripts/fetch_sample_300.py`.
Reference transcripts derive from those upstream sources; respect their
respective licenses (Common Voice CC0, USC, UzbekVoice, etc.).

## Citation

If you use UzbekPER, cite the benchmark and the `uzg2p` G2P converter.
(Details TBD — preprint in `paper/`.)
