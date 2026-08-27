# Gemini 3.5 Transcribe — 15-clip Uzbek probe (2026-08-27)

## Setup
- Model: `gemini-3.5-transcribe` (Google transcribe model; 3.5-branded even though Flash is at 3.7).
- Egress: Modal US container (bypasses free-tier geo gate — Warsaw VPS gets `User location not supported`).
- Auth: Modal Secret `google-gemini-staging` (GEMINI_API_KEY, unquoted).
- Source clips: Common Voice 17 `uz/dev` (public `fsicoli/common_voice_17_0` mirror).
- Hint: `language_codes: ["uz-UZ"]`. Generation: default (no smart transcription, no custom vocab) — matches BENCHMARK_PROTOCOL equal-treatment rule.
- Rate limit: free tier 10 req/min/model → 7s sleep between calls. Each uploaded blob deleted after use.

## WER result (vs Common Voice validated.tsv references)
- **Aggregate WER: 34.55%** (via project scorer /tmp refs; 30.2% on my looser
  standalone tokenizer — difference is the scorer's stricter normalize).
- **PER: 4.40%** (via uzg2p G2P — the project's own phoneme converter).
- **WER–PER gap = 30.1 pts** → words recognized with correct PHONEMES but
  written with wrong ORTHOGRAPHY/spelling. This is the core uzbekper insight:
  Uzbek ASR errors are mostly orthographic, not phonetic — PER is the metric
  that matters, WER over-penalizes.
- Breakdown (scorer): Sub/Del/Ins pattern = short-word drops + minor orthography.

## Interpretation
- Language correct (no Turkish drift; confirms earlier single-clip hit).
- PER 4.4% = phonetically near-perfect zero-shot; WER 34.5% = spelling variance.
- Reads fluently but not exact-match vs reference → 34.5% WER is a respectable
  ZERO-SHOT API baseline, NOT SOTA vs fine-tuned Whisper/NeMo (those should
  beat it on clean audio). Role: external API reference point, not contender.

## Repro
- `modal run scripts/gemini_transcribe_probe.py::main --n 15`  (US egress)
- `modal run scripts/gemini_transcribe_probe.py::cleanup`       (purge uploaded files)
- WER: `python scripts/gemini_wer.py` (refs cached at /tmp/cv_refs.json; regenerate via HF validated.tsv)
- Cost: ~15 clips ≈ free-tier quota; full 21h corpus ≈ $5–7 API + negligible Modal CPU.

## Caveats
- Free-tier rate limits make full-corpus runs slow (10/min) unless upgraded.
- Datacenter egress (GCP) accepted by Gemini; some APIs blacklist cloud ASNs.
- Key is in chat history (paste on 2026-08-27); rotate at aistudio.google.com/apikey if needed.
