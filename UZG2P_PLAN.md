# UzbekPER — G2P Benchmark Project Plan

## Goal

Build `uzg2p` (Uzbek grapheme-to-phoneme library) + `UzbekPER` (phoneme error rate
benchmark) so Uzbek ASR/TTS can be evaluated on equal phonetic footing. Nothing like
this exists; every current WER number for Uzbek is inflated by orthographic chaos.

## Status: FOUNDATION ALREADY EXISTS

Key discovery (2026-08-24): **Epitran already supports Uzbek** (`uzb-Latn` + `uzb-Cyrl`).
We do NOT build core G2P from scratch. We audit, patch, extend, wrap.

### Epitran audit results (first pass)

Installed in ~/uzbek-tts/g2p_env. Mapping file:
`epitran/data/map/uzb-Latn.csv`

Working correctly:
- All single letters: a,b,d̪,e,f,ɡ,h,i,j(=y),k,l,m,n,p,q,r,s,t̪,u,v,χ(x),z
- Digraphs: sh→ʃ, ch→t͡ʃ, ng→ŋ, ts→t͡s
- gʻ→ʁ (both U+02BB okina, U+2018, and ASCII apostrophe forms!)
- ʼ/’ → ʔ (glottal stop)
- oo→u, ë→ja (preprocessor rules)

Known issues found:
1. **oʻ→o vs o→ɒ**: distinction preserved but subtle (/o/ vs /ɒ/); ASCII-apostrophe
   input makes them identical (expected — matches our training data reality)
2. No stress assignment at all
3. No Russian loanword handling (aspirated consonants, э/e words)
4. Vowel harmony not modeled (matters for suffix allophony)

Scholarly references for validation:
- Sjoberg, "Uzbek Structural Grammar" (1963) — full PDF freely available
  (theswissbay.ch), has complete phoneme tables + alphabet correspondences
- Journal of the IPA "Uzbek" Illustration paper (Cambridge)
- oxuscom.com/orthography.html — orthography rules with examples

## Roadmap

### Phase 1: Audit (this week, free)
- [x] Install epitran, smoke-test basic mappings
- [ ] Extract vocabulary from our 111h corpus (~50-100k unique words)
- [ ] Run Epitran over full vocab; dump all outputs
- [ ] Hand-check stratified sample (~500 words) against Sjoberg tables
- [ ] Diff Epitran output vs Whisper-large-v3 hypotheses on sample audio
      (misalignments = candidate exceptions or Epitran bugs)
- Deliverable: `audit_report.md` with error taxonomy

### Phase 2: uzg2p package (1 week)
- [ ] Wrap Epitran with pre/post processing:
      - input normalization (all apostrophe variants → canonical)
      - Russian loanword exception dict (auto-harvested from Phase 1 diffs)
      - optional stress marking (final-syllable default + exceptions dict)
- [ ] Config flag: strict-o (keep o/oʻ split) vs merged-o (practical mode)
- [ ] Unit tests against hand-checked word list
- Deliverable: `uzg2p` pip-installable package, MIT

### Phase 3: UzbekPER benchmark (1 week)
- [ ] Define protocol: PER = phoneme Levenshtein / ref length, computed after
      G2P normalization of both reference and hypothesis
- [ ] Fixed test set: uzinfocom test split (10.2k rows, already held out,
      zero overlap with train) + our 10-sentence TTS set
- [ ] Score all public Uzbek ASR models:
      - nvidia/stt_uz_fastconformer_hybrid_large_pc (16.46% WER baseline)
      - uzinfocom-edu-ai/asr-uz-fastconformer-large
      - whisper-small-uz fine-tunes (BlueRaccoon etc.)
      - openai/whisper-large-v3 zero-shot
- [ ] Publish WER + PER side by side; measure the orthographic-inflation gap
- Deliverable: benchmark leaderboard + analysis notebook

### Phase 4 (optional): upstream + paper
- PR okina/stress fixes to Epitran upstream
- Short paper/workshop submission: "UzbekPER: Phoneme-level evaluation
  reveals orthographic inflation in Uzbek ASR benchmarks"

## Hardware needs

Phase 1 diff requires Whisper inference on sample audio: Colab T4 free tier
is sufficient. Everything else is CPU (VPS).

## Data assets already on disk

- 701h pool manifests: ~/uzbek-tts/data/*.json (338k rows, source-tagged)
- Our filtered 111h corpus vocab: extractable from training metadata
- TTS test sentences: fixed 10-sentence set (in repo docs)

## Success criteria

- uzg2p passes 95%+ on hand-checked word list
- UzbekPER correlates with human judgment better than raw WER
  (spot-check: models humans rank better should score lower PER)
- At least 4 external models scored on the public leaderboard
