# UzbekPER Benchmark Protocol v1.1
Status: ACTIVE — reconstruction complete (9,320/10,151 clips joined, 2026-08-25);
GPU inference not yet run.
Date: 2026-08-25

## 1. Test set

- **Canonical set:** `uzinfocom-edu-ai/uzbek-asr-curated-701h` official test
 split — 10,151 utterances, ~21.0 h upstream. Our local copy
 (`data/test_manifest.json`) verified BYTE-IDENTICAL to the HF file.
 **Evaluated benchmark set = 9,320 reconstructed clips (~19.2 h)** — see below.
  Zero overlap with the train split by construction (upstream's split).
- **Known contamination caveat (recorded, per paper §5):** NVIDIA's model card
  states training on a composite corpus that likely includes the same upstream
  sources; its scores may be optimistically biased. Affects all systems'
  comparability equally; noted in results.
- **OPEN BLOCKER:** the 701h repo's audio upload is incomplete (only shard
  audios/00/ present, 250/337,920 files).
  **DECIDED:** reconstruct the test clips from the six upstream sources
  (all publicly available) by joining source tag + normalized text against their
  manifests/parquets. No dependency on uzinfocom finishing the upload.
  **DONE 2026-08-25: 9,320/10,151 (91.8%)** — uzbekvoice 98.7% (ID),
  usc 93.4% (ID+text), common_voice 92.6% (text, yakhyo mirror),
  it_youtube 87.7% (text), news_youtube 71.8% (text), podcasts 70.7% (text).
  Join outputs: data/join_*.json. Optional fuzzy second pass may recover more.
- Sub-sampling allowed for replication (paper §5.2), must state subset size.

## 2. Systems under evaluation — CLOSED LIST v1

Inclusion criteria: publicly downloadable weights on HF, Uzbek-capable,
zero-shot evaluation only (no fine-tuning on test speakers). Chosen to cover:
official NVIDIA model, dataset-author model, best multilingual zero-shot,
and credible community fine-tunes.

| # | Model ID | Params | Why included |
|---|---|---|---|
| 1 | nvidia/stt_uz_fastconformer_hybrid_large_pc | 115M | Official NVIDIA Uzbek model; claims 16.46% WER |
| 2 | uzinfocom-edu-ai/asr-uz-fastconformer-large | ~115M | Test-set authors' own ASR; same family as #1 |
| 3 | openai/whisper-large-v3 | 1.5B | Multilingual zero-shot ceiling |
| 4 | BlueRaccoon/whisper-small-uz | 244M | Most-downloaded community Whisper FT (74 dl, since 2022) |
| 5 | GitNazarov/whisper-large-uz | 1.5B | Community large FT (low downloads — verify quality before final inclusion) |

Candidates considered but EXCLUDED for now (revisit if community requests):
zafarrr/uzbek-stt-fastconformer-v1.2/v1.5 (<10 dl each, unverified),
Asrorxon/whisper-small-uz (0 dl), qwen3-asr-uzbek variants (new, unproven,
different API class), colorlessideas/xlsr-300m-uz-asr (wav2vec class needs
LM for fair comparison).

Rule: any leaderboard addition must state model ID, commit hash, decode params.

## 3. Inference rules (identical for every system)

- Audio fed as-is from test set (16 kHz mono wav); NO noise reduction,
  no VAD trimming, no enhancement of any kind.
- Language forced to Uzbek where the API allows (Whisper: language='uz').
- Decoding: greedy or default beam (whichever is the model default);
  if beam > 1 used, record it. No external LM fusion.
- Numbers/punctuation: kept as model outputs them. Normalization applied
  IDENTICALLY to ref and hyp before scoring (see §4), so this cancels out.
- One run per system; temperature/fallback quirks of Whisper recorded if hit.
- Hardware: single consumer GPU (RTX 4090-class) on Vast rental; batch size
  recorded; wall-clock and s/utt logged per system.
- **Artifact policy:** every system produces one transcripts JSONL
  (`transcripts_<model>.json`) — the canonical artifact for all scoring,
  leaderboard entries, and public release. Each record stores: idx, ref text,
  hyp raw, source tag, model id + HF revision hash, decode config, timestamp.
  JSONLs are pushed incrementally to the VPS during inference (checkpoint
  every 200 utterances → immediate upload in parallel with compute), so pod
  loss never loses more than one checkpoint and the pod can be destroyed as
  soon as inference ends without a final transfer step.
- Every transcript artifact stores: idx, ref text raw, hyp raw, source tag,
  model id + HF revision hash, decode config, date.

## 4. Scoring rules

Both metrics computed after IDENTICAL normalization of ref and hyp:

**Text normalization (applied to both sides):**
1. lowercase
2. apostrophe variants → U+02BB okina
3. strip punctuation
4. collapse whitespace

**WER:** Levenshtein distance over word sequences / total ref words.
Foreign tokens (digits, non-Uzbek latin junk) are NOT removed for WER —
they count as errors against whichever side produced them.

**PER:** word sequences → phoneme sequences via uzg2p G2P (v0.2 settings:
strict oʻ=/ɵ/, ng-split exceptions, keep-ts rule). Phoneme sequences joined
with a word-boundary marker '#'. PER = phoneme Levenshtein / total ref
phonemes. Words whose G2P output is empty (foreign tokens like "ok", numbers)
are excluded from BOTH sides symmetrically — so an ASR error that invents
or drops such tokens still shows up via alignment shift.

**Inflation gap** = WER − PER, reported per system and per source subset.

**Scoring matrix (mandatory):** every system is scored separately on each
evaluation set (primary reconstructed split / FLEURS / diagnostic hour) and
on the primary set as a whole. No cross-set averaging. The primary set is
reported two ways: micro (all clips pooled) and macro (unweighted mean of
the six per-source scores), since micro lets common_voice's 3,645 matched clips
dominate. Cross-set comparisons of the SAME system across tiers expose
contamination effects; within-set comparisons across systems expose quality.

**Statistical rigor:** report bootstrap 95% CI (1000 resamples) on WER and
PER for the main table. Resampling unit = utterance (sample utterances with
replacement, recompute metric each replicate); paired system comparisons
resample the same utterance indices for both systems.

## 5. Success criteria & claims discipline

- The paper's claim is NOT "NVIDIA is/isn't best". It IS: "word-level metrics
  systematically inflate apparent error vs phoneme-level reality; here is the
  measured gap across N systems." Rankings are secondary observations.
- If our WER for NVIDIA reproduces ≈16.46% on this test set → strong external
  validity point (independent reproduction).
- Benchmark declared "successful" when: ≥3 systems produce clean transcripts
  (no mass <ERROR> rows), scoring pipeline passes a hand-audited sample of
  ≥50 alignments, and CIs are tight enough to separate systems.

## 6. Public release plan (if benchmark succeeds)

- Code: github.com/Uazimov/uzg2p (+ benchmark harness folder)
- Leaderboard: static site (GitHub Pages) with per-system table, per-source
  breakdown, submission instructions = "open a PR with your transcripts JSONL
  produced by the pinned harness version".
- Data: we do NOT redistribute test audio (license chain belongs to upstream);
  we publish the manifest + reconstruction recipe + hashes so anyone can
  materialize it identically.

## Open decisions for Utkirbek

1. Confirm the 5-model closed list (drop GitNazarov? add zafarrr v1.5?)
2. Audio blocker path: discussion post vs reconstruction vs both
3. OK with GitHub-Pages leaderboard as the public face?
