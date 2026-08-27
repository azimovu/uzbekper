# UzbekPER — Inference Run Log & Post-Mortem (2026-08-25)

## Environment
- Pod: Vast.ai contract `48634331`, RTX 4090 (24GB), 125GB RAM, 40GB disk, ~900 Mbps down / ~650 Mbps up.
- Cost: $0.26/hr → total rental ≈ 4.5h ≈ **$1.17** (including idle GPU time — see failures).
- Base image: PyTorch 2.4.1 cu121 + transformers preinstalled. We added NeMo, datasets, librosa.

## What was attempted
Reconstruct 9,320 matched test clips from upstream repos (uzbekvoice, usc, common_voice,
news/it/podcasts youtube sets), run 5 ASR models zero-shot, score WER+PER with uzg2p.

## What actually completed
| Artifact | State | Records |
|---|---|---|
| Audio reconstructed on pod | partial | ~8,600 wavs (≈92% of 9,320) |
| `transcripts_GitNazarov_whisper-large-uz.json` | full first pass | 8,271 (idx 0–8270) |
| `transcripts_openai_whisper-large-v3.json` | first pass only | 4,051 → +169 delta = 4,220 (idx 0–4219) |
| `transcripts_BlueRaccoon_whisper-small-uz.json` | first pass only | 4,051 → +169 delta = 4,220 (idx 0–4219) |
| nvidia FastConformer / uzinfocom FastConformer | NOT RUN | dropped (see below) |

## Chronology of failures
1. **Unauthenticated HF** — first fetch ran without HF token → rate-limited, usc crawled
   at 250 clips/2min. Fixed mid-run by installing token in `/root/.cache/huggingface/token`.
   Lesson: always set `HF_TOKEN` before any datasets pull.

2. **Dependency clash** — installing NeMo upgraded transformers to 5.x (removed Whisper class
   import) and torch to 2.13. Downgraded to transformers 4.46.3 + torch 2.4.1. This broke NeMo
   (needs newer torch) — NeMo models were therefore never run. Lesson: separate venvs per stack.

3. **Batch=1 bottleneck** — initial whisper loop did one clip at a time, GPU at 7% util,
   1.2s/utt. Fixed with bs=16 → 0.16s/utt (8× speedup).

4. **Disk exhaustion (40GB)** — two large model caches (24GB v3 + 11GB GitNazarov) + audio
   + streaming parquet caches filled the disk. Inference crashed mid-checkpoint
   (`No space left on device`). Large-v3 cache was deleted and re-downloaded twice.
   Lesson: 40GB is too small for 2 large models + audio; pick ≥80GB or delete weights between runs.

5. **Resume-position bug (CRITICAL)** — the `--skip-done` logic keyed records by *list position*
   in the manifest, but the manifest grew (4,051→8,271) between runs. The delta run therefore
   re-ran clips 0–4219 (overwriting the first pass) instead of the missing 4,220–8,648.
   Result: v3/small-uz lost full coverage, only kept idx 0–4219. GitNazarov was a single full
   pass so it is intact at 8,271. Fix would have been keying by manifest `idx` field, not position.

6. **`cd` scoping bug** — launching two backgrounded jobs on one SSH line: the `cd` only applied
   to the first; the second failed with "can't open file /root/benchmark_infer.py". Relaunched
   separately. Cosmetic, cost one retry.

7. **Pod died** (contract exited) before the corrected full-pass reruns finished. All pod-side
   data (audio + in-progress transcripts) lost. VPS retained: first-pass JSONLs, all chunk files,
   and the 169-clip delta overlap.

## Language-confusion artifact (affects results)
Scoring revealed OpenAI whisper-large-v3 and GitNazarov whisper-large-uz output substantial
non-Uzbek text (Turkish/Azerbaijani forms: `Büyük`, `saglık`, `tashkilatı`):
- BlueRaccoon small-uz: 1/4220 hyps with Turkish-specific chars → clean.
- OpenAI v3: 554/4220 (13%) → partially confused.
- GitNazarov: 2948/8271 (36%) → heavily confused.
Likely cause: `forced_decoder_ids` with `language='uz'` did not pin the language reliably in the
transformers 4.46 Whisper path (whisper's tokenizer may default-detect). These two models' WER/PER
are NOT valid measures of Uzbek ASR quality and must not be reported as such.

## What is trustworthy
- BlueRaccoon whisper-small-uz: full first pass, clean Uzbek, WER 31.09% / PER 9.78% (idx 0–4219).
- GitNazarov raw transcripts exist (8,271) but are language-contaminated; usable only after a
  re-run with correct language pinning (not available now).

## To finish properly (future rental)
1. Rent ≥80GB disk pod. 2. Separate venvs: `uv venv` for NeMo (torch 2.5+), base for whisper
   (transformers 4.46). 3. Set HF_TOKEN first. 4. Re-fetch audio from join files (reproducible).
5. Run each model as ONE full pass on the fixed 9,320 manifest; key resume by manifest `idx`.
6. Pin language explicitly: for whisper use `proc.get_decoder_prompt_ids(language='uz')` AND verify
   output has no Turkish chars before trusting; consider `model.config.forced_decoder_ids`.
7. Pull JSONLs after every model, not just at end.
