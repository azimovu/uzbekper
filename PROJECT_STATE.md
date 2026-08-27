# PROJECT STATE — UzbekPER (STT benchmark + G2P) — split from uzbek-tts 2026-08-24

This folder is now its OWN project. `~/uzbek-tts` is frozen: TTS work is done
(released on HF), remaining there = make HF repo public + finish/publish paper.

## What this project IS

**UzbekPER**: phoneme error rate benchmark for Uzbek ASR, built on `uzg2p`
(open-source Uzbek G2P). END GOAL: benchmark results + paper preprint
(TechRxiv/Zenodo → SIGUL/LREC workshop).

## Layout

| Thing | Path |
|---|---|
| uzg2p package | ~/uzbekper/uzg2p/ (venv at .venv; 12/12 tests green) |
| g2p env (epitran patched) | ~/uzbekper/g2p_env/ |
| Audit + validation bot | ~/uzbekper/g2p_audit/ (bot.py, verdicts.db) |
| Test manifest | ~/uzbekper/data/test_manifest.json (10,151 rows) |
| Pod-side inference | ~/uzbekper/scripts/benchmark_infer.py |
| Local WER+PER scoring | ~/uzbekper/scripts/benchmark_score.py |
| Paper | ~/uzbekper/paper/uzbekper_benchmark.md |

## Verified provenance (2026-08-24)

- data/test_manifest.json is BYTE-IDENTICAL to the official test_manifest.json
  of uzinfocom-edu-ai/uzbek-asr-curated-701h on HF. It IS the held-out test
  split (3%, 10,151 utts, ~21h).
- BLOCKER: that HF repo's audio upload is INCOMPLETE — commit says "sharded
  00-99" but only audios/00/ exists (250 of 337,920 wavs). All test-split
  audio paths 404.
- Options decided-in-principle: (1) ask repo owners to finish upload via HF
  discussion, (2) reconstruct test audio from upstream sources by joining on
  source tag + text, (3) switch benchmark set (weakens paper framing - avoid).

## What REMAINS (in order)

1. Native validation finish: ~240 words left in verdicts.db (238 yes / 9 no /
   13 unsure so far). Bot runs from THIS folder now:
   cd ~/uzbekper && ./g2p_env/bin/python g2p_audit/bot.py
2. Audio blocker resolution (see above)
3. GPU scoring run (Vast rental, ~$0.33/hr all-in, inference only)
4. Score locally with benchmark_score.py → populate paper §6
5. GitHub push of uzg2p (blocked on Utkirbek: repo + PAT)

## Rules (carried over from Utkirbek)

1. Ack messages before tool calls
2. ts: native words keep /ts/ always; foreign loans → s only if curated
3. oʻ ≠ o: strict /ɵ/
4. genitive -ning keeps single /ŋ/
5. Don't invent download lineage — verify provenance
6. Pod wrap-up: destroy immediately when uploads finish
7. Budget: wall-clock × all-in rate (RAM billed separately)
