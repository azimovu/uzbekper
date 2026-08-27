# SESSION HANDOFF — UzbekPER inference (2026-08-25, verified)

## TL;DR
Pod (Vast contract 48634331) died mid-run. We have partial transcript JSONLs on the
VPS. Only ONE model produced valid Uzbek (BlueRaccoon whisper-small-uz). Two others
(OpenAI v3, GitNazarov large-uz) are language-contaminated (Turkish/Azerbaijani output).
NeMo/FastConformer never ran (dependency clash). Paper §6 NOT written with results yet.

## What is on the VPS right now (verified via disk inspection)

### Transcripts in `final/`
- `transcripts_GitNazarov_whisper-large-uz.json` — 8,271 records (idx 0–8270)
- `transcripts_openai_whisper-large-v3.json` — 4,220 records (idx 0–4219)
- `transcripts_BlueRaccoon_whisper-small-uz.json` — 4,220 records (idx 0–4219)

### Audio
Gone — was on the pod only. Reproducible from `data/join_*.json` (6 sources) +
upstream HF repos. Join files: common_voice 3645 / uzbekvoice 3030 / usc 1445 /
news_youtube 438 / it_youtube 484 / podcasts 278.

### Manifest
`data/test_manifest.json` — 10,151 rows (the full official test split).
Source distribution: common_voice 3979 / uzbekvoice 3070 / usc 1547 / it_youtube 552
/ news_youtube 610 / podcasts_dialect 393.

## Scoring results (verified — run with benchmark_score.py)

### Single-model scores (each model scores its own records)
| Model | n | WER | PER | Gap | Turkish-char hyps | Verdict |
|---|---|---|---|---|---|---|
| BlueRaccoon whisper-small-uz | 4,220 | 31.09% | 9.78% | 21.31 | 1/4220 | ✅ CLEAN |
| OpenAI whisper-large-v3 | 4,220 | 94.53% | 31.26% | 63.27 | 554/4220 (13%) | ❌ Contaminated |
| GitNazarov whisper-large-uz | 8,271 | 113.51% | 37.24% | 76.27 | 2948/8271 (36%) | ❌ Contaminated |

### Matched comparison (4,099 clips shared by all 3 models, by ref text)
| Model | n | WER | PER | Gap |
|---|---|---|---|---|
| BlueRaccoon whisper-small-uz | 4,155 | 29.71% | 9.10% | 20.61 |
| OpenAI whisper-large-v3 | 4,155 | 94.58% | 30.93% | 63.65 |
| GitNazarov whisper-large-uz | 4,116 | 112.92% | 37.60% | 75.32 |

Per-source (matched set, BlueRaccoon):
- common_voice: WER 15.46 / PER 3.69 (n=1815)
- uzbekvoice: WER 18.30 / PER 3.99 (n=1505)
- usc: WER 50.55 / PER 18.32 (n=713)
- news_youtube: WER 64.50 / PER 23.89 (n=122)
- (it_youtube excluded — no overlap in matched set)

## The three bugs that killed us (root-caused)

### 1. Resume-position bug (CRITICAL)
`benchmark_infer.py` resume logic (old line 204) used:
  `todo = [r for i, r in enumerate(rows) if i not in done_idx]`
where `done_idx` came from `d['idx']` — and `idx` was set to the **list position**
at inference time (line 151: `rec.add(start + j, ...)` → idx = position).

The manifest GREW between runs (4,051 → 8,271 → 10,151) and reordered.
So old idx values point to DIFFERENT clips in the current manifest.

**Verified:** BlueRaccoon idx=0 ref found at manifest position 4983 (not 0).
GitNazarov idx=0 ref found at manifest position 0. The three transcript files
are aligned to different manifest orderings — 0/4220 refs from BlueRaccoon/v3
matched their manifest[idx], only 13/8271 from GitNazarov.

**FIX applied (benchmark_infer.py):** Added `clip_id = audio_filepath` (stable ID)
to each manifest row. Resume now keys by `clip_id`, not position. Also fixed the
merge logic to deduplicate by `clip_id` instead of `idx`. Backward-compatible with
old transcripts (falls back to idx for records lacking clip_id).

### 2. Language-pin not holding (CRITICAL)
Whisper's `forced_decoder_ids = proc.get_decoder_prompt_ids(language='uz')`
(line 120) is supposed to pin language. But v3 (94.61% WER) and GitNazarov
(112.92% WER) output Turkish chars in 13% and 36% of hyps respectively.
The language prompt is silently failing in the transformers 4.46 Whisper path.

**FIX applied (benchmark_infer.py):** Added `verify_pin=True` flag that runs 5
probe clips before full inference. Checks output for Turkish chars
(ü/İ/ı/Ş/Ğ/Ç); aborts if any found. Use `--no-verify-pin` only for debugging.
Also wrote `scripts/verify_language_pin.py` as a stand-alone 5-clip test.

### 3. Dependency clash (NeMo broken)
Installing NeMo upgraded transformers→5.x (removed WhisperForConditionalGeneration)
and torch→2.13. Downgrade to transformers 4.46.3 + torch 2.4.1 fixed Whisper
but broke NeMo (needs newer torch). FastConformer models NEVER RAN.

**FIX for next pod:** Separate venvs — `uv venv` for NeMo (torch 2.5+), base env
for Whisper (transformers 4.46). Do NOT mix in one env.

### Other issues from INFERENCE_LOG.md
- HF token not set initially → rate-limited CV fetch. Fix: set HF_TOKEN before any pull.
- batch=1 → 7% GPU util. Fix: bs=16 (8× speedup). Now default.
- 40GB disk filled by 2 model caches + audio + parquet. Fix: ≥80GB disk or
  delete weights between model runs.
- `cd` scoping bug on multi-command SSH line. Fix: separate commands.

## What to do next session (to get clean full-coverage results)

1. **Rent FRESH pod:** ≥80GB disk, RTX 4090-class (or A6000/A100).
2. **Separate venvs FIRST:**
   - `uv venv ~/.venv/whisper && source ~/.venv/whisper/bin/activate && pip install torch==2.4.1 transformers==4.46.3 librosa`
   - `uv venv ~/.venv/nemo && source ~/.venv/nemo/bin/activate && pip install nemo-asr torch==2.5+`
3. **Set HF_TOKEN before anything else:** `export HF_TOKEN=<your-token>`
4. **Run language-pin verification on v3 and GitNazarov BEFORE full inference:**
   ```
   python scripts/verify_language_pin.py --model v3 --manifest data/test_manifest.json --audio-root ./audios
   python scripts/verify_language_pin.py --model gitnazarov --manifest data/test_manifest.json --audio-root ./audios
   ```
   If these FAIL (Turkish chars in 5 clips), do NOT run full inference.
   Fix the language pin — try `model.config.forced_decoder_ids = proc.get_decoder_prompt_ids(language='uz')`
   and explicitly set it before generate().
5. **Re-fetch audio from join files** using `scripts/fetch_test_audio.py` (token-aware, resumable).
6. **Run each model as ONE full pass** on the complete 10,151 manifest:
   ```
   python scripts/benchmark_infer.py --manifest data/test_manifest.json \
       --audio-root ./audios --models whisper_large_v3 --push-target user@host:/dest
   python scripts/benchmark_infer.py --manifest data/test_manifest.json \
       --audio-root ./audios --models whisper_large_uz --push-target user@host:/dest
   python scripts/benchmark_infer.py --manifest data/test_manifest.json \
       --audio-root ./audios --models whisper_small_uz --push-target user@host:/dest
   ```
   (NeMo models in separate venv — see step 2.)
7. **Pull JSONLs after EVERY model** (incremental chunk push is in benchmark_infer.py).
   Do NOT wait until end — pod can die.
8. **Score with the (fixed) benchmark_score.py** — use `--matched` flag for fair
   cross-model comparison on shared clips:
   ```
   ./uzg2p/.venv/bin/python scripts/benchmark_score.py --matched \
       final/transcripts_*.json --out final/scores_matched.json
   ```
9. **Then** write §6 of paper with real numbers.

## Files of interest
- `scripts/benchmark_infer.py` — FIXED: clip_id-based resume, language-pin verification
- `scripts/benchmark_score.py` — FIXED: --matched flag for fair cross-model scoring
- `scripts/verify_language_pin.py` — stand-alone 5-clip language-pin test
- `scripts/fetch_test_audio.py` — pod audio fetch from join files (token-aware, resumable)
- `data/join_*.json` — matched clip maps per upstream source
- `final/transcripts_*.json` — existing (partial, 2 contaminated) transcripts
- `final/scores_matched.json` — matched-set scores (just generated)
- `uzg2p/` — G2P (patched epitran in uzg2p/.venv)
- `INFERENCE_LOG.md` — full failure chronology
- `paper/uzbekper_benchmark.md` — §6 tables still EMPTY, do not populate yet

## Update 2026-08-27 (continuing from Modal rewrite session)

### Modal CLI authenticated on this VPS
- `~/.modal.toml` has profile `azimov2398`, `modal profile list` shows it active.
- No Modal app deployed yet (`modal app list` empty).
- $30 credit available (per previous session); ~$10 spent on the failed Vast run, ~$20 remaining.
- Modal SDK importable; this session confirmed via `modal --version` (1.5.4).

### Task 7 (`modal_app.py`) — GREEN, 7/7 new tests, 40/40 total
- `modal_app.py` exists at repo root, importable offline (no real `modal` SDK required for tests).
- `tests/test_modal_config.py` locks the contract:
  - two distinct Images (whisper/nemo) — never shared deps
  - three persistent Volumes (audio/hf_cache/artifacts)
  - CPU functions: prepare_audio, warm_model_cache, verify_inventory, export_artifacts
  - GPU functions: run_whisper (L4, 1h), run_nemo (A10G, 1h), concurrency=1, no warm pool
  - `dry_run("blue_raccoon_whisper_small_uz")` returns real receipt without launching compute
  - `run_model(full=True)` refuses without `receipts/<model>.smoke-pass.json`
  - CLI dict has prepare/smoke/run-model/export/status
- `dry_run` resolves `clip_count=10151` from `data/test_manifest.json` (JSONL-formatted; a same-shape `ready_manifest.json` will be preferred when CPU prep produces it).

### What's NEXT (resume from here)
1. Wire real Modal images + `@app.function` decorators (replace stubs). Currently stubs; tests don't need real `modal` import.
2. Implement CPU prep function bodies (Task 3 of the rewrite plan: `prepare_audio`, `verify_inventory`).
3. Implement smoke runner: 20-clip probe, language-pin check, write `receipts/<model>.smoke-pass.json` on pass.
4. Then `run-model --full` unlocks.

### Open gotchas (carried forward)
- Whisper language pin still unreliable (see references/whisper-language-pin.md). smoke gate must check Turkish chars before full runs.
- GitNazarov large-uz is still listed but excluded per Task 10 of the rewrite plan.

## Update 2026-08-27 (b) — Task 3 audio validation core, GREEN 51/51

### New module: `uzbekper_pipeline/audio_inventory.py` (stdlib-only)
- `probe_audio(path)` — structural WAV validation (RIFF header walk, fmt/data
  chunk bounds checks, truncation guards, PCM-only), returns sample_rate /
  channels / duration_seconds / bits / sha256. Raises `AudioProbeError` on
  garbage/empty/truncated; `FileNotFoundError` on missing.
- `build_inventory(rows, audio_root)` — per-record classify
  present/missing/corrupt (+error_kind), sorted by clip_id, never throws on
  individual bad files.
- `summarize_inventory(inv)` — present/missing/corrupt counts +
  total_present_seconds.
- Replaces the old `size > 1KB` acceptance that let HTML error pages pose as
  audio in the 2026-08-25 run.

### Tests: `tests/test_audio_inventory.py` — 11 tests, all green
Catches real bugs: truncated data chunk, corrupt mid-file garbage, content-hash
determinism.

### Test totals: 51 passing (33 fetch_targets + 7 modal_config + 11 audio_inventory).

### NEXT (resume from here)
1. Wire `audio_inventory` into a CPU Modal function (prepare_audio body) that
   writes `inventory.jsonl` to the artifacts Volume (plan Task 3 items 4–5).
2. Real Modal images replace stubs (Task 4) — whisper + nemo pinned deps.
3. Smoke runner (20-clip probe, Turkish-char gate → smoke-pass receipt).

## Update 2026-08-27 (c) — prepare-step core GREEN, 57/57

### New module: `uzbekper_pipeline/prepare.py`
- `run_prepare(rows, audio_root, out_dir, dataset_jsonl=None, force=False)`:
  probes every ready-manifest row via audio_inventory, writes
  content-addressed `inventory_<dataset-sha8>.jsonl` (records + summary line),
  atomic tmp→rename publish. REFUSES to overwrite same-digest inventory
  unless force=True (append-only spirit). Returns receipt dict.
- Digest: sha256 over sorted clip_ids of the row set (or explicit dataset
  file's sha256 when dataset_jsonl= is passed).
- This is the body the Modal CPU prepare_audio function will call.

### Tests: tests/test_prepare.py — 6 green (content-addressing, refusal,
force-regenerate, no .tmp leftovers, per-dataset files, empty-rows rejection).
Test-harness lesson: importlib was missing from the test file itself — RED
showed NameError not ModuleNotFoundError; fixed the test first.

### Totals: 57 passing (33 fetch_targets + 7 modal_config + 11 audio_inventory + 6 prepare).

### NEXT (resume here)
1. Real Modal wiring (Task 4): replace modal_app.py stubs with actual
   Image.debian_slim pip-pins + @app.function CPU prepare_audio that calls
   uzbekper_pipeline.prepare.run_prepare against the artifacts Volume.
2. Smoke runner: 20-clip probe + Turkish-char gate -> receipts/<model>.smoke-pass.json
3. Then run-model --full unlocks.

## Update 2026-08-27 (d) — Task 4 real Modal wiring GREEN, 62/62

### modal_app.py now builds REAL Modal objects when SDK present
- Project .venv has modal 1.5.4 → registries hold real modal.Image / Volume
  instances (offline construction; network only on deploy/run).
  Without SDK, stub fallback keeps tests green anywhere.
- WHISPER_PINS (torch 2.4.1 + transformers 4.46.3 — known-good pairing) and
  NEMO_PINS (torch 2.5.1 + nemo_toolkit[asr] 2.1.0) recorded as dicts;
  never shared envs (handoff mandate).
- FUNCTIONS dict: plain-callable bodies usable locally AND wrappable by
  @app.function later. prepare_audio_body(ready_manifest, audio_root,
  artifacts_dir) end-to-end produces the content-addressed inventory.

### Root-cause fix caught by TDD (not papered over)
prepare_audio_body first used load_manifest() which requires
audio_filepath — but FROZEN ready manifests carry clip_id/text/source only.
Would have crashed on real data. Added manifest.load_ready_manifest()
(validates clip_id presence+uniqueness); reverted my test-fixture edit that
masked it.

### NEW RULE for future slices
Frozen ready manifests = clip_id-keyed. Raw manifests = audio_filepath-keyed.
Always load with the matching loader.

### Totals: 62 passing (33 fetch_targets + 7 config + 11 inventory + 6 prepare + 5 wiring)

### NEXT (resume here)
1. Add GPU runner bodies to FUNCTIONS (whisper/nemo, batched inference,
   ArtifactStore chunks) + @app.function wrappers; smoke probe w/ Turkish-char gate.
2. First REAL modal run: `modal run` CPU prepare over a small fixture to verify
   image build + volume writes (billable seconds ~0; no GPU).
3. Then the 9,320-clip reconstruction + smoke gates before any full pass.

## Update 2026-08-27 (e) — Gemini 3.5 Transcribe: 15-clip probe + WER

### Viability: CONFIRMED, with measured WER
- Ran 15 real Common Voice 17 `uz/dev` clips through `gemini-3.5-transcribe`
  via Modal US egress (bypasses free-tier geo gate). uz-UZ hint, default mode.
- **Aggregate WER = 30.2%** (39/129 ref words; Sub=20 Del=19 Ins=0).
- Language correct (no Turkish contamination). Reads fluently but not exact
  vs CV reference → solid ZERO-SHOT API baseline, NOT SOTA vs fine-tuned
  Whisper/NeMo. Role: external API reference, not best-in-class.
- Eyeball "15/15 nailed" was wrong: I checked grammar, not exact match.
  WER is the honest metric — the user was right to ask for it.

### Artifacts (this session)
- `scripts/gemini_transcribe_probe.py` — Modal batch probe (n clips, throttle,
  cleanup). ::main --n 15 / ::cleanup entrypoints.
- `scripts/gemini_wer.py` — WER from /tmp/cv_refs.json (CV validated.tsv refs).
- `logs/gemini_probe_2026-08-27.md` — full setup + WER breakdown + repro.
- Modal Secret `google-gemini-staging` holds GEMINI_API_KEY (unquoted).

### Notes
- Free tier = 10 req/min/model → full 21h corpus slow unless upgraded.
- Cost full corpus ≈ $5–7 API + negligible Modal CPU.

### PER computed (the actual uzbekper metric) — 2026-08-27
- Scorer `scripts/benchmark_score.py` already does PER via uzg2p G2P.
  `from uzg2p import G2P` works only with uzg2p/.venv + sys.path insert
  (run the scorer, not raw import).
- Built `final/transcripts_gemini.json` (ref/hyp/source per clip) via
  `scripts/build_gemini_transcripts.py`.
- **Gemini 3.5 Transcribe on 15 CV clips: WER 34.55% / PER 4.40%**
  (WER-PER gap = 30.1 pts). The gap is the CORE FINDING: errors are
  orthographic, not phonetic — PER is the metric that matters for Uzbek.
- This corrects my earlier "15/15 nailed" (grammar check) and standalone
  WER 30.2% (looser tokenizer). Project scorer is the source of truth.
- Artifacts: final/transcripts_gemini.json, scripts/build_gemini_transcripts.py,
  logs/gemini_probe_2026-08-27.md (updated with PER).

### Fixed 300-clip benchmark sample — 2026-08-27
- `data/sample_300.jsonl` = frozen, stratified 300-clip sample (seed 20260827).
  Built by `scripts/sample_300.py` from data/test_manifest.json.
  Strata: common_voice 100, uzbekvoice 70, usc 50, it_youtube 40, news_youtube 40
  (podcasts ABSENT from manifest -> 0 rows, its 20 redistributed to common_voice).
  Each row: audio_filepath, duration, text (reference), source. ~50 min audio total.
- `scripts/fetch_sample_300.py` = fetch the 300 to ./audios (reuses
  fetch_test_audio.py parquet machinery, scoped to sample). --dry-run verified
  (300 mapped, no errors). Real fetch running (background proc_c51b573c4860).
- Disk health: 21G free on /dev/sda1 (72% used) — ample for ~50MB audio.
- Storage plan: sample manifest -> GitHub public (citable); audio -> VPS
  (+ optional HF). No GitHub repo scaffolded yet — pending user decision.
- Next: verify 300/300 on disk, then API + local-model eval through uzg2p.
