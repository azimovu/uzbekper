"""Fetch the frozen 300-clip sample to VPS disk (audios/), reusing the
per-source HF parquet logic from fetch_test_audio.py but scoped to
data/sample_300.jsonl (not the full 9,320 manifest).

For each sample row we stream the source's parquet and match by clip stem
(ID) or normalized text, then write to audios/<audio_filepath> so the sample
manifest resolves unchanged. Resumable: skips files already present >1KB.

Usage:
  python scripts/fetch_sample_300.py            # fetch all 5 sources
  python scripts/fetch_sample_300.py --dry-run # count only, no download
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys_path = ROOT
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from fetch_test_audio import (  # reuse proven machinery
    norm, iter_parquet_batches, find_audio_col, save_wav, REPOS,
)

SAMPLE = os.path.join(ROOT, "data", "sample_300.jsonl")
OUT_ROOT = os.path.join(ROOT, "audios")


def stem_of(relpath):
    return os.path.basename(relpath).rsplit(".", 1)[0]


def load_sample():
    rows = []
    with open(SAMPLE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(dry_run: bool):
    rows = load_sample()
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)

    total = len(rows)
    done = 0
    for src, srows in by_src.items():
        if src not in REPOS:
            print(f"[{src}] no repo mapping, skip ({len(srows)} clips)")
            continue
        repo, splits = REPOS[src]
        needed = set(stem_of(r["audio_filepath"]) for r in srows)
        # also index by normalized text for text-joined sources
        by_text = {norm(r["text"]): r for r in srows}
        fetched_stems = set()
        print(f"[{src}] need {len(needed)} clips from {repo}", flush=True)
        if dry_run:
            done += len(needed)
            continue
        for split in splits:
            if needed <= fetched_stems:
                break
            try:
                ds = iter_parquet_batches(repo, split)
            except Exception as e:
                print(f"  split {split} skip: {e}", flush=True)
                continue
            for batch in ds:
                cols = batch.column_names
                acol = find_audio_col(cols)
                texts = (batch.column("sentence").to_pylist()
                         if "sentence" in cols else
                         batch.column("text").to_pylist()
                         if "text" in cols else [None] * batch.num_rows)
                ids = (batch.column("id").to_pylist()
                       if "id" in cols else
                       batch.column("path").to_pylist()
                       if "path" in cols else [""] * batch.num_rows)
                audios = batch.column(acol).to_pylist()
                for rid, t, aud in zip(ids, texts, audios):
                    if needed <= fetched_stems:
                        break
                    base = str(rid).split("/")[-1].rsplit(".", 1)[0]
                    alt = base[4:] if base.startswith("usc_") else base
                    match_row = None
                    for cand in (base, alt):
                        for r in by_src[src]:
                            if stem_of(r["audio_filepath"]) == cand:
                                match_row = r
                                break
                        if match_row:
                            break
                    if match_row is None and t is not None:
                        match_row = by_text.get(norm(t))
                    if match_row is None:
                        continue
                    stem = stem_of(match_row["audio_filepath"])
                    if stem in fetched_stems:
                        continue
                    out_path = os.path.join(OUT_ROOT, match_row["audio_filepath"])
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                        fetched_stems.add(stem)
                        done += 1
                        continue
                    if save_wav(aud, out_path):
                        fetched_stems.add(stem)
                        done += 1
                if needed <= fetched_stems:
                    break
        print(f"  [{src}] fetched {len(fetched_stems & needed)}/{len(needed)}",
              flush=True)

    print(f"\n=== {'DRY-RUN ' if dry_run else ''}SUMMARY ===")
    print(f"sample size: {total}")
    if not dry_run:
        # verify on disk
        ok = 0
        for r in rows:
            p = os.path.join(OUT_ROOT, r["audio_filepath"])
            if os.path.exists(p) and os.path.getsize(p) > 1024:
                ok += 1
        print(f"on disk OK: {ok}/{total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.dry_run)


if __name__ == "__main__":
    main()
