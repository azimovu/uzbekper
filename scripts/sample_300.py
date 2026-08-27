"""Build the frozen 300-clip benchmark sample for uzbekper.

Reads the ready manifest (data/test_manifest.json, JSONL with
audio_filepath/duration/text/source) and emits a STRATIFIED, FIXED-SEED
300-clip subset to data/sample_300.jsonl. Deterministic: same seed ->
same 300 clips every run, so the sample is reproducible and citable.

Strata (by source, proportional to availability, min 20 each):
  common_voice 80 | uzbekvoice 70 | usc 50 | news_youtube 40 |
  it_youtube 40 | podcasts 20   = 300

Usage: python scripts/sample_300.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20260827
TARGET = 300
SRC_ALLOC = {
    "common_voice": 80,
    "uzbekvoice": 70,
    "usc": 50,
    "news_youtube": 40,
    "it_youtube": 40,
    "podcasts": 20,
}

HERE = Path(__file__).resolve().parent
MANIFEST = Path(HERE.parent) / "data" / "test_manifest.json"
OUT = Path(HERE.parent) / "data" / "sample_300.jsonl"


def main() -> int:
    by_src = defaultdict(list)
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_src[r["source"]].append(r)

    rnd = random.Random(SEED)
    chosen = []
    shortfall = 0
    for src, n in SRC_ALLOC.items():
        pool = by_src.get(src, [])
        if len(pool) < n:
            print(f"WARN {src}: only {len(pool)} available, requested {n}")
            shortfall += n - len(pool)
            n = len(pool)
        chosen.extend(rnd.sample(pool, n))
    # Redistribute shortfall (e.g. podcasts absent) across available sources.
    if shortfall:
        avail = [s for s in SRC_ALLOC if by_src.get(s)]
        i = 0
        while shortfall > 0:
            s = avail[i % len(avail)]
            pool = by_src[s]
            # only add clips not already chosen
            chosen_ids = {c["audio_filepath"] for c in chosen}
            extra = [c for c in pool if c["audio_filepath"] not in chosen_ids]
            if extra:
                take = min(shortfall, len(extra))
                chosen.extend(rnd.sample(extra, take))
                shortfall -= take
            i += 1
            if i > len(avail) * 50:  # safety
                break

    rnd.shuffle(chosen)  # mix sources in output order; sampling already fixed
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(r["source"] for r in chosen)
    total_dur = sum(r.get("duration", 0) for r in chosen)
    print(f"wrote {len(chosen)} clips -> {OUT}")
    for s, c in dist.most_common():
        print(f"  {s:16} {c}")
    print(f"total audio duration: {total_dur/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
