"""Smoke-probe selection and output-quality gates.

Replaces the old "5 arbitrary clips + Turkish-char regex" gate, which passed
runs whose outputs were 36% contaminated. Principles:

* Probes are deterministic and stratified across all six sources, biased to
  longer (harder) clips where language confusion shows first.
* No single signal proves correctness. Empty output, repetition loops, error
  rows, length mismatch, Turkish-specific characters, and non-Uzbek character
  rates are tracked separately; any hard-signal failure blocks the full run.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

# Turkish-specific letters that essentially never appear in Uzbek orthography.
_TURKISH_RE = re.compile(r"[çğİıÖöşŞüÇ]")
_ERROR_RE = re.compile(r"<ERROR")
_ALLOWED_CHARS_RE = re.compile(r"[A-Za-zʻʼ''0-9 .,!?%:-]")


def turkish_char_hits(text: str) -> int:
    return len(_TURKISH_RE.findall(text or ""))


def select_probe_clips(rows: list[dict], per_source: int = 4) -> list[dict]:
    """Deterministic stratified probe: longest clips per source, stable order."""
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    probe = []
    for src in sorted(by_src):
        ranked = sorted(
            by_src[src],
            key=lambda r: (-(r.get("duration") or 0.0), r["clip_id"]),
        )
        probe.extend(ranked[:per_source])
    return sorted(probe, key=lambda r: r["clip_id"])


def _repetition_ratio(hyp: str) -> float:
    toks = hyp.split()
    if len(toks) < 8:
        return 0.0
    uniq = len(set(toks))
    return 1.0 - uniq / len(toks)


def _suspiciously_short(ref: str, hyp: str) -> bool:
    """Hyp misses >= half the ref words when ref has >= 3 words."""
    n_ref, n_hyp = len(ref.split()), len(hyp.split())
    return n_ref >= 3 and n_hyp * 2 <= n_ref


class QualityReport:
    def __init__(self, flags: dict[str, list[str]], stats: dict,
                 samples: dict[str, dict]):
        self.flags = flags
        self.stats = stats
        self.samples = samples

    @property
    def passed(self) -> bool:
        hard_failures = ("empty_output", "repetitive_loop", "error_row",
                         "turkish_chars", "high_non_uzbek_chars",
                         "suspiciously_short")
        return not any(self.flags[k] for k in hard_failures)

    def report_lines(self) -> list[str]:
        lines = [f"probe n={self.stats['n']} passed={self.passed}"]
        for k, v in sorted(self.flags.items()):
            if v:
                lines.append(f"  FLAG {k}: {len(v)} {v[:5]}")
        for k, v in self.stats.items():
            if k != "n":
                lines.append(f"  stat {k}: {v}")
        for cid, s in list(self.samples.items())[:10]:
            lines.append(f"  sample {cid}: ref={s['ref']!r} hyp={s['hyp']!r}")
        return lines

    def to_json(self) -> str:
        return json.dumps(
            {"passed": self.passed, "flags": self.flags,
             "stats": self.stats, "samples": self.samples},
            ensure_ascii=False, indent=2)


def evaluate_probe_outputs(outs: list[dict]) -> QualityReport:
    flags: dict[str, list[str]] = {
        "empty_output": [],
        "repetitive_loop": [],
        "error_row": [],
        "turkish_chars": [],
        "high_non_uzbek_chars": [],
        "suspiciously_short": [],
    }
    exact = 0
    char_ratios: list[float] = []
    samples: dict[str, dict] = {}
    for o in outs:
        cid = str(o["clip_id"])
        ref, hyp = str(o.get("ref") or ""), str(o.get("hyp") or "")
        stripped = hyp.strip()
        if _ERROR_RE.search(stripped):
            flags["error_row"].append(cid)
        if not stripped:
            flags["empty_output"].append(cid)
            continue
        if _repetition_ratio(stripped) > 0.5:
            flags["repetitive_loop"].append(cid)
        if turkish_char_hits(stripped) > 0:
            flags["turkish_chars"].append(cid)
        if _suspiciously_short(ref, stripped):
            flags["suspiciously_short"].append(cid)
        allowed = sum(1 for c in stripped if _ALLOWED_CHARS_RE.match(c))
        ratio = 1.0 - allowed / max(1, len(stripped))
        char_ratios.append(ratio)
        if ratio > 0.30:
            flags["high_non_uzbek_chars"].append(cid)
        if stripped == ref.strip():
            exact += 1
        samples[cid] = {"ref": ref[:80], "hyp": stripped[:80],
                        "source": o.get("source", "")}
    n = len(outs)
    stats = {
        "n": n,
        "exact_match_rate": round(exact / n, 4) if n else 0.0,
        "mean_non_uzbek_char_ratio":
            round(sum(char_ratios) / len(char_ratios), 4) if char_ratios else 0.0,
        "max_non_uzbek_char_ratio":
            round(max(char_ratios), 4) if char_ratios else 0.0,
    }
    return QualityReport(flags, stats, samples)
