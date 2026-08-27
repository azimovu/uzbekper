"""Tests for uzbekper_pipeline.quality — smoke probes and output sanity gates.

The old gate (5 arbitrary clips, Turkish-chars regex) let contamination slip
through: Uzbek shares most Turkish letters' ABSENCE patterns, and 36% of
GitNazarov's output was junk despite a "verify_pin" pass. The new gates are
deterministic, stratified across all six sources, multi-signal, and every
signal is individually unit-tested here.
"""
import json

import pytest

from uzbekper_pipeline.quality import (
    QualityReport,
    evaluate_probe_outputs,
    select_probe_clips,
    turkish_char_hits,
)

SOURCES = ["common_voice", "uzbekvoice", "usc", "it_youtube",
           "news_youtube", "podcasts_dialect"]


def _manifest_rows():
    rows = []
    for i, src in enumerate(SOURCES):
        for j in range(10):
            rows.append({
                "clip_id": f"audios/{i}_{j}.wav",
                "text": f"clip {i} {j}",
                "source": src,
                "duration": 3.0 + j * 0.1,
            })
    return rows


def test_probe_selection_is_deterministic_and_stratified():
    rows = _manifest_rows()
    p1 = select_probe_clips(rows, per_source=3)
    p2 = select_probe_clips(rows, per_source=3)
    assert [r["clip_id"] for r in p1] == [r["clip_id"] for r in p2]
    by_src = {}
    for r in p1:
        by_src.setdefault(r["source"], []).append(r)
    assert set(by_src) == set(SOURCES)          # all six sources covered
    assert all(len(v) == 3 for v in by_src.values())


def test_probe_prefers_longer_harder_clips():
    """Dialect/noisy sources are where contamination hides; probe longest."""
    rows = _manifest_rows()
    for k, r in enumerate(rows):
        r["duration"] = 1.0 if k % 2 else 30.0
    probe = select_probe_clips(rows, per_source=2)
    for r in probe:
        assert r["duration"] == 30.0


def test_turkish_char_hits_exact_matches_only():
    assert turkish_char_hits("Büyük sağlık tashkilatı") > 0
    assert turkish_char_hits("oʻzbek tili goʻzal") == 0   # okina is Uzbek
    assert turkish_char_hits("Salom dunyo") == 0
    assert turkish_char_hits("it's 100%") == 0            # ascii junk not TR


def test_eval_flags_empty_and_repeat_outputs():
    outs = [
        {"clip_id": "a", "ref": "salom bolalar", "hyp": ""},
        {"clip_id": "b", "ref": "salom bolalar", "hyp": "salom"},
        {"clip_id": "c", "ref": "bir ikki uch", "hyp": "bir"},
    ]
    rep = evaluate_probe_outputs(outs)
    assert rep.flags["empty_output"] == ["a"]
    assert rep.flags["suspiciously_short"] == ["c"]
    # 'b' is fine
    assert rep.passed is False                     # empty output => hard fail


def test_eval_flags_mass_repetition():
    tok = "salom salom salom salom salom salom salom salom"
    outs = [{"clip_id": "x", "ref": "yaxshi kun", "hyp": tok}]
    rep = evaluate_probe_outputs(outs)
    assert rep.flags["repetitive_loop"] == ["x"]
    assert rep.passed is False


def test_eval_flags_error_rows_and_contamination():
    outs = [
        {"clip_id": "e1", "ref": "x", "hyp": "<ERROR:CUDA OOM>"},
        {"clip_id": "tr", "ref": "x", "hyp": "Büyük health merkez"},
        {"clip_id": "ok", "ref": "x", "hyp": "normal oʻzbek gap"},
    ]
    rep = evaluate_probe_outputs(outs)
    assert rep.flags["error_row"] == ["e1"]
    assert rep.flags["turkish_chars"] == ["tr"]
    assert rep.passed is False


def test_clean_outputs_pass_with_stats():
    outs = [
        {"clip_id": "a", "ref": "salom dunyo", "hyp": "salom dunyo"},
        {"clip_id": "b", "ref": "bir ikki", "hyp": "bir ikki uch"},
        {"clip_id": "c", "ref": "kitob", "hyp": "kitob"},
    ]
    rep = evaluate_probe_outputs(outs)
    assert rep.passed is True
    assert rep.stats["n"] == 3
    assert rep.stats["exact_match_rate"] >= 0.3
    assert rep.report_lines()               # reviewable rendering exists


def test_report_is_json_serializable_with_all_context(tmp_path):
    outs = [{"clip_id": "a", "ref": "r", "hyp": "h",
             "source": "usc"}]
    rep = evaluate_probe_outputs(outs)
    blob = json.dumps({"passed": rep.passed,
                       "flags": rep.flags,
                       "stats": rep.stats})
    assert '"passed"' in blob
