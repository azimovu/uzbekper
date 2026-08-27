"""Regression tests for the fetch_test_audio.py counting bug (extracted logic).

Bug: `need = sum(len(v) for v in targets.get(src, {}).values())` summed the
NUMBER OF FIELDS per matched record (2-3), not the number of matched records.
Progress printed "6/3645" style nonsense and the summary under-counted.

The fix extracts target counting into uzbekper_pipeline.fetch_targets and is
verified here against hand-built fixtures with known counts.
"""
import json

from uzbekper_pipeline.fetch_targets import count_needed, load_targets


def _write_join(tmp_path, src, matched_ids, unmatched_ids):
    rows = {}
    idx = 0
    for i in matched_ids:
        rows[str(i)] = {"idx": idx, "matched": True, "up_field": str(i)}
        idx += 1
    for i in unmatched_ids:
        rows[str(i)] = {"idx": idx, "matched": False}
        idx += 1
    (tmp_path / f"join_{src}.json").write_text(
        json.dumps({"scanned": 0, "by_id": len(matched_ids),
                    "total": idx, "rows": rows}))


def test_count_needed_counts_clips_not_fields(tmp_path):
    # 4 sources; each with a known number of matched clips.
    # unmatched ids occupy a disjoint range so they never shadow matches
    _write_join(tmp_path, "uzbekvoice", range(3030), range(10**6, 10**6 + 40))
    _write_join(tmp_path, "usc", range(1445), [])
    _write_join(tmp_path, "common_voice", range(3645),
                [10**7, 10**7 + 1])
    _write_join(tmp_path, "podcasts", range(278), [])
    targets = load_targets(str(tmp_path))
    need = count_needed(targets)
    assert need == 3030 + 1445 + 3645 + 278          # 8398
    assert targets["uzbekvoice"]["0"]["idx"] == 0


def test_real_join_files_total_9320():
    targets = load_targets("/home/ubuntu/uzbekper/data")
    total = sum(len(v) for v in targets.values())
    assert total == 9320                              # the frozen benchmark set


def test_unmatched_records_excluded(tmp_path):
    _write_join(tmp_path, "tiny", [1], [2, 3, 4])
    targets = load_targets(str(tmp_path))
    assert len(targets["tiny"]) == 1
    assert count_needed(targets) == 1
