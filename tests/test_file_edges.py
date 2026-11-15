"""tools/analysis/file_edges.py stats: edge-touching calls counted, successors and continuity found."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import tools.analysis.file_edges as fe

FIELDS = ["call_id", "recording_id", "start_time", "duration_s", "start_s", "end_s", "snr_db", "peak_freq_hz",
          "confidence", "quality"]


def test_stats_counts_edge_calls(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "files").mkdir(parents=True)
    for rid, t0 in (("a", "2021-05-03T10:00:00"), ("b", "2021-05-03T10:10:00"), ("c", "2021-05-03T11:00:00")):
        (run / "files" / f"{rid}.json").write_text(json.dumps({"recording_id": rid, "start_time": t0, "duration_s": 600.0,
                                                               "path": f"/x/{rid}.WAV"}))
    (run / "calls_summary.json").write_text(json.dumps({"duplicates_dropped": []}))
    rows = []
    for rid, s, e in (("a", 100, 101), ("a", 599.7, 599.98), ("b", 0.0, 0.3), ("b", 50, 51), ("c", 599.9, 600.0)):
        rows.append({"call_id": f"{rid}_{s}", "recording_id": rid, "start_time": "2021-05-03T10:00:00", "duration_s": e - s,
                     "start_s": s, "end_s": e, "snr_db": 20, "peak_freq_hz": 8000, "confidence": 0.9, "quality": "ok"})
    for name in ("calls.tsv", "calls_train.tsv"):
        with open(run / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    res = fe.stats(run, tmp_path / "out")
    assert res["end_touching"]["n"] == 2 and res["start_touching"]["n"] == 1
    assert res["continuous_successor"] == 1                      # a -> b is continuous, b -> c is not
    assert res["end_touching_with_start_touching_successor"] == 1
