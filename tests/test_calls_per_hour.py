"""tools/analysis/calls_per_hour.py: hour splitting and rates per recorded hour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.analysis.calls_per_hour import aggregate, rates


def test_a_recording_across_an_hour_boundary(tmp_path: Path) -> None:
    (tmp_path / "files").mkdir()
    (tmp_path / "calls").mkdir()
    (tmp_path / "files" / "r.json").write_text(json.dumps(
        {"recording_id": "r", "start_time": "2021-05-01T10:30:00", "duration_s": 5400.0}))
    (tmp_path / "calls" / "r.csv").write_text(
        "start_s,start_time\n1,2021-05-01T10:40:00.000\n2,2021-05-01T11:10:00.000\n3,2021-05-01T11:20:00.000\n")
    rows = aggregate(tmp_path)
    assert [(r["hour"], r["n_calls"], r["recorded_s"]) for r in rows] == [(10, 1, 1800.0), (11, 2, 3600.0)]
    _, m = rates(rows)
    assert m[0, 10] == 2.0 and m[0, 11] == 2.0 and np.isnan(m[0, 12])
