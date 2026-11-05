"""Calls per hour of the day: every recording date as a thin line, and their mean.

Counts [ours]. Every call of a corpus run (``calls/<id>.csv``, all quality flags), by the
date and hour of its start time. Rates are per recorded hour: each recording's span
(``files/<id>.json``: start time and duration) is split at the hour boundaries, so an hour
recorded only in part is scaled up, and one recorded for less than ``MIN_RECORDED_S`` is
left out (too little audio for a rate). The mean is over the dates that recorded that hour.

``aggregate`` runs where the run is (the cluster) and writes ``calls_per_hour.tsv``
(date, hour, n_calls, recorded_s); ``plot`` draws it anywhere.

Usage:
    python tools/analysis/calls_per_hour.py aggregate RUN OUT.tsv
    python tools/analysis/calls_per_hour.py plot OUT.tsv FIG.png
    (the v4 figure and its table are kept in figures/; see figures/README.md)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MIN_RECORDED_S = 1800.0

Cell = Tuple[str, int]


def aggregate(run: Path) -> List[Dict]:
    calls: Dict[Cell, int] = defaultdict(int)
    recorded: Dict[Cell, float] = defaultdict(float)
    for p in sorted((run / "files").glob("*.json")):
        f = json.loads(p.read_text())
        t = dt.datetime.fromisoformat(f["start_time"])
        end = t + dt.timedelta(seconds=f["duration_s"])
        while t < end:
            nxt = min(end, t.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1))
            recorded[(t.date().isoformat(), t.hour)] += (nxt - t).total_seconds()
            t = nxt
        path = run / "calls" / f"{f['recording_id']}.csv"
        if path.exists():
            for r in csv.DictReader(open(path)):
                calls[(r["start_time"][:10], int(r["start_time"][11:13]))] += 1
    return [{"date": d, "hour": h, "n_calls": calls.get((d, h), 0), "recorded_s": round(s, 1)}
            for (d, h), s in sorted(recorded.items())]


def rates(rows: List[Dict]) -> Tuple[List[str], np.ndarray]:
    """``(dates, (n_dates, 24) calls per recorded hour)``, NaN where too little was recorded."""
    dates = sorted({r["date"] for r in rows})
    m = np.full((len(dates), 24), np.nan)
    ix = {d: i for i, d in enumerate(dates)}
    for r in rows:
        if float(r["recorded_s"]) >= MIN_RECORDED_S:
            m[ix[r["date"]], int(r["hour"])] = int(r["n_calls"]) * 3600 / float(r["recorded_s"])
    return dates, m


def plot(rows: List[Dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates, m = rates(rows)
    hours = np.arange(24)
    mean = np.nanmean(m, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=130)
    for ax, log in zip(axes, (False, True)):
        for row in m:
            ax.plot(hours, np.where(row > 0, row, np.nan) if log else row, color="#8a8984", lw=0.6, alpha=0.3)
        ax.plot(hours, mean, color="#2a78d6", lw=2.4, marker="o", ms=4, label=f"mean over dates (n = {len(dates)})")
        ax.plot([], [], color="#8a8984", lw=0.8, label="one recording date")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlim(-0.3, 23.3)
        ax.set_xlabel("hour of day (start of the call)")
        ax.set_ylabel("calls per recorded hour" + (" (log scale)" if log else ""))
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_title("Linear", fontsize=10, loc="left")
    axes[1].set_title("Log (hours with zero calls are not drawn)", fontsize=10, loc="left")
    fig.suptitle("Detected calls per hour of the day, v4 corpus", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["aggregate", "plot"])
    ap.add_argument("src", type=Path)
    ap.add_argument("dest", type=Path)
    a = ap.parse_args()
    if a.step == "aggregate":
        rows = aggregate(a.src)
        with open(a.dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "hour", "n_calls", "recorded_s"], delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} date-hours, {sum(r['n_calls'] for r in rows)} calls -> {a.dest}")
    else:
        plot(list(csv.DictReader(open(a.src), delimiter="\t")), a.dest)


if __name__ == "__main__":
    main()
