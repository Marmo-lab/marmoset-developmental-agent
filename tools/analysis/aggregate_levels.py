"""Aggregate per-file level_audit.py JSONs into one corpus-wide table.

Answers the two questions the sweep exists for:

1. Is Talia's absolute AMP_THRESHOLD = 0.02 dropping data across the corpus, or was
   210503_001 an outlier? -> the `zero40` and `maxmag` columns, ordered by date.
2. Does the best single capsule preserve more signal than the 4-capsule mean? -> the
   `gain` column, and the paired count at the bottom.

It also separates the two explanations for a level trend. If the recorder's gain was
raised partway through the campaign, the noise floor and the peak rise together and
`DR` (dynamic range) stays flat. If the colony genuinely got louder, the peak rises
while the stationary floor does not, so `DR` climbs.

Usage:
    python tools/analysis/aggregate_levels.py <dir-of-jsons>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def load(d: str) -> List[Dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "level_*.json"))):
        with open(p) as f:
            r = json.load(f)
        w = r["windows"]
        mm = np.array([x["mean_mix_frac_above"] for x in w])
        bc = np.array([x["best_channel_frac_above"] for x in w])
        fl = np.array([x["floor_db"] for x in w])
        pk = np.array([x["peak_db"] for x in w])
        rows.append({
            "name": os.path.basename(p)[len("level_"):-len(".json")],
            "start": r.get("start_time", ""),
            "maxmag": r["mag_percentiles_mean_mix"]["max"],
            "zero_mean": int((mm == 0).sum()),
            "zero_best": int((bc == 0).sum()),
            "n": len(w),
            "gain": float(bc.sum() / mm.sum()) if mm.sum() > 0 else float("inf"),
            "floor_db": float(np.median(fl)),
            "peak_db": float(np.median(pk)),
            "dr_db": float(np.median(pk - fl)),
        })
    rows.sort(key=lambda r: r["start"] or r["name"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir")
    a = ap.parse_args()
    rows = load(a.dir)
    if not rows:
        raise SystemExit(f"no level_*.json under {a.dir}")

    print(f"{'start':<20}{'file':<26}{'maxmag':>9}{'zero40':>8}{'zeroBest':>9}"
          f"{'gain':>7}{'floor':>8}{'peak':>8}{'DR':>7}")
    for r in rows:
        g = "inf" if r["gain"] == float("inf") else f"{r['gain']:.2f}x"
        print(f"{(r['start'] or '?'):<20}{r['name'][:25]:<26}{r['maxmag']:9.4f}"
              f"{r['zero_mean']:5d}/{r['n']:<2d}{r['zero_best']:6d}/{r['n']:<2d}"
              f"{g:>7}{r['floor_db']:8.1f}{r['peak_db']:8.1f}{r['dr_db']:7.1f}")

    n = len(rows)
    blind = [r for r in rows if r["zero_mean"] >= 0.5 * r["n"]]
    print(f"\n--- Q1: is the absolute threshold dropping data corpus-wide? ---")
    print(f"files where the 4-ch mean is silent in >=50% of windows: {len(blind)}/{n}")
    print(f"max |STFT| below the 0.02 threshold in:                  "
          f"{sum(r['maxmag'] < 0.02 for r in rows)}/{n} files")

    g = np.array([r["gain"] for r in rows if np.isfinite(r["gain"])])
    better = sum(r["zero_best"] < r["zero_mean"] for r in rows)
    print(f"\n--- Q2: does the best single capsule beat the 4-capsule mean? ---")
    print(f"files where best capsule has fewer silent windows: {better}/{n}")
    if g.size:
        print(f"bin-survival gain  median {np.median(g):.2f}x   "
              f"min {g.min():.2f}x   max {g.max():.2f}x")

    print(f"\n--- gain change or behaviour change? ---")
    fl = np.array([r["floor_db"] for r in rows])
    pk = np.array([r["peak_db"] for r in rows])
    dr = np.array([r["dr_db"] for r in rows])
    k = max(1, n // 3)
    print(f"{'':<14}{'floor':>9}{'peak':>9}{'DR':>9}")
    print(f"{'earliest ' + str(k):<14}{fl[:k].mean():9.1f}{pk[:k].mean():9.1f}{dr[:k].mean():9.1f}")
    print(f"{'latest ' + str(k):<14}{fl[-k:].mean():9.1f}{pk[-k:].mean():9.1f}{dr[-k:].mean():9.1f}")
    d_fl, d_pk, d_dr = fl[-k:].mean() - fl[:k].mean(), pk[-k:].mean() - pk[:k].mean(), dr[-k:].mean() - dr[:k].mean()
    print(f"{'change':<14}{d_fl:+9.1f}{d_pk:+9.1f}{d_dr:+9.1f}  dB")
    print("\ninterpretation: floor and peak rising together with DR flat => recorder gain"
          "\nchanged (an artefact we must normalise out). Peak rising with the floor flat"
          "\n=> the colony really did get louder (a property to model, not correct).")


if __name__ == "__main__":
    main()
