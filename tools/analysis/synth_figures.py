"""Figures of the synthetic evaluation, from a results folder ``analysis/synth_eval/<set>/``.

The folder holds what ``tools/synth/score.py`` reads (``recordings.tsv``, ``events.tsv``,
``detections/``), its ``scores.json``, and the frequency-shift probe of
``tools/analysis/synth_shift_probe.py`` (``shift_probe.tsv``). All figures use the front-end
view (``short_low_conf`` detections removed: what reaches the training set) and the main
matching of ``score.py`` (IoU >= 0.5; phee, twitter and trill by any overlap).

* ``<set>_snr_curve.png``: recall and precision against SNR, with 95% intervals from a
  bootstrap over recordings (``N_BOOT`` resamples within each level) [ours: recordings, not
  calls, are the independent units, since calls in one minute share a background].
* ``<set>_recall_by_type.png``: recall per call type and SNR, a one-hue heat map, rows
  sorted by recall at the highest SNR.
* ``<set>_frequency_shift.png``: share found at +20 dB, each donor type as it is and shifted
  5 kHz (normal calls up, short high calls down).

Usage:
    python tools/analysis/synth_figures.py analysis/synth_eval/v1 figures [--name synth_v1]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tools.synth.score import score_recording, split_ok

N_BOOT = 2000
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequential blue 100-700
TYPE_NAMES = {"alarm_1": "alarm", "loud_shrill": "loud shrill", "phee_2": "phee (2 notes)", "phee_3": "phee (3 notes)",
              "phee_4": "phee (4 notes)", "tsik_ek": "tsik-ek"}


def per_recording(folder: Path) -> List[Dict]:
    """Front-end, main matching, per recording: snr, calls, calls found, detections, detections right."""
    events: Dict[str, List[Dict]] = defaultdict(list)
    for e in csv.DictReader(open(folder / "events.tsv"), delimiter="\t"):
        events[e["rec_id"]].append(e)
    dets: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for p in sorted((folder / "detections").glob("*.tsv")):
        for d in csv.DictReader(open(p), delimiter="\t"):
            if d["quality"] != "short_low_conf":
                dets[d["rec_id"]].append((float(d["start_s"]), float(d["end_s"])))
    out = []
    for r in csv.DictReader(open(folder / "recordings.tsv"), delimiter="\t"):
        ev = events[r["rec_id"]]
        s = score_recording(dets[r["rec_id"]], [(float(e["onset_s"]), float(e["offset_s"])) for e in ev],
                            [split_ok(e["call_type"]) for e in ev])
        out.append({"snr": float(r["snr"]), "n_calls": len(ev), "found": sum(s["call_main"]),
                    "n_det": len(s["det_main"]), "right": sum(s["det_main"])})
    return out


def bootstrap(rows: List[Dict], seed: int = 0) -> Dict[float, Dict[str, Tuple[float, float, float]]]:
    """Per SNR: recall and precision (pooled over recordings) with 2.5 and 97.5 percentiles."""
    rng = np.random.default_rng(seed)
    out = {}
    for snr in sorted({r["snr"] for r in rows}):
        m = np.array([[r["found"], r["n_calls"], r["right"], r["n_det"]] for r in rows if r["snr"] == snr], float)
        idx = rng.integers(0, len(m), (N_BOOT, len(m)))
        s = m[idx].sum(axis=1)
        rec, prec = s[:, 0] / s[:, 1], s[:, 2] / np.maximum(s[:, 3], 1)
        tot = m.sum(0)
        out[snr] = {"recall": (tot[0] / tot[1], *np.percentile(rec, [2.5, 97.5])),
                    "precision": (tot[2] / max(tot[3], 1), *np.percentile(prec, [2.5, 97.5]))}
    return out


def _db(v: float) -> str:
    """An SNR level in dB to the nearest half: +20, +3.5, -3."""
    return f"{round(v * 2) / 2:+g}"


def _style(ax) -> None:
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)


def snr_curve(folder: Path, path: Path) -> Dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = per_recording(folder)
    b = bootstrap(rows)
    scores = json.loads((folder / "scores.json").read_text())["views"]["front_end"]["by_snr"]
    snrs = sorted(b)
    x = [20 * np.log10(s) for s in snrs]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for key, color, label in (("precision", ORANGE, "precision"), ("recall", BLUE, "recall")):
        y = [b[s][key][0] for s in snrs]
        ax.fill_between(x, [b[s][key][1] for s in snrs], [b[s][key][2] for s in snrs], color=color, alpha=0.18, lw=0)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, mec="white", mew=1.2)
        ax.annotate(label, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
    fa = [scores[k]["false_alarms_per_h"] for k in sorted(scores, key=float)]
    ax.text(0.0, -0.2, "false alarms per hour (no planted call), low to high SNR: " + " / ".join(f"{v:g}" for v in fa),
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.set_xlim(min(x) - 1, max(x) + 4.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, [_db(v) for v in x])
    ax.set_xlabel("SNR (dB, 5-20 kHz, over the call)", color=INK2, fontsize=9.5)
    _style(ax)
    n_rec, n_calls = len(rows), sum(r["n_calls"] for r in rows)
    ax.set_title(f"Learned detector on synthetic recordings: {n_rec:,} one-minute recordings, {n_calls:,} planted calls\n"
                 "front end; IoU ≥ 0.5, multi-syllable calls by overlap; band = 95% bootstrap over recordings",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {str(s): {k: [round(float(v), 4) for v in b[s][k]] for k in b[s]} for s in snrs}


def recall_by_type(folder: Path, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    v = json.loads((folder / "scores.json").read_text())["views"]["front_end"]["by_type"]
    snrs = sorted({k for per in v.values() for k in per}, key=lambda k: -float(k))
    types = sorted(v, key=lambda t: -v[t][snrs[0]]["recall_main"])
    m = np.array([[v[t][s]["recall_main"] for s in snrs] for t in types])
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd"] + RAMP)
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=150)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(types)):
        for j in range(len(snrs)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if m[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(snrs)), [_db(20 * np.log10(float(s))) for s in snrs])
    ax.set_yticks(range(len(types)), [TYPE_NAMES.get(t, t) + (" *" if split_ok(t) else "") for t in types])
    ax.tick_params(length=0, colors=INK2, labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(len(snrs) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(types) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("SNR (dB)", color=INK2, fontsize=9.5)
    n = min(sum(per[s]["n"] for s in snrs) // len(snrs) for per in v.values())
    ax.set_title("Recall by call type and SNR (front end)", loc="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.01, f"* scored by any overlap (split calls count as found); others IoU ≥ 0.5. "
             f"At least {n} calls per cell.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def frequency_shift(folder: Path, path: Path, snr: str = "10.0") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(folder / "shift_probe.tsv"), delimiter="\t"))
    found: Dict[Tuple[str, bool], List[bool]] = defaultdict(list)
    peak: Dict[Tuple[str, bool], List[float]] = defaultdict(list)
    for r in rows:
        if r["snr"] == snr:
            key = (r["type"], r["version"] != "as is")
            found[key].append(float(r["max_p"]) >= 0.5)
            peak[key].append(float(r["peak_hz"]))
    short = ("tsik", "tsik_ek", "seep")
    normal = sorted({t for t, _ in found if t not in short}, key=lambda t: -np.mean(found[(t, False)]))
    order = normal + [None] + sorted(short, key=lambda t: -np.mean(found[(t, False)]))
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=150)
    ys, labels = [], []
    for i, t in enumerate(order):
        if t is None:
            continue
        a, s = np.mean(found[(t, False)]), np.mean(found[(t, True)])
        y = len(order) - 1 - i
        ax.annotate("", xy=(s, y), xytext=(a, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2,
                                                                   shrinkA=5, shrinkB=5, mutation_scale=9))
        ax.plot(a, y, "o", ms=8, color=BLUE, mec="white", mew=1.2, zorder=3)
        ax.plot(s, y, "o", ms=8, color=ORANGE, mec="white", mew=1.2, zorder=3)
        ys.append(y)
        labels.append(f"{TYPE_NAMES.get(t, t)}  ({np.median(peak[(t, False)]) / 1000:.0f} → "
                      f"{np.median(peak[(t, True)]) / 1000:.0f} kHz)")
    ax.set_yticks(ys, labels)
    gap = len(order) - 1 - order.index(None)
    ax.axhline(gap, color=GRID, lw=1)
    ax.text(1.0, gap + 0.35, "usual calls, shifted up 5 kHz", ha="right", fontsize=8.5, color=INK2)
    ax.text(1.0, gap - 0.65, "short high calls, shifted down 5 kHz", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel(f"share found at {20 * np.log10(float(snr)):+.0f} dB (max frame probability ≥ 0.5)", color=INK2, fontsize=9.5)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", color=BLUE, label="as recorded")
    ax.plot([], [], "o", color=ORANGE, label="frequency-shifted")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    ax.set_title("Frequency alone does not explain the missed short high calls", loc="left", fontsize=10.5,
                 color=INK, pad=22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--name", default=None, help="file prefix (default synth_<folder name>)")
    a = ap.parse_args()
    name = a.name or f"synth_{a.folder.name}"
    ci = snr_curve(a.folder, a.out / f"{name}_snr_curve.png")
    (a.folder / "bootstrap.json").write_text(json.dumps(ci, indent=1))
    recall_by_type(a.folder, a.out / f"{name}_recall_by_type.png")
    if (a.folder / "shift_probe.tsv").exists():
        frequency_shift(a.folder, a.out / f"{name}_frequency_shift.png")


if __name__ == "__main__":
    main()
