"""Browse a merged call corpus by ear: one page per recording date, calls in time order.

For each date, the page shows a stretch of consecutive calls from the busiest hour, the
way they were heard that day: each call's audio (with its stored margin), its
spectrogram with the call edges marked, its wall-clock time, the gap since the previous
call, and the quality flag. Hearing consecutive calls is the point: the corpus is meant
to be read as "sentences" of calls later. Every page carries the verdict buttons of
``tools/labels/listen_feedback.py``.

Usage:
    python tools/corpus/browse_calls.py RUN OUT [--per-date 40]
    (RUN is a run directory with calls.tsv from merge_calls.py and call_audio/)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

from tools.corpus.merge_calls import read_call  # noqa: E402
from tools.labels.listen_feedback import add_feedback  # noqa: E402

STYLE = ("body{font-family:system-ui,sans-serif;margin:16px;max-width:1250px}.ex{display:inline-block;margin:6px;"
         "font-size:11px;vertical-align:top;width:290px}img{width:290px;height:140px}audio{width:290px}"
         ".gap{color:#555}.flag{color:#c62828;font-weight:bold}")


def _stretch(rows: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    """``n`` consecutive calls starting in the date's busiest clock hour."""
    hours = defaultdict(int)
    for r in rows:
        hours[r["start_time"][11:13]] += 1
    busiest = max(hours, key=hours.get)
    first = next(i for i, r in enumerate(rows) if r["start_time"][11:13] == busiest)
    return rows[first:first + n]


def _render(run: Path, r: Dict[str, str], out: Path) -> None:
    x = read_call(run, r, with_margin=True)
    sf.write(out / f"{r['call_id']}.wav", 0.7 * x / (np.abs(x).max() + 1e-9), 48_000, subtype="PCM_16")
    f, t, s = signal.stft(x, fs=48_000, nperseg=1024, noverlap=784)
    p = 20 * np.log10(np.abs(s) + 1e-12)
    p -= np.median(p, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    ax.imshow(p, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=35, extent=[0, len(x) / 48_000, 0, 24])
    lead, d = float(r["audio_lead_s"]), float(r["duration_s"])
    ax.axvline(lead, color="c", lw=1)
    ax.axvline(lead + d, color="c", lw=1)
    ax.set_yticks([0, 8, 16, 24])
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out / f"{r['call_id']}.png", dpi=70)
    plt.close(fig)


def build(run: Path, out: Path, per_date: int) -> int:
    rows = list(csv.DictReader(open(run / "calls.tsv"), delimiter="\t"))
    by_date: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_date[r["start_time"][:10]].append(r)
    (out / "a").mkdir(parents=True, exist_ok=True)
    index = [f"<!doctype html><html><head><meta charset='utf-8'><title>Call corpus</title><style>{STYLE}</style>"
             f"</head><body><h1>Call corpus: {len(rows)} calls, {len(by_date)} dates</h1><p>Each date page plays "
             f"{per_date} consecutive calls from that day's busiest hour, in the order they happened.</p><ul>"]
    for date, rs in sorted(by_date.items()):
        stretch = _stretch(rs, per_date)
        hours = sum(float(r["duration_s"]) for r in rs) / 3600
        index.append(f"<li><a href='{date}.html'>{date}</a>: {len(rs)} calls, {hours:.2f} call-hours</li>")
        page = [f"<!doctype html><html><head><meta charset='utf-8'><title>{date}</title><style>{STYLE}</style>"
                f"</head><body><p><a href='index.html'>all dates</a></p><h1>{date}: {len(stretch)} consecutive calls "
                f"from {stretch[0]['start_time'][11:19]}</h1>"]
        prev = None
        for r in stretch:
            _render(run, r, out / "a")
            t = dt.datetime.fromisoformat(r["start_time"])
            gap = "" if prev is None else f"<span class='gap'>+{(t - prev).total_seconds():.2f} s after previous</span><br>"
            prev = t + dt.timedelta(seconds=float(r["duration_s"]))
            flag = "" if r["quality"] == "ok" else f" · <span class='flag'>{r['quality']}</span>"
            page.append(f"<div class='ex'>{gap}<img src='a/{r['call_id']}.png'><audio controls preload='none' "
                        f"src='a/{r['call_id']}.wav'></audio>{r['start_time'][11:23]} · {float(r['duration_s']):.2f} s"
                        f" · conf {float(r['confidence']):.2f}{flag}</div>")
        page.append("</body></html>")
        (out / f"{date}.html").write_text("\n".join(page))
        add_feedback(out / f"{date}.html")
    index.append("</ul></body></html>")
    (out / "index.html").write_text("\n".join(index))
    return len(by_date)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--per-date", type=int, default=40)
    a = ap.parse_args()
    print(f"{build(Path(a.run), Path(a.out), a.per_date)} date pages -> {a.out}/index.html")


if __name__ == "__main__":
    main()
