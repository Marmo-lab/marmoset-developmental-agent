"""Listening page for the phee offset disagreement of DETECTOR_EVAL finding 15.

On the lab's phee database the learned detector ends phees a median 0.36 s after the
database's offsets, while on the colony test set it ends calls 41 ms before the hand
labels. Whether the extra tail is the call, the experiment room's reverberation, or the
database's detector stopping early is a question for the ear. This page shows ``--n``
phees, one per experiment (seeded), each matched one-to-one by a single detection on its
caller's channel (as ``score_phee.score_exp`` counts them):

* the caller's channel from 0.4 s before the earlier onset to 0.6 s after the later
  offset, at 48 kHz, peak-normalised (loudness across examples means nothing);
* the disputed tail alone, from the database offset to the detector offset, at the same
  gain as the whole clip plus ``TAIL_GAIN_DB``, so a faint tail is audible and its level
  can be compared with the call's;
* a spectrogram of the clip and a zoom on the end, with the database's edges (cyan) and
  the detector's (red, dashed);
* buttons for a verdict per example (database end right / detector end right / between /
  unclear), kept in the browser and saved with "save" as ``phee_edges_verdicts.json``.

Usage (where the experiment audio is, i.e. the cluster; writes only under OUT):
    python tools/analysis/listen_phee_edges.py $WORK_DIR/phee_score OUT [--n 20] [--seed 0]
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

from marmoset_developmental_agent.preprocessing.filter_5khz import resample_audio  # noqa: E402

SR = 48_000
PRE_S, POST_S = 0.4, 0.6
TAIL_GAIN_DB = 12.0  # [ours] audible without clipping the clip-normalised tail


def one_to_one(x: Dict[str, object]) -> List[Tuple[int, Tuple[float, float], Tuple[float, float, float]]]:
    """``(channel, phee, detection)`` for every phee matched by exactly one detection that overlaps no other phee."""
    out = []
    for ch, dets in x["detections"].items():
        mine = [(a, b) for c, a, b, _ in x["phees"] if str(c) == ch]
        for p in mine:
            hit = [d for d in dets if min(d[1], p[1]) > max(d[0], p[0])]
            if len(hit) == 1 and sum(min(hit[0][1], q[1]) > max(hit[0][0], q[0]) for q in mine) == 1:
                out.append((int(ch), p, tuple(hit[0])))
    return out


def _spec(ax: plt.Axes, y: np.ndarray, t0: float, lo: float, hi: float, edges: Dict[str, Tuple[float, float]]) -> None:
    f, t, s = signal.spectrogram(y, SR, nperseg=1024, noverlap=1024 - 240)
    db = 10 * np.log10(s + 1e-20)
    ax.pcolormesh(t + t0, f / 1000, db, shading="auto", cmap="magma", vmin=db.max() - 70, vmax=db.max())
    for name, (a, b) in edges.items():
        style = dict(color="cyan", lw=1.6) if name == "db" else dict(color="red", lw=1.6, ls="--")
        ax.axvline(a, **style)
        ax.axvline(b, **style)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 24)
    ax.set_ylabel("kHz")


def build(run: Path, out: Path, n: int, seed: int) -> None:
    xs = [json.loads(p.read_text()) for p in sorted((run / "exps").glob("*.json"))]
    rng = random.Random(seed)
    picks = []
    for x in rng.sample(xs, min(n, len(xs))):
        cands = one_to_one(x)
        if cands:
            picks.append((x, *rng.choice(cands)))
    picks.sort(key=lambda p: p[0]["exp"])
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    items = []
    for k, (x, ch, (pa, pb), (da, db_, conf)) in enumerate(picks, 1):
        info = sf.info(x["wav"])
        lo, hi = max(0.0, min(pa, da) - PRE_S), max(pb, db_) + POST_S
        with sf.SoundFile(x["wav"]) as f:
            f.seek(int(lo * info.samplerate))
            raw = f.read(int((hi - lo) * info.samplerate), dtype="float32", always_2d=True)[:, ch - 1]
        y = resample_audio(raw, info.samplerate, SR)
        gain = 0.9 / (np.abs(y).max() + 1e-9)
        sid = f"{k:02d}_{x['exp']}_ch{ch}"
        sf.write(out / "audio" / f"{sid}.wav", y * gain, SR, subtype="PCM_16")
        ta, tb = sorted((pb, db_))
        tail = y[int((ta - lo) * SR):int((tb - lo) * SR)] * gain * 10 ** (TAIL_GAIN_DB / 20)
        sf.write(out / "audio" / f"{sid}_tail.wav", np.clip(tail, -1, 1), SR, subtype="PCM_16")
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [2, 1]})
        edges = {"db": (pa, pb), "det": (da, db_)}
        _spec(axs[0], y, lo, lo, hi, edges)
        _spec(axs[1], y, lo, ta - 0.3, tb + 0.3, edges)
        axs[0].set_title("whole call (cyan = database, red dashed = detector)", fontsize=9)
        axs[1].set_title("the end, zoomed", fontsize=9)
        for a in axs:
            a.set_xlabel("s in recording")
        fig.tight_layout()
        fig.savefig(out / "img" / f"{sid}.png", dpi=90)
        plt.close(fig)
        items.append({"id": sid, "exp": x["exp"], "ch": ch, "db": [pa, pb], "det": [da, db_], "conf": conf,
                      "offset_err_s": db_ - pb, "onset_err_s": da - pa})
        print(f"{sid}: offset {db_ - pb:+.2f} s, onset {da - pa:+.2f} s", flush=True)
    (out / "items.json").write_text(json.dumps(items, indent=1))
    (out / "index.html").write_text(_page(items))


def _page(items: List[Dict[str, object]]) -> str:
    rows = []
    for it in items:
        sid = html.escape(str(it["id"]))
        rows.append(
            f'<div class="ex" data-id="{sid}"><h3>{sid}</h3>'
            f'<p>database {it["db"][0]:.2f}–{it["db"][1]:.2f} s · detector {it["det"][0]:.2f}–{it["det"][1]:.2f} s · '
            f'<b>detector ends {it["offset_err_s"]:+.2f} s</b> after the database (starts {it["onset_err_s"]:+.2f} s)</p>'
            f'<img src="img/{sid}.png"><div class="pl"><span>whole call</span><audio controls preload="none" '
            f'src="audio/{sid}.wav"></audio><span>disputed tail only (+{TAIL_GAIN_DB:g} dB)</span><audio controls '
            f'preload="none" src="audio/{sid}_tail.wav"></audio></div><div class="v">'
            + "".join(f'<button data-v="{v}">{t}</button>' for v, t in
                      (("db", "the database end is right"), ("det", "the detector end is right"),
                       ("between", "between them"), ("unclear", "unclear")))
            + '<input placeholder="note"></div></div>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Phee call ends</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;max-width:1100px;padding-bottom:60px}}
.ex{{border-top:1px solid #ccc;padding:8px 0}}h3{{margin:4px 0;font-size:14px}}p{{margin:2px 0;font-size:13px}}
img{{width:100%;max-width:1000px}}.pl{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12px}}
.v button{{margin:4px 4px 0 0}}.v button.on{{background:#2e7d32;color:#fff}}.v input{{width:260px}}
#bar{{position:fixed;bottom:0;left:0;right:0;background:#222;color:#fff;padding:8px 16px}}</style></head><body>
<h2>Phee call ends: database vs detector</h2>
<p>The detector ends phees a median 0.36 s after the phee database (DETECTOR_EVAL finding 15). For each call: is
the stretch between the two ends still the call, or room echo / silence? Each clip is normalised on its own.</p>
{''.join(rows)}
<div id="bar"><span id="n"></span> <button id="save">save verdicts</button></div>
<script>
const K="phee_edges"; let st={{}}; try{{st=JSON.parse(localStorage.getItem(K)||"{{}}")}}catch(e){{}}
function put(){{try{{localStorage.setItem(K,JSON.stringify(st))}}catch(e){{}}
 document.getElementById("n").textContent=Object.values(st).filter(s=>s.v).length+" / {len(items)} judged"}}
document.querySelectorAll(".ex").forEach(ex=>{{const id=ex.dataset.id, s=st[id]||(st[id]={{}});
 const bs=ex.querySelectorAll("button"), inp=ex.querySelector("input"); inp.value=s.note||"";
 bs.forEach(b=>{{if(b.dataset.v===s.v)b.classList.add("on"); b.onclick=()=>{{s.v=b.dataset.v;
  bs.forEach(x=>x.classList.toggle("on",x===b)); put()}}}}); inp.oninput=()=>{{s.note=inp.value; put()}}}});
document.getElementById("save").onclick=()=>{{const a=document.createElement("a");
 a.href=URL.createObjectURL(new Blob([JSON.stringify(st,null,1)],{{type:"application/json"}}));
 a.download="phee_edges_verdicts.json"; a.click()}}; put();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="score_phee.py output directory (exps/*.json)")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(Path(a.run), Path(a.out), a.n, a.seed)


if __name__ == "__main__":
    main()
