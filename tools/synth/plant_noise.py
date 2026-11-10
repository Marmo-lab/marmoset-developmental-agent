"""Planted noise events: real colony noises added to the synthetic set, to make it more like real recordings.

What it is for. The synthetic held-out set plants calls into quiet night background. A real colony
minute also holds non-call sounds; the user heard and named a sample of them
(``tools/analysis/find_noise_events.py``, the library's ``/review`` page): about 90% cage rattles and
scratches, a few people, doors and beeps. This module adds such noises to the synthetic recordings, at
their real level and in the proportions heard, so the detector is scored on minutes that sound more like
the colony [ours, approved by the user 2026-11-10]. The scoring is the usual one (``tools/synth/score.py``,
``tools/analysis/presence_table.py``): a detection on a noise is simply a false alarm, a call lost under
a noise simply a miss.

Each new recording is an existing recording of the held-out set (``v2test``) plus noises, keeping its
``rec_id``, its calls and their answer.

Steps (on the cluster, tools/cluster/plant_noise.sbatch):

* ``extract REVIEW RUN NOISES``: cut every labelled noise of the held-out half out of the raw recording.
* ``make NOISES SET OUT``: SET's recordings at the chosen SNR levels, with noises, written to OUT.

Usage:
    python tools/synth/plant_noise.py extract REVIEW_DIR RUN NOISES
    python tools/synth/plant_noise.py make NOISES SET OUT [--snr 1.5] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf

from tools.synth.generate import LENGTH_S, SR, place

# The user's answers on the /review page that name a noise. "call" (a call the run missed) and
# "discard" (unclear) are not noises and are never planted. [ours, the user's labels]
NOISE_TYPES = ("cage", "human", "door", "beep", "other_noise")

# Mean number of noises per one-minute recording, drawn as a Poisson count so that some minutes hold
# none and some several, as in a real hour. Estimate [ours, 2026-11-10]: the candidate rule found 462
# loud non-call events per hour over 81 h; weighting the noise share the user heard in each
# frequency x duration cell by that cell's size gives ~5 noises a minute over day and night, nearly all
# of them by day (70 of 72 heard noises), so ~9 a minute by day, roughly (5-11 labels per cell; and a
# lower bound, since events next to calls were not counted). The synthetic calls are at the day rate,
# so the noises should be too; the user chose 7, between the all-day and the daytime estimate.
NOISES_PER_MIN = 7.0

# Each noise is cut exactly to the span the candidate rule found (where it was >= 15 dB above the
# band median), with a 10 ms raised-cosine fade at each end. Why tight: the cut carries the day
# background that surrounded the noise; outside the noise that background would be heard as a "patch"
# of a different room tone on the night background. Inside the span it is at least 15 dB below the
# noise (that is the rule's threshold), so it is masked. Why 10 ms: long enough that the edge is not
# itself a click (a 5 ms fade, as for the calls, still clicks on a broadband noise cut mid-sound),
# short enough not to soften a door's real attack. [ours]
NOISE_FADE_S = 0.010

# Output clipping guard, as in generate.py: if a noise pushes the peak over 0.99 the whole recording
# is scaled down uniformly, which leaves every ratio (call to background, noise to background) as it was.
PEAK = 0.99


def faded(x: np.ndarray, fade_s: float = NOISE_FADE_S) -> np.ndarray:
    """``x`` with raised-cosine ramps of ``fade_s`` at both ends (shorter if ``x`` is very short)."""
    n = min(int(fade_s * SR), len(x) // 2)
    # 0.5 - 0.5 cos goes smoothly from 0 to 1: its slope is zero at both ends, so no spectral splatter.
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    y = x.astype(np.float64).copy()
    y[:n] *= ramp
    y[len(y) - n:] *= ramp[::-1]
    return y


# ---------------------------------------------------------------- extract

def labelled_noises(review: Path) -> List[Dict]:
    """Items of the review the user named as a noise (the last answer per item wins, as on the page)."""
    verdicts: Dict[str, str] = {}
    for line in (review / "review_feedback.jsonl").read_text().splitlines():
        d = json.loads(line)
        verdicts[d["item_id"]] = d.get("verdict") or ""
    items = list(csv.DictReader(open(review / "items.tsv"), delimiter="\t"))
    return [{**r, "type": verdicts[r["item_id"]]} for r in items if verdicts.get(r["item_id"]) in NOISE_TYPES]


def type_shares(noises: Sequence[Dict]) -> Dict[str, float]:
    """The share of each noise type among *all* labelled noises, both halves.

    Why both halves: the user asked that noises be planted in the proportions heard, which are the best
    estimate of the colony's proportions. Only held-out clips are planted (see ``extract``), but their own
    proportions are a smaller, noisier sample of the same thing, so the shares come from everything heard.
    """
    n: Dict[str, int] = defaultdict(int)
    for r in noises:
        n[r["type"]] += 1
    total = sum(n.values())
    return {t: n[t] / total for t in sorted(n)}


def extract(review: Path, run: Path, out: Path) -> None:
    """Write ``out/audio/<item_id>.flac`` (mono 48 kHz, the 4-capsule mean) and ``out/noises.tsv``.

    Only noises from recordings of the held-out (``test``) half are cut. Why: the train half's noises are
    kept unheard by any evaluation, so a later detector may train on them without being tested on the very
    sounds it learned. ``noises.tsv`` also stores the type shares over both halves, for ``make``.
    """
    noises = labelled_noises(review)
    shares = type_shares(noises)
    paths = {json.loads(p.read_text())["recording_id"]: json.loads(p.read_text())["path"] for p in (run / "files").glob("*.json")}
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in noises:
        if r["half"] != "test":
            continue
        a, d = float(r["start_s"]), float(r["dur_s"])
        with sf.SoundFile(paths[r["recording_id"]]) as f:
            assert f.samplerate == SR, (r["recording_id"], f.samplerate)
            f.seek(int(round(a * SR)))
            # The mean of the 4 capsules: the same mix the detector hears and the backgrounds are made of,
            # so the noise and the background it lands on are in the same units and the same "microphone".
            x = f.read(int(round(d * SR)), dtype="float32", always_2d=True).mean(axis=1)
        sf.write(out / "audio" / f"{r['item_id']}.flac", x, SR, subtype="PCM_24")
        rows.append({"noise_id": r["item_id"], "type": r["type"], "duration_s": round(len(x) / SR, 3),
                     "recording_id": r["recording_id"], "start_s": a, "share": round(shares[r["type"]], 4)})
    _write(out / "noises.tsv", rows)
    (out / "shares.json").write_text(json.dumps({"all_halves": shares, "n_labelled": len(noises),
                                                 "n_cut_test_half": len(rows)}, indent=1))
    print(f"{len(rows)} held-out noises cut; type shares over all {len(noises)} labelled: {shares}", flush=True)


# ---------------------------------------------------------------- make

def load_noises(folder: Path) -> Tuple[Dict[str, List[Tuple[str, np.ndarray]]], Dict[str, float]]:
    """Noise clips by type, and the type shares to draw them with.

    A type heard by the user but with no clip in the held-out half cannot be planted; it is dropped and
    the other shares renormalised (and printed, so the drop is visible), rather than silently planting
    a train-half clip.
    """
    by_type: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    for r in csv.DictReader(open(folder / "noises.tsv"), delimiter="\t"):
        x, sr = sf.read(folder / "audio" / f"{r['noise_id']}.flac", dtype="float64")
        assert sr == SR
        by_type[r["type"]].append((r["noise_id"], faded(x)))
    shares = json.loads((folder / "shares.json").read_text())["all_halves"]
    kept = {t: s for t, s in shares.items() if by_type.get(t)}
    if set(kept) != set(shares):
        print(f"no held-out clip for {sorted(set(shares) - set(kept))}: dropped, shares renormalised", flush=True)
    total = sum(kept.values())
    return dict(by_type), {t: s / total for t, s in sorted(kept.items())}


def draw(n: int, by_type: Dict[str, List[Tuple[str, np.ndarray]]], shares: Dict[str, float],
         rng: np.random.Generator) -> List[Tuple[str, str, np.ndarray]]:
    """``n`` noises: the type by its share, then a clip uniformly within the type.

    Two stages, not a uniform draw over all clips, so that the planted proportions are the heard ones
    (``type_shares``) even though the held-out half holds a different mix of clips.
    """
    types = list(shares)
    picks = rng.choice(len(types), size=n, p=[shares[t] for t in types])
    out = []
    for k in picks:
        t = types[k]
        nid, x = by_type[t][rng.integers(len(by_type[t]))]
        out.append((t, nid, x))
    return out


def plant(n_noise: int, by_type: Dict, shares: Dict[str, float], y0: np.ndarray, scale0: float,
          rng: np.random.Generator) -> Tuple[np.ndarray, List[Dict]]:
    """An existing recording + ``n_noise`` noises at random places in the minute.

    Where: uniform over the minute, without regard to the calls. A noise lands on a call only as often
    as it would by chance in a real minute; no noise is aimed at or kept away from a call. The one
    constraint is between noises: ``generate.place`` (the rule the calls are placed with) keeps GAP_S
    (0.5 s) between two noises and EDGE_S (0.5 s) from the minute's edges, so two noises never fuse
    into one longer, louder sound that was never heard.

    Level: ``y0`` is the recording as written, already multiplied by its clipping scale ``scale0``, so the
    noise is added times ``scale0`` as well. That keeps it at the level it was recorded at relative to the
    background (gain 1, approved as "natural level"): same microphones, same room.
    """
    noises = draw(n_noise, by_type, shares, rng)
    onsets = place([len(x) / SR for *_, x in noises], rng)
    while onsets is None:                  # too much noise for one minute: drop the longest and retry
        noises.remove(max(noises, key=lambda nz: len(nz[2])))
        onsets = place([len(x) / SR for *_, x in noises], rng)
    y = y0.copy()
    events = []
    for (t, nid, x), on in zip(noises, onsets):
        i = int(round(on * SR))
        y[i:i + len(x)] += scale0 * x
        events.append({"noise_id": nid, "type": t, "onset_s": round(i / SR, 4), "offset_s": round((i + len(x)) / SR, 4)})
    return y, events


def make(noises_dir: Path, src: Path, out: Path, levels: Sequence[float], seed: int) -> None:
    """``src``'s recordings at the chosen SNR levels, each with noises added, under ``out`` (same ``rec_id``)."""
    by_type, shares = load_noises(noises_dir)
    recs = [r for r in csv.DictReader(open(src / "recordings.tsv"), delimiter="\t") if float(r["snr"]) in levels]
    src_events: Dict[str, List[Dict]] = defaultdict(list)
    for e in csv.DictReader(open(src / "events.tsv"), delimiter="\t"):
        src_events[e["rec_id"]].append(e)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    new_recs, events, noises = [], [], []
    for j, r in enumerate(recs):
        rid = r["rec_id"]
        # One generator per recording, seeded by its position and independent of the source set's
        # generator: the source recordings are untouched, and rerunning gives the same noises.
        rng = np.random.default_rng([seed, j])
        y0, sr = sf.read(src / "audio" / f"{rid}.flac", dtype="float64")
        assert sr == SR
        scale0 = float(r["scale"])
        y, ev = plant(int(rng.poisson(NOISES_PER_MIN * LENGTH_S / 60)), by_type, shares, y0, scale0, rng)
        s = _write_audio(out / "audio" / f"{rid}.flac", y)
        # If the clipping guard scaled the recording down, the calls' gains go down by the same factor: the
        # answer file must describe the audio as written.
        new_recs.append({**r, "scale": round(scale0 * s, 6), "n_noises": len(ev)})
        events += [{**e, "gain": round(float(e["gain"]) * s, 6)} for e in src_events[rid]]
        noises += [{"rec_id": rid, **e, "gain": round(scale0 * s, 6)} for e in ev]
        if (j + 1) % 100 == 0:
            print(f"{j + 1}/{len(recs)} recordings", flush=True)
    _write(out / "recordings.tsv", new_recs)
    # events.tsv keeps the calls only, so tools/synth/score.py scores this set unchanged; the noises go to
    # their own file and never count as calls.
    _write(out / "events.tsv", events)
    _write(out / "noises.tsv", noises)
    (out / "plant.json").write_text(json.dumps({"source": str(src), "levels": list(levels), "seed": seed,
                                                "noises_per_min": NOISES_PER_MIN, "shares": shares,
                                                "fade_s": NOISE_FADE_S}, indent=1))


def _write_audio(path: Path, y: np.ndarray) -> float:
    """Write ``y`` as mono 24-bit FLAC, scaled down uniformly if its peak exceeds ``PEAK``; returns that factor."""
    peak = float(np.abs(y).max())
    s = PEAK / peak if peak > PEAK else 1.0
    sf.write(path, (y * s).astype(np.float32), SR, subtype="PCM_24")
    return s


def _write(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "make"])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--snr", default="1.5")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.step == "extract":
        extract(*a.paths)
    else:
        make(*a.paths, levels=[float(s) for s in a.snr.split(",")], seed=a.seed)


if __name__ == "__main__":
    main()
