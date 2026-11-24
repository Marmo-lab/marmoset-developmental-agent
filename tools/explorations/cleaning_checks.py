"""Measurements behind the cleaning parameters, before any tuning (user, 2026-11-24). Nothing here changes a setting.

C1  Noise, synthetic against real. The synthetic set (``tools/synth``) plants donor calls into colony background; if
    the cleaning is to be tuned on it, its background must look like the background real calls are cleaned against.
    Per call, on the 4-capsule mean (the only signal the synthetic set has), with the cleaning's own front end
    (1 kHz high-pass, STFT 1024 / 240, ``clean_calls.noise_profile`` over +-3 s, ``clean_calls.declick``):
    noise level in six bands, the spread of the per-frame background level (non-stationarity, which a fixed N(f)
    cannot follow), the share of click frames, and the call's SNR (median over its frames of the loudest bin's
    P / N in 2-20 kHz, in dB). The real SNR distribution also says how much each synthetic SNR level weighs.
C2  How fast calls move in frequency. On the loudest beam (production): the dominant ridge (the loudest 2-20 kHz bin
    of each call frame, where it stands >= 12 dB above N(f)) and its step between consecutive 5 ms frames; steps of
    3 kHz or more (below the 4.6 kHz minimum harmonic spacing) are switches to another harmonic, counted apart. Per
    call: percentiles of the step and the share of steps above each candidate ``LINK_HZ``. Per candidate link (6 dB
    peaks, 20 ms): the share of the dominant ridge's energy the mask keeps, and the share of the call's peak energy
    in tracks dropped as shorter than 20 ms. On real v6 calls with SNR >= 20 dB and on the 321 Turesson donors.
C3  What ridges let through on background. Same beam and window, frames outside every call (by the cleaning's
    guard): for PEAK_DB x LINK_HZ x MIN_LEN_S, the share of cells and of energy the mask keeps, and kept tracks per
    second. The cost side of C2.
C4  The 44 calls of the examples page: C2's retention and track counts at the current setting (local, from blocks).

Usage (cluster; ``pc.sbatch`` wrappers in the scratchpad; array tasks split the items):
    python tools/explorations/cleaning_checks.py real RUN OUT [--n 600]
    python tools/explorations/cleaning_checks.py synth SET OUT [--n 50]
    python tools/explorations/cleaning_checks.py donors DONOR_DIR OUT
    python tools/explorations/cleaning_checks.py examples BLOCKS OUT          (local)
    python tools/explorations/cleaning_checks.py report OUT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

from marmoset_developmental_agent.config.agent_config import CleaningConfig, load_config
from marmoset_developmental_agent.preprocessing import clean_calls as cc
from marmoset_developmental_agent.preprocessing.filter_5khz import resample_audio
from tools.explorations.ridge_listen import _beam, ridge_tracks, tracks_to_mask

SR = 48_000
CTX_S = 3.0                                       # = CleaningConfig.context_s
BANDS = [(1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 24000)]
LINKS = [250, 500, 750, 1000, 1500, 2000]
PEAKS = [3.0, 6.0, 9.0]
MIN_LENS = [0.010, 0.020, 0.040]
STRICT_DB, LOOSE_HZ = 12.0, 3000.0
SEED = 2026_11_24
Interval = Tuple[float, float]


def _task(items: List) -> List:
    k, n = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    return items[k::n]


def _front(x: np.ndarray, calls: Sequence[Interval], cfg: CleaningConfig):
    """The cleaning's front end: high-pass, STFT, N(f) from call-free frames, click capping."""
    x = signal.sosfiltfilt(signal.butter(5, cfg.highpass_hz, "highpass", fs=SR, output="sos"), x.astype(np.float64))
    f, t, X = cc._stft(x, SR, cfg)
    noise, _ = cc.noise_profile(np.abs(X) ** 2, t, calls, cfg)
    X, click = cc.declick(X, f, noise, cfg)
    return f, t, np.abs(X) ** 2, noise, click


def _outside(t: np.ndarray, calls: Sequence[Interval], guard: float) -> np.ndarray:
    out = np.ones(len(t), bool)
    for a, b in calls:
        out &= ~((t >= a - guard) & (t <= b + guard))
    return out


def noise_stats(x: np.ndarray, call: Interval, calls: Sequence[Interval], cfg: CleaningConfig) -> Dict:
    """C1 on one window of the capsule mean."""
    f, t, P, N, click = _front(x, calls, cfg)
    out = _outside(t, calls, cfg.guard_s)
    inband = (f >= 2000) & (f <= 20000)
    frame_db = 10 * np.log10(P[inband][:, out].sum(0) + 1e-20)
    inside = (t >= call[0]) & (t <= call[1])
    snr = 10 * np.log10(P[inband][:, inside] / (N[inband, None] + 1e-20) + 1e-12).max(0) if inside.any() else [np.nan]
    return {"band_db": [round(float(10 * np.log10(N[(f >= lo) & (f < hi)].sum() + 1e-20)), 2) for lo, hi in BANDS],
            "bg_db": round(float(10 * np.log10(N[inband].sum() + 1e-20)), 2),
            "bg_iqr_db": round(float(np.subtract(*np.percentile(frame_db, [75, 25]))), 2) if out.any() else None,
            "click_frac": round(float(click[out].mean()), 4) if out.any() else None,
            "snr_db": round(float(np.median(snr)), 2), "dur": round(call[1] - call[0], 3)}


def ridge_stats(x: np.ndarray, call: Interval, calls: Sequence[Interval], cfg: CleaningConfig, leak: bool) -> Dict:
    """C2 (always) and C3 (``leak``) on one window of the beam."""
    f, t, P, N, _ = _front(x, calls, cfg)
    hop = cfg.hop_length / SR
    df = float(f[1] - f[0])
    inside = (t >= call[0]) & (t <= call[1])
    # C2: the dominant ridge, the loudest 2-20 kHz bin of each call frame where it stands >= STRICT_DB above N(f).
    # (Loose-linked tracks were tried first: in a call's dense peak field the greedy link joins unrelated peaks, so
    # their steps measured the linker, not the call.) A step of LOOSE_HZ or more between two such frames is a switch
    # to another harmonic or source, counted apart.
    fi = np.flatnonzero((f >= 2000) & (f <= 20000))
    cols = np.flatnonzero(inside)
    dom = fi[P[fi][:, cols].argmax(0)] if cols.size else np.zeros(0, int)
    strong = 10 * np.log10(P[dom, cols] / (N[dom] + 1e-20) + 1e-12) >= STRICT_DB if cols.size else np.zeros(0, bool)
    pair = strong[:-1] & strong[1:]
    jumps = np.abs(np.diff(dom))[pair] * df
    steps = jumps[jumps < LOOSE_HZ]
    res: Dict = {"n_steps": int(steps.size), "switches": int((jumps >= LOOSE_HZ).sum())}
    if steps.size:
        res.update({"step_p50": float(np.percentile(steps, 50)), "step_p90": float(np.percentile(steps, 90)),
                    "step_p99": float(np.percentile(steps, 99)), "step_max": float(steps.max()),
                    "over": {str(L): float((steps > L).mean()) for L in LINKS}})
    pts = [(int(k), int(j)) for k, j, s in zip(dom, cols, strong) if s]
    e_strict = sum(P[k, j] for k, j in pts)
    # Per candidate link (6 dB peaks, 20 ms): the share of the dominant ridge's energy the mask keeps, and the share of
    # all peak energy inside the call that sits in tracks dropped as too short.
    by_link = {L: ridge_tracks(P, N, f, 6.0, L) for L in LINKS}
    min_frames = int(np.ceil(0.020 / hop))
    if e_strict > 0:
        res["retained"] = {str(L): float(sum(P[k, j] for k, j in pts if m[k, j]) / e_strict)
                           for L, m in ((L, tracks_to_mask(by_link[L], P.shape, hop, 0.020)) for L in LINKS)}
    dropped = {}
    for L, trs in by_link.items():
        e_in = [sum(P[k, j] for k, j in tr if inside[j]) for tr in trs]
        tot = sum(e_in)
        dropped[str(L)] = float(sum(e for e, tr in zip(e_in, trs) if len(tr) < min_frames) / tot) if tot else None
    res["dropped_short"] = dropped
    if leak:
        out = _outside(t, calls, cfg.guard_s)
        sec = out.sum() * hop
        e_out = P[:, out].sum()
        leakage = {}
        for pk in PEAKS:
            for L in LINKS:
                trs = by_link[L] if pk == 6.0 else ridge_tracks(P, N, f, pk, L)
                for ml in MIN_LENS:
                    m = tracks_to_mask(trs, P.shape, hop, ml)
                    n_tr = sum(len(tr) >= int(np.ceil(ml / hop)) and np.mean([out[j] for _, j in tr]) > 0.5 for tr in trs)
                    leakage[f"{pk:g}|{L}|{ml * 1000:g}"] = [round(float(m[:, out].mean()), 5),
                                                            round(float(P[:, out][m[:, out]].sum() / e_out), 5),
                                                            round(n_tr / sec, 3) if sec else None]
        res["leak"] = leakage
        res["bg_s"] = float(sec)
    return res


# ----------------------------------------------------------------------------------------------------------------- #
def real(run: Path, out: Path, n: int, leak_n: int) -> None:
    """A uniform random sample of v6 day calls (``ok``, 07:00-17:59): C1 on the mean, C2 / C3 on the beam."""
    cfg = load_config().cleaning
    rows = [r for r in csv.DictReader(open(run / "calls.tsv"), delimiter="\t")
            if r["quality"] == "ok" and 7 <= dt.datetime.fromisoformat(r["start_time"]).hour < 18]
    rng = random.Random(SEED)
    rows = rng.sample(rows, n)
    for i, r in _task(list(enumerate(rows))):
        dst = out / "real" / f"{i:04d}.json"
        if dst.exists():
            continue
        info = json.loads((run / "files" / f"{r['recording_id']}.json").read_text())
        s, e = float(r["start_s"]), float(r["end_s"])
        with sf.SoundFile(info["path"]) as fh:
            sr = fh.samplerate
            lo = max(0.0, s - CTX_S)
            fh.seek(int(lo * sr))
            caps = fh.read(int((e + CTX_S - lo) * sr), dtype="float32", always_2d=True).T
        hi = lo + caps.shape[1] / sr
        cat = list(csv.DictReader(open(run / "calls" / f"{r['recording_id']}.csv")))
        calls = [(float(c["start_s"]) - lo, float(c["end_s"]) - lo) for c in cat
                 if float(c["end_s"]) > lo and float(c["start_s"]) < hi]
        call = (s - lo, e - lo)
        mean = resample_audio(caps.mean(0), sr, SR)
        beam = resample_audio(_beam(caps, sr, call[0], call[1]), sr, SR)
        rec = {"call_id": r["call_id"], "corpus_snr_db": float(r["snr_db"]), "confidence": float(r["confidence"]),
               **noise_stats(mean, call, calls, cfg)}
        if rec["snr_db"] >= 20 or i < leak_n:
            rec.update(ridge_stats(beam, call, calls, cfg, leak=i < leak_n))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(rec))


def synth(set_dir: Path, out: Path, n: int) -> None:
    """C1 on planted calls: ``n`` per SNR level, the same window and measures as ``real``."""
    cfg = load_config().cleaning
    ev = [r for r in csv.DictReader(open(set_dir / "events.tsv"), delimiter="\t") if r["kind"] == "call"]
    by_rec: Dict[str, List[Interval]] = {}
    for r in ev:
        by_rec.setdefault(r["rec_id"], []).append((float(r["onset_s"]), float(r["offset_s"])))
    rng = random.Random(SEED)
    chosen = []
    for lev in sorted({r["snr"] for r in ev}, key=float):
        chosen += rng.sample([r for r in ev if r["snr"] == lev], n)
    for i, r in _task(list(enumerate(chosen))):
        dst = out / "synth" / f"{i:04d}.json"
        if dst.exists():
            continue
        x, sr = sf.read(set_dir / "audio" / f"{r['rec_id']}.flac", dtype="float32")
        s, e = float(r["onset_s"]), float(r["offset_s"])
        lo, hi = max(0.0, s - CTX_S), min(len(x) / sr, e + CTX_S)
        w = resample_audio(x[int(lo * sr):int(hi * sr)], sr, SR)
        calls = [(a - lo, b - lo) for a, b in by_rec[r["rec_id"]] if b > lo and a < hi]
        rec = {"rec_id": r["rec_id"], "call_type": r["call_type"], "level": float(r["snr"]),
               **noise_stats(w, (s - lo, e - lo), calls, cfg)}
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(rec))


def donors(donor_dir: Path, out: Path) -> None:
    """C2 on every Turesson donor: the clip, its padding as the call-free context."""
    cfg = load_config().cleaning
    rows = list(csv.DictReader(open(donor_dir / "donors.tsv"), delimiter="\t"))
    for r in _task(rows):
        dst = out / "donors" / f"{r['donor_id']}.json"
        if dst.exists():
            continue
        x, sr = sf.read(donor_dir / "audio" / f"{r['donor_id']}.flac", dtype="float32")
        x = resample_audio(x if x.ndim == 1 else x.mean(1), sr, SR)
        call = (float(r["onset_s"]), float(r["offset_s"]))
        rec = {"donor_id": r["donor_id"], "call_type": r["call_type"], **ridge_stats(x, call, [call], cfg, leak=False)}
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(rec))


def examples(blocks: Path, out: Path) -> None:
    """C4 on the examples page: C2's measures plus the number and length of kept tracks inside the call."""
    cfg = load_config().cleaning
    rows = []
    for p in sorted(blocks.glob("*.npz")):
        d = np.load(p)
        x, call = d["audio"], tuple(float(v) for v in d["call"])
        calls = [tuple(c) for c in d["calls"]]
        calls = calls + [call] if not any(abs(c[0] - call[0]) < 0.05 and abs(c[1] - call[1]) < 0.05 for c in calls) else calls
        meta = json.loads(str(d["meta"]))
        rows.append({"item": p.stem, "group": meta.get("group"), "name": meta.get("name"),
                     **ridge_stats(x, call, calls, cfg, leak=False)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "examples.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("real"); a.add_argument("run", type=Path); a.add_argument("out", type=Path)
    a.add_argument("--n", type=int, default=600); a.add_argument("--leak-n", type=int, default=150)
    b = sub.add_parser("synth"); b.add_argument("set", type=Path); b.add_argument("out", type=Path)
    b.add_argument("--n", type=int, default=50)
    c = sub.add_parser("donors"); c.add_argument("dir", type=Path); c.add_argument("out", type=Path)
    d = sub.add_parser("examples"); d.add_argument("blocks", type=Path); d.add_argument("out", type=Path)
    g = ap.parse_args()
    if g.cmd == "real":
        real(g.run, g.out, g.n, g.leak_n)
    elif g.cmd == "synth":
        synth(g.set, g.out, g.n)
    elif g.cmd == "donors":
        donors(g.dir, g.out)
    else:
        examples(g.blocks, g.out)


if __name__ == "__main__":
    main()
