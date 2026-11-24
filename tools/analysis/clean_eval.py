"""Cleaning evaluation by shadow filtering: what a cleaning setting does to a known call and to known noise.

Why (user, 2026-11-24). The cleaning settings were chosen by ear: spectral subtraction (``clean_calls.clean_call``,
alpha = 3) in October, and ridge resynthesis (``tools/explorations/ridge_listen.py``) at 9 dB / 20 ms after the
listening comparison of 2026-11-24. Listening cannot say how much of a call a setting damages and how much noise it
leaves, separately. This measures both, on mixtures where the two are known.

Mixtures [ours]. y = b + g * s.
* s: a Turesson donor call (``tools/synth/donor_calls.py``) between its bounds, with the synthetic set's 5 ms
  raised-cosine ramps (``generate.faded``). Donors whose bounds the user found wrong (``BAD_DONORS``) are left out.
* b: real colony background, not the synthetic set's night background: the cleaning checks (C1,
  ``tools/explorations/cleaning_checks.py``) found that background 7-8 dB quieter in the call band, steadier, and
  free of clicks, all of which favour a cleaning. Here b is a stretch of a 2021 day recording (07:00-17:59) with no
  v6 detection within ``GUARD_S`` of it, ``CTX_S`` on each side of the call, on one of the 8 virtual beams drawn at
  random. Production cleans the loudest beam over the call, which the call's direction decides, so with respect to
  the noise the beam is random. "No detection" is not "no call": a faint call the detector missed can remain in b,
  where it counts as noise. Quiet stretches are also likelier to hold long gaps, so b may be quieter than the
  background around real calls; each mixture records b's level for comparison with C1.
* g: set by bisection so that the mixture's call SNR, measured as in C1 (``call_snr_db``), hits a target. Targets
  are the 10/25/50/75/90th percentiles of that SNR over 600 random v6 day calls (C1, measured on the 4-capsule mean;
  the beam is about 2 dB clearer, so these targets are, if anything, slightly hard).

Shadow filtering. Every cleaning step after the STFT multiplies each time-frequency cell by a real gain computed
from the mixture: click capping (``declick`` scales a cell by cap / |X|), the spectral gain or the ridge mask, and
the fade outside the call. Their product H is computed once, from Y. The high-pass and the STFT are linear, so
S + B = Y, and H applied to S and to B separately splits the cleaned output exactly into the cleaned call and the
leftover noise: istft(H S) + istft(H B) = istft(H Y), which ``tests/test_clean_eval.py`` checks, together with
istft(H Y) being exactly what the production functions output.

Metrics, over the call's frames and every bin from the high-pass cutoff up (dB):
* ``call_kept``: 10 log10(sum |H S|^2 / sum |S|^2), the call energy that survives.
* ``call_sdr``: 10 log10(sum |S|^2 / sum |H S - S|^2), the call against everything the cleaning did to it (cut
  energy and changed shape alike). Higher is better.
* ``noise_left``: 10 log10(sum |H B|^2 / sum |B|^2), the share of the noise under the call that survives. Lower is
  better.
* ``snr_gain``: call_kept - noise_left, the improvement in call-to-noise ratio.
The donor carries its own recording noise (its clip floor, at least about 18 dB below the call for most types), and
that noise is part of S. A cleaning that removes it is scored as damaging the call, so ``call_sdr`` has a ceiling of
roughly that floor; every setting meets the same ceiling, so the comparison holds.

Usage (cluster, array tasks split the mixtures):
    python tools/analysis/clean_eval.py run RUN DONOR_DIR OUT
    python tools/analysis/clean_eval.py report OUT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

from marmoset_developmental_agent.config.agent_config import CleaningConfig, load_config
from marmoset_developmental_agent.io.channels import ambisonic_beams
from marmoset_developmental_agent.preprocessing import clean_calls as cc
from marmoset_developmental_agent.preprocessing.filter_5khz import resample_audio
from tools.explorations.ridge_listen import ridge_mask
from tools.synth.generate import faded

SR = 48_000
# [ours] Background on each side of the call: CleaningConfig.context_s, the span production takes the noise from.
CTX_S = 3.0
# [ours] No v6 detection this close to the background stretch, so a detected call's tail is not counted as noise.
GUARD_S = 0.5
# [ours] C1 (cleaning_checks.py, 600 v6 day calls): p10 / p25 / p50 / p75 / p90 of call_snr_db.
TARGET_SNR_DB = (16.0, 19.4, 23.9, 30.3, 38.5)
N_PER_TYPE = 10
SEED = 2026_11_24
# [ours] The user's donor_check review (tools/explorations/donor_check.py, its README row): bounds wrong or the call
# missed. The fix to their bounds is pending, so they are left out rather than planted with wrong edges.
BAD_DONORS = {"seep_000", "trill_002", "trill_005", "trill_006", "trill_011", "trill_013", "trill_015", "trill_016",
              "trill_017", "trill_018", "trill_021", "trill_022", "trill_023", "trill_024", "trill_025"}
# Settings: the spectral subtraction in use, and ridges at the user's choice (9 dB / 20 ms, 2026-11-24) with one step
# each way on both parameters that C3 / C4 found to matter. LINK_HZ stays 500: C2 / C3 found it insensitive.
Setting = Tuple[str, Optional[Tuple[float, float, float]]]
SETTINGS: Dict[str, Setting] = {
    "spectral": ("spectral", None),
    "ridge_6dB_20ms": ("ridge", (6.0, 500.0, 0.020)),
    "ridge_9dB_20ms": ("ridge", (9.0, 500.0, 0.020)),
    "ridge_12dB_20ms": ("ridge", (12.0, 500.0, 0.020)),
    "ridge_9dB_10ms": ("ridge", (9.0, 500.0, 0.010)),
    "ridge_9dB_40ms": ("ridge", (9.0, 500.0, 0.040)),
}
Interval = Tuple[float, float]


# ----------------------------------------------------------------------------------------------------------------- #
# The cleaning as a per-cell gain
# ----------------------------------------------------------------------------------------------------------------- #
def front(x: np.ndarray, cfg: CleaningConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """High-pass and STFT exactly as ``clean_call`` does them: ``(f, t, X)``. Both steps are linear in ``x``, which is
    what lets the call and the noise be filtered separately and still add up to the mixture."""
    x = signal.sosfiltfilt(signal.butter(5, cfg.highpass_hz, "highpass", fs=SR, output="sos"), x.astype(np.float64))
    return cc._stft(x, SR, cfg)


def cell_gain(Y: np.ndarray, f: np.ndarray, t: np.ndarray, call: Interval, calls: Sequence[Interval],
              cfg: CleaningConfig, setting: Setting) -> np.ndarray:
    """H, the real gain the cleaning applies to every cell of the mixture's STFT ``Y``.

    Each factor is computed from the mixture only, exactly as production computes it, so that ``H * Y`` is the
    production output's STFT and ``H`` itself can then be applied to the call and to the noise apart.
    """
    noise, _ = cc.noise_profile(np.abs(Y) ** 2, t, calls, cfg)          # N(f) from the call-free frames of the mixture
    X, _ = cc.declick(Y, f, noise, cfg)
    # declick returns Y * (cap / |Y|): a real factor per cell, at most 1. Recovering it as |X| / |Y| is exact wherever
    # |Y| > 0; where |Y| = 0 the cell contributes nothing to any of S, B or Y's outputs, so its factor is irrelevant.
    mag = np.abs(Y)
    D = np.where(mag > 0, np.abs(X) / np.where(mag > 0, mag, 1.0), 1.0)
    P = np.abs(X) ** 2                                                    # power after click capping, as clean_call
    kind, params = setting
    if kind == "spectral":
        # clean_call's gain, step for step: subtraction with over-subtraction alpha and a floor, then the median
        # filter over (bins x frames) and the moving average over frames that suppress isolated surviving cells.
        G = np.clip(1 - cfg.over_subtraction * noise[:, None] / (P + 1e-20), cfg.gain_floor, 1.0)
        G = uniform_filter1d(median_filter(G, size=tuple(cfg.gain_median)), cfg.gain_smooth_frames, axis=1)
    else:
        peak_db, link_hz, min_len_s = params
        # ridge_clean's mask: 1 within +-2 bins of every ridge point, 0 elsewhere (binary; no smoothing).
        G = ridge_mask(P, noise, f, cfg.hop_length / SR, peak_db=peak_db, link_hz=link_hz,
                       min_len_s=min_len_s).astype(float)
    a, b = call
    # The fade both cleanings apply: full gain inside the call, a linear ramp to 0 over fade_s outside it.
    fade = np.clip(np.minimum(t - a, b - t) / cfg.fade_s + 1, 0, 1)
    return D * G * fade[None, :]


def inverse(Z: np.ndarray, n: int, cfg: CleaningConfig) -> np.ndarray:
    """Inverse STFT to ``n`` samples, as ``clean_call`` trims and pads its output."""
    y = signal.istft(Z, SR, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length)[1]
    return np.pad(y, (0, max(0, n - len(y))))[:n]


# ----------------------------------------------------------------------------------------------------------------- #
# SNR, scaling and metrics
# ----------------------------------------------------------------------------------------------------------------- #
def call_snr_db(y: np.ndarray, call: Interval, calls: Sequence[Interval], cfg: CleaningConfig) -> float:
    """The call's SNR as C1 measured it (``cleaning_checks.noise_stats``): over the call's frames, the loudest
    2-20 kHz cell above that frequency's noise level, after click capping; the median over frames, in dB."""
    f, t, Y = front(y, cfg)
    noise, _ = cc.noise_profile(np.abs(Y) ** 2, t, calls, cfg)
    X, _ = cc.declick(Y, f, noise, cfg)
    band = (f >= 2000) & (f <= 20000)
    inside = (t >= call[0]) & (t <= call[1])
    rel = np.abs(X[band][:, inside]) ** 2 / (noise[band, None] + 1e-20)
    return float(np.median(10 * np.log10(rel.max(axis=0) + 1e-12)))


def scale_to(s: np.ndarray, b: np.ndarray, call: Interval, target_db: float, cfg: CleaningConfig,
             tol_db: float = 0.25) -> Tuple[float, float]:
    """``(g, achieved SNR)`` with ``call_snr_db(b + g s)`` within ``tol_db`` of ``target_db``, by bisection on g in dB.

    The SNR rises with g (the call's cells grow, the noise level N(f) comes from call-free frames and stays), so
    bisection converges; 40 halvings of a 160 dB bracket reach far below the tolerance, and the loop stops early.
    """
    lo, hi = -120.0, 40.0                         # dB relative to unit gain; donors and backgrounds are both << 1 FS
    snr = float("nan")
    for _ in range(40):
        mid = (lo + hi) / 2
        snr = call_snr_db(b + 10 ** (mid / 20) * s, call, [call], cfg)
        if abs(snr - target_db) <= tol_db:
            break
        lo, hi = (mid, hi) if snr < target_db else (lo, mid)
    return 10 ** (mid / 20), snr


def _db(num: float, den: float) -> float:
    return float(10 * np.log10((num + 1e-30) / (den + 1e-30)))


def metrics(s: np.ndarray, b: np.ndarray, call: Interval, cfg: CleaningConfig,
            settings: Dict[str, Setting] = SETTINGS) -> Dict[str, Dict[str, float]]:
    """Every setting's metrics on the mixture ``b + s`` (``s`` already scaled), by shadow filtering."""
    f, t, S = front(s, cfg)
    _, _, B = front(b, cfg)
    Y = S + B                                    # = front(b + s) up to rounding: the high-pass and STFT are linear
    sel = np.ix_(f >= cfg.highpass_hz, (t >= call[0]) & (t <= call[1]))
    s2, b2 = float((np.abs(S[sel]) ** 2).sum()), float((np.abs(B[sel]) ** 2).sum())
    out = {}
    for name, setting in settings.items():
        H = cell_gain(Y, f, t, call, [call], cfg, setting)[sel]
        hs, hb = H * S[sel], H * B[sel]
        kept, left = _db(float((np.abs(hs) ** 2).sum()), s2), _db(float((np.abs(hb) ** 2).sum()), b2)
        out[name] = {"call_kept": round(kept, 3), "call_sdr": round(_db(s2, float((np.abs(hs - S[sel]) ** 2).sum())), 3),
                     "noise_left": round(left, 3), "snr_gain": round(kept - left, 3)}
    return out


# ----------------------------------------------------------------------------------------------------------------- #
# Mixtures
# ----------------------------------------------------------------------------------------------------------------- #
def day_recordings(run: Path) -> List[str]:
    """2021 day recordings of the run (start 07:00-17:59), by id."""
    out = []
    for p in sorted((run / "files").glob("Rec_2021_*.json")):
        info = json.loads(p.read_text())
        if 7 <= dt.datetime.fromisoformat(info["start_time"]).hour < 18 and not info.get("night_skipped"):
            out.append(p.stem)
    return out


def background(run: Path, recs: Sequence[str], need_s: float, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """``need_s`` of one random beam of a random day recording, ``GUARD_S`` clear of every v6 detection."""
    for _ in range(200):
        rid = recs[int(rng.integers(len(recs)))]
        info = json.loads((run / "files" / f"{rid}.json").read_text())
        dets = sorted((float(c["start_s"]), float(c["end_s"])) for c in csv.DictReader(open(run / "calls" / f"{rid}.csv")))
        edges = [0.0] + [x for d in dets for x in d] + [float(info["duration_s"])]
        gaps = [(edges[i] + GUARD_S, edges[i + 1] - GUARD_S) for i in range(0, len(edges), 2)]
        gaps = [(a, e) for a, e in gaps if e - a >= need_s]
        if not gaps:
            continue
        a, e = gaps[int(rng.integers(len(gaps)))]
        start = a + float(rng.uniform(0, e - a - need_s))
        with sf.SoundFile(info["path"]) as fh:
            sr = fh.samplerate
            fh.seek(int(start * sr))
            caps = fh.read(int(round(need_s * sr)), dtype="float32", always_2d=True).T
        if caps.shape[0] != 4 or caps.shape[1] < int(need_s * sr) - 1:
            continue
        k = int(rng.integers(8))
        x = ambisonic_beams(caps)[k]
        if sr != SR:
            x = resample_audio(x, sr, SR)
        return x.astype(np.float64)[: int(round(need_s * SR))], {"recording_id": rid, "start_s": round(start, 3), "beam": k}
    raise RuntimeError(f"no call-free stretch of {need_s:.1f} s found")


def jobs(donor_dir: Path) -> List[Dict]:
    """``N_PER_TYPE`` donors per type (fixed draw), each at every target SNR: the same calls at every level."""
    rows = [r for r in csv.DictReader(open(donor_dir / "donors.tsv"), delimiter="\t") if r["donor_id"] not in BAD_DONORS]
    rng = np.random.default_rng(SEED)
    out = []
    for ti, typ in enumerate(sorted({r["call_type"] for r in rows})):
        pool = sorted((r for r in rows if r["call_type"] == typ), key=lambda r: r["donor_id"])
        pick = [pool[i] for i in sorted(rng.choice(len(pool), min(N_PER_TYPE, len(pool)), replace=False))]
        for di, r in enumerate(pick):
            for li, target in enumerate(TARGET_SNR_DB):
                out.append({"id": f"{typ}_{di:02d}_L{li}", "seed": [SEED, ti, di, li], "target": target, **r})
    return out


def run(run_dir: Path, donor_dir: Path, out: Path) -> None:
    cfg = load_config().cleaning
    todo = jobs(donor_dir)
    k, n = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    recs = day_recordings(run_dir)
    (out / "mixtures").mkdir(parents=True, exist_ok=True)
    for job in todo[k::n]:
        dst = out / "mixtures" / f"{job['id']}.json"
        if dst.exists():
            continue
        rng = np.random.default_rng(job["seed"])
        x, sr = sf.read(donor_dir / "audio" / f"{job['donor_id']}.flac", dtype="float32")
        assert sr == SR, (job["donor_id"], sr)
        s0 = faded(x[int(float(job["onset_s"]) * SR):int(float(job["offset_s"]) * SR)])
        dur = len(s0) / SR
        b, where = background(run_dir, recs, dur + 2 * CTX_S, rng)
        s = np.zeros_like(b)
        i0 = int(CTX_S * SR)
        s[i0:i0 + len(s0)] = s0
        call = (CTX_S, CTX_S + dur)
        g, snr = scale_to(s, b, call, job["target"], cfg)
        f, t, B = front(b, cfg)
        bg_noise, _ = cc.noise_profile(np.abs(B) ** 2, t, [call], cfg)
        res = {"id": job["id"], "donor_id": job["donor_id"], "call_type": job["call_type"], "duration_s": round(dur, 3),
               "target_snr_db": job["target"], "snr_db": round(snr, 2), "gain": g, **where,
               "bg_db": round(10 * np.log10(bg_noise[(f >= 2000) & (f <= 20000)].sum() + 1e-20), 2),
               "settings": metrics(g * s, b, call, cfg)}
        dst.write_text(json.dumps(res))
        print(job["id"], f"snr {snr:.1f}", flush=True)


# ----------------------------------------------------------------------------------------------------------------- #
# Report
# ----------------------------------------------------------------------------------------------------------------- #
FAST = {"chirp", "seep", "tsik", "tsik_ek", "twitter"}     # C2: >= 10% of steps above 500 Hz per 5 ms
SHORT = {"alarm_1", "seep", "tsik", "trill", "tsik_ek"}    # median donor duration under 0.35 s (SYNTH_METHODS.md)


def report(out: Path) -> None:
    rows = [json.loads(p.read_text()) for p in sorted((out / "mixtures").glob("*.json"))]
    names = list(SETTINGS)
    keys = ("call_sdr", "call_kept", "noise_left", "snr_gain")

    def table(sub: List[Dict], title: str) -> None:
        print(f"\n{title} (n={len(sub)}; medians, dB)")
        print(f"{'setting':18s}" + "".join(f"{k:>12s}" for k in keys))
        for nm in names:
            print(f"{nm:18s}" + "".join(f"{np.median([r['settings'][nm][k] for r in sub]):12.2f}" for k in keys))

    print(f"{len(rows)} mixtures; achieved SNR - target: median {np.median([r['snr_db'] - r['target_snr_db'] for r in rows]):.2f} dB;"
          f" background 2-20 kHz median {np.median([r['bg_db'] for r in rows]):.1f} dB (C1 real: -75.0)")
    table(rows, "all")
    for t in TARGET_SNR_DB:
        table([r for r in rows if r["target_snr_db"] == t], f"target SNR {t} dB")
    for label, grp in (("short types", SHORT), ("fast-FM types", FAST)):
        table([r for r in rows if r["call_type"] in grp], label)
    for typ in sorted({r["call_type"] for r in rows}):
        sub = [r for r in rows if r["call_type"] == typ]
        print(f"  {typ:12s} call_sdr " + " ".join(f"{nm.replace('ridge_', 'r'):>10s} {np.median([r['settings'][nm]['call_sdr'] for r in sub]):5.1f}" for nm in names))
    # Paired: the chosen setting against each other one, per mixture.
    ref = "ridge_9dB_20ms"
    print(f"\npaired against {ref} (median difference per mixture, other - {ref}; share of mixtures where other is better)")
    for nm in names:
        if nm == ref:
            continue
        d_sdr = np.array([r["settings"][nm]["call_sdr"] - r["settings"][ref]["call_sdr"] for r in rows])
        d_left = np.array([r["settings"][nm]["noise_left"] - r["settings"][ref]["noise_left"] for r in rows])
        print(f"  {nm:18s} call_sdr {np.median(d_sdr):+6.2f} (better {np.mean(d_sdr > 0):.0%})   "
              f"noise_left {np.median(d_left):+6.2f} (better {np.mean(d_left < 0):.0%})")
    _figure(rows, names, out / "clean_eval.png")


def _figure(rows: List[Dict], names: List[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"spectral": "#2a6fdb", "ridge_6dB_20ms": "#9a9a9a", "ridge_9dB_20ms": "#d6452b",
              "ridge_12dB_20ms": "#7a1f12", "ridge_9dB_10ms": "#e8a33d", "ridge_9dB_40ms": "#1a9a5a"}
    groups = [("all calls", rows)] + [(f"SNR {t:g} dB", [r for r in rows if r["target_snr_db"] == t])
                                      for t in (TARGET_SNR_DB[0], TARGET_SNR_DB[2], TARGET_SNR_DB[4])]
    fig, axs = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.0), sharey=True)
    for ax, (title, sub) in zip(axs, groups):
        for nm in names:
            x = np.median([r["settings"][nm]["noise_left"] for r in sub])
            y = np.median([r["settings"][nm]["call_sdr"] for r in sub])
            ax.plot(x, y, "o", ms=9, color=colors[nm], label=nm)
            ax.annotate(nm.replace("ridge_", ""), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set(title=title, xlabel="noise left under the call (dB, lower = cleaner)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("call preserved, SDR (dB, higher = less damage)")
    axs[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("figure", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run", type=Path)
    r.add_argument("donors", type=Path)
    r.add_argument("out", type=Path)
    p = sub.add_parser("report")
    p.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.run, a.donors, a.out)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
