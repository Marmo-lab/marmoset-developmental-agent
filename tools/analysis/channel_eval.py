"""Compare the five capsule-combination strategies on real recordings, gain-invariantly.

There is no labelled ground truth yet (OPEN_DECISIONS C9), so this cannot measure
detection accuracy. What it can measure without being fooled by gain -- a single
on-axis cardioid has twice the gain of the 4-capsule mean, so any *absolute* threshold
favours single-capsule strategies for reasons unrelated to SNR (see io/channels.py) --
are two gain-invariant quantities, per strategy, on the same windows of the same files:

* **Dynamic range** in the call band: p99.9 minus p10 of per-frame band energy, in dB,
  after the 5 kHz high-pass at the detection rate. Higher means transient calls stand
  further above the stationary background. Scaling the signal leaves it unchanged.
* **Spectral-detector yield**: calls (and call-seconds) found by our level-relative
  ``frame_features="spectral"`` detector. Its thresholds are uncalibrated, so counts
  are only comparable *between strategies*, not meaningful in absolute terms.

``per_channel`` runs the detector on every capsule and unions the detections; its
dynamic range is computed on the per-frame maximum across capsules, the envelope that
union effectively sees.

Usage:
    python tools/analysis/channel_eval.py run <raw.wav> --json-out out.json
    python tools/analysis/channel_eval.py aggregate <dir-of-jsons>
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.io.channels import STRATEGIES, combine, pick_best_channel
from marmoset_developmental_agent.io.raw_reader import read_bext_datetime
from marmoset_developmental_agent.preprocessing.filter_5khz import (
    compute_frame_activity, highpass_filter, resample_audio)
from marmoset_developmental_agent.preprocessing.segmentation import activity_to_calls

BAND_HZ = (5_000.0, 11_000.0)  # call band available below the 22050 Hz detection Nyquist


def _band_energy_frames(x: np.ndarray, sr: int, cfg: PreprocessingConfig) -> np.ndarray:
    """Per-STFT-frame energy inside BAND_HZ. Shape (T,)."""
    f, _, z = signal.stft(x, fs=sr, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length,
                          boundary=None, padded=False)
    sel = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    return (np.abs(z[sel]) ** 2).sum(axis=0)


def _merge(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def evaluate_file(path: str, n_windows: int = 10, window_s: float = 30.0) -> Dict:
    info = sf.info(path)
    sr = info.samplerate
    cfg = dataclasses.replace(PreprocessingConfig.talia(), frame_features="spectral")
    det_sr = cfg.detection_sample_rate
    span = int(window_s * sr)
    starts = np.linspace(0, max(0, info.frames - span), n_windows).astype(np.int64)

    # best_snr fixes one capsule per file, chosen from excerpts spread across the file.
    with sf.SoundFile(path) as f:
        exc = []
        for s0 in starts:
            f.seek(int(s0))
            exc.append(f.read(sr * 3, dtype="float32", always_2d=True))
    best = pick_best_channel(np.concatenate(exc), sr)

    t_file = read_bext_datetime(path)
    windows = []
    with sf.SoundFile(path) as f:
        for s0 in starts:
            f.seek(int(s0))
            block = f.read(span, dtype="float32", always_2d=True)
            row: Dict[str, Dict[str, float]] = {}
            for strat in STRATEGIES:
                sigs = combine(block, strat, sr, channel=best if strat == "best_snr" else None)
                energies, calls = [], []
                for x in sigs:
                    d = highpass_filter(resample_audio(x, sr, det_sr), det_sr,
                                        cfg.highpass_cutoff_hz, cfg.highpass_order)
                    energies.append(_band_energy_frames(d, det_sr, cfg))
                    act = compute_frame_activity(d, det_sr, cfg)
                    calls += [(c.start_s, c.end_s) for c in activity_to_calls(act, cfg)]
                env = np.max(np.stack(energies), axis=0)
                db = 10 * np.log10(env + 1e-30)
                merged = _merge(calls)
                row[strat] = {
                    "dr_db": float(np.percentile(db, 99.9) - np.percentile(db, 10)),
                    "n_calls": len(merged),
                    "call_s": float(sum(e - s for s, e in merged)),
                }
            hour = None
            if t_file is not None:
                hour = (t_file + dt.timedelta(seconds=float(s0) / sr)).hour
            windows.append({"t_s": float(s0) / sr, "hour": hour, "strategies": row})
    return {"file": path, "start_time": t_file.isoformat() if t_file else "",
            "best_snr_channel": best, "windows": windows}


def aggregate(d: str) -> None:
    files = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "chan_*.json")))]
    if not files:
        raise SystemExit(f"no chan_*.json under {d}")
    print(f"{len(files)} files, {sum(len(f['windows']) for f in files)} windows\n")

    for label, keep in (("ALL", lambda h: True),
                        ("DAY 06-19", lambda h: h is not None and 6 <= h < 19),
                        ("NIGHT", lambda h: h is not None and not 6 <= h < 19)):
        ws = [w for f in files for w in f["windows"] if keep(w["hour"])]
        if not ws:
            continue
        print(f"=== {label}: {len(ws)} windows ===")
        print(f"{'strategy':<16}{'median DR':>11}{'dDR vs mean':>13}{'DR better':>11}"
              f"{'calls':>8}{'call-s':>9}{'calls vs mean':>15}")
        base_dr = np.array([w["strategies"]["mean"]["dr_db"] for w in ws])
        base_n = sum(w["strategies"]["mean"]["n_calls"] for w in ws)
        for s in STRATEGIES:
            dr = np.array([w["strategies"][s]["dr_db"] for w in ws])
            n = sum(w["strategies"][s]["n_calls"] for w in ws)
            cs = sum(w["strategies"][s]["call_s"] for w in ws)
            ratio = f"{n / base_n:.2f}x" if base_n else "n/a"
            print(f"{s:<16}{np.median(dr):11.1f}{np.median(dr - base_dr):+13.2f}"
                  f"{int((dr > base_dr + 0.1).sum()):>6}/{len(ws):<4}{n:>8}{cs:>9.0f}{ratio:>15}")
        print()
    print("dDR: median paired difference in dynamic range vs `mean`, dB (gain-invariant).\n"
          "DR better: windows where the strategy beats `mean` by > 0.1 dB.\n"
          "calls: spectral (level-relative) detector, uncalibrated -- compare across rows only.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("wav")
    r.add_argument("--json-out", required=True)
    r.add_argument("--windows", type=int, default=10)
    r.add_argument("--window-s", type=float, default=30.0)
    g = sub.add_parser("aggregate")
    g.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "run":
        res = evaluate_file(a.wav, a.windows, a.window_s)
        with open(a.json_out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {a.json_out} ({len(res['windows'])} windows, best_snr ch {res['best_snr_channel']})")
    else:
        aggregate(a.dir)


if __name__ == "__main__":
    main()
