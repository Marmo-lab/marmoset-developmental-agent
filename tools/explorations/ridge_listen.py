"""Listening comparison: original vs. the corpus's spectral cleaning vs. ridge-based resynthesis (Didi's proposal).

Why (user, 2026-11-19). Didi proposed cleaning calls by keeping only their spectral ridges: the dominant peaks and
harmonics in the STFT, with their complex coefficients (magnitude and phase), masking everything else, and
resynthesizing by inverse STFT. Before any metric, the user wants to hear it next to the original and the current
cleaning (``preprocessing/clean_calls.py``) on the same calls. This script is the listening stage only; it does not
touch the production path. If ridges win, they move into ``clean_calls`` as a proper Track 1 change.

What is shared, so the comparison isolates one decision. Both cleanings use the same beam (the loudest of 8 virtual
cardioids, as ``clean_corpus``), the same 1 kHz high-pass, the same STFT (``CleaningConfig.n_fft`` / ``hop_length``:
21 ms Hann window, 5 ms hop), the same per-frequency noise profile N(f) from call-free frames, the same click capping,
and the same fade to silence outside the call. Only the per-cell gain differs:
* spectral cleaning: G = max(1 - alpha * N / P, floor), a continuous gain by how far a cell stands above the noise;
* ridges: G = 1 within ``HALF_WIDTH_BINS`` of a ridge, 0 elsewhere.

Ridges [ours, from the user's and Didi's description, 2026-11-19]:
1. Peaks: in each frame, a bin that is a local maximum across frequency and at least ``PEAK_DB`` above the noise at
   its frequency (P >= 10^(PEAK_DB/10) * N). A tone, fundamental or harmonic, is a narrow peak in every frame.
2. Tracks: a peak continues a track from the previous frame when it lies within ``LINK_HZ`` of the track's last
   bin; pairs are matched greedily, nearest first, one-to-one. 500 Hz per 5 ms hop admits a trill's fast frequency
   modulation (about 30 cycles/s over a few kHz moves up to ~200 Hz per hop) with room to spare.
3. Length: a track shorter than ``MIN_LEN_S`` is dropped. A click or a noise blip makes a peak in one or two frames;
   a call's tone persists.
4. Mask width: a pure tone under a Hann window occupies the main lobe, 4 bins wide, so the mask keeps
   ``HALF_WIDTH_BINS`` = 2 bins on each side of every ridge bin. Keeping the peak bin alone would cut the tone's
   energy and distort it on resynthesis.

Items: the 8 calls of the October cleaning demo (local hand-labelled clips; their call bounds and neighbouring calls
are the hand labels) and ``N_V6`` calls of the v6 corpus run, spread over call duration (terciles) and tonality
(the detector's tonal fraction below or above 0.5, so broadband calls are included), drawn with ``SEED``.

Usage:
    python tools/explorations/ridge_listen.py extract-v6 RUN OUT     # cluster: v6 calls -> OUT/blocks/*.npz
    python tools/explorations/ridge_listen.py extract-demo OUT       # local: the 8 October demo calls
    python tools/explorations/ridge_listen.py build OUT              # anywhere: clean both ways, write OUT/index.html
    python tools/explorations/ridge_listen.py compare BLOCKS OUT     # anywhere: five versions (VERSIONS_COMPARE)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

from marmoset_developmental_agent.config.agent_config import CleaningConfig, load_config
from marmoset_developmental_agent.io.channels import ambisonic_beams, band_energy
from marmoset_developmental_agent.preprocessing import clean_calls as cc
from marmoset_developmental_agent.preprocessing.filter_5khz import resample_audio

SR = 48_000
BLOCK_S = 10.0            # audio kept on each side of a call: covers the cleaning's wide noise context
PAD_S = 0.25              # audio around the call in the clips the user hears (as the October demo)
PEAK_DB = 9.0             # user's choice by ear, 2026-11-24 (compare page): 6 dB kept twice the background
LINK_HZ = 500.0
MIN_LEN_S = 0.020
HALF_WIDTH_BINS = 2
SEED = 2026_11_19
N_V6 = 36                 # 6 per cell: duration tercile x tonal / broadband
DEMO = [("ho_06", 45.26, 46.45, "phee"), ("clip_11", 57.86, 59.62, "phee"), ("ho_06", 14.03, 16.30, "tonal (K3)"),
        ("clip_16", 13.53, 13.93, "short (K6)"), ("clip_24", 12.97, 14.10, "trill?"), ("clip_28", 42.69, 44.23, "infant cry"),
        ("ho_14", 41.68, 43.73, "pulsed / mobbing?"), ("clip_13", 24.77, 27.81, "twitter-like")]

Interval = Tuple[float, float]


# --------------------------------------------------------------------------------------------------------------- #
# The ridge mask and the ridge cleaning
# --------------------------------------------------------------------------------------------------------------- #
def ridge_tracks(power: np.ndarray, noise: np.ndarray, f: np.ndarray, peak_db: float = PEAK_DB,
                 link_hz: float = LINK_HZ) -> List[List[Tuple[int, int]]]:
    """Every track as a list of (bin, frame) points, before the length test; steps 1-2 of the module docstring."""
    n_bins, n_frames = power.shape
    df = float(f[1] - f[0])
    # 1. Peaks: a local maximum across frequency (strictly above the lower neighbour, at least the upper one, so a
    #    flat top yields one peak) that clears the noise at its own frequency by peak_db.
    above = power >= (10 ** (peak_db / 10)) * noise[:, None]
    local = np.zeros_like(above)
    local[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    peaks = [np.flatnonzero(above[:, t] & local[:, t]) for t in range(n_frames)]
    # 2. Tracks: each open track remembers its points; a frame's peaks extend the nearest open tracks within link_hz.
    max_jump = link_hz / df
    open_tracks: List[List[Tuple[int, int]]] = []
    done: List[List[Tuple[int, int]]] = []
    for t in range(n_frames):
        pairs = sorted((abs(int(k) - tr[-1][0]), i, j) for i, tr in enumerate(open_tracks) for j, k in enumerate(peaks[t])
                       if abs(int(k) - tr[-1][0]) <= max_jump)
        used_tr, used_pk = set(), set()
        nxt: List[List[Tuple[int, int]]] = []
        for _, i, j in pairs:                      # nearest first, each track and each peak used once
            if i in used_tr or j in used_pk:
                continue
            used_tr.add(i); used_pk.add(j)
            nxt.append(open_tracks[i] + [(int(peaks[t][j]), t)])
        done += [tr for i, tr in enumerate(open_tracks) if i not in used_tr]   # a track with no continuation ends
        nxt += [[(int(k), t)] for j, k in enumerate(peaks[t]) if j not in used_pk]  # an unmatched peak starts one
        open_tracks = nxt
    return done + open_tracks


def tracks_to_mask(tracks: Sequence[List[Tuple[int, int]]], shape: Tuple[int, int], hop_s: float,
                   min_len_s: float = MIN_LEN_S, half_width: int = HALF_WIDTH_BINS) -> np.ndarray:
    """Steps 3-4: drop tracks shorter than ``min_len_s``, then mark ``half_width`` bins around every kept point."""
    min_frames = int(np.ceil(min_len_s / hop_s))
    mask = np.zeros(shape, bool)
    for tr in tracks:
        if len(tr) >= min_frames:
            for k, t in tr:
                mask[max(0, k - half_width):k + half_width + 1, t] = True
    return mask


def ridge_mask(power: np.ndarray, noise: np.ndarray, f: np.ndarray, hop_s: float,
               peak_db: float = PEAK_DB, link_hz: float = LINK_HZ, min_len_s: float = MIN_LEN_S,
               half_width: int = HALF_WIDTH_BINS) -> np.ndarray:
    """Boolean (bins x frames) mask: True within ``half_width`` bins of a ridge.

    ``power`` is |X|^2 of the STFT and ``noise`` the per-bin noise power N(f); see the module docstring for the rule.
    """
    return tracks_to_mask(ridge_tracks(power, noise, f, peak_db, link_hz), power.shape, hop_s, min_len_s, half_width)


def ridge_clean(audio: np.ndarray, sr: int, call: Interval, calls: Sequence[Interval], cfg: CleaningConfig,
                peak_db: float = PEAK_DB, min_len_s: float = MIN_LEN_S) -> np.ndarray:
    """The ridge cleaning of one call, with every step but the gain identical to ``clean_calls.clean_call``."""
    x = signal.sosfiltfilt(signal.butter(5, cfg.highpass_hz, "highpass", fs=sr, output="sos"), audio.astype(np.float64))
    f, t, X = cc._stft(x, sr, cfg)
    noise, _ = cc.noise_profile(np.abs(X) ** 2, t, calls, cfg)
    X, _ = cc.declick(X, f, noise, cfg)
    G = ridge_mask(np.abs(X) ** 2, noise, f, cfg.hop_length / sr, peak_db=peak_db, min_len_s=min_len_s).astype(float)
    a, b = call
    fade = np.clip(np.minimum(t - a, b - t) / cfg.fade_s + 1, 0, 1)
    # The kept cells carry their original complex values (real and imaginary parts, i.e. magnitude and phase), so the
    # inverse STFT rebuilds the ridges exactly as recorded; overlap-add of the Hann windows restores the waveform.
    y = signal.istft(X * G * fade[None, :], sr, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop_length)[1]
    return np.pad(y, (0, max(0, len(audio) - len(y))))[: len(audio)].astype(np.float32)


# --------------------------------------------------------------------------------------------------------------- #
# Items: one mono block (the call's beam, +-BLOCK_S) per call, with the call and its neighbours in block seconds
# --------------------------------------------------------------------------------------------------------------- #
def _beam(capsules: np.ndarray, sr: int, s: float, e: float) -> np.ndarray:
    """The loudest of the 8 virtual cardioids over the call, as ``clean_corpus._beam_index`` chooses it."""
    seg = capsules[:, int(s * sr):int(e * sr)]
    k = int(np.argmax(band_energy(ambisonic_beams(seg), sr)))
    return ambisonic_beams(capsules)[k]


def _save(out: Path, item: str, x: np.ndarray, call: Interval, calls: List[Interval], meta: Dict) -> None:
    (out / "blocks").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "blocks" / f"{item}.npz", audio=x.astype(np.float32), call=np.array(call),
                        calls=np.array(calls).reshape(-1, 2), meta=json.dumps(meta))


def extract_demo(out: Path) -> None:
    labels: Dict[str, List[Interval]] = {}
    for p in ("data/labels/handlabels.csv", "data/labels/heldout_labels.csv"):
        for r in csv.DictReader(open(p)):
            labels.setdefault(r["clip_id"], []).append((float(r["clip_start_s"]), float(r["clip_end_s"])))
    for n, (cid, a, b, name) in enumerate(DEMO):
        d = np.load(f"labeling/{'heldout/audio48' if cid.startswith('ho_') else 'clip_audio48'}/{cid}.npz")
        sr, lead, caps = int(d["sr"]), float(d["lead_s"]), d["capsules"]
        lo, hi = max(-lead, a - BLOCK_S), min(caps.shape[1] / sr - lead, b + BLOCK_S)
        x = _beam(caps[:, int((lo + lead) * sr):int((hi + lead) * sr)], sr, a - lo - PAD_S, b - lo + PAD_S)
        x = resample_audio(x, sr, SR)
        others = [(s - lo, e - lo) for s, e in labels.get(cid, []) if e > lo and s < hi]
        _save(out, f"demo_{n:02d}", x, (a - lo, b - lo), others,
              {"source": f"{cid} @{a:.2f}-{b:.2f} s", "name": name, "group": "hand-labelled clips"})


def extract_v6(run: Path, out: Path) -> None:
    rows = [r for r in csv.DictReader(open(run / "calls.tsv"), delimiter="\t")
            if r["quality"] == "ok" and float(r["confidence"]) >= 0.8 and float(r["snr_db"]) >= 10
            and 7 <= dt.datetime.fromisoformat(r["start_time"]).hour < 18]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = rows[:4000]
    feats: Dict[str, Dict[str, Dict]] = {}
    for r in rows:                                   # the detector's tonal fraction, from the per-recording catalogue
        rid = r["recording_id"]
        if rid not in feats:
            feats[rid] = {c["start_s"]: c for c in csv.DictReader(open(run / "calls" / f"{rid}.csv"))}
        r["tonal"] = float(feats[rid].get(r["start_s"], {}).get("tonal_fraction", "nan"))
    rows = [r for r in rows if np.isfinite(r["tonal"])]
    cuts = np.percentile([float(r["duration_s"]) for r in rows], [100 / 3, 200 / 3])
    cells: Dict[Tuple[int, bool], List[Dict]] = {}
    for r in rows:
        cells.setdefault((int(np.searchsorted(cuts, float(r["duration_s"]))), r["tonal"] >= 0.5), []).append(r)
    chosen = [(key, r) for key in sorted(cells) for r in cells[key][: N_V6 // 6]]
    for n, ((tercile, tonal), r) in enumerate(chosen):
        info = json.loads((run / "files" / f"{r['recording_id']}.json").read_text())
        s, e = float(r["start_s"]), float(r["end_s"])
        with sf.SoundFile(info["path"]) as fh:
            sr = fh.samplerate
            lo = max(0.0, s - BLOCK_S)
            fh.seek(int(lo * sr))
            caps = fh.read(int((e + BLOCK_S - lo) * sr), dtype="float32", always_2d=True).T
        x = resample_audio(_beam(caps, sr, s - lo - PAD_S, e - lo + PAD_S), sr, SR)
        hi = lo + x.size / SR
        others = [(float(c["start_s"]) - lo, float(c["end_s"]) - lo) for c in feats[r["recording_id"]].values()
                  if float(c["end_s"]) > lo and float(c["start_s"]) < hi]
        _save(out, f"v6_{n:02d}", x, (s - lo, e - lo), others,
              {"source": f"{r['call_id']}", "group": f"{('short', 'medium', 'long')[tercile]}, {'tonal' if tonal else 'broadband'}",
               "name": f"{float(r['duration_s']):.2f} s, peak {float(r['peak_freq_hz']) / 1000:.1f} kHz, tonal {r['tonal']:.2f}"})


# --------------------------------------------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------------------------------------------- #
VERSIONS = [("orig", "Original"), ("spectral", "Spectral subtraction (current)"), ("ridge", "Ridge resynthesis")]
# ``compare`` (user, 2026-11-24): the two ridge settings that matter (docs: cleaning_checks C3 / C4), one step each way
# from the current 6 dB / 20 ms: stricter peaks (less background kept) and shorter tracks (more of short calls kept).
VERSIONS_COMPARE = VERSIONS[:2] + [("ridge", "Ridges 6 dB, 20 ms (previous)"), ("ridge9", "Ridges 9 dB, 20 ms (cleaner)"),
                                   ("ridge10ms", "Ridges 6 dB, 10 ms (short calls)")]
RIDGE_SETTINGS = {"ridge": (6.0, MIN_LEN_S), "ridge9": (9.0, MIN_LEN_S), "ridge10ms": (6.0, 0.010)}


def _png(path: Path, x: np.ndarray, vmin: float, vmax: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, t, Z = signal.stft(x, SR, nperseg=1024, noverlap=1024 - 240)
    fig, ax = plt.subplots(figsize=(3.4, 1.8), dpi=110)
    ax.pcolormesh(t, f / 1000, 20 * np.log10(np.abs(Z) + 1e-9), vmin=vmin, vmax=vmax, cmap="magma", shading="auto")
    ax.set_ylim(1, 22)
    ax.tick_params(labelsize=6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build(out: Path, blocks: Path | None = None, compare: bool = False) -> None:
    cfg = load_config().cleaning
    (out / "a").mkdir(parents=True, exist_ok=True)
    versions = VERSIONS_COMPARE if compare else VERSIONS
    cards, group = [], None
    for p in sorted((blocks or out / "blocks").glob("*.npz")):
        d = np.load(p)
        x, (a, b), calls, meta = d["audio"], tuple(d["call"]), [tuple(c) for c in d["calls"]], json.loads(str(d["meta"]))
        calls = calls + [(a, b)] if not any(abs(c[0] - a) < 0.05 and abs(c[1] - b) < 0.05 for c in calls) else calls
        s0, s1 = int(max(0.0, a - PAD_S) * SR), int(min(x.size / SR, b + PAD_S) * SR)
        ver = {"orig": signal.sosfiltfilt(signal.butter(5, cfg.highpass_hz, "highpass", fs=SR, output="sos"), x)[s0:s1],
               "spectral": cc.clean_call(x, SR, (a, b), calls, cfg)[0][s0:s1]}
        for key, _ in versions[2:]:
            pk, ml = RIDGE_SETTINGS[key]
            ver[key] = ridge_clean(x, SR, (a, b), calls, cfg, peak_db=pk, min_len_s=ml)[s0:s1]
        peak = max(float(np.abs(v).max()) for v in ver.values()) or 1.0
        ref = 20 * np.log10(np.abs(signal.stft(ver["orig"], SR, nperseg=1024, noverlap=784)[2]) + 1e-9)
        vmin, vmax = float(np.percentile(ref, 40)), float(ref.max())
        cols = []
        for key, label in versions:
            y = (ver[key] / peak * 0.9).astype(np.float32)        # one gain for the three, so loudness compares
            sf.write(out / "a" / f"{p.stem}_{key}.flac", y, SR, subtype="PCM_16")   # FLAC: half the size, plays in every browser
            _png(out / "a" / f"{p.stem}_{key}.png", ver[key], vmin, vmax)
            cols.append(f'<div class="ex"><b>{label}</b><img src="a/{p.stem}_{key}.png" alt="">'
                        f'<audio controls preload="none" src="a/{p.stem}_{key}.flac"></audio></div>')
        if meta.get("group") != group:
            group = meta.get("group")
            cards.append(f'<h2>{html.escape(group.capitalize() if group else "")}</h2>')
        cards.append(f'<h3>{html.escape(meta["name"])} <small>{html.escape(meta["source"])}</small></h3><div class="row">{"".join(cols)}</div>')
    (out / "index.html").write_text((PAGE_COMPARE if compare else PAGE).replace("__CARDS__", "\n".join(cards)),
                                    encoding="utf-8")
    print(out / "index.html", sum(c.startswith("<h3") for c in cards), "calls")


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Call cleaning examples</title>
<style>body{font-family:system-ui,sans-serif;margin:16px auto;max-width:1000px;padding:0 16px;color:#222;background:#fff;line-height:1.45}
.row{display:flex;gap:10px;flex-wrap:wrap}.ex{width:310px;max-width:100%;font-size:13px;display:flex;flex-direction:column;gap:4px}
img,audio{width:100%}h2{margin:34px 0 4px;border-bottom:1px solid #ddd;padding-bottom:4px}h3{margin:18px 0 6px;font-size:15px}
small{color:#777;font-weight:normal}ul{padding-left:20px}</style></head><body>
<h1>Call cleaning: original, spectral subtraction, ridge resynthesis</h1>
<p>Marmoset calls from our colony recordings, each in three versions:</p>
<ul><li><b>Original</b>: the call on the loudest of eight virtual beams of the four-capsule recorder, high-passed at 1 kHz.</li>
<li><b>Spectral subtraction</b> (the cleaning in use): every time-frequency cell is kept with a soft gain
G = max(1 &minus; 3N/P, 0.01), where P is the cell's power and N the noise level at its frequency, estimated from the
call-free audio around the call.</li>
<li><b>Ridge resynthesis</b>: in each 5 ms frame, the spectral peaks at least 9 dB above N are linked over time into
ridges (the fundamental and its harmonics; tracks shorter than 20 ms are dropped). Only the complex STFT coefficients
within &plusmn;2 bins of a ridge are kept, everything else is set to zero, and the waveform is rebuilt by inverse STFT.</li></ul>
<p>Both cleanings share the same STFT (21 ms Hann window, 5 ms hop), noise estimate and click suppression, and keep the
recorded phase. The three versions of a call are played at the same gain and drawn on the same colour scale as the
original, with 0.25 s before and after the call. The first group are hand-labelled clips; the others are calls the
detector found in the 2021 recordings, six per combination of length (short / medium / long, by tercile) and tonality
(tonal / broadband).</p>
__CARDS__
</body></html>"""


PAGE_COMPARE = """<!doctype html><html lang="he"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cleaning settings</title>
<style>body{font-family:system-ui,sans-serif;margin:16px auto;max-width:1240px;padding:0 16px;color:#222;background:#fff;line-height:1.45}
.row{display:flex;gap:8px;flex-wrap:wrap}.ex{width:236px;max-width:100%;font-size:12px;display:flex;flex-direction:column;gap:4px}
img,audio{width:100%}h2{margin:34px 0 4px;border-bottom:1px solid #ddd;padding-bottom:4px}h3{margin:18px 0 6px;font-size:15px}
small{color:#777;font-weight:normal}ul{padding-right:20px}.intro{direction:rtl;max-width:900px}</style></head><body>
<div class="intro"><h1>השוואת הגדרות ניקוי</h1>
<p>אותן 44 קריאות של דף הדוגמאות, כל אחת בחמש גרסאות, באותה עוצמה ובאותו סולם צבע:</p>
<ul><li><b>Original</b>: הביים החזק, אחרי high-pass של 1 kHz.</li>
<li><b>Spectral subtraction</b>: הניקוי הנוכחי (a = 3).</li>
<li><b>Ridges 6 dB, 20 ms</b>: רכסים בהגדרה של היום.</li>
<li><b>Ridges 9 dB, 20 ms</b>: "נקי יותר". רק שיאים חזקים יותר הופכים לרכס, ולכן נשאר בערך חצי מהרעש סביב הקריאה (4% מהרקע במקום 9%), במחיר חלקים חלשים של הקריאה.</li>
<li><b>Ridges 6 dB, 10 ms</b>: "עדין לקריאות קצרות". נשמרים גם רכסים של 10 ms, ולכן נשאר יותר מקריאות קצרות ומהירות, במחיר יותר רעש (12% מהרקע).</li></ul>
<p>מה לחפש: האם הקריאה נשמעת שלמה וטבעית, וכמה רעש נשאר בתוכה ובין ההרמוניות. במיוחד בקבוצות של הקריאות הקצרות וה-broadband.</p></div>
__CARDS__
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract-v6", "extract-demo", "build", "compare"])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path, nargs="?")
    x = ap.parse_args()
    if x.step == "extract-v6":
        extract_v6(x.a, x.b)
    elif x.step == "extract-demo":
        extract_demo(x.a)
    elif x.step == "compare":                      # compare BLOCKS OUT: the five-version page for choosing a setting
        build(x.b, blocks=x.a, compare=True)
    else:
        build(x.a)


if __name__ == "__main__":
    main()
