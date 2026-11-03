"""Call map: a 2-D embedding of a corpus run's calls, to see whether they form clouds by type.

Exploration only. The developmental agent stays unsupervised and nothing here feeds the
pipeline (CLAUDE.md non-goals: no call-type classification). The map is a check on the
corpus: if the detector cuts whole calls, calls of one type should land together, and
noise or fragments should form their own islands. It is also the reference against
which the agent's units can later be compared.

Sample [ours]. Up to ``--n`` calls from ``calls_train.tsv`` (seeded), spread evenly over
the recording dates, so busy days do not fill the map.

Features [ours]. Each call becomes a fixed-size image: an STFT at 48 kHz (1024-point
window, 5 ms hop), dB interpolated onto ``N_BANDS`` log-spaced frequencies from
``F_LO_HZ`` to ``F_HI_HZ`` (the call fundamentals and their first harmonics, DETECTOR_EVAL
finding 11), then linearly resampled to ``N_STEPS`` time steps. Resampling makes the
image show a call's shape, whatever its length, so the duration is added separately as
``DUR_WEIGHT * log10(duration)``. Each image is in dB below its own peak, clipped at
``FLOOR_DB``: loudness, i.e. distance to the microphone, is removed.

Embedding. PCA to ``N_PCA`` dimensions, then t-SNE to 2-D, then HDBSCAN on the 2-D map.
That is the usual recipe for vocal repertoires (Sainburg et al. 2020, PLoS Comput Biol,
with UMAP; UMAP is not in our environment, and t-SNE keeps local structure the same
way). Clustering the 2-D map, not the PCA space, follows the same paper. Its clusters
are islands of the picture, not call types; they are named by ear on the map page.

Writes ``RUN/call_map/map.json`` (points, clusters, mean image per cluster) for the page
that ``tools/corpus/call_library.py`` serves at ``/map``.

Usage (on the cluster, via tools/cluster/call_map.sbatch):
    python tools/analysis/call_map.py RUN [--n 30000] [--seed 0] [--version orig|clean]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

N_BANDS, N_STEPS = 64, 32  # [ours] 2048 pixels per call
F_LO_HZ, F_HI_HZ = 2_000.0, 20_000.0  # [ours] fundamentals 4.6-9.7 kHz and harmonics (DETECTOR_EVAL 11)
FLOOR_DB = 50.0  # [ours] image range below the call's peak
DUR_WEIGHT = 10.0  # [ours] log10(duration) sd ~0.35 -> ~10% of the image variance
N_PCA = 50  # [ours]


def call_image(x: np.ndarray, sr: int) -> np.ndarray:
    """``(N_BANDS, N_STEPS)`` image in [0, 1]: dB below the call's peak on log frequencies, resampled in time."""
    n_fft, hop = 1024, sr // 200
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    f, _, s = signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    db = 20 * np.log10(np.abs(s) + 1e-12)
    centres = np.geomspace(F_LO_HZ, min(F_HI_HZ, sr / 2), N_BANDS)
    db = np.stack([np.interp(centres, f, col) for col in db.T], axis=1)
    t_old = np.linspace(0, 1, db.shape[1])
    t_new = np.linspace(0, 1, N_STEPS)
    db = np.stack([np.interp(t_new, t_old, row) for row in db]) if db.shape[1] > 1 else np.repeat(db, N_STEPS, 1)
    return (np.clip(db - db.max(), -FLOOR_DB, 0) / FLOOR_DB + 1).astype(np.float32)


def features(images: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Rows of flattened images with the weighted log-duration appended."""
    return np.hstack([images.reshape(len(images), -1), DUR_WEIGHT * np.log10(durations)[:, None]]).astype(np.float32)


def sample_rows(run: Path, n: int, seed: int) -> List[Dict[str, str]]:
    """Up to ``n`` training calls, the same number from each date where possible."""
    by_date: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with open(run / "calls_train.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            by_date[r["start_time"][:10]].append(r)
    rng = random.Random(seed)
    per_date = -(-n // max(1, len(by_date)))
    rows = [r for d in sorted(by_date) for r in rng.sample(by_date[d], min(per_date, len(by_date[d])))]
    return sorted(rng.sample(rows, min(n, len(rows))), key=lambda r: (r["audio_path"], float(r["audio_offset_s"])))


def read_images(run: Path, rows: List[Dict[str, str]], version: str) -> np.ndarray:
    """Images of ``rows`` (sorted by recording), each FLAC opened once."""
    out = np.zeros((len(rows), N_BANDS, N_STEPS), np.float32)
    f, path = None, None
    for i, r in enumerate(rows):
        p = run / (r["audio_path"].replace("call_audio/", "call_audio_clean/") if version == "clean" else r["audio_path"])
        if p != path:
            if f is not None:
                f.close()
            f, path = sf.SoundFile(p), p
        sr = f.samplerate
        f.seek(int(round((float(r["audio_offset_s"]) + float(r["audio_lead_s"])) * sr)))
        out[i] = call_image(f.read(int(round(float(r["duration_s"]) * sr)), dtype="float32"), sr)
        if i % 2000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    if f is not None:
        f.close()
    return out


def embed(x: np.ndarray, seed: int, perplexity: float, min_cluster: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(2-D map (N, 2), cluster labels (N,), -1 = unclustered)``."""
    z = PCA(min(N_PCA, *x.shape), random_state=seed).fit_transform(x)
    xy = TSNE(2, perplexity=perplexity, init="pca", random_state=seed, n_jobs=-1).fit_transform(z)
    return xy.astype(np.float32), HDBSCAN(min_cluster_size=min_cluster, copy=True).fit_predict(xy)


def build(run: Path, n: int, seed: int, version: str, perplexity: float, min_cluster: int) -> Dict[str, object]:
    t0 = time.time()
    rows = sample_rows(run, n, seed)
    print(f"{len(rows)} calls sampled", flush=True)
    images = read_images(run, rows, version)
    dur = np.array([float(r["duration_s"]) for r in rows])
    xy, labels = embed(features(images, dur), seed, perplexity, min_cluster)
    num = lambda r, k: round(float(r[k]), 1) if r.get(k) not in (None, "", "nan") else None  # noqa: E731
    points = [[r["call_id"], round(float(a), 2), round(float(b), 2), int(c), round(float(r["duration_s"]), 3),
               num(r, "peak_freq_hz"), num(r, "snr_db"), r["start_time"][:10], int(r["start_time"][11:13])]
              for r, (a, b), c in zip(rows, xy, labels)]
    clusters = []
    for c in sorted(set(labels.tolist())):
        m = labels == c
        clusters.append({"id": int(c), "n": int(m.sum()), "median_dur_s": round(float(np.median(dur[m])), 3),
                         "mean_image": np.round(images[m].mean(0), 2).tolist()})
    out = {"meta": {"n": len(rows), "seed": seed, "version": version, "perplexity": perplexity,
                    "min_cluster": min_cluster, "n_bands": N_BANDS, "n_steps": N_STEPS, "f_lo_hz": F_LO_HZ,
                    "f_hi_hz": F_HI_HZ, "fields": ["call_id", "x", "y", "cluster", "duration_s", "peak_freq_hz",
                                                   "snr_db", "date", "hour"]},
           "points": points, "clusters": clusters}
    (run / "call_map").mkdir(exist_ok=True)
    (run / "call_map" / "map.json").write_text(json.dumps(out))
    print(f"{len(clusters) - (-1 in labels)} clusters, {int((labels == -1).sum())} unclustered, "
          f"{time.time() - t0:.0f} s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", choices=("orig", "clean"), default="orig")
    ap.add_argument("--perplexity", type=float, default=50.0)
    ap.add_argument("--min-cluster", type=int, default=150, help="HDBSCAN min_cluster_size")
    a = ap.parse_args()
    build(Path(a.run), a.n, a.seed, a.version, a.perplexity, a.min_cluster)


if __name__ == "__main__":
    main()
