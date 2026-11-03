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

``--cluster C`` re-embeds one cluster of a map on its own (``map_cC.json``; nested maps
chain, ``map_c3_c2.json``), with HDBSCAN's ``leaf`` selection, to split a large mass.

Writes ``RUN/call_map/map.json`` (points, clusters, mean image per cluster) for the page
that ``tools/corpus/call_library.py`` serves at ``/map``, and caches the images
(``images_<version>.npy``) for ``--compare``.

``--compare`` draws the same sample with other embeddings next to the map, each coloured
by the map's clusters and by duration (``RUN/call_map/compare.png``): if a cloud is real,
it should hold together whatever the method. PCA alone is linear; t-SNE at other
perplexities shows how much of the picture is the setting; Isomap (on a subsample, it
is quadratic in memory) keeps global distances that t-SNE discards.

Usage (on the cluster, via tools/cluster/call_map.sbatch):
    python tools/analysis/call_map.py RUN [--n 30000] [--seed 0] [--version orig|clean]
    python tools/analysis/call_map.py RUN --cluster 3 [--parent map] [--min-cluster 60]
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


def cached_images(run: Path, rows: List[Dict[str, str]], version: str) -> np.ndarray:
    """``read_images``, kept in ``RUN/call_map/images_<version>.npy`` with the call ids beside it."""
    d = run / "call_map"
    d.mkdir(exist_ok=True)
    ids = [r["call_id"] for r in rows]
    img, idp = d / f"images_{version}.npy", d / f"images_{version}_ids.json"
    if img.exists() and idp.exists() and json.loads(idp.read_text()) == ids:
        return np.load(img)
    images = read_images(run, rows, version)
    np.save(img, images)
    idp.write_text(json.dumps(ids))
    return images


def compare(run: Path, isomap_n: int = 8_000) -> Path:
    """``compare.png``: the map's sample under PCA, t-SNE at three perplexities and Isomap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import Isomap

    m = json.loads((run / "call_map" / "map.json").read_text())
    meta, pts = m["meta"], m["points"]
    with open(run / "calls_train.tsv") as f:
        by_id = {r["call_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    rows = [by_id[p[0]] for p in pts]
    images = cached_images(run, rows, meta["version"])
    dur = np.array([p[4] for p in pts])
    labels = np.array([p[3] for p in pts])
    z = PCA(min(N_PCA, len(rows)), random_state=meta["seed"]).fit(features(images, dur))
    zz = z.transform(features(images, dur))
    rng = np.random.default_rng(meta["seed"])
    sub = np.sort(rng.choice(len(rows), min(isomap_n, len(rows)), replace=False))
    views: List[Tuple[str, np.ndarray, np.ndarray]] = [
        (f"PCA, components 1-2 ({100 * z.explained_variance_ratio_[:2].sum():.0f}% of variance)", np.arange(len(rows)),
         zz[:, :2]),
        (f"t-SNE perplexity {meta['perplexity']:g} (the map)", np.arange(len(rows)), np.array([p[1:3] for p in pts]))]
    for perp in (15.0, min(200.0, (len(rows) - 1) / 3)):
        t0 = time.time()
        xy = TSNE(2, perplexity=perp, init="pca", random_state=meta["seed"], n_jobs=-1).fit_transform(zz)
        views.append((f"t-SNE perplexity {perp:g}", np.arange(len(rows)), xy))
        print(f"t-SNE {perp:g}: {time.time() - t0:.0f} s", flush=True)
    views.append((f"Isomap, 15 neighbours ({len(sub):,} calls)", sub, Isomap(n_neighbors=15).fit_transform(zz[sub])))
    cmap = plt.get_cmap("tab10")
    col = np.array([(0.75, 0.77, 0.77, 1) if c < 0 else cmap(c % 10) for c in labels])
    fig, axs = plt.subplots(2, len(views), figsize=(4.2 * len(views), 8.4))
    for k, (title, idx, xy) in enumerate(views):
        for row, (c, cm) in enumerate(((col[idx], None), (np.log10(dur[idx]), "viridis"))):
            ax = axs[row, k]
            ax.scatter(xy[:, 0], xy[:, 1], s=0.6, c=c, cmap=cm, vmin=-1 if cm else None, vmax=1 if cm else None,
                       linewidths=0, rasterized=True)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(title if row == 0 else "same, colour = log10 duration (0.1-10 s)", fontsize=9)
    fig.suptitle("Colour (top) = the map's HDBSCAN clusters, grey = unclustered", fontsize=10)
    fig.tight_layout()
    out = run / "call_map" / "compare.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def embed(x: np.ndarray, seed: int, perplexity: float, min_cluster: int,
          selection: str = "eom") -> Tuple[np.ndarray, np.ndarray]:
    """``(2-D map (N, 2), cluster labels (N,), -1 = unclustered)``.

    ``selection`` is HDBSCAN's ``cluster_selection_method``: ``eom`` keeps the most stable
    clusters of the tree, so one large stable mass stays whole; ``leaf`` takes the tree's
    leaves, the finest clusters it supports, which is what splits such a mass.
    """
    z = PCA(min(N_PCA, *x.shape), random_state=seed).fit_transform(x)
    xy = TSNE(2, perplexity=min(perplexity, (len(x) - 1) / 3), init="pca", random_state=seed,
              n_jobs=-1).fit_transform(z)
    hdb = HDBSCAN(min_cluster_size=min_cluster, cluster_selection_method=selection, copy=True)
    return xy.astype(np.float32), hdb.fit_predict(xy)


def _write_map(run: Path, name: str, info: List[list], images: np.ndarray, xy: np.ndarray, labels: np.ndarray,
               meta: Dict[str, object]) -> Dict[str, object]:
    """``RUN/call_map/<name>.json``; ``info`` rows are ``[call_id, duration_s, peak_freq_hz, snr_db, date, hour]``."""
    dur = np.array([i[1] for i in info])
    points = [[i[0], round(float(a), 2), round(float(b), 2), int(c), *i[1:]] for i, (a, b), c in zip(info, xy, labels)]
    clusters = []
    for c in sorted(set(labels.tolist())):
        m = labels == c
        clusters.append({"id": int(c), "n": int(m.sum()), "median_dur_s": round(float(np.median(dur[m])), 3),
                         "mean_image": np.round(images[m].mean(0), 2).tolist()})
    out = {"meta": {**meta, "name": name, "n": len(info), "n_bands": N_BANDS, "n_steps": N_STEPS, "f_lo_hz": F_LO_HZ,
                    "f_hi_hz": F_HI_HZ, "fields": ["call_id", "x", "y", "cluster", "duration_s", "peak_freq_hz",
                                                   "snr_db", "date", "hour"]},
           "points": points, "clusters": clusters}
    (run / "call_map").mkdir(exist_ok=True)
    (run / "call_map" / f"{name}.json").write_text(json.dumps(out))
    print(f"{name}: {len(clusters) - (-1 in labels)} clusters, {int((labels == -1).sum())} unclustered", flush=True)
    return out


def build(run: Path, n: int, seed: int, version: str, perplexity: float, min_cluster: int,
          selection: str = "eom") -> Dict[str, object]:
    """The map of a fresh sample: ``RUN/call_map/map.json``."""
    t0 = time.time()
    rows = sample_rows(run, n, seed)
    print(f"{len(rows)} calls sampled", flush=True)
    images = cached_images(run, rows, version)
    dur = np.array([float(r["duration_s"]) for r in rows])
    xy, labels = embed(features(images, dur), seed, perplexity, min_cluster, selection)
    num = lambda r, k: round(float(r[k]), 1) if r.get(k) not in (None, "", "nan") else None  # noqa: E731
    info = [[r["call_id"], round(float(r["duration_s"]), 3), num(r, "peak_freq_hz"), num(r, "snr_db"),
             r["start_time"][:10], int(r["start_time"][11:13])] for r in rows]
    out = _write_map(run, "map", info, images, xy, labels, {"seed": seed, "version": version, "perplexity": perplexity,
                                                             "min_cluster": min_cluster, "selection": selection})
    print(f"{time.time() - t0:.0f} s", flush=True)
    return out


def submap(run: Path, parent: str, cluster: int, perplexity: float, min_cluster: int,
           selection: str = "leaf") -> Dict[str, object]:
    """A map of one cluster of ``parent`` alone, re-embedded: ``RUN/call_map/<parent>_c<cluster>.json``.

    In the full map a cluster shares the plane with every other call; on its own, its
    internal differences get the whole plane (PCA and t-SNE are refit on its calls only).
    The images come from the cache that ``build`` wrote, so no audio is read.
    """
    m = json.loads((run / "call_map" / f"{parent}.json").read_text())
    meta = m["meta"]
    pts = [p for p in m["points"] if p[3] == cluster]
    if not pts:
        raise ValueError(f"{parent} has no cluster {cluster}")
    version = meta["version"]
    ids = json.loads((run / "call_map" / f"images_{version}_ids.json").read_text())
    pos = {c: i for i, c in enumerate(ids)}
    images = np.load(run / "call_map" / f"images_{version}.npy")[[pos[p[0]] for p in pts]]
    dur = np.array([p[4] for p in pts])
    xy, labels = embed(features(images, dur), meta["seed"], perplexity, min_cluster, selection)
    return _write_map(run, f"{parent}_c{cluster}", [[p[0], *p[4:]] for p in pts], images, xy, labels,
                      {"seed": meta["seed"], "version": version, "perplexity": perplexity, "min_cluster": min_cluster,
                       "selection": selection, "parent": parent, "cluster": cluster})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", choices=("orig", "clean"), default="orig")
    ap.add_argument("--perplexity", type=float, default=50.0)
    ap.add_argument("--min-cluster", type=int, default=150, help="HDBSCAN min_cluster_size")
    ap.add_argument("--selection", choices=("eom", "leaf"), help="HDBSCAN cluster selection (default eom; leaf for --cluster)")
    ap.add_argument("--compare", action="store_true", help="draw other embeddings of map.json's sample instead")
    ap.add_argument("--parent", default="map", help="with --cluster: the map to take the cluster from")
    ap.add_argument("--cluster", type=int, help="re-embed this cluster of --parent alone (a sub-map)")
    a = ap.parse_args()
    if a.compare:
        compare(Path(a.run))
    elif a.cluster is not None:
        submap(Path(a.run), a.parent, a.cluster, a.perplexity, a.min_cluster, a.selection or "leaf")
    else:
        build(Path(a.run), a.n, a.seed, a.version, a.perplexity, a.min_cluster, a.selection or "eom")


if __name__ == "__main__":
    main()
