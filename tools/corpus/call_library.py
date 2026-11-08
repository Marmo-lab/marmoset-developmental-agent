"""Call library: browse, filter and listen to a corpus run's calls, original and cleaned, in a browser.

A small HTTP server over a run directory (``detect_corpus.py`` output, optionally cleaned by
``clean_corpus.py``). It reads ``calls/<id>.csv`` and the FLAC packs; nothing is copied.
The page (``call_library.html``) lists calls with filters (date, duration, confidence,
SNR, quality flag, cleaned only), plays each call as original and cleaned, draws both
spectrograms on one scale, and records verdicts.

``/map`` shows the call maps of ``tools/analysis/call_map.py`` (``RUN/call_map/map*.json``;
``?m=map_c3`` picks a sub-map): click a point to hear it; names given to a map's clusters go
to ``RUN/call_map/names.json`` for ``map``, ``names_<map>.json`` for the others.

``--donors DIR`` adds ``/donors``: the donor calls of ``tools/synth/donor_calls.py`` (whole clip,
proposed call bounds, the call alone), with verdicts in ``DIR/donor_feedback.jsonl``.
``--backgrounds DIR`` adds ``/backgrounds``: the night candidates of ``tools/synth/backgrounds.py``
(the hour's long-term spectrum; any 15 s of it, the 4-capsule mean, played from the raw file),
with verdicts in ``DIR/bg_feedback.jsonl``.
``--synth DIR`` adds ``/synth``: a set of synthetic recordings of ``tools/synth/generate.py``
(the minute, its spectrogram with the planted calls marked), with verdicts in ``DIR/synth_feedback.jsonl``.

``--short-high DIR`` adds ``/short_high``: candidates of ``tools/analysis/find_short_high.py`` (a 2 s
clip around each, its spectrogram, played at normal and half speed), with verdicts in
``DIR/short_high_feedback.jsonl``.

``--review DIR`` adds ``/review``: a generic listening review (e.g. ``tools/analysis/merge_review.py``):
``review.json`` (title, intro, answer options), ``items.tsv`` (``item_id``, any info columns,
``marks`` as JSON ``[[start_s, end_s, kind, label], ...]`` in clip seconds) and
``clips/<item_id>.flac``; verdicts in ``DIR/review_feedback.jsonl``. The server also runs
locally on an empty run folder, for a review of local files.

Verdicts are appended to ``RUN/library_feedback.jsonl`` (one JSON object per click; the
last one per call wins), so Claude can read them from the run directory.

Every request must carry the token printed at start-up (``?t=...``): on the cluster the
server listens on a compute node that other users can reach.

Usage:
    python tools/corpus/call_library.py RUN [--port 8765] [--host 127.0.0.1] [--all] [--donors DIR] [--backgrounds DIR] [--synth DIR] [--short-high DIR] [--review DIR]
    (--all: every recording, not only the cleaned ones)
On the cluster: tools/cluster/call_library.sbatch, then an SSH tunnel; see its header.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import random
import re
import secrets
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

HERE = Path(__file__).resolve().parent
NUMERIC = ("start_s", "duration_s", "snr_db", "confidence", "peak_freq_hz", "audio_offset_s", "audio_duration_s",
           "audio_lead_s")
SHORT_S, LOW_CONF, LONG_S = 0.18, 0.6, 8.0  # merge_calls.quality


class Library:
    def __init__(self, run: Path, include_all: bool) -> None:
        self.run = run
        cleaned = {p.stem for p in (run / "clean").glob("*.json")} if (run / "clean").exists() else set()
        rids = sorted(p.stem for p in (run / "files").glob("*.json")) if include_all else sorted(cleaned)
        self.calls: List[Dict] = []
        for rid in rids:
            path = run / "calls" / f"{rid}.csv"
            if not path.exists():
                continue
            for k, r in enumerate(csv.DictReader(open(path))):
                c = {f: float(r[f]) for f in NUMERIC if r.get(f) not in (None, "")}
                c.update(call_id=f"{rid}_{k:05d}", recording_id=rid, start_time=r["start_time"],
                         notes=r.get("notes", ""), cleaned=rid in cleaned)
                d = c["duration_s"]
                c["quality"] = ("short_low_conf" if d < SHORT_S and c.get("confidence", 1) < LOW_CONF
                                else "long" if d >= LONG_S else "ok")
                self.calls.append(c)
        self.calls.sort(key=lambda c: c["start_time"])
        self.by_id = {c["call_id"]: c for c in self.calls}
        self.feedback_path = run / "library_feedback.jsonl"
        self.lock = threading.Lock()

    def meta(self) -> Dict:
        dates: Dict[str, int] = {}
        for c in self.calls:
            dates[c["start_time"][:10]] = dates.get(c["start_time"][:10], 0) + 1
        return {"n_calls": len(self.calls), "n_cleaned": sum(c["cleaned"] for c in self.calls),
                "dates": [{"date": d, "n": n} for d, n in sorted(dates.items())]}

    def query(self, q: Dict[str, str]) -> Dict:
        def num(key: str, default: float) -> float:
            try:
                return float(q.get(key, default))
            except ValueError:
                return default
        rows = self.calls
        if q.get("date"):
            rows = [c for c in rows if c["start_time"].startswith(q["date"])]
        if q.get("hour"):
            rows = [c for c in rows if c["start_time"][11:13] == q["hour"].zfill(2)]
        lo, hi = num("min_dur", 0), num("max_dur", 1e9)
        rows = [c for c in rows if lo <= c["duration_s"] <= hi]
        rows = [c for c in rows if c.get("confidence", 1) >= num("min_conf", 0) and c.get("snr_db", 99) >= num("min_snr", -99)]
        if q.get("quality") in ("ok", "short_low_conf", "long"):
            rows = [c for c in rows if c["quality"] == q["quality"]]
        if q.get("cleaned") == "1":
            rows = [c for c in rows if c["cleaned"]]
        fb = self.feedback()
        if q.get("verdict") == "none":
            rows = [c for c in rows if c["call_id"] not in fb]
        elif q.get("verdict"):
            rows = [c for c in rows if fb.get(c["call_id"], {}).get("verdict") == q["verdict"]]
        sort = q.get("sort", "time")
        if sort == "random":
            rows = rows[:]
            random.Random(int(num("seed", 0))).shuffle(rows)
        elif sort in ("duration_s", "snr_db", "confidence"):
            rows = sorted(rows, key=lambda c: c.get(sort, 0), reverse=q.get("desc", "1") == "1")
        size = max(1, min(200, int(num("size", 30))))
        page = max(0, int(num("page", 0)))
        out = []
        for c in rows[page * size:(page + 1) * size]:
            out.append({**{k: c[k] for k in ("call_id", "recording_id", "start_time", "duration_s", "quality", "cleaned",
                                              "notes")},
                        **{k: c.get(k) for k in ("snr_db", "confidence", "peak_freq_hz")},
                        "feedback": fb.get(c["call_id"])})
        return {"total": len(rows), "page": page, "size": size, "rows": out}

    def audio(self, call_id: str, version: str) -> Tuple[np.ndarray, int]:
        c = self.by_id[call_id]
        folder = "call_audio_clean" if version == "clean" else "call_audio"
        path = self.run / folder / f"{c['recording_id']}.flac"
        if not path.exists():
            raise FileNotFoundError(f"{folder}/{c['recording_id']}.flac")
        sr = sf.info(path).samplerate
        with sf.SoundFile(path) as f:
            f.seek(int(round(c["audio_offset_s"] * sr)))
            return f.read(int(round(c["audio_duration_s"] * sr)), dtype="float32"), sr

    def feedback(self) -> Dict[str, Dict]:
        return _read_feedback(self.feedback_path, "call_id")

    def add_feedback(self, d: Dict) -> None:
        _append_feedback(self.feedback_path, self.lock, "call_id", d)


class Donors:
    """The donor calls of ``tools/synth/donor_calls.py``: ``DIR/donors.tsv`` and ``DIR/audio/<donor_id>.flac``."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.rows: List[Dict] = []
        for r in csv.DictReader(open(folder / "donors.tsv"), delimiter="\t"):
            self.rows.append({k: (v if k in ("donor_id", "call_type", "source") else float(v)) for k, v in r.items()})
        self.by_id = {r["donor_id"]: r for r in self.rows}
        self.feedback_path = folder / "donor_feedback.jsonl"
        self.lock = threading.Lock()

    def audio(self, donor_id: str, part: str) -> Tuple[np.ndarray, int]:
        """The whole clip, or (``part="call"``) the call alone between its bounds."""
        r = self.by_id[donor_id]
        x, sr = sf.read(self.folder / "audio" / f"{donor_id}.flac", dtype="float32")
        if part == "call":
            x = x[int(r["onset_s"] * sr):int(r["offset_s"] * sr)]
        return x, sr

    def feedback(self) -> Dict[str, Dict]:
        return _read_feedback(self.feedback_path, "donor_id")

    def add_feedback(self, d: Dict) -> None:
        _append_feedback(self.feedback_path, self.lock, "donor_id", d)


class Backgrounds:
    """The background candidates of ``tools/synth/backgrounds.py``: ``DIR/candidates.json`` and ``DIR/png/``."""

    MAX_S = 30.0

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        data = json.loads((folder / "candidates.json").read_text())
        self.info = {k: v for k, v in data.items() if k != "candidates"}
        self.rows = [{k: v for k, v in c.items() if k != "level_db"} for c in data["candidates"] if c["selected"]]
        self.by_id = {r["recording_id"]: r for r in self.rows}
        self.feedback_path = folder / "bg_feedback.jsonl"
        self.lock = threading.Lock()

    def audio(self, rid: str, start_s: float, dur_s: float) -> Tuple[np.ndarray, int]:
        """``dur_s`` (at most ``MAX_S``) of the raw recording from ``start_s``, as the 4-capsule mean."""
        path = self.by_id[rid]["path"]
        sr = sf.info(path).samplerate
        with sf.SoundFile(path) as f:
            f.seek(max(0, int(start_s * sr)))
            return f.read(int(min(dur_s, self.MAX_S) * sr), dtype="float32", always_2d=True).mean(axis=1), sr

    def feedback(self) -> Dict[str, Dict]:
        return _read_feedback(self.feedback_path, "recording_id")

    def add_feedback(self, d: Dict) -> None:
        _append_feedback(self.feedback_path, self.lock, "recording_id", d)


class SynthSet:
    """A synthetic set of ``tools/synth/generate.py``: ``recordings.tsv``, ``events.tsv``, ``audio/``."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.rows = list(csv.DictReader(open(folder / "recordings.tsv"), delimiter="\t"))
        self.by_id = {r["rec_id"]: r for r in self.rows}
        self.events: Dict[str, List[Dict]] = {r["rec_id"]: [] for r in self.rows}
        for e in csv.DictReader(open(folder / "events.tsv"), delimiter="\t"):
            self.events[e["rec_id"]].append(e)
        self.feedback_path = folder / "synth_feedback.jsonl"
        self.lock = threading.Lock()

    def audio(self, rec_id: str) -> Tuple[np.ndarray, int]:
        if rec_id not in self.by_id:  # only the set's own files, never a path from the URL
            raise KeyError(rec_id)
        x, sr = sf.read(self.folder / "audio" / f"{rec_id}.flac", dtype="float32")
        return x, sr

    def feedback(self) -> Dict[str, Dict]:
        return _read_feedback(self.feedback_path, "rec_id")

    def add_feedback(self, d: Dict) -> None:
        _append_feedback(self.feedback_path, self.lock, "rec_id", d)


class ShortHigh:
    """Candidates of ``tools/analysis/find_short_high.py``: ``candidates.tsv`` and ``clips/<cand_id>.flac``."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.rows = list(csv.DictReader(open(folder / "candidates.tsv"), delimiter="\t"))
        self.by_id = {r["cand_id"]: r for r in self.rows}
        summary = folder / "summary.json"
        self.summary = json.loads(summary.read_text()) if summary.exists() else {}
        self.feedback_path = folder / "short_high_feedback.jsonl"
        self.lock = threading.Lock()

    def audio(self, cand_id: str) -> Tuple[np.ndarray, int]:
        if cand_id not in self.by_id:  # only the candidates' clips, never a path from the URL
            raise KeyError(cand_id)
        x, sr = sf.read(self.folder / "clips" / f"{cand_id}.flac", dtype="float32")
        return x, sr

    def feedback(self) -> Dict[str, Dict]:
        return _read_feedback(self.feedback_path, "cand_id")

    def add_feedback(self, d: Dict) -> None:
        _append_feedback(self.feedback_path, self.lock, "cand_id", d)


class Review:
    """A generic review folder: ``review.json``, ``items.tsv``, ``clips/<item_id>.flac``."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.info = json.loads((folder / "review.json").read_text())
        self.rows = list(csv.DictReader(open(folder / "items.tsv"), delimiter="\t"))
        self.by_id = {r["item_id"]: r for r in self.rows}
        self.feedback_path = folder / "review_feedback.jsonl"
        self.lock = threading.Lock()

    def audio(self, item_id: str) -> Tuple[np.ndarray, int]:
        if item_id not in self.by_id:  # only the review's clips, never a path from the URL
            raise KeyError(item_id)
        x, sr = sf.read(self.folder / "clips" / f"{item_id}.flac", dtype="float32")
        return x, sr

    def feedback(self) -> Dict[str, Dict]:
        return _read_feedback(self.feedback_path, "item_id")

    def add_feedback(self, d: Dict) -> None:
        _append_feedback(self.feedback_path, self.lock, "item_id", d)


REVIEW_COLORS = {"label": "#4fc1b8", "a": "#b9b8b2", "b": "#3987e5"}


@lru_cache(maxsize=512)
def _review_png(item_id: str) -> bytes:
    """The clip, 0-24 kHz, each band in dB above its median; one strip of marks per kind, on top."""
    rv = REVIEW[0]
    x, sr = rv.audio(item_id)
    marks = json.loads(rv.by_id[item_id]["marks"])
    f, t, s = signal.stft(x, sr, nperseg=512, noverlap=432)
    p = 20 * np.log10(np.abs(s) + 1e-12)
    p -= np.median(p, axis=1, keepdims=True)
    fig = plt.figure(figsize=(7.5, 2.6), dpi=80)
    ax = fig.add_axes([0.05, 0.1, 0.94, 0.88])
    ax.imshow(p, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=30, extent=[0, len(x) / sr, 0, sr / 2000])
    kinds = list(dict.fromkeys(m[2] for m in marks))
    for a, b, kind, _ in marks:
        k = kinds.index(kind)
        ax.axvspan(a, b, ymin=0.97 - 0.045 * k, ymax=1.0 - 0.045 * k, color=REVIEW_COLORS.get(kind, "#e0a24a"))
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=6, colors="#555")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


@lru_cache(maxsize=512)
def _short_high_png(cand_id: str) -> bytes:
    """The 2 s clip, 0-24 kHz, each band in dB above its median; the candidate marked on top."""
    sh = SHORT_HIGH[0]
    x, sr = sh.audio(cand_id)
    r = sh.by_id[cand_id]
    f, t, s = signal.stft(x, sr, nperseg=512, noverlap=432)
    p = 20 * np.log10(np.abs(s) + 1e-12)
    p -= np.median(p, axis=1, keepdims=True)
    fig = plt.figure(figsize=(5.2, 2.2), dpi=80)
    ax = fig.add_axes([0.06, 0.11, 0.93, 0.86])
    ax.imshow(p, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=30, extent=[0, len(x) / sr, 0, sr / 2000])
    a = float(r["clip_offset_s"])
    ax.axvspan(a, a + float(r["duration_s"]), ymin=0.96, ymax=1.0, color="#4fc1b8")
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=6, colors="#555")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


@lru_cache(maxsize=256)
def _synth_png(rec_id: str, answers: bool) -> bytes:
    """The minute's spectrogram, 0-24 kHz; with ``answers``, each planted call shaded and named."""
    st = SYNTH[0]
    x, sr = st.audio(rec_id)
    f, t, s = signal.stft(x, sr, nperseg=1024, noverlap=544)
    p = 20 * np.log10(np.abs(s) + 1e-12)
    p -= np.median(p, axis=1, keepdims=True)
    fig = plt.figure(figsize=(14, 2.6), dpi=80)
    ax = fig.add_axes([0.03, 0.12, 0.965, 0.8])
    ax.imshow(p, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=30, extent=[0, len(x) / sr, 0, sr / 2000])
    if answers:
        for k, e in enumerate(st.events[rec_id]):
            a, b = float(e["onset_s"]), float(e["offset_s"])
            ax.axvspan(a, b, ymin=0.95, ymax=1.0, color="#4fc1b8")
            ax.text((a + b) / 2, 22.6 - 1.4 * (k % 2), e["call_type"], color="#e8fffd", fontsize=6.5, ha="center", va="top")
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xticks(range(0, 61, 5))
    ax.tick_params(labelsize=7, colors="#555")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _read_feedback(path: Path, key: str) -> Dict[str, Dict]:
    """The last verdict per item from a JSONL log; an empty verdict with no note clears it."""
    out: Dict[str, Dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("verdict") is None and not d.get("note"):
                out.pop(d[key], None)
            else:
                out[d[key]] = d
    return out


def _append_feedback(path: Path, lock: threading.Lock, key: str, d: Dict) -> None:
    d = {key: d[key], "verdict": d.get("verdict"), "note": d.get("note", ""),
         "time": dt.datetime.now().isoformat(timespec="seconds")}
    with lock, open(path, "a") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")


@lru_cache(maxsize=2048)
def _spectrogram_png(lib_id: int, call_id: str, version: str) -> bytes:
    lib = LIBS[lib_id]
    orig, sr = lib.audio(call_id, "orig")
    x = orig if version == "orig" else lib.audio(call_id, "clean")[0]
    ref = 20 * np.log10(np.abs(signal.stft(orig, sr, nperseg=1024, noverlap=784)[2]).max() + 1e-12)
    f, t, s = signal.stft(x, sr, nperseg=1024, noverlap=784)
    p = 20 * np.log10(np.abs(s) + 1e-12) - ref
    c = lib.by_id[call_id]
    fig = plt.figure(figsize=(4.4, 2.0), dpi=80)
    ax = fig.add_axes([0.07, 0.12, 0.92, 0.85])
    ax.imshow(p, origin="lower", aspect="auto", cmap="magma", vmin=-60, vmax=0, extent=[0, len(x) / sr, 0, sr / 2000])
    for edge in (c["audio_lead_s"], c["audio_lead_s"] + c["duration_s"]):
        ax.axvline(edge, color="#4fc1b8", lw=0.9)
    ax.set_yticks([0, 8, 16, 24])
    ax.tick_params(labelsize=6, colors="#555")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


@lru_cache(maxsize=1024)
def _donor_png(donor_id: str) -> bytes:
    """The whole clip's spectrogram, 0-24 kHz, with the proposed call bounds."""
    x, sr = DONORS[0].audio(donor_id, "clip")
    r = DONORS[0].by_id[donor_id]
    f, t, s = signal.stft(x, sr, nperseg=1024, noverlap=784)
    p = 20 * np.log10(np.abs(s) + 1e-12)
    fig = plt.figure(figsize=(5.2, 2.0), dpi=80)
    ax = fig.add_axes([0.06, 0.12, 0.93, 0.85])
    ax.imshow(p - p.max(), origin="lower", aspect="auto", cmap="magma", vmin=-70, vmax=0,
              extent=[0, len(x) / sr, 0, sr / 2000])
    for edge in (r["onset_s"], r["offset_s"]):
        ax.axvline(edge, color="#4fc1b8", lw=1.1)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=6, colors="#555")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _wav(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, 0.8 * x / (np.abs(x).max() + 1e-12), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


LIBS: List[Library] = []
DONORS: List[Donors] = []
BACKGROUNDS: List[Backgrounds] = []
SYNTH: List[SynthSet] = []
SHORT_HIGH: List[ShortHigh] = []
REVIEW: List[Review] = []
MAP_NAME = re.compile(r"map(_c\d+)*")


def _map_name(q: Dict[str, str]) -> str:
    """The ``m`` query value, checked: only ``map`` and its sub-maps (``map_c3``, ``map_c3_c1``) are served."""
    m = q.get("m", "map")
    if not MAP_NAME.fullmatch(m):
        raise KeyError(m)
    return m


def _names_path(run: Path, name: str) -> Path:
    return run / "call_map" / ("names.json" if name == "map" else f"names_{name}.json")


def _maps(run: Path) -> List[Dict]:
    """The run's maps: name, size and, for a sub-map, its parent and cluster (read from each file's head)."""
    out = []
    for p in sorted((run / "call_map").glob("map*.json")):
        if MAP_NAME.fullmatch(p.stem):
            with open(p) as f:
                head = f.read(4096)
            try:  # call_map writes {"meta": {...}, "points": ...}, meta first
                info = json.loads(head[head.index("{", 1):head.index(', "points"')])
            except ValueError:
                info = {}
            out.append({"name": p.stem, "n": info.get("n"), "parent": info.get("parent"), "cluster": info.get("cluster")})
    return out


def make_handler(lib_id: int, token: str):
    lib = LIBS[lib_id]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # quiet
            return

        def _send(self, body: bytes, ctype: str, code: int = 200, cache: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _ok(self) -> Optional[Dict[str, str]]:
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            if not secrets.compare_digest(q.get("t", ""), token):
                self._send(b"forbidden: open the URL printed by the server (it carries ?t=...)", "text/plain", 403)
                return None
            return q

        def do_GET(self) -> None:
            q = self._ok()
            if q is None:
                return
            path = urlparse(self.path).path.rstrip("/")
            try:
                if path in ("", "/index.html"):
                    self._send((HERE / "call_library.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/map":
                    self._send((HERE / "call_map.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/maps":
                    self._send(json.dumps(_maps(lib.run)).encode(), "application/json")
                elif path == "/api/map":
                    self._send((lib.run / "call_map" / f"{_map_name(q)}.json").read_bytes(), "application/json")
                elif path == "/api/map_names":
                    names = _names_path(lib.run, _map_name(q))
                    self._send(names.read_bytes() if names.exists() else b"{}", "application/json")
                elif path == "/donors":
                    self._send((HERE / "donors.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/donors":
                    fb = DONORS[0].feedback()
                    self._send(json.dumps([{**r, "feedback": fb.get(r["donor_id"])} for r in DONORS[0].rows]).encode(),
                               "application/json")
                elif path.startswith("/api/donor_audio/"):
                    x, sr = DONORS[0].audio(path.split("/")[-1], q.get("part", "clip"))
                    self._send(_wav(x, sr), "audio/wav", cache=True)
                elif path.startswith("/api/donor_spec/"):
                    self._send(_donor_png(path.split("/")[-1]), "image/png", cache=True)
                elif path == "/backgrounds":
                    self._send((HERE / "backgrounds.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/backgrounds":
                    bgs = BACKGROUNDS[0]
                    fb = bgs.feedback()
                    self._send(json.dumps({**bgs.info, "rows": [{**r, "feedback": fb.get(r["recording_id"])}
                                                                 for r in bgs.rows]}).encode(), "application/json")
                elif path.startswith("/api/bg_audio/"):
                    x, sr = BACKGROUNDS[0].audio(path.split("/")[-1], float(q.get("start", 0)), float(q.get("dur", 15)))
                    self._send(_wav(x, sr), "audio/wav", cache=True)
                elif path.startswith("/api/bg_png/"):
                    rid = path.split("/")[-1]
                    if rid not in BACKGROUNDS[0].by_id:  # only the candidates' images, never a path from the URL
                        raise KeyError(rid)
                    self._send((BACKGROUNDS[0].folder / "png" / f"{rid}.png").read_bytes(), "image/png", cache=True)
                elif path == "/synth":
                    self._send((HERE / "synth.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/synth":
                    st = SYNTH[0]
                    fb = st.feedback()
                    self._send(json.dumps([{**r, "events": st.events[r["rec_id"]], "feedback": fb.get(r["rec_id"])}
                                           for r in st.rows]).encode(), "application/json")
                elif path.startswith("/api/synth_audio/"):
                    x, sr = SYNTH[0].audio(path.split("/")[-1])
                    self._send(_wav(x, sr), "audio/wav", cache=True)
                elif path.startswith("/api/synth_spec/"):
                    self._send(_synth_png(path.split("/")[-1], q.get("answers", "1") == "1"), "image/png", cache=True)
                elif path == "/review":
                    self._send((HERE / "review.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/review":
                    rv = REVIEW[0]
                    fb = rv.feedback()
                    self._send(json.dumps({**rv.info, "colors": REVIEW_COLORS,
                                           "rows": [{**r, "feedback": fb.get(r["item_id"])} for r in rv.rows]},
                                          ensure_ascii=False).encode(), "application/json")
                elif path.startswith("/api/review_audio/"):
                    x, sr = REVIEW[0].audio(path.split("/")[-1])
                    self._send(_wav(x, sr), "audio/wav", cache=True)
                elif path.startswith("/api/review_spec/"):
                    self._send(_review_png(path.split("/")[-1]), "image/png", cache=True)
                elif path == "/short_high":
                    self._send((HERE / "short_high.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/short_high":
                    sh = SHORT_HIGH[0]
                    fb = sh.feedback()
                    self._send(json.dumps({"summary": sh.summary, "rows": [{**r, "feedback": fb.get(r["cand_id"])}
                                                                           for r in sh.rows]}).encode(), "application/json")
                elif path.startswith("/api/short_high_audio/"):
                    x, sr = SHORT_HIGH[0].audio(path.split("/")[-1])
                    self._send(_wav(x, sr), "audio/wav", cache=True)
                elif path.startswith("/api/short_high_spec/"):
                    self._send(_short_high_png(path.split("/")[-1]), "image/png", cache=True)
                elif path == "/api/meta":
                    self._send(json.dumps(lib.meta()).encode(), "application/json")
                elif path == "/api/calls":
                    self._send(json.dumps(lib.query(q)).encode(), "application/json")
                elif path.startswith("/api/audio/"):
                    x, sr = lib.audio(path.split("/")[-1], q.get("v", "orig"))
                    self._send(_wav(x, sr), "audio/wav", cache=True)
                elif path.startswith("/api/spec/"):
                    self._send(_spectrogram_png(lib_id, path.split("/")[-1], q.get("v", "orig")), "image/png", cache=True)
                else:
                    self._send(b"not found", "text/plain", 404)
            except (KeyError, IndexError, FileNotFoundError) as e:
                self._send(f"not found: {e}".encode(), "text/plain", 404)

        def do_POST(self) -> None:
            if self._ok() is None:
                return
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            if urlparse(self.path).path == "/api/map_names":
                names = {str(int(k)): str(v)[:80] for k, v in json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0)))).items()}
                with lib.lock:
                    _names_path(lib.run, _map_name(q)).write_text(json.dumps(names, ensure_ascii=False, indent=1))
                self._send(b'{"ok":true}', "application/json")
                return
            if urlparse(self.path).path == "/api/review_feedback" and REVIEW:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if d.get("item_id") not in REVIEW[0].by_id:
                    self._send(b"unknown item", "text/plain", 400)
                    return
                REVIEW[0].add_feedback(d)
                self._send(b'{"ok":true}', "application/json")
                return
            if urlparse(self.path).path == "/api/short_high_feedback" and SHORT_HIGH:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if d.get("cand_id") not in SHORT_HIGH[0].by_id:
                    self._send(b"unknown candidate", "text/plain", 400)
                    return
                SHORT_HIGH[0].add_feedback(d)
                self._send(b'{"ok":true}', "application/json")
                return
            if urlparse(self.path).path == "/api/synth_feedback" and SYNTH:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if d.get("rec_id") not in SYNTH[0].by_id:
                    self._send(b"unknown recording", "text/plain", 400)
                    return
                SYNTH[0].add_feedback(d)
                self._send(b'{"ok":true}', "application/json")
                return
            if urlparse(self.path).path == "/api/bg_feedback" and BACKGROUNDS:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if d.get("recording_id") not in BACKGROUNDS[0].by_id:
                    self._send(b"unknown recording", "text/plain", 400)
                    return
                BACKGROUNDS[0].add_feedback(d)
                self._send(b'{"ok":true}', "application/json")
                return
            if urlparse(self.path).path == "/api/donor_feedback" and DONORS:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if d.get("donor_id") not in DONORS[0].by_id:
                    self._send(b"unknown donor", "text/plain", 400)
                    return
                DONORS[0].add_feedback(d)
                self._send(b'{"ok":true}', "application/json")
                return
            if urlparse(self.path).path != "/api/feedback":
                self._send(b"not found", "text/plain", 404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            d = json.loads(body)
            if d.get("call_id") not in lib.by_id:
                self._send(b"unknown call", "text/plain", 400)
                return
            lib.add_feedback(d)
            self._send(b'{"ok":true}', "application/json")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--all", action="store_true", help="every recording, not only the cleaned ones")
    ap.add_argument("--token", help="fixed access token (default: random)")
    ap.add_argument("--donors", type=Path, help="donor-call folder of tools/synth/donor_calls.py, served at /donors")
    ap.add_argument("--backgrounds", type=Path, help="folder of tools/synth/backgrounds.py, served at /backgrounds")
    ap.add_argument("--synth", type=Path, help="a set of tools/synth/generate.py, served at /synth")
    ap.add_argument("--short-high", type=Path, help="folder of tools/analysis/find_short_high.py, served at /short_high")
    ap.add_argument("--review", type=Path, help="a review folder (review.json, items.tsv, clips/), served at /review")
    a = ap.parse_args()
    LIBS.append(Library(Path(a.run), a.all))
    if a.donors:
        DONORS.append(Donors(a.donors))
        print(f"{len(DONORS[0].rows)} donor calls at /donors", flush=True)
    if a.backgrounds:
        BACKGROUNDS.append(Backgrounds(a.backgrounds))
        print(f"{len(BACKGROUNDS[0].rows)} background candidates at /backgrounds", flush=True)
    if a.synth:
        SYNTH.append(SynthSet(a.synth))
        print(f"{len(SYNTH[0].rows)} synthetic recordings at /synth", flush=True)
    if a.short_high:
        SHORT_HIGH.append(ShortHigh(a.short_high))
        print(f"{len(SHORT_HIGH[0].rows)} short high candidates at /short_high", flush=True)
    if a.review:
        REVIEW.append(Review(a.review))
        print(f"{len(REVIEW[0].rows)} items to review at /review", flush=True)
    token = a.token or secrets.token_urlsafe(12)
    lib = LIBS[0]
    print(f"{len(lib.calls)} calls from {len(set(c['recording_id'] for c in lib.calls))} recordings", flush=True)
    print(f"open: http://localhost:{a.port}/?t={token}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(0, token)).serve_forever()


if __name__ == "__main__":
    main()
