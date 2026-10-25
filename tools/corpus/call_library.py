"""Call library: browse, filter and listen to a corpus run's calls, original and cleaned, in a browser.

A small HTTP server over a run directory (``detect_corpus.py`` output, optionally cleaned by
``clean_corpus.py``). It reads ``calls/<id>.csv`` and the FLAC packs; nothing is copied.
The page (``call_library.html``) lists calls with filters (date, duration, confidence,
SNR, quality flag, cleaned only), plays each call as original and cleaned, draws both
spectrograms on one scale, and records verdicts.

Verdicts are appended to ``RUN/library_feedback.jsonl`` (one JSON object per click; the
last one per call wins), so Claude can read them from the run directory.

Every request must carry the token printed at start-up (``?t=...``): on the cluster the
server listens on a compute node that other users can reach.

Usage:
    python tools/corpus/call_library.py RUN [--port 8765] [--host 127.0.0.1] [--all]
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
SHORT_S, LOW_CONF = 0.18, 0.6  # merge_calls.quality


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
                c["quality"] = "short_low_conf" if c["duration_s"] < SHORT_S and c.get("confidence", 1) < LOW_CONF else "ok"
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
        if q.get("quality") in ("ok", "short_low_conf"):
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
        out: Dict[str, Dict] = {}
        if self.feedback_path.exists():
            for line in self.feedback_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("verdict") is None and not d.get("note"):
                    out.pop(d["call_id"], None)
                else:
                    out[d["call_id"]] = d
        return out

    def add_feedback(self, d: Dict) -> None:
        d = {"call_id": d["call_id"], "verdict": d.get("verdict"), "note": d.get("note", ""),
             "time": dt.datetime.now().isoformat(timespec="seconds")}
        with self.lock, open(self.feedback_path, "a") as f:
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


def _wav(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, 0.8 * x / (np.abs(x).max() + 1e-12), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


LIBS: List[Library] = []


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
            except (KeyError, FileNotFoundError) as e:
                self._send(f"not found: {e}".encode(), "text/plain", 404)

        def do_POST(self) -> None:
            if self._ok() is None:
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
    a = ap.parse_args()
    LIBS.append(Library(Path(a.run), a.all))
    token = a.token or secrets.token_urlsafe(12)
    lib = LIBS[0]
    print(f"{len(lib.calls)} calls from {len(set(c['recording_id'] for c in lib.calls))} recordings", flush=True)
    print(f"open: http://localhost:{a.port}/?t={token}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(0, token)).serve_forever()


if __name__ == "__main__":
    main()
