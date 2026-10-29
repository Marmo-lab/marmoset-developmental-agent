"""Serve the labelling clips to the phone app (``apps/labeler``) over the local Wi-Fi, and take its labels back.

The phone app (MarmoLabel) downloads every clip of every labelling set once, while on
Wi-Fi, and labels offline; when the user taps "send to computer" it posts each finished
clip's labels here. Labels are written in exactly the Audacity export format of
LABELING.md (``start<TAB>end<TAB>text``, ``none`` for a clip with no calls), into the
set's own ``labels/`` folder, so ``import_audacity.py`` reads phone and Audacity labels
alike. A label file that already exists and differs is first copied to
``labels/.bak/<clip>.<time>.txt``: nothing labelled in Audacity is ever lost.

Sets (``name=folder``; each folder holds ``clips.csv``, ``clips/<file>`` and ``labels/``):
calibration = ``labeling``, heldout = ``labeling/heldout``, test = ``labeling/test``.
The app never runs a detector, so serving the blind test set keeps TEST_PROTOCOL.md.

Every request carries the pairing code printed at start-up (``?t=...``); the phone
asks for it once. The APK itself is served at ``/app.apk`` for installing.

Usage:
    python tools/labels/phone_server.py [--port 8766] [--code 123456] [--set name=folder ...]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETS = {"calibration": ROOT / "labeling", "heldout": ROOT / "labeling" / "heldout",
                "test": ROOT / "labeling" / "test"}
APK = ROOT / "apps" / "labeler" / "build" / "marmolabel.apk"
TEXTS = {"c", "?", "n", "none"}  # LABELING.md; import_audacity.VALID plus the no-calls marker
MAX_BODY = 1 << 20


def manifest(sets: Dict[str, Path]) -> Dict[str, object]:
    """Every set's clips with their audio size and current labels (``None`` = not labelled yet)."""
    out: List[Dict[str, object]] = []
    for name, folder in sets.items():
        clips = []
        for r in csv.DictReader(open(folder / "clips.csv")):
            audio = folder / "clips" / r["file"]
            if not audio.exists():
                continue
            lab = folder / "labels" / f"{r['clip_id']}.txt"
            clips.append({"id": r["clip_id"], "wall_time": r["wall_time"], "duration_s": float(r["duration_s"]),
                          "bytes": audio.stat().st_size, "labels": lab.read_text() if lab.exists() else None})
        out.append({"name": name, "clips": clips})
    return {"sets": out}


def parse_labels(text: str) -> str:
    """Validate a label file from the phone and return it normalised; ``ValueError`` if malformed.

    The phone sends only finished clips, so an empty file is refused: in Audacity terms
    an empty file means "checked, no calls", which the app says explicitly with ``none``.
    """
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3 or parts[2] not in TEXTS:
            raise ValueError(f"bad label line: {line!r}")
        s, e = float(parts[0]), float(parts[1])
        if not 0 <= s <= e:
            raise ValueError(f"bad label times: {line!r}")
        lines.append(f"{s:.6f}\t{e:.6f}\t{parts[2]}")
    if not lines:
        raise ValueError("empty label file (use 'none' for a clip without calls)")
    return "\n".join(lines) + "\n"


def write_labels(folder: Path, clip_id: str, text: str) -> Optional[Path]:
    """Write ``labels/<clip_id>.txt``; return the backup path if a different file was replaced."""
    path = folder / "labels" / f"{clip_id}.txt"
    path.parent.mkdir(exist_ok=True)
    backup = None
    if path.exists() and path.read_text() != text:
        backup = path.parent / ".bak" / f"{clip_id}.{dt.datetime.now():%Y%m%d-%H%M%S}.txt"
        backup.parent.mkdir(exist_ok=True)
        backup.write_text(path.read_text())
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return backup


def lan_address() -> str:
    """The Mac's address on the Wi-Fi (the interface that routes outward); no packet is sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def make_handler(sets: Dict[str, Path], code: str, lock: threading.Lock) -> type:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            if "/api/labels" in self.path:
                print(f"{self.address_string()} {fmt % args}", flush=True)

        def _send(self, body: bytes, ctype: str, code_: int = 200, extra: Optional[Dict[str, str]] = None) -> None:
            self.send_response(code_)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            if secrets.compare_digest(q.get("t", ""), code):
                return True
            self._send("wrong or missing pairing code".encode(), "text/plain; charset=utf-8", 403)
            return False

        def _clip(self, set_name: str, clip_id: str) -> Optional[Dict[str, str]]:
            folder = sets.get(set_name)
            if folder is None:
                return None
            return next((r for r in csv.DictReader(open(folder / "clips.csv")) if r["clip_id"] == clip_id), None)

        def do_GET(self) -> None:
            if not self._authorised():
                return
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/")]
            if parts == ["api", "manifest"]:
                self._send(json.dumps(manifest(sets)).encode(), "application/json")
            elif len(parts) == 3 and parts[0] == "audio" and (row := self._clip(parts[1], parts[2])):
                self._send((sets[parts[1]] / "clips" / row["file"]).read_bytes(), "audio/wav")
            elif parts == ["app.apk"] and APK.exists():
                self._send(APK.read_bytes(), "application/vnd.android.package-archive",
                           extra={"Content-Disposition": 'attachment; filename="marmolabel.apk"'})
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:
            if not self._authorised():
                return
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/")]
            n = int(self.headers.get("Content-Length", 0))
            if len(parts) != 4 or parts[:2] != ["api", "labels"] or not self._clip(parts[2], parts[3]) or n > MAX_BODY:
                self._send(b"unknown clip", "text/plain", 404)
                return
            try:
                text = parse_labels(self.rfile.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                self._send(str(e).encode(), "text/plain; charset=utf-8", 400)
                return
            with lock:
                backup = write_labels(sets[parts[2]], parts[3], text)
            note = f" (previous file kept as {backup.name})" if backup else ""
            print(f"labels: {parts[2]}/{parts[3]}: {text.count(chr(10))} lines{note}", flush=True)
            self._send(json.dumps({"ok": True, "backup": backup is not None}).encode(), "application/json")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--code", help="pairing code (default: a random 6-digit one)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=FOLDER", help="replace the default sets")
    a = ap.parse_args()
    sets = {k: Path(v) for k, v in (s.split("=", 1) for s in a.set)} if a.set else DEFAULT_SETS
    code = a.code or f"{secrets.randbelow(10 ** 6):06d}"
    m = manifest(sets)
    for s in m["sets"]:
        done = sum(c["labels"] is not None for c in s["clips"])
        print(f"{s['name']}: {len(s['clips'])} clips, {done} labelled", flush=True)
    url = f"http://{lan_address()}:{a.port}"
    print(f"\nin the app:  address {url}   code {code}", flush=True)
    print(f"install APK: {url}/app.apk?t={code}" + ("" if APK.exists() else "  (not built yet)"), flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(sets, code, threading.Lock())).serve_forever()


if __name__ == "__main__":
    main()
