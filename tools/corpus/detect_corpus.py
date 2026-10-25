"""Run the production front-end over the raw corpus and write the call catalogue.

One SLURM array task takes every ``n_tasks``-th recording of the manifest (interleaved, so
each task gets a similar mix of busy days and quiet nights) and, per recording, writes:

``calls/<recording_id>.csv``
    One row per detected call (``streaming.CallRecord``): position in the file, wall-clock
    time, duration and quality features.
``files/<recording_id>.json``
    Per-recording summary: duration, call and chunk counts, wall time. Written last and
    atomically, so its presence marks the recording done: a re-queued or re-submitted task
    skips it (the cluster preempts with REQUEUE).
``call_audio/<recording_id>.flac`` (only with ``--export-calls``)
    Every call of the recording, back to back, at ``call_audio_sample_rate`` (48 kHz), 24-bit
    lossless, each with ``call_audio_margin_s`` of context on both sides. The catalogue's
    ``audio_offset_s`` / ``audio_duration_s`` locate each call's audio in the file and
    ``audio_lead_s`` is how far before the call's ``start_s`` that audio begins. One file per
    recording, not per call: ~1.8 M small files would strain the shared file system, which
    allocates 2.3x the size of small files (SESSION_STATE, v2_day).
``chunks/<recording_id>/`` (only with ``--export-audio``)
    The 16 kHz training chunks as WAVs, plus ``chunks.tsv``. The GmSLM-style product,
    superseded on 2026-10-22 by individual calls; kept for the comparison arm.

A recording that fails is logged to ``errors/<recording_id>.txt`` and the task moves on.
Raw files are only read. The configuration is the production default unless ``--config``
overlays a YAML, and is saved once as ``run_config.yaml``.

Usage:
    python tools/corpus/detect_corpus.py MANIFEST OUT [--task-index I --n-tasks N]
        [--only PATHS.txt] [--config config/x.yaml] [--export-calls] [--export-audio]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig, load_config
from marmoset_developmental_agent.preprocessing.dataset_builder import RecordingFile, load_recordings_from_manifest
from marmoset_developmental_agent.preprocessing.segmentation import plan_chunks, recording_entirely_at_night
from marmoset_developmental_agent.preprocessing.streaming import (
    CallRecord, FileSource, detect_recording, read_call_audio, read_chunks)

CALL_FIELDS = ["start_s", "end_s", "duration_s", "start_time", "snr_db", "tonal_fraction",
               "peak_freq_hz", "peak_freq_iqr_hz", "confidence", "notes",
               "audio_offset_s", "audio_duration_s", "audio_lead_s"]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # unique: array tasks share run_config.yaml
    tmp.write_text(text)
    os.replace(tmp, path)


def _calls_csv(records: List[CallRecord], audio: Optional[List[Tuple[float, float, float]]] = None) -> str:
    rows = [",".join(CALL_FIELDS)]
    for k, r in enumerate(records):
        pack = ",".join(f"{v:.4f}" for v in audio[k]) if audio else ",,"
        rows.append(f"{r.start_s:.4f},{r.end_s:.4f},{r.duration_s:.4f},{r.start_time.isoformat(timespec='milliseconds')},"
                    f"{r.snr_db:.2f},{r.tonal_fraction:.3f},{r.peak_freq_hz:.1f},{r.peak_freq_iqr_hz:.1f},"
                    f"{r.confidence:.3f},{r.notes},{pack}")
    return "\n".join(rows) + "\n"


def _export_calls(out: Path, rec: RecordingFile, src: FileSource, records: List[CallRecord],
                  cfg: PreprocessingConfig) -> List[Tuple[float, float, float]]:
    """Write the recording's calls back to back as one FLAC; return each call's
    ``(audio_offset_s, audio_duration_s, audio_lead_s)`` in it."""
    import numpy as np
    import soundfile as sf

    sr = cfg.call_audio_sample_rate
    waves = read_call_audio(src, records, cfg)
    index, pos = [], 0
    for r, w in zip(records, waves):
        index.append((pos / sr, len(w) / sr, min(cfg.call_audio_margin_s, r.start_s)))
        pos += len(w)
    path = out / "call_audio" / f"{rec.recording_id}.flac"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.flac")
    data = np.concatenate(waves) if waves else np.zeros(0, np.float32)
    sf.write(tmp, np.clip(data, -1.0, 1.0), sr, subtype="PCM_24", format="FLAC")
    os.replace(tmp, path)
    return index


def _export_audio(out: Path, rec: RecordingFile, src: FileSource, plans: list, cfg: PreprocessingConfig) -> None:
    import soundfile as sf

    d = out / "chunks" / rec.recording_id
    d.mkdir(parents=True, exist_ok=True)
    lines = ["path\toffset_s\tstart_time\tcalls"]
    for k, c in enumerate(read_chunks(src, plans, cfg, rec.recording_id, rec.start_time)):
        sf.write(d / f"{k:05d}.wav", c.waveform, c.sample_rate, subtype="PCM_16")
        calls = ";".join(f"{x.start_s:.3f}-{x.end_s:.3f}" for x in c.calls)
        lines.append(f"{rec.recording_id}/{k:05d}.wav\t{c.offset_s:.4f}\t{c.start_time.isoformat()}\t{calls}")
    _atomic_write(d / "chunks.tsv", "\n".join(lines) + "\n")


def process(rec: RecordingFile, out: Path, cfg: PreprocessingConfig, export_audio: bool,
            export_calls: bool = False) -> Dict[str, object]:
    t0 = time.time()
    src = FileSource(rec.path)
    duration_s = src.frames / src.sample_rate
    night = cfg.exclude_night and recording_entirely_at_night(rec.start_time, duration_s, cfg)
    records = detect_recording(src, rec.start_time, cfg)
    plans = plan_chunks([r.segment() for r in records], cfg, duration_s)
    audio = _export_calls(out, rec, src, records, cfg) if export_calls else None
    _atomic_write(out / "calls" / f"{rec.recording_id}.csv", _calls_csv(records, audio))
    if export_audio:
        _export_audio(out, rec, src, plans, cfg)
    summary = {
        "recording_id": rec.recording_id, "path": rec.path, "start_time": rec.start_time.isoformat(),
        "duration_s": duration_s, "n_calls": len(records), "n_chunks": len(plans),
        "call_s": sum(r.duration_s for r in records), "night_skipped": night, "elapsed_s": time.time() - t0,
    }
    _atomic_write(out / "files" / f"{rec.recording_id}.json", json.dumps(summary))
    return summary


def _save_run_config(out: Path, cfg: PreprocessingConfig) -> None:
    path = out / "run_config.yaml"
    if path.exists():
        return
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    _atomic_write(path, yaml.safe_dump({"git_commit": commit, "preprocessing": dataclasses.asdict(cfg)},
                                       sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="recordings.csv from tools/build_manifest.py")
    ap.add_argument("out", help="output directory (under $WORK_DIR, never the raw data)")
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--only", help="text file of raw paths: process just these (pilot runs)")
    ap.add_argument("--config", help="YAML overlaid on the defaults (e.g. config/gmslm_arm.yaml)")
    ap.add_argument("--export-calls", action="store_true", help="also write each recording's calls as one 48 kHz FLAC")
    ap.add_argument("--export-audio", action="store_true", help="also write 16 kHz chunk WAVs (GmSLM arm)")
    a = ap.parse_args()

    cfg = load_config(a.config).preprocessing
    out = Path(a.out)
    recs = load_recordings_from_manifest(a.manifest)
    if a.only:
        wanted = {line.strip() for line in open(a.only) if line.strip()}
        recs = [r for r in recs if r.path in wanted]
        missing = wanted - {r.path for r in recs}
        if missing:
            print(f"warning: {len(missing)} --only paths are not in the manifest", file=sys.stderr)
    mine = recs[a.task_index :: a.n_tasks]
    _save_run_config(out, cfg)
    print(f"task {a.task_index}/{a.n_tasks}: {len(mine)} of {len(recs)} recordings -> {out}", flush=True)

    done = failed = 0
    for rec in mine:
        if (out / "files" / f"{rec.recording_id}.json").exists():
            done += 1
            continue
        try:
            s = process(rec, out, cfg, a.export_audio, a.export_calls)
            done += 1
            print(f"{rec.recording_id}: {s['duration_s'] / 60:.1f} min, {s['n_calls']} calls, "
                  f"{s['n_chunks']} chunks, {s['elapsed_s']:.0f} s", flush=True)
        except Exception:  # one bad file must not stop the other recordings of this task
            failed += 1
            _atomic_write(out / "errors" / f"{rec.recording_id}.txt", traceback.format_exc())
            print(f"{rec.recording_id}: FAILED, see errors/", flush=True)
    print(f"task {a.task_index}: {done} done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
