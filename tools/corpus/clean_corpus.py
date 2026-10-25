"""Spectrally clean the calls of a corpus run, writing a cleaned copy of each recording's call audio.

For every recording done in RUN (``files/<id>.json``), and every call in ``calls/<id>.csv``:

* the beam is chosen exactly as for the stored audio (``read_call_audio``): the loudest
  of the 8 virtual cardioids over the call plus its margin, so original and cleaned
  versions come from the same direction;
* that beam is read over ``context_s`` around the call, widening to ``wide_context_s``
  when fewer than ``min_noise_s`` of it lie outside every detected call (busy periods);
* ``clean_calls.clean_call`` cleans it, and the slice that the stored audio covers
  (``audio_lead_s`` before the call to the end of its margin) is kept.

Output, next to the run's own files and never touching them:

``call_audio_clean/<id>.flac``
    The cleaned calls, back to back, with exactly the original pack's lengths, so the
    catalogue's ``audio_offset_s`` / ``audio_duration_s`` / ``audio_lead_s`` index both.
``clean/<id>.json``
    Per-recording diagnostics, written last (the done marker): calls, median noise
    seconds per call, calls that needed the wide context, click frames inside calls.

Usage:
    python tools/corpus/clean_corpus.py RUN [--one-per-date] [--task-index I --n-tasks N] [--config x.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from marmoset_developmental_agent.config.agent_config import CleaningConfig, PreprocessingConfig, load_config
from marmoset_developmental_agent.io.channels import ambisonic_beams, band_energy
from marmoset_developmental_agent.preprocessing.clean_calls import calls_in_window, clean_call
from marmoset_developmental_agent.preprocessing.filter_5khz import resample_audio
from marmoset_developmental_agent.preprocessing.streaming import FileSource


def _beam_index(block: np.ndarray, sr: int) -> int:
    """The loudest beam over ``block``, as ``combine(..., "ambisonic_beam")`` chooses it."""
    return int(np.argmax(band_energy(ambisonic_beams(block.T), sr)))


def clean_recording(run: Path, rid: str, pre: PreprocessingConfig, cfg: CleaningConfig) -> Dict[str, object]:
    t0 = time.time()
    info = json.loads((run / "files" / f"{rid}.json").read_text())
    rows = list(csv.DictReader(open(run / "calls" / f"{rid}.csv")))
    src = FileSource(info["path"])
    sr = src.sample_rate
    if pre.training_channel_strategy != "ambisonic_beam" or src.channels != 4:
        raise SystemExit("clean_corpus expects 4-capsule audio and the ambisonic_beam training channel")
    calls = [(float(r["start_s"]), float(r["end_s"])) for r in rows]
    # Exact per-call sample counts of the original pack: the catalogue rounds times to 0.1 ms,
    # so cut the pack at its rounded offsets and take the last call to the pack's end.
    total = sf.info(run / "call_audio" / f"{rid}.flac").frames
    starts = [int(round(float(r["audio_offset_s"]) * pre.call_audio_sample_rate)) for r in rows]
    lengths = [b - a for a, b in zip(starts, starts[1:] + [total])]
    out, noise_s, wide, clicks = [], [], 0, 0
    for r, (a, b), n in zip(rows, calls, lengths):
        lead = float(r["audio_lead_s"])
        s0 = int(round((a - lead) * sr))
        n_src = int(round(n * sr / pre.call_audio_sample_rate))
        k = _beam_index(src.read(s0, n_src), sr)
        for ctx in (cfg.context_s, cfg.wide_context_s):
            lo = max(0.0, a - ctx)
            hi = min(src.frames / sr, b + ctx)
            x = ambisonic_beams(src.read(int(round(lo * sr)), int(round((hi - lo) * sr))).T)[k]
            y, d = clean_call(x, sr, (a - lo, b - lo), calls_in_window(calls, lo, hi), cfg)
            if d["noise_s"] >= cfg.min_noise_s or ctx == cfg.wide_context_s:
                break
        wide += ctx == cfg.wide_context_s
        noise_s.append(d["noise_s"])
        clicks += d["click_frames_in_call"]
        i0 = s0 - int(round(lo * sr))
        seg = resample_audio(y[i0:i0 + n_src], sr, pre.call_audio_sample_rate)
        out.append(np.pad(seg, (0, max(0, n - len(seg))))[:n])
    path = run / "call_audio_clean" / f"{rid}.flac"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.flac")
    data = np.concatenate(out) if out else np.zeros(0, np.float32)
    sf.write(tmp, np.clip(data, -1.0, 1.0), pre.call_audio_sample_rate, subtype="PCM_24", format="FLAC")
    os.replace(tmp, path)
    summary = {"recording_id": rid, "n_calls": len(rows), "median_noise_s": float(np.median(noise_s)) if noise_s else None,
               "wide_context_calls": wide, "click_frames_in_calls": clicks, "elapsed_s": time.time() - t0}
    marker = run / "clean" / f"{rid}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(summary))
    os.replace(tmp, marker)
    return summary


def select(run: Path, one_per_date: bool) -> List[str]:
    """Done recordings with calls; with ``one_per_date``, the one with the most calls per date."""
    files = [json.loads(p.read_text()) for p in (run / "files").glob("*.json")]
    files = [f for f in files if f["n_calls"] > 0]
    if not one_per_date:
        return sorted(f["recording_id"] for f in files)
    by_date: Dict[str, List[Dict]] = defaultdict(list)
    for f in files:
        by_date[f["start_time"][:10]].append(f)
    return sorted(max(fs, key=lambda f: f["n_calls"])["recording_id"] for fs in by_date.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--one-per-date", action="store_true")
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--config", help="YAML overlaid on the defaults (preprocessing / cleaning sections)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run = Path(a.run)
    rids = select(run, a.one_per_date)[a.task_index::a.n_tasks]
    print(f"task {a.task_index}/{a.n_tasks}: {len(rids)} recordings", flush=True)
    for rid in rids:
        if (run / "clean" / f"{rid}.json").exists():
            continue
        try:
            s = clean_recording(run, rid, cfg.preprocessing, cfg.cleaning)
            print(f"{rid}: {s['n_calls']} calls, {s['elapsed_s']:.0f} s", flush=True)
        except Exception as e:  # one bad recording must not stop the task
            (run / "errors").mkdir(exist_ok=True)
            (run / "errors" / f"clean_{rid}.txt").write_text(repr(e))
            print(f"{rid}: FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
