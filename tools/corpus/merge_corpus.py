"""Merge a corpus run's per-recording chunk lists into one chronological training manifest.

``detect_corpus.py --export-audio`` writes ``chunks/<recording_id>/chunks.tsv`` per
recording, in whatever order the array tasks ran. The agent reads one ``manifest.tsv`` plus
``day_sizes.json`` (``dataset_builder.load_manifest``, the format ``export_manifest``
writes). This step builds them in place, next to the chunk WAVs, without copying audio:

* rows are sorted by wall-clock start time (guardrail 5: the agent lives its days in order;
  ties keep recording order, so equal timestamps are deterministic);
* simulated days are cut as in ``build_chronological_dataset``: by the exposure schedule
  F(d) (``exposure``, Hasson ``generate_cycle_sizes.py``) or one per recording date
  (``calendar``), from the ``exposure`` section of ``--config``;
* duplicate copies are kept once. The shared drive holds the same file under two folders
  (``Rec_2021_07_14-17`` repeats all 116 files of ``Rec_2021_07_08-13``; ``210627_075`` is in
  ``Rec_2021_06_24-27`` and, cut short with zeros after ~10 min, in ``Rec_2021_06_29``).
  Recordings with the same file name and start time are one recording; the copy with the
  most detected calls is kept (a zero-filled copy loses calls), ties going to the
  earliest folder name;
* a recording without ``chunks.tsv`` (catalogue-only run, or unfinished) is an error,
  since silently dropping it would leave a hole in the chronology.

Usage:
    python tools/corpus/merge_corpus.py OUT [--config config/x.yaml]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

from marmoset_developmental_agent.config.agent_config import load_config
from marmoset_developmental_agent.preprocessing.dataset_builder import DailyExposureSchedule, _runs

MANIFEST_FIELDS = ["path", "n_samples", "day", "recording_id", "start_time", "offset_s", "calls"]


def deduplicate(files: List[Dict]) -> Tuple[List[str], List[str]]:
    """Recording ids to keep (sorted) and the duplicate copies dropped."""
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for f in files:
        groups.setdefault((Path(f["path"]).name, f["start_time"]), []).append(f)
    keep, dropped = [], []
    for g in groups.values():
        g.sort(key=lambda f: (-f["n_calls"], str(Path(f["path"]).parent)))
        keep.append(g[0]["recording_id"])
        dropped += [f["recording_id"] for f in g[1:]]
    return sorted(keep), sorted(dropped)


def merge(out: Path, config: str | None = None) -> Dict[str, object]:
    """Write ``out/manifest.tsv`` and ``out/day_sizes.json``; return a short summary."""
    cfg = load_config(config)
    pre, exp = cfg.preprocessing, cfg.exposure
    n_samples = int(round(pre.chunk_duration_s * pre.target_sample_rate))
    files = [json.loads(p.read_text()) for p in (out / "files").glob("*.json")]
    rids, dropped = deduplicate(files)
    missing = [r for r in rids if not (out / "chunks" / r / "chunks.tsv").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} recordings have no chunks.tsv (first: {missing[0]}); "
                                "was the run made with --export-audio, and is it finished?")
    rows: List[Dict[str, str]] = []
    for rid in rids:
        with open(out / "chunks" / rid / "chunks.tsv") as f:
            rows += [{**r, "recording_id": rid} for r in csv.DictReader(f, delimiter="\t")]
    if not rows:
        raise RuntimeError(f"no chunks under {out / 'chunks'}")
    times = [dt.datetime.fromisoformat(r["start_time"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: times[i])  # stable: ties keep recording order

    if exp.day_assignment == "exposure":
        sizes = DailyExposureSchedule(exp, pre.chunk_duration_s).day_sizes(len(rows))
    else:
        sizes = [len(g) for g in _runs([times[i].date() for i in order])]
    days = [d for d, n in enumerate(sizes) for _ in range(n)]

    tmp = out / "manifest.tsv.tmp"
    with open(tmp, "w") as f:
        f.write("\t".join(MANIFEST_FIELDS) + "\n")
        for day, i in zip(days, order):
            r = rows[i]
            f.write(f"chunks/{r['path']}\t{n_samples}\t{day}\t{r['recording_id']}\t{r['start_time']}\t"
                    f"{r['offset_s']}\t{r['calls']}\n")
    (out / "day_sizes.json").write_text(json.dumps(sizes))
    tmp.replace(out / "manifest.tsv")  # last: a manifest on disk always matches day_sizes.json
    return {"n_chunks": len(rows), "n_recordings": len(rids), "duplicates_dropped": dropped, "n_days": len(sizes),
            "first": times[order[0]].isoformat(), "last": times[order[-1]].isoformat(),
            "day_assignment": exp.day_assignment}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="corpus run directory (detect_corpus.py --export-audio)")
    ap.add_argument("--config", help="YAML whose exposure section sets the day assignment")
    a = ap.parse_args()
    s = merge(Path(a.out), a.config)
    print(f"{len(s['duplicates_dropped'])} duplicate copies dropped")
    print(f"{s['n_chunks']} chunks from {s['n_recordings']} recordings, {s['first']} .. {s['last']}; "
          f"{s['n_days']} simulated days ({s['day_assignment']}) -> {Path(a.out) / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
