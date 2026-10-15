"""Verification gate: does our "talia" detector port reproduce her call_indicies.pkl?

This is the only available check that we understand what the published pipeline
actually did. Her detections for a file live in
``firstTrial/<folder>/<stem>/call_indicies.pkl`` as a DataFrame of
``high_sampled_start`` / ``high_sampled_end`` at 48 kHz. We re-run our port over the
same raw file and compare.

Faithfulness notes -- places we cannot be bit-exact, and why:
  * She called ``librosa.load`` (soxr_hq resampling); we use ``resample_poly``. The
    anti-aliasing filters differ slightly, so individual bins differ slightly.
  * She chunked the 22050 Hz signal into 1,000,000-sample subparts and processed each
    independently, so detections could not span a subpart boundary. We replicate that
    chunking here, because it changes the answer near boundaries.
  * Her mono mixdown averaged all 4 capsules; we use ``mean`` for this comparison so
    the channel strategy is not a confound.

Agreement is therefore judged by overlap matching, not by exact sample equality.

Usage:
    python tools/analysis/verify_talia_port.py <raw.wav> <call_indicies.pkl>
"""

from __future__ import annotations

import argparse
import pickle
from typing import List, Tuple

import numpy as np
import soundfile as sf

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.io.channels import combine
from marmoset_developmental_agent.preprocessing.filter_5khz import (
    compute_frame_activity, highpass_filter, resample_audio)
from marmoset_developmental_agent.preprocessing.segmentation import activity_to_calls

SUBPART = 1_000_000  # [talia] main.py:196 length_of_sub_audios, at the detection rate


def detect(path: str, cfg: PreprocessingConfig, strategy: str = "mean") -> List[Tuple[int, int]]:
    """Run the port over one raw file. Returns (start, end) pairs in native-rate samples."""
    info = sf.info(path)
    native_sr = info.samplerate
    ratio = native_sr / cfg.detection_sample_rate
    # One subpart of detection-rate audio at a time, mirroring her memory chunking.
    native_chunk = int(round(SUBPART * ratio))

    out: List[Tuple[int, int]] = []
    with sf.SoundFile(path) as f:
        start_native = 0
        while start_native < info.frames:
            f.seek(start_native)
            block = f.read(native_chunk, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            mono = combine(block, strategy, native_sr)[0]
            det = resample_audio(mono, native_sr, cfg.detection_sample_rate)
            filt = highpass_filter(det, cfg.detection_sample_rate,
                                   cfg.highpass_cutoff_hz, cfg.highpass_order)
            act = compute_frame_activity(filt, cfg.detection_sample_rate, cfg)
            for c in activity_to_calls(act, cfg):
                out.append((start_native + int(c.start_s * native_sr),
                            start_native + int(c.end_s * native_sr)))
            start_native += native_chunk
    return out


def load_reference(pkl: str) -> List[Tuple[int, int]]:
    df = pickle.load(open(pkl, "rb"))
    s = df["high_sampled_start"].astype(np.int64).values
    e = df["high_sampled_end"].astype(np.int64).values
    return sorted(zip(s.tolist(), e.tolist()))


def compare(ours: List[Tuple[int, int]], theirs: List[Tuple[int, int]], sr: int,
            min_iou: float = 0.3) -> dict:
    """Overlap-match our detections against hers. Not an accuracy metric -- neither
    set is ground truth; this measures whether we reproduced *her behaviour*."""
    def iou(a, b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union else 0.0

    matched_t = set()
    hits = 0
    for a in ours:
        best, bi = 0.0, -1
        for j, b in enumerate(theirs):
            v = iou(a, b)
            if v > best:
                best, bi = v, j
        if best >= min_iou and bi not in matched_t:
            matched_t.add(bi)
            hits += 1
    return {
        "n_ours": len(ours), "n_theirs": len(theirs), "matched": hits,
        "recall_of_hers": hits / len(theirs) if theirs else 0.0,
        "precision_vs_hers": hits / len(ours) if ours else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("pkl")
    ap.add_argument("--strategy", default="mean")
    ap.add_argument("--min-iou", type=float, default=0.3)
    a = ap.parse_args()

    cfg = PreprocessingConfig(frame_features="talia")
    sr = sf.info(a.wav).samplerate
    theirs = load_reference(a.pkl)
    print(f"reference: {len(theirs)} calls from {a.pkl}")
    ours = detect(a.wav, cfg, a.strategy)
    print(f"ours     : {len(ours)} calls (strategy={a.strategy})\n")

    r = compare(ours, theirs, sr, a.min_iou)
    for k, v in r.items():
        print(f"  {k:20s} {v:.3f}" if isinstance(v, float) else f"  {k:20s} {v}")

    print(f"\n{'#':>3} {'HERS (s)':>22} {'OURS (s)':>22}")
    for i in range(max(len(ours), len(theirs))):
        t = f"{theirs[i][0]/sr:8.2f}-{theirs[i][1]/sr:8.2f}" if i < len(theirs) else " " * 17
        o = f"{ours[i][0]/sr:8.2f}-{ours[i][1]/sr:8.2f}" if i < len(ours) else ""
        print(f"{i:3d} {t:>22} {o:>22}")
        if i > 40:
            print("    ... truncated")
            break


if __name__ == "__main__":
    main()
