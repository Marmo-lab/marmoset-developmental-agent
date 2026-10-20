"""Verification gate: does our "talia" detector port reproduce her call_indicies.pkl?

This is the only available check that we understand what the published pipeline
actually did. Her detections for a file live in
``firstTrial/<folder>/<stem>/call_indicies.pkl`` as a DataFrame of
``high_sampled_start`` / ``high_sampled_end`` at 48 kHz. We re-run our port over the
same raw file and compare.

Faithfulness notes -- places we cannot be bit-exact, and why:
  * Resampling matches (whole-file soxr_hq, as librosa.load does), but her librosa
    version is unknown, and the HPF uses second-order sections where she used (b, a)
    coefficients -- numerically tiny, but bins sitting exactly at the absolute
    threshold can flip.
  * She chunked the 22050 Hz signal into 1,000,000-sample subparts and processed each
    independently, so detections could not span a subpart boundary. We replicate that.
  * Her mono mixdown averaged all 4 capsules; we use ``mean`` for this comparison so
    the channel strategy is not a confound.

Agreement is therefore judged by overlap matching, not by exact sample equality.

Usage:
    python tools/analysis/verify_talia_port.py <raw.wav> <call_indicies.pkl>
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

from marmoset_developmental_agent.config.agent_config import PreprocessingConfig
from marmoset_developmental_agent.io.channels import combine
from marmoset_developmental_agent.preprocessing.filter_5khz import (
    compute_frame_activity, highpass_filter)
from marmoset_developmental_agent.preprocessing.segmentation import activity_to_calls

SUBPART = 1_000_000  # [talia] main.py:196 length_of_sub_audios, at the detection rate


def detect(path: str, cfg: PreprocessingConfig, strategy: str = "mean") -> List[Tuple[int, int]]:
    """Run the port over one raw file. Returns (start, end) pairs in native-rate samples.

    Mirrors her wav_processor: the whole file is mixed to mono and resampled to the
    detection rate in one pass with librosa's default soxr_hq (what ``librosa.load``
    does), *then* cut into 1,000,000-sample subparts that are each high-passed and
    segmented independently. Resampling per chunk instead would add filter edge
    effects at every subpart boundary that her pipeline never had.

    Peak memory is the mono signal: ~0.7 GB at 48 kHz for a 62-minute file.
    """
    import librosa

    info = sf.info(path)
    native_sr, det_sr = info.samplerate, cfg.detection_sample_rate
    parts = [combine(b, strategy, native_sr)[0]
             for b in sf.blocks(path, blocksize=native_sr * 60, dtype="float32", always_2d=True)]
    mono = librosa.resample(np.concatenate(parts), orig_sr=native_sr, target_sr=det_sr,
                            res_type="soxr_hq")
    del parts

    out: List[Tuple[int, int]] = []
    for k in range(0, len(mono), SUBPART):
        filt = highpass_filter(mono[k:k + SUBPART], det_sr, cfg.highpass_cutoff_hz, cfg.highpass_order)
        act = compute_frame_activity(filt, det_sr, cfg)
        for c in activity_to_calls(act, cfg):
            # her save_calls: (k + frame*512 +- 500) / 22050 * 48000, truncated
            out.append((int((k + c.start_s * det_sr) / det_sr * native_sr),
                        int((k + c.end_s * det_sr) / det_sr * native_sr)))
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


# Each variant turns off one stage that could reject or split a call, so the change in
# recall-of-hers localises where our port diverges from her pipeline.
ABLATIONS: Dict[str, Dict[str, float]] = {
    "full port": {},
    "no crossing rejection": {"max_bandwidth_crossings": 10**9},
    "no abrupt-up rejection": {"abrupt_up_freq_range_hz": float("inf")},
    "no shape rejections": {"max_bandwidth_crossings": 10**9,
                            "abrupt_up_freq_range_hz": float("inf")},
    "wider merge gap (x2)": {"merge_gap_s": 2 * PreprocessingConfig.talia().merge_gap_s},
}


def ablate(wav: str, theirs: List[Tuple[int, int]], sr: int, strategy: str,
           min_iou: float) -> None:
    """Re-run detection under each variant in ABLATIONS and tabulate agreement."""
    print(f"\n{'variant':<26}{'ours':>6}{'matched':>9}{'recall_hers':>13}{'prec_vs_hers':>14}")
    for name, overrides in ABLATIONS.items():
        cfg = dataclasses.replace(PreprocessingConfig.talia(), **overrides)
        r = compare(detect(wav, cfg, strategy), theirs, sr, min_iou)
        print(f"{name:<26}{r['n_ours']:>6}{r['matched']:>9}"
              f"{r['recall_of_hers']:>13.3f}{r['precision_vs_hers']:>14.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("pkl")
    ap.add_argument("--strategy", default="mean")
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--ablate", action="store_true",
                    help="also re-run with each rejection stage disabled")
    a = ap.parse_args()

    cfg = PreprocessingConfig.talia()
    sr = sf.info(a.wav).samplerate
    theirs = load_reference(a.pkl)
    print(f"reference: {len(theirs)} calls from {a.pkl}")
    ours = detect(a.wav, cfg, a.strategy)
    print(f"ours     : {len(ours)} calls (strategy={a.strategy})\n")

    r = compare(ours, theirs, sr, a.min_iou)
    for k, v in r.items():
        print(f"  {k:20s} {v:.3f}" if isinstance(v, float) else f"  {k:20s} {v}")

    print(f"\n{'HERS (s)':>20} {'dur':>6}   best-overlapping OURS (s)")
    for b in theirs:
        best, hit = 0.0, None
        for o in ours:
            inter = max(0, min(o[1], b[1]) - max(o[0], b[0]))
            union = max(o[1], b[1]) - min(o[0], b[0])
            if union and inter / union > best:
                best, hit = inter / union, o
        tag = f"{hit[0]/sr:8.2f}-{hit[1]/sr:8.2f}  IoU {best:.2f}" if hit else "-- missed --"
        print(f"{b[0]/sr:9.2f}-{b[1]/sr:9.2f} {(b[1]-b[0])/sr:6.2f}   {tag}")

    if a.ablate:
        ablate(a.wav, theirs, sr, a.strategy, a.min_iou)


if __name__ == "__main__":
    main()
