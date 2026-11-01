"""Phee-database score: paths map to the cluster share, and each phee is counted on its caller's channel only."""

from __future__ import annotations

from tools.analysis.score_phee import _stats, cluster_path, score_exp


def test_windows_paths_map_to_the_cluster_share():
    p = r"Z:\Omer-Lab-Shared\data\experiments\phee2monkeys\exp_2021_02_23_001\Audio\ExpAudio.wav"
    assert cluster_path(p) == ("/ems/elsc-labs/omer-d/Omer-Lab-Shared/data/experiments/phee2monkeys/"
                               "exp_2021_02_23_001/Audio/ExpAudio.wav")


def test_phees_scored_on_their_own_channel_and_other_detections_sorted():
    x = {"duration_s": 60.0,
         "phees": [(1, 10.0, 11.0, "A"), (1, 20.0, 21.0, "A"), (2, 30.0, 31.0, "B")],
         "detections": {"1": [(10.02, 10.95, 0.9),   # phee 1: one-to-one
                              (30.0, 31.0, 0.9),     # B's phee heard on A's mic: crosstalk
                              (40.0, 40.8, 0.9),     # nothing listed, long
                              (45.0, 45.1, 0.5)],    # nothing listed, short_low_conf
                        "2": [(30.5, 30.7, 0.9)]}}   # B's phee, IoU < 0.5
    s = _stats([score_exp(x, drop_short_low_conf=False)])
    assert s["phees"] == 3
    assert s["R"] == 2 / 3 and s["R_iou"] == 1 / 3
    # 2 channel-minutes of audio
    assert s["unlisted_per_min"] == 1.0 and s["unlisted_long_per_min"] == 0.5
    assert s["crosstalk_per_min"] == 0.5
    assert round(s["onset_med_ms"]) == 260  # one-to-one onsets +20 and +500 ms
    dropped = _stats([score_exp(x, drop_short_low_conf=True)])
    assert dropped["unlisted_per_min"] == 0.5
