# Explorations

One-off investigations. Each answered a question once; the answer is recorded in `docs/`, and the script is kept so the
result can be traced and rerun. None of them is on the production path. Cluster jobs for them are in `cluster/`.

| Script | Question | Result in |
|---|---|---|
| `level_audit.py`, `aggregate_levels.py` | Where does Talia's absolute STFT threshold fall in our recordings? | `DETECTOR_EVAL.md` |
| `channel_eval.py` | Which combination of the four capsules should detection run on? | `DETECTOR_EVAL.md` |
| `listen_phee_edges.py` | Phee offsets: detector or labels? (a listening page) | `DETECTOR_EVAL.md`, finding 15 |
| `synth_shift_probe.py` | Is the detector blind to high frequency, or to short high calls themselves? | `SYNTH_EVAL.md` |
| `find_short_high.py` | Does the colony make short high calls that the detector misses? Also the real-minute training share of v2. | `SYNTH_EVAL.md` |
| `merge_review.py` | Which merges does the retrained detector make? (a listening page) | `SYNTH_EVAL.md` |
| `find_noise_events.py` | Loud non-call events in the colony, for the user to label (input to `tools/synth/plant_noise.py`) | `SYNTH_EVAL.md` |
| `file_edges.py` | Are detections at recording-file edges an artifact? (No: calls cut by the file split.) | `DETECTOR_V2A_SCORES.md`, section 4 |
| `figure_candidates.py`, `repertoire_figure.py` | The repertoire figure (`figures/repertoire_v5.png`) | `figures/README.md` |
| `donor_check.py` | Are the shortest synthetic donors and the trills really calls with the right edges? (a listening page) | every one is a call, but 14 trill donors and `seep_000` have wrong edges (8 are 10 ms clicks from `call_bounds`' loudest-frame fallback); fix pending, `SYNTH_METHODS.md` to follow |
