# Open decisions before Stage 1 implementation

Everything I need to know to start building. Answer inline under each `**A:**`.
Questions marked **[BLOCKING]** stop me from writing code at all; the rest I can work
around with a stated assumption if you'd rather not decide yet.

Context for why each question exists is in `gmslm_audit/GMSLM_CODE_AUDIT.md` and the
adjudication already merged into `config/agent_config.py`.

---

## A. Access and execution — how we actually run things

### A1. How do I execute work on the cluster? **[BLOCKING]**

This is the single most important question. Right now I'm running on your laptop
(`/Users/zvimarmor/Omer_Lab_Project`, macOS). The raw data is on ELSC storage
(`/ems/elsc-labs/omer-d/...`). I have no network access to the cluster from here.

So before anything else I need to know which of these we're doing:

1. **You are the executor.** I write code and `sbatch` scripts locally; you copy them
   over and run them, pasting back output. Slow round-trips, but zero setup.
2. **I have an SSH path.** There's a key/alias configured so I can run commands on the
   login node myself. Fastest, if it exists. (If you want this, you'd need to run the
   login yourself once — `! ssh ...` in the prompt puts the output in this session.)
3. **We work on a local data sample** for all of Stage 1 development, and only move to
   the cluster for the full run. This is my recommendation — see A2.

**A: ANSWERED (2026-10-14) — option 2, direct SSH.** `ssh elsc-login-direct` works while
Zvi is in the lab on lab wifi. Production environment: read-only everywhere except
`$HOME`. Falls back to "Zvi executes" when off-site.

Surveyed: login node is 2 cores / 11 GB free (never install or compute there);
`python/3.11.13` at `/ems/elsc-labs/apps/python/3.11.13` (needs `LD_LIBRARY_PATH`, see
`tools/cluster/env.sh`); CUDA 12.1-13.1; driver 535 (use a cu121 torch build);
42 TB free on `/ems/elsc-labs`.

SLURM: `gpu.q` walltime 6 days, holds **one** a100 node (4 cards) plus two
`rtxpro6000` nodes (15 cards, 96 GB each, Blackwell -- newer and less contended than
the A100s, benchmark before committing). `elscn.q` is the CPU partition (6 d).
`MaxArraySize = 1001`. **`PreemptMode = REQUEUE`** -- jobs can be preempted, so
checkpoint/resume is mandatory, not optional.

### A2. Has the ~9 GB data sample been copied to this laptop yet? **[BLOCKING]**

The audit wrote an rsync template at §5 but explicitly notes it was **not run**, and I
see no raw audio in the project folder. Without at least one real recording locally, I
cannot run the verification gate, cannot test the 4-channel reader against real
ambisonic files, and cannot calibrate anything — I'd be writing code blind.

The minimum useful sample is three items from the audit's plan:

| item | size | why |
|---|---|---|
| `AnimalRoomRecordings/Rec_2021_05_03/210503_001.WAV` | 2.05 GB | the only raw file with a complete matching processed output |
| `segmentations_results/firstTrial/Rec_2021_05_03/210503_001/**` | 74 MB | her `call_indicies.pkl` + logs + `online_plots/` — the verification target |
| `Rec_2021_06_28_ultrasonic.../M12_F14_210628103427_001.WAV` | 1.05 GB | the 250 kHz anomaly, to confirm our reader rejects it |

That's ~3.2 GB and unblocks all of Stage 1. The other items (P2.2, P2.3, PHEEDB) can
come later.

**A: ANSWERED (2026-10-14) — analyse in place on the cluster.** No bulk copy to the
laptop. Only small artifacts pulled locally for inspection (her `call_indicies.pkl`,
logs, `online_plots/` for `210503_001`) under `gmslm_sample/` (gitignored).

### A3. What is the Python environment on the cluster?

Locally we're on 3.11.8 with a CPU-only torch. For the cluster I need to know: is there
a `module load` system (and what are the module names), or do we build our own conda
env? Which CUDA version, and therefore which torch build? Is there internet access from
the compute nodes for `pip install`, or do we need to vendor wheels?

**A: ANSWERED (2026-10-14) — built and verified.** venv at `~/venvs/marmoset311`,
created by `tools/cluster/setup_env.sh` on a compute node. Verified: torch 2.14.1+cpu,
numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, soundfile 0.14.0, librosa, pyyaml,
pytest. CPU-only torch is deliberate -- Stage 1 needs no GPU; swap in a cu121 build
before training.

Three environment gotchas found, all now handled in `tools/cluster/`:
1. `module` is a shell function, absent from non-interactive shells -> set `PATH` and
   `LD_LIBRARY_PATH` directly (the interpreter will not even start without the latter).
2. `[ -f x ] && source x` as the **last line of a sourced file** returns 1 when the file
   is absent, aborting any `set -e` caller.
3. **`sbatch` copies the script to a spool dir**, so `$(dirname "$0")` does not resolve
   -- use `$SLURM_SUBMIT_DIR`. Every batch script needs this preamble.

### A4. Storage quota for our outputs?

The numbers nobody has budgeted, and they're large:

| artifact | size |
|---|---|
| 10 s chunk WAVs, 16 kHz, 216,660 segments | ~69 GB (int16) / ~138 GB (float32) |
| **encoder features cached for k-means** (108M frames × 768 dims × float32) | **~332 GB** |
| k-means subset only (50% of train+val) | ~130 GB |
| checkpoints over a 100 h run | tens of GB |

That feature cache is why Talia's `.npy` files totalled 9.1 TB. I need to know our quota
and where to write (scratch vs. project space, and whether scratch is purged) so I can
design the pipeline to shard-and-delete rather than discover the limit at 95% through a
job.

**A: ANSWERED (2026-10-14) — sufficient.** 42 TB free on `/ems/elsc-labs`, no visible
per-user quota. Our outputs go to `$WORK_DIR` (`~/marmoset_work`). Still shard-and-delete
the ~332 GB feature cache rather than materialising it all at once.

### A5. Should this project become a git repo?

It currently isn't (`CLAUDE.md` even notes what to do about the nested clones if it
becomes one). For a multi-month project with cluster runs I'd strongly recommend
`git init` plus a remote — it's how we'll keep laptop and cluster in sync, and how we'll
know which commit produced which results. Also need to decide: do `gmslm_audit/` and
`references/` go in, or stay local-only? (I'd gitignore both — they're large and
read-only.)

**A: ANSWERED (2026-10-14) — yes.** Remote:
`https://github.com/Zvimarmor/marmoset-developmental-agent.git`. `.gitignore` excludes
`references/`, `gmslm_audit/{index,code_mirror}/`, `outputs/`, `gmslm_sample/` and all
audio/checkpoint binaries; 45 files / 464 KB tracked. The audit report itself IS tracked.

Note: a stale pre-audit snapshot sits at `~/Omer_Lab_Project` on the cluster
(Sept 27-29, not a git repo). Left untouched; delete once the git workflow is live. The
live cluster checkout is `~/marmoset-developmental-agent`.

---

## B. The raw data

### B6. Channel strategy for the 4-capsule ambisonic recordings? **[BLOCKING]**

The most consequential question in Stage 1. Raw files are Zoom H3-VR: 48 kHz, 4 channels
(capsules tagged FLU/FRD/BLD/BRU), one device in one room. Talia's pipeline let
`librosa.load` silently average all four. Our own code does the same thing at
`dataset_builder.py:221` (`audio.mean(axis=1)`) — so this is currently a hardcoded
accident on both sides, not a decision.

Why it matters: averaging four capsules with inter-capsule phase differences causes
comb-filtering that attenuates directional sources (an actual marmoset call from one
side of the room) while summing uncorrelated noise. I believe this is the main reason
recall was 0.78, and it's our biggest improvement opportunity.

Candidate strategies:

| strategy | what it does |
|---|---|
| `mean` | current behaviour — average all 4. Needed as the baseline. |
| `max_energy` | per analysis window, take the capsule with most in-band energy |
| `best_snr` | pick one capsule per *file* by in-band SNR |
| `per_channel_or` | detect independently on all 4, union the detections |
| `ambisonic_w` | reconstruct the omnidirectional W component properly from the A-format capsules |

My recommendation: **implement all five as a swept config option.** The reader has to
handle 4 channels either way, so the marginal cost of each extra strategy is small, and
then the ablation against hand labels decides it empirically rather than by argument.

Do you agree, or do you want to commit to one on acoustic/biological grounds first?
(If you know how the recorder was physically positioned relative to the cages, that would
genuinely inform this — tell me if you do.)

**A: ANSWERED (2026-10-15) — implement all five and evaluate the improvement.** Four were implemented already (`mean`, `max_energy`, `best_snr`, `per_channel`). The fifth as originally phrased, ambisonic W, equals `mean` up to a constant (W = FLU+FRD+BLD+BRU), so the fifth is implemented as a genuinely different ambisonic option instead: `ambisonic_beam`, virtual cardioids steered around the horizon from the A->B-format conversion, choosing the loudest per block.

### B7. The ultrasonic folder — exclude, or handle separately?

`Rec_2021_06_28_ultrasonic_recordings_colony/` is a different instrument: 250 kHz, mono,
16-bit, no `bext` chunk, tracking two named individuals (M12/F14). It sits inside the
same `AnimalRoomRecordings/*` glob that Talia's array job (`--array=0-55`) iterated over
with a hardcoded 48 kHz assumption — so if it was processed, every sample index from it
is wrong by ~5.2×.

Three options: (a) exclude it entirely from the corpus; (b) process it through a separate
code path with correct rates; (c) exclude for now, decide later. I recommend (a) for the
reproduction and (c)'s flexibility for later — but it's a scientific call about whether
those two individuals' data matters to you.

Either way our reader will **hard-fail** on any file whose format doesn't match the
expected 48 kHz / 4 ch, rather than silently mis-processing it. That part isn't optional.

**A: ANSWERED (2026-10-15) — (a) exclude the ultrasonic folder.** Already enforced: `io.raw_reader.probe` rejects it (250 kHz, 1 ch); the manifest lists its 6 files under rejects.

### B8. Is the `firstTrial` vs `second_trial` question settled for you?

The audit concludes `firstTrial` + 10 s segments is final (its manifests total exactly
216,660 = the paper's figure, 80/10/10). But `second_trial/downsample` has 197,141 files
and a log shows 10-second paths under `second_trial` too — audit open question 6.

For our purposes I plan to **regenerate everything from raw** rather than consume either
tree, using `firstTrial`'s manifests only as a cross-check on segment counts. Confirm
that's what you want, or say if you'd rather we reuse her processed segments directly
(faster, but inherits every issue we just catalogued).

**A: ANSWERED (2026-10-15) — regenerate everything from raw**; her `firstTrial` manifests are a cross-check on counts only.

---

## C. Labelling — the blocking prerequisite

### C9. Who labels, and do we have labelling capacity? **[BLOCKING]**

The paper's precision 0.975 / recall 0.78 came from a hold-out set that **does not exist
anywhere** in Talia's tree (audit §3.2, exhaustively grepped). Until we have labels, any
change to preprocessing is unmeasurable — we literally cannot tell improvement from
regression. This paces the entire project, and it's human wall-clock time, so it should
start immediately and in parallel with my coding.

I need to know: is there someone in the lab who can reliably identify marmoset calls in a
spectrogram, or are we doing it ourselves? If the latter, I should build a simple
labelling tool (spectrogram view + click-drag to mark onset/offset + keyboard shortcuts,
writing a plain CSV). That's maybe a day of work and worth it.

**A: OPEN (2026-10-15).** Zvi can label but cannot tell call types apart. Proposal in chat: label onset/offset only (call vs not-call), no types; detector-proposed candidates plus blind windows to measure misses.

### C10. How much audio, and sampled how?

My proposal: **30–60 minutes total**, drawn as many short excerpts rather than one block
— e.g. 30 × 2-minute windows stratified across day/night, across the date range
(2021-05-03 → 2021-10-12), and across quiet/busy periods. Stratifying matters because a
single contiguous hour could be unrepresentatively quiet or noisy, and we'd calibrate to
an artifact.

Confirm the budget, or tell me if you can only get e.g. 15 minutes (workable, just
noisier estimates).

**A: DEFERRED (2026-10-15)** — revisit later.

### C11. What exactly counts as "a call" for labelling purposes?

This has to be pinned down *before* labelling starts, or the labels won't mean one thing.
Decisions needed:

- **Twitter calls** are sequences of syllables. One label spanning the whole twitter, or
  one label per syllable? (Our `merge_gap_s = 0.0929 s` bridges gaps up to ~93 ms, which
  implicitly treats closely-spaced syllables as one call — labels must match that
  convention or the metric is meaningless.)
- **Overlapping calls** from two animals: one label or two? Talia's pipeline has no
  overlap handling at all (audit §3.1 step 5).
- **Partial calls** clipped at a window boundary: label or skip?
- **Non-marmoset sounds** — cage noise, human speech, door slams: do we label these as
  explicit negatives? Very useful for diagnosing false positives if so.
- **Call types**: do we label type (phee/trill/twitter/tsik/ek/...) as well as
  onset/offset? Costs maybe 30% more labelling time and gives us per-type recall plus
  ground truth for the probe tasks. I think it's worth it, but it's your time budget.

**A:**

### C12. How is a detection matched to a label?

The metric definition, which the paper never states. This choice alone is worth roughly
±0.05 in recall, so it must be fixed and documented rather than chosen after seeing
results. Options:

- **IoU ≥ threshold** (e.g. 0.5) — standard, strict about boundaries
- **Midpoint inside label** — lenient, measures "did you notice the call" not "did you
  get the edges right"
- **Any overlap ≥ some fraction of the shorter segment** — what our code currently does
  (`detection_scores`, `min_overlap=0.5`)

And the edge cases: if one detection covers two labels (merge), or two detections split
one label, what counts as TP/FP/FN?

My recommendation: report **both** a lenient midpoint-based score (does the detector see
calls?) and a strict IoU ≥ 0.5 score (does it get boundaries right?), because they
diagnose different failures. Primary number for decisions: IoU ≥ 0.5.

**A:**

### C13. What's the optimisation target — precision or recall?

Talia got 0.975 / 0.78. Those aren't balanced; she was clearly tuned toward precision.
For a language model I'd argue the opposite: scattered false positives add noise the uLM
can average over, whereas losing 22% of all calls permanently removes structure it could
have learned. So I'd target **recall at fixed high precision** (e.g. maximise recall
subject to precision ≥ 0.95).

But this is a scientific judgement about your data and I may be wrong — if false
positives are mostly cage-rattle that looks call-like, they could be actively harmful.
Your call.

**A:**

---

## D. Experiment design

### D14. Which encoder is the primary path? **[BLOCKING for the training stage]**

Two options, with very different amounts of remaining work:

| | **DINO-SR** (trainable) | **Marmoset HuBERT** (frozen) |
|---|---|---|
| extra work | none — trains from scratch in our loop | recover her checkpoint, convert fairseq→HF, resolve the layer question |
| compute | part of the 80–100 h agent run | reuses ~1.1 GB of existing pretraining |
| open questions it inherits | none | audit Q3: layer 1 (paper) vs 6 (her launch script) vs 9 (vocoder name) vs 12 (downstream scripts) — four disagreeing signals |
| scientific fit | encoder learns *alongside* the decoder — the point of the developmental framing | faithful GmSLM reproduction |

My recommendation: **DINO-SR primary**, because it removes the largest unresolved
question from the critical path and is the developmentally honest choice. Keep HuBERT as
the GmSLM-comparison arm, and only spend effort on the conversion once DINO-SR runs.

Do you agree? And relatedly: **do you want to try to recover her HuBERT checkpoint at
all**, or treat GmSLM reproduction as out of scope and build only forward?

**A: ANSWERED (2026-10-15) — frozen HuBERT first.** First runs are plumbing / timing / sanity, so the cheaper frozen path wins. DINO-SR later.

### D15. Unit vocabulary sweep — which k, and decided by what metric?

You said the unit count is part of the research around Uri's paper. To run it I need the
grid and the decision rule. Proposal: **k ∈ {50, 100, 200}** (50 = GmSLM's value,
200 = the upper end GSLM found useful for English), decided by next-week surprisal on
held-out days plus the zero-shot structural benchmarks — *not* by training loss, which
falls trivially with larger k.

One caveat I want on record: because we don't deduplicate, a 10 s segment is 499 highly
autocorrelated units, so a large share of achievable loss reduction is just "the current
unit repeats." I'd like to also report a **run-length-corrected** surprisal so we don't
mistake duration modelling for sequence learning. Tell me if you want that as a primary
metric or a diagnostic.

**A: ANSWERED (2026-10-15) — sweep k in {50, 100, 200}, implemented in code.**

### D16. Developmental time-scale: what does one "simulated day" mean? **[BLOCKING for the training stage]**

Currently `day_assignment: "exposure"` with `days_per_period: 30` and an exposure
schedule rising 2 h/day → 8 h/day over 13 periods — i.e. Hasson's *human infant*
schedule, copied directly. `CLAUDE.md` already flags this as needing calibration:
marmoset development is much faster than human (vocal maturation over weeks-to-months,
not years).

Two sub-questions:

1. **Exposure or calendar mode?** In exposure mode the recordings are ordered by
   timestamp but a "day" is a fixed audio budget, so simulated days don't map to real
   developmental age. In calendar mode one real recording day = one simulated day, but
   our data is only ~55 recording days across 5.3 months with five multi-day gaps, giving
   a short lumpy trajectory. Exposure is the official 1kd behaviour and my default.
2. **What exposure curve for marmosets?** Do you have a developmental reference for how
   much vocal input an infant marmoset receives over its first months? If not, I'll keep
   Hasson's curve shape and compress the time-scale, documenting it as an unvalidated
   assumption.

Also note: **our recordings are of a colony, not of a single developing infant.** The
chronology is calendar time across the colony, not one animal's age. Is that the intended
interpretation, or is there per-individual data I don't know about?

**A: ANSWERED (2026-10-15) — keep the human exposure curve for now**; a marmoset curve comes later. `day_assignment: exposure` (current default) stays.

### D17. What's the actual deliverable, and in what order?

These are different projects sharing a codebase:

- (a) **Reproduce GmSLM** — validate we can hit the paper's numbers
- (b) **Build the developmental agent** — the novel Omer × Hasson contribution
- (c) **Fix preprocessing** — the thing you named as underperforming

(c) serves both. But (a) and (b) compete for GPU time and pull the encoder decision in
opposite directions (D14). Which is the priority, and is (a) a gate we must pass before
(b), or an optional comparison arm?

**A: ANSWERED (2026-10-15) — (c) fix preprocessing first, then (b) the developmental agent.** GmSLM reproduction (a) is a comparison arm, not a gate.

### D18. PHEEDB — use it, and how?

Three sessions, 96 kHz / 24-bit / stereo (not the paper's stated 16 kHz), ~51–72 min
each, 4 named marmosets, with per-call labels in `calls.csv`/`df_merged.csv`. Two uses:

- **Out-of-domain eval set** for the trajectory metric (our `ood_waveforms` hook exists
  and is unused) — cheap and valuable
- **Labelled call data** — it has real onset/offset labels! These are *not* a substitute
  for colony hold-out labels (different recording setup, different noise, close-miked
  rather than room ambisonic), but they're free labelled marmoset calls for sanity checks

Also: the audit flags an **asymmetric mono mixdown** in her PHEE scripts
(`left_norm + right_norm/2` — right channel halved, left not). Bug or intentional? If you
don't know, I'll treat it as a bug and use a symmetric mix.

**A:**

### D19. Vocoder — retrain, or defer?

Her final checkpoint (`hub_9_emb_1/g_00400000`, 400k steps) was trained on **layer-9**
units. Fed layer-1 or DINO-SR units it will produce noise. Options: retrain (400k steps,
~2–4 days GPU), or accept that we can't listen to generations yet and defer.

I'd **defer** — it's a presentation tool, not a measurement tool, and it costs GPU days
we need elsewhere. But if you need audible outputs for a talk or a lab meeting soon, say
so and I'll plan for it.

**A:**

---

## E. Infrastructure details

### E20. SLURM specifics beyond what you've given me

You gave me `gpu.q` / `a100:1` / 8 CPUs / 64 GB / 100 h. For preprocessing I also need:

- **The CPU partition name** — detection is pure CPU work, needs no GPU, and shouldn't
  queue behind GPU jobs. 55 raw folders is a natural array job.
- **Array job limits** — max concurrent tasks, and max array size.
- **Max walltime on the CPU partition** (one 62-min file through detection is maybe
  10–30 min of CPU, so per-file tasks are safe either way, but I need to know).
- **Is `gpu.q` preemptible?** This decides how aggressively I need checkpoint/requeue.
  A 100 h job on a preemptible partition without requeue is a coin flip.
- **Can we realistically get 4 A100s at once**, or is 1 the practical allocation? The
  difference is 20–25 h vs 80–100 h per run, and it changes whether DDP is worth building
  now or later.

**A:**

### E21. Experiment logging — wandb or files?

Talia used wandb (the audit found 5 run configs there, which is how we recovered the
vocoder's 400k steps — logging saved that stage from being as unrecoverable as the uLM).
Strong argument for using it. Do you have a wandb account/entity we should log to, or
should I write structured local logs plus a results JSON instead? Do compute nodes have
outbound internet for wandb to sync?

**A:**

### E22. Who else touches this code, and what are the review expectations?

Is this just us, or will Talia / other lab members read or run it? Affects how much I
invest in docs and CLI ergonomics versus raw speed. And: is there anyone I should assume
will take over the cluster-side work?

**A:**

### E23. Any deadline or milestone I should plan around?

A lab meeting, a submission, a committee review. Changes sequencing — e.g. if you need a
figure in three weeks, I'd prioritise differently than for a six-month build.

**A:**

---

## What I'll start on immediately once answered

Given answers to the four **[BLOCKING]** items (A1, A2, B6, C9) I can begin without
waiting for the rest:

1. **Streaming 4-channel reader** (`io/raw_reader.py`) — per-file, per-chunk, never
   loading 2.9 GB into RAM; `bext`/`iXML` timestamp extraction; hard-fail on unexpected
   formats. Replaces the `.mean(axis=1)` at `dataset_builder.py:221`.
2. **Channel strategies** (`io/channels.py`) — the five from B6 behind one config field.
3. **Manifest builder** — walks `AnimalRoomRecordings/*`, reads each `bext` start time,
   emits the `path,start_time,recording_id` CSV our pipeline already expects, skipping
   anything that fails format validation.
4. **Labelling tool**, if C9 says we're labelling ourselves.

All four are CPU-only and need no cluster GPU allocation. Items 1–3 are also exactly what
unblocks the verification gate against her `call_indicies.pkl`.
