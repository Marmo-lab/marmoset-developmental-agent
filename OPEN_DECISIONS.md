# Open decisions

**How to use this file:** answer on the `Your answer:` line under each open question.
One line is enough. Writing **`default`** accepts my recommendation. Skip anything you
want to leave for later; nothing below blocks current work unless marked **[blocks]**.
Answered questions move to the *Decided* log at the bottom.

Background for every question: `gmslm_audit/GMSLM_CODE_AUDIT.md` and `SESSION_STATE.md`.

---

## Still open (9)

### Labelling and evaluation

**C11. Labelling conventions — are the rules in `LABELING.md` OK?**
Before you start labelling, the conventions have to be fixed so the labels mean one
thing. I wrote defaults into `LABELING.md`: mark onset/offset only (no call types);
labels `c` = call, `?` = unsure, `n` = optional obvious non-call; one label over a
syllable sequence whose gaps are < ~0.1 s (matches the detector's merge gap); overlapping
callers = one label; calls cut by the clip edge still labelled.
*Default:* keep those rules as written.
Your answer:

**C12. How is a detection matched to a label?**
The paper never says, and the choice moves recall by roughly ±0.05, so it must be fixed
*before* we see results. Options: strict (overlap ≥ 50% of the union, "IoU ≥ 0.5"), or
lenient (the detection's midpoint falls inside a label).
*Default:* report both; use IoU ≥ 0.5 as the number decisions are based on.
Your answer:

**C13. What do we optimise the detector for — fewer false alarms or fewer misses?**
Talia's detector was tuned for few false alarms (precision 0.975) at the cost of missing
22% of calls (recall 0.78). For a language model, missed calls are lost structure, while
occasional false alarms are noise it can average over.
*Default:* maximise recall while keeping precision ≥ 0.95.
Your answer:

**C10. How much to label in total?** *(deferred by you — revisit after the first batch)*
The first batch is 30 one-minute clips. After labelling those we'll know how long it
takes and whether more is needed.
Your answer:

### Experiment design

**D18. PHEEDB — use it, and is the uneven stereo mix a bug?**
Three ~1 h sessions with 4 named marmosets and real per-call labels. Uses: (1) an
out-of-domain test set for the developmental trajectory (the code hook exists, unused);
(2) free labelled calls for sanity checks. Also, Talia's scripts mix it to mono as
`left + right/2` — the right channel is halved and the left is not.
*Default:* use it for both; treat the uneven mix as a bug and use an even mix.
Your answer:

**D19. Vocoder — retrain it, or postpone?**
Talia's trained vocoder (turns units back into sound) was trained on units from a
different HuBERT layer than we'll use, so it will produce noise with ours. Retraining
costs ~2–4 GPU-days. It's only needed to *listen* to generated output, not to measure
anything.
*Default:* postpone, unless you need audible examples for a talk soon.
Your answer:

### Infrastructure and planning

**E20. GPUs — which ones, and how many at once?**
Already found from the cluster itself: CPU work runs on `elscn.q`; `gpu.q` allows 6-day
jobs; arrays up to 1001 tasks; jobs **can be preempted** (so long runs need
save-and-resume, which I'll build). Still open: `gpu.q` has only **one** A100 node
(4 cards) but also **15 newer RTX PRO 6000 cards** (96 GB each), likely less contended.
Can we realistically get 4 GPUs at once? And is it OK to use the RTX PRO 6000s?
*Default:* use whatever is free, preferring RTX PRO 6000; plan for 1 GPU, add multi-GPU
later if we can actually get 4.
Your answer:

**E21. Experiment logging — Weights & Biases (wandb) or plain files?**
Talia's wandb logs are the only reason one of her training stages could be recovered at
all. Do you have a wandb account to log to?
*Default:* plain local log files + a results JSON per run now; add wandb if you have an account.
Your answer:

**E22. Who else will read or run this code?**
Just us, or Talia / other lab members too? Decides how much effort goes into docs and
ease of use versus speed.
*Default:* just us for now, with docs good enough for a lab member to follow.
Your answer:

**E23. Any deadline or milestone to plan around?**
A lab meeting, a talk, a submission, a committee review?
*Default:* none — plan for steady progress.
Your answer:

---

## Decided

| # | Question | Decision | Date |
|---|---|---|---|
| A1 | How do I run things on the cluster? | Direct SSH (`ssh elsc-login-direct`, lab network only). Production environment: write only inside your home dir. Falls back to you running commands when off-site. | 2026-10-14 |
| A2 | Copy data to the laptop? | No bulk copy — analyse on the cluster; only small artifacts come back. | 2026-10-14 |
| A3 | Python environment | venv at `~/venvs/marmoset311` built by `tools/cluster/setup_env.sh`; CPU torch for now, a CUDA build before training. Cluster quirks recorded in `SESSION_STATE.md`. | 2026-10-14 |
| A4 | Storage | Enough (42 TB free). Outputs in `~/marmoset_work`. | 2026-10-14 |
| A5 | Git | Yes — `github.com/Zvimarmor/marmoset-developmental-agent`. No Claude co-author lines in commits (history rewritten 2026-10-15). | 2026-10-14 |
| B6 | Combining the 4 microphone capsules | Implement all five strategies and evaluate. Done: `mean`, `max_energy`, `best_snr`, `per_channel`, `ambisonic_beam` (the planned "ambisonic W" equals `mean`, so a steered-beam option replaced it). Result: ~2 dB better daytime contrast than `mean`; provisional default `ambisonic_beam`. Final choice after labels. | 2026-10-15 |
| B7 | Ultrasonic folder | Exclude. Already enforced by the reader (rejected on format). | 2026-10-15 |
| B8 | Reuse Talia's segments or regenerate? | Regenerate everything from raw; her counts are only a cross-check. | 2026-10-15 |
| C9 | Who labels? | You, using Audacity, marking only start/end of calls (no call types). Package: 30 one-minute clips + picture guide; instructions in `LABELING.md`. | 2026-10-15 |
| D14 | Which encoder first? | Frozen HuBERT first (cheaper; first runs are for timing and sanity). DINO-SR later. | 2026-10-15 |
| D15 | Unit vocabulary sizes | Test k = 50, 100, 200. Implemented: `tools/sweep_units.py`. | 2026-10-15 |
| D16 | Exposure schedule | Human curve for now; marmoset curve later. | 2026-10-15 |
| D17 | Priority order | (c) fix preprocessing → (b) developmental agent. GmSLM reproduction is a comparison, not a requirement. | 2026-10-15 |
