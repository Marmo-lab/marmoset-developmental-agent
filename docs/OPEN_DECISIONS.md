# Open decisions

**How to use this file:** answer on the `Your answer:` line under each open question.
One line is enough. Writing **`default`** accepts my recommendation. Skip anything you
want to leave for later; nothing below blocks current work unless marked **[blocks]**.
Answered questions move to the *Decided* log at the bottom.

Background for every question: `docs/audit/GMSLM_CODE_AUDIT.md` and `SESSION_STATE.md`.

---

## Still open (5)

### Labelling and evaluation

**C12. How is a detection matched to a label?**
The paper never says, and the choice moves recall by roughly ±0.05, so it must be fixed
*before* we see results. Options: strict (overlap ≥ 50% of the union, "IoU ≥ 0.5"), or
lenient (the detection's midpoint falls inside a label).
*Default:* report both; use IoU ≥ 0.5 as the number decisions are based on.
Your answer:

**C10. How much to label in total?** *(deferred by you — revisit after the first batch)*
The first batch is 30 one-minute clips. After labelling those we'll know how long it
takes and whether more is needed.
Your answer:

### Experiment design

**D19. Vocoder — retrain it, or postpone?**
Talia's trained vocoder (turns units back into sound) was trained on units from a
different HuBERT layer than we'll use, so it will produce noise with ours. Retraining
costs ~2–4 GPU-days. It's only needed to *listen* to generated output, not to measure
anything.
*Default:* postpone, unless you need audible examples for a talk soon.
Your answer:

**D24. How many units? A data-driven prior from the call acoustics** *(raised 2026-10-19)*
Idea: estimate a "phoneme-like" inventory in the calls and size the encoder from it, by
analogy with the 1kd paper's units-vs-phonemes analysis. Note that analysis measures
*alignment* (purity P(phone|unit), P(unit|phone); `references/hasson_dev/analysis/`),
with a 512-entry codebook against ~39 English phones, i.e. ~13 units per phone, not a
matching count. First measurement (`tools/analysis/call_inventory.py`, 298 labelled
calls, `outputs/call_inventory.json`):
- calls: median 0.56 s (= 28 units at 50 Hz), dominant 6.7-10.3 kHz; 69% have >= 2 notes.
- **notes** (the phoneme analogue; median 0.09 s): BIC finds **~5-10 note types**.
- **20 ms frames** (the encoder's grain): ~20-30 spectral states inside calls, ~5 in
  background. Background is ~75% of day-clip time, so units are also spent on it.
- These are lower bounds (crude 10-D features, 12k frames; BIC under-fits with little data).
*Default:* keep the sweep around GmSLM's 50, widened at both ends to **k in {25, 50, 100,
200}** (+ DINO-SR's 512, reporting how many codewords are actually used). Choose k by
measured alignment, as in 1kd: unit purity against note types from this analysis (and
against PHEEDB call types, D18), units per note, effective vocabulary size (unigram
perplexity), the share of units spent on background, and normalised surprisal.
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
| C11 | Labelling conventions | As written in `LABELING.md`, plus: the labeller's ear decides (unlabelled = noise). 30/30 clips labelled. | 2026-10-19 |
| C13 | Optimise for misses or false alarms? | **Recall first.** False alarms inside a chunk that also holds a call cost nothing (the chunk keeps all its audio anyway); only chunks with *no* call dilute the data. Constraint: **chunk precision >= 0.85** (not event precision >= 0.95). Call-level evaluation sets use a stricter operating point or the hand labels. | 2026-10-19 |
| D18 | PHEEDB | Use for both: out-of-domain trajectory test set and labelled sanity checks. Uneven `left + right/2` mix treated as a bug; use an even mix (default, not objected to). | 2026-10-19 |
| E20 | GPUs / compute | Use whatever is free on `gpu.q` (A100s or RTX PRO 6000s); queues can be long, so plan for waiting and for preemption (save-and-resume). Any free CPU on the cluster is fair game. | 2026-10-19 |
| E21 | Experiment logging | wandb: account `zvimarmor` (entity `zvimarmor-the-hebrew-university-of-jerusalem`), logged in on the cluster; wandb 0.31.0 in the venv; host api.forge.coreweave.com (key in ~/.netrc, never in git). Login and compute nodes both reach it (checked 2026-10-19), so online mode works. Integration: optional `LoggingConfig` + thin logger, off in mock/tests. | 2026-10-19 |
| E22 | Who reads the code? | Just us for now; docs good enough for a lab member to follow. | 2026-10-19 |
| E23 | Deadline | None: steady progress. | 2026-10-19 |
