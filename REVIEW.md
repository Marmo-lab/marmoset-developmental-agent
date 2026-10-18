# REVIEW.md: Code review standard

This applies to every review: of a human's change, of Claude's own change, and of a diff before it is committed. The goal is to catch what matters and spend as little reviewer time as possible on anything else.

## 1. What counts as a failure (blocking)
A change fails review if it does any of the following:

| Category | Concrete failures |
|---|---|
| **Guardrails** (`CLAUDE.md`) | Collapses repeated units on a path to the uLM or the evaluations. Shuffles across days. Reads config from a raw dict, or adds a setting outside `agent_config.py`. Imports another stage's internals. Leaves a public function untyped. Adds a feature without a mock path or a test. |
| **Science / provenance** | Changes a constant or formula without a `[paper]` / `[talia]` / `[ours]` tag or a cited source. Stores a duration in frames instead of seconds. Mixes STFT scalings. Changes detector behaviour so the Talia port gate drops below 23/23 at IoU 1.00. Silently changes what a reported metric means. |
| **Cluster safety** | A script writes outside `$HOME` / `WORK_DIR`. A script deletes, moves or modifies shared or raw data. Heavy compute runs on the login node. A path in a shell or sbatch script is unquoted or unvalidated. |
| **Tests / mock** | `pytest tests/` fails. `ruff check .` is not clean. `run_pipeline.py --mock` fails or takes longer than ~2 min. A test is weakened or deleted to make a change pass. |
| **Security** | Secrets, tokens or personal absolute paths are committed. `pickle` / `torch.load` is used on files from an untrusted source. `subprocess` / `shell=True` is built from unvalidated input. A Claude co-author trailer appears in a commit. |

Security findings are always **Important** and always blocking.

## 2. How to review
1. **The core first.** Does the change solve the stated problem at its root cause, not just the symptom? Does it fit the stage architecture? What else does it break: callers, configs, YAML, tests, cluster scripts?
2. **A security pass** against the Security row above, on every review.
3. **Verify, don't guess.** Run the commands from `CLAUDE.md` where possible, and report what was run and what it printed.

## 3. Findings format
Each finding has:
- **Tag:** `Important` (must fix before merge), `Nit` (optional), or `Pre-existing` (a real bug the change did not introduce).
- **Evidence:** the exact `path/to/file.py:LINE`, and the code or command output that shows the problem. No finding without a file and line. If you can't point to the line, drop the finding.
- **Failure scenario:** the concrete input or state that leads to the wrong output, crash or unsafe action.
- **Fix:** one sentence. A patch only if it is short.

Rules:
- **At most 3 Nits per review.** Keep only the ones that affect readability of scientific or numerical logic, and drop the rest silently.
- **Drop purely stylistic findings** (naming taste, formatting, comment wording, line length) unless they violate something written in `CLAUDE.md`.
- **Drop diagnostic or cleanup suggestions** that are unrelated to the change's purpose.
- **Pre-existing bugs** are listed separately, at the end, and never counted against the change's author. Report one only if it is Important.

## 4. Review rounds
- **Cap: 3 rounds** per change, counting the first review.
- After round 1, review **only the new diff** and check whether each open finding is resolved. Don't re-review untouched code.
- If an Important finding is still open after round 3, stop. Escalate to the human with a short note: what keeps failing, why patching hasn't worked, and the architectural question to reconsider.

## 5. Verdict
End every review with one line: **Approve**, **Approve with nits**, or **Request changes (N Important)**.
