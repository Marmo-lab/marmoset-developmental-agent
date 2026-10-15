"""Unit-vocabulary sweep: run the pipeline once per k-means size and compare.

OPEN_DECISIONS D15: k in {50, 100, 200} (``HuBERTEncoderConfig.n_clusters_sweep``), with
the frozen marmoset HuBERT encoder (D14). Each run writes to ``<output-dir>/k<K>/`` and
the comparison table to ``<output-dir>/sweep_summary.json``.

Comparing across k needs care, because larger k trivially changes some numbers:

* **Raw surprisal is not comparable.** Bits per unit grow with vocabulary size, and a
  random model scores log2(k + 3). The table therefore also reports *normalised*
  surprisal, bits / log2(k + 3): the fraction of maximum entropy, 1.0 = chance.
* **Zero-shot accuracies are comparable** (always pairwise, chance = 50%), and are the
  primary decision metric, together with probe accuracy when probes are run.
* With no deduplication, much of the loss reduction is "the current unit repeats" --
  larger k cuts calls into more distinct units and changes run lengths, so expect the
  surprisal ordering and the zero-shot ordering to disagree. Prefer zero-shot.

Usage:
    python tools/sweep_units.py --mock                       # CPU smoke test
    python tools/sweep_units.py --config config/default.yaml --output-dir outputs/k_sweep
    python tools/sweep_units.py --mock --k 50 100            # override the list
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pipeline  # noqa: E402
from marmoset_developmental_agent.config.agent_config import load_config  # noqa: E402


def summarise(res: Dict[str, Any]) -> Dict[str, Any]:
    k = int(res["n_clusters"])
    traj = [t for t in res.get("trajectory", []) if "next_period" in t]
    bits = traj[-1]["next_period"]["mean_surprisal_bits"] if traj else float("nan")
    return {
        "k": k,
        "final_surprisal_bits": bits,
        "normalised_surprisal": bits / math.log2(k + 3),
        "zero_shot": {name: v["accuracy"] for name, v in res.get("zero_shot", {}).items()},
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--config", type=str)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--k", type=int, nargs="+", help="override n_clusters_sweep")
    ap.add_argument("--days", type=int)
    ap.add_argument("--skip-extended", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, mock=a.mock)
    ks = a.k or cfg.encoder.hubert.n_clusters_sweep
    base = Path(a.output_dir or f"{cfg.output_dir}_k_sweep")

    rows = []
    for k in ks:
        out = base / f"k{k}"
        args = ["--encoder", "hubert", "--n-clusters", str(k), "--output-dir", str(out)]
        args += ["--mock"] if a.mock else []
        args += ["--config", a.config] if a.config else []
        args += ["--days", str(a.days)] if a.days is not None else []
        args += ["--skip-extended"] if a.skip_extended else []
        print(f"\n######## k = {k} -> {out} ########", flush=True)
        if run_pipeline.main(args) != 0:
            raise SystemExit(f"pipeline failed for k={k}")
        rows.append(summarise(json.loads((out / "results.json").read_text())))

    tasks = sorted({t for r in rows for t in r["zero_shot"]})
    print(f"\n{'k':>5}{'surprisal':>11}{'norm.':>8}" + "".join(f"{t[:12]:>14}" for t in tasks))
    for r in rows:
        print(f"{r['k']:>5}{r['final_surprisal_bits']:>11.3f}{r['normalised_surprisal']:>8.3f}"
              + "".join(f"{r['zero_shot'].get(t, float('nan')):>14.3f}" for t in tasks))
    print("\nnorm. = bits / log2(k+3), 1.0 = chance. Zero-shot: chance = 0.5; primary metric.")
    (base / "sweep_summary.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {base / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
