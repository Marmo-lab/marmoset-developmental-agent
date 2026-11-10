"""Draw analysis/presence_table/presence_table.json as a table in the layout of Hasson et al.'s Table 1.

Usage: python tools/analysis/presence_table_figure.py  ->  figures/presence_table_v2a.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROWS = [("Call presence, 1 min", "blind test (15 min, frozen v2a) | 60 s"),
        ("Call presence, 10 s", "blind test (15 min, frozen v2a) | 10 s"),
        ("Call presence, 1 s", "blind test (15 min, frozen v2a) | 1 s"),
        ("Call presence, 1 min (CV)", "cross-validated (45 min, out-of-fold v2a) | 60 s"),
        ("Call presence, 10 s (CV)", "cross-validated (45 min, out-of-fold v2a) | 10 s"),
        ("Call presence, 1 s (CV)", "cross-validated (45 min, out-of-fold v2a) | 1 s")]
HEAD = "#c9dcee"


def main() -> None:
    r = json.loads(Path("analysis/presence_table/presence_table.json").read_text())
    cols = ["Precision", "Recall", "F1-Score", "Precision\nLift", "Cohen's\nKappa", "Cohen's\nKappa"]
    sub = ["Machine-\nHuman"] * 5 + ["Human-\nHuman"]
    cells = []
    for name, key in ROWS:
        x = r[key]
        cells.append([name, f"{x['precision']:.2f}", f"{x['recall']:.2f}", f"{x['f1']:.2f}", f"{x['lift']:.1f}",
                      f"{x['kappa']:.2f}", "—"])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")
    t = ax.table(cellText=[["Algorithm"] + cols, [""] + sub] + cells, loc="upper center", cellLoc="left",
                 colWidths=[0.27] + [0.12] * 6)
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    for (i, _j), c in t.get_celld().items():
        c.set_linewidth(1.6 if i > 1 else 0.8)
        c.set_height(0.11 if i < 2 else 0.095)
        if i < 2:
            c.set_facecolor(HEAD)
            c.get_text().set_fontweight("bold")
    caption = ("Table 1 (our analogue). Call presence of detector v2a vs. the user's hand labels, per segment of the labelled\n"
               "one-minute clips. Top rows: the 15 blind test minutes, frozen model, scored once (all day minutes). (CV) rows: the\n"
               "45 calibration + held-out minutes, cross-validated out-of-fold (39 day, 6 night; no detection in any night minute).\n"
               "Precision lift = precision / prevalence of call-positive segments. Pooled over minutes. Human-human kappa: not yet\n"
               "measured (needs a second rater).")
    fig.text(0.04, 0.02, caption, fontsize=9.5, style="italic", va="bottom")
    out = Path("figures/presence_table_v2a.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
