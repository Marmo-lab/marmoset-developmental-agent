"""Draw analysis/presence_table/presence_table.json as a table, after the layout of Hasson et al.'s Table 1.

Real colony minutes are scored against the user's hand labels; synthetic minutes against the exactly known
planted calls. One minute-level row only (a minute almost always holds a call); the rest at 10 s and 1 s, the
resolution our detector works at. Synthetic rows only at 1 s: with ~21 planted calls a minute every 10 s segment
holds one, so 10 s says nothing there.

Usage: python tools/analysis/presence_table_figure.py  ->  figures/presence_table_v2a.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BT, CV = "blind test (15 min, frozen v2a)", "cross-validated (45 min, out-of-fold v2a)"
ROWS = [("Real colony, blind test: 15 minutes, hand labels", None),
        ("Call presence, 1 min", f"{BT} | 60 s"), ("Call presence, 10 s", f"{BT} | 10 s"), ("Call presence, 1 s", f"{BT} | 1 s"),
        ("Real colony, cross-validated: 45 minutes, hand labels", None),
        ("Call presence, 10 s", f"{CV} | 10 s"), ("Call presence, 1 s", f"{CV} | 1 s"),
        ("Synthetic, held-out half: 1,800 minutes, planted calls in night background", None),
        ("All SNRs, 1 s", "synthetic, all SNRs | 1 s")] + [
        (f"SNR {lab}, 1 s", f"synthetic, {key} | 1 s") for lab, key in
        [("+20 dB", "+20 dB"), ("+10 dB", "+10.1 dB"), ("+6 dB", "+6.02 dB"), ("+3.5 dB", "+3.52 dB"),
         ("0 dB", "+0 dB"), ("−3 dB", "-3.1 dB")]]
HEAD, SECTION = "#c9dcee", "white"


def main() -> None:
    r = json.loads(Path("analysis/presence_table/presence_table.json").read_text())
    rows = [["Segment", "Precision", "Recall", "F1-Score", "Precision\nLift", "Cohen's\nKappa", "Cohen's\nKappa"],
            ["", "Machine vs.\nreference", "Machine vs.\nreference", "Machine vs.\nreference", "Machine vs.\nreference",
             "Machine vs.\nreference", "Human-\nHuman"]]
    kinds = ["head", "head"]
    for name, key in ROWS:
        if key is None:
            rows.append([name] + [""] * 6)
            kinds.append("section")
            continue
        x = r[key]
        hh = "exact" if key.startswith("synthetic") else "—"
        rows.append([f"   {name}", f"{x['precision']:.2f}", f"{x['recall']:.2f}", f"{x['f1']:.2f}", f"{x['lift']:.1f}",
                     f"{x['kappa']:.2f}", hh])
        kinds.append("row")
    fig, ax = plt.subplots(figsize=(12, 8.2))
    ax.axis("off")
    t = ax.table(cellText=rows, loc="upper center", cellLoc="left", colWidths=[0.34] + [0.11] * 6)
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    for (i, j), c in t.get_celld().items():
        c.set_height(0.075 if kinds[i] == "head" else 0.052)
        if kinds[i] == "head":
            c.set_facecolor(HEAD)
            c.get_text().set_fontweight("bold")
        elif kinds[i] == "section":
            c.set_facecolor(SECTION)
            c.get_text().set_fontweight("bold")
            c.visible_edges = "TBL" if j == 0 else ("TBR" if j == 6 else "TB")
            if j == 0:
                c.get_text().set_clip_on(False)
    caption = ("Call presence of detector v2a, per segment of one-minute recordings. Reference: the user's hand labels (real colony)\n"
               "or the planted calls (synthetic: unseen donor calls planted into unseen call-free night background at six SNRs).\n"
               "A segment is positive when a call or a detection overlaps it. Blind test: frozen model, scored once; cross-validated:\n"
               "each minute scored by a model that never trained on it. Precision lift = precision / prevalence of positive segments.\n"
               "Pooled over minutes. Human-human kappa not yet measured (needs a second rater). No noise events are planted yet.")
    fig.text(0.04, 0.02, caption, fontsize=9.5, style="italic", va="bottom")
    out = Path("figures/presence_table_v2a.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
