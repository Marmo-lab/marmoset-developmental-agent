"""Listening check of suspect donor bounds: is there a call in the clip, and do our bounds catch it?

Why (2026-11-16). Scoring against audible bounds showed that two "trill" donors are 10 ms clicks, and the user asked
to hear them before dropping them, since Turesson et al. 2016 labelled every clip as a call. In fact 8 of the 27
trill donors have 10 ms bounds. The bounds inside each clip were proposed by our ``tools/synth/donor_calls.py``, not by
the paper, so a failed proposal is a likelier cause than a wrong label.

The page shows each *whole* donor clip as the paper ships it, with our bounds marked. Items: every trill donor, plus
every other donor shorter than ``SHORT_S``. Answers: the bounds catch the call; there is a call but the bounds miss it;
no call at all; unsure.

Usage: python tools/explorations/donor_check.py DONORS OUT     (DONORS = $WORK_DIR/synth/donors/turesson)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import soundfile as sf

SHORT_S = 0.08
QUESTION = {
    "title": "בדיקת תורמים: האם יש קריאה בקליפ, והאם הגבולות שלנו תופסים אותה?",
    "intro": ("כל פריט הוא קליפ שלם מהמאמר (Turesson 2016). הפס הצבעוני הוא הגבולות שהקוד שלנו הציע בתוך הקליפ. "
              "כל קריאות ה-trill, ועוד תורמים שקצרים מ-80 מילישניות."),
    "options": [["ok", "הגבולות תופסים את הקריאה"], ["missed", "יש קריאה, הגבולות מפספסים"],
                ["none", "אין קריאה בכלל"], ["unsure", "לא בטוח"]],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("donors", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.donors / "donors.tsv"), delimiter="\t")
            if r["call_type"] == "trill" or float(r["offset_s"]) - float(r["onset_s"]) < SHORT_S]
    rows.sort(key=lambda r: (r["call_type"] != "trill", r["call_type"], r["donor_id"]))
    (a.out / "clips").mkdir(parents=True, exist_ok=True)
    items = []
    for r in rows:
        src = a.donors / "audio" / f"{r['donor_id']}.flac"
        shutil.copy(src, a.out / "clips" / f"{r['donor_id']}.flac")
        items.append({"item_id": r["donor_id"], "type": r["call_type"],
                      "bounds_ms": round(1000 * (float(r["offset_s"]) - float(r["onset_s"]))),
                      "clip_s": round(sf.info(src).duration, 2),
                      "marks": json.dumps([[float(r["onset_s"]), float(r["offset_s"]), "a", "our bounds"]])})
    with open(a.out / "items.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]), delimiter="\t")
        w.writeheader()
        w.writerows(items)
    (a.out / "review.json").write_text(json.dumps(QUESTION | {"models": ["our bounds", ""]}, ensure_ascii=False, indent=1))
    print(f"{len(items)} donors -> {a.out}")


if __name__ == "__main__":
    main()
