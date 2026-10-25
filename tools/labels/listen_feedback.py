"""Add verdict buttons to a listening page, so the user can mark each example as they listen.

Every example on the page (an element with class ``ex`` that holds an ``<audio>``) gets four
buttons — call / not a call / wrong edges / unsure — and a free-text note. Choices are kept
in the browser (``localStorage``, keyed by the page's path, so reopening the page keeps
them), and a bar at the bottom shows progress and saves everything as
``feedback_<page>.json``. The download is the hand-off: Claude reads it from ``~/Downloads``.

The item id is the audio file's name, so a verdict can be traced back to the call it
belongs to (the pages name audio files after call ids or cluster examples).

Usage:
    python tools/labels/listen_feedback.py labeling/pilot_learned/listen/index.html [more pages...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARK = "<!-- listen-feedback -->"

WIDGET = MARK + """
<style>
.fb{margin-top:4px;display:flex;flex-wrap:wrap;gap:3px;direction:rtl}
.fb button{font-size:11px;padding:2px 6px;border:1px solid #bbb;border-radius:4px;background:#f6f6f6;cursor:pointer}
.fb button.on[data-v=call]{background:#2e7d32;color:#fff}.fb button.on[data-v=noise]{background:#c62828;color:#fff}
.fb button.on[data-v=edges]{background:#ef6c00;color:#fff}.fb button.on[data-v=unsure]{background:#616161;color:#fff}
.fb input{width:100%;font-size:11px;direction:rtl}
#fbbar{position:fixed;bottom:0;left:0;right:0;background:#222;color:#fff;padding:8px 16px;font-size:14px;direction:rtl;z-index:9}
#fbbar button{margin-inline-start:12px;font-size:14px}
body{padding-bottom:56px}
</style>
<div id="fbbar"><span id="fbcount"></span><button id="fbsave">💾 שמור משוב לקובץ</button><button id="fbclear">נקה הכל</button></div>
<script>
(function(){
  const KEY = "fb:" + location.pathname;
  const page = location.pathname.split("/").slice(-2, -1)[0] || "page";
  let st = {};
  try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} count(); };
  const items = [...document.querySelectorAll(".ex")].filter(d => d.querySelector("audio"));
  function count(){
    const n = Object.values(st).filter(v => v.v).length;
    document.getElementById("fbcount").textContent = `סומנו ${n} מתוך ${items.length}`;
  }
  const LABELS = [["call","✓ קריאה"],["noise","✗ לא קריאה"],["edges","↔ גבולות לא נכונים"],["unsure","? לא בטוח"]];
  items.forEach(div => {
    const src = div.querySelector("audio").getAttribute("src");
    const id = src.split("/").pop().replace(/\\.wav$/, "");
    const box = document.createElement("div"); box.className = "fb";
    const s = st[id] || {};
    LABELS.forEach(([v, t]) => {
      const b = document.createElement("button"); b.textContent = t; b.dataset.v = v;
      if (s.v === v) b.classList.add("on");
      b.onclick = () => {
        const cur = st[id] || {};
        cur.v = cur.v === v ? undefined : v;
        st[id] = cur; box.querySelectorAll("button").forEach(x => x.classList.toggle("on", x.dataset.v === cur.v)); save();
      };
      box.appendChild(b);
    });
    const note = document.createElement("input"); note.placeholder = "הערה (לא חובה)"; note.value = s.note || "";
    note.oninput = () => { st[id] = Object.assign(st[id] || {}, {note: note.value}); save(); };
    box.appendChild(note); div.appendChild(box);
  });
  document.getElementById("fbsave").onclick = () => {
    const data = {page: location.pathname, saved: new Date().toISOString(), items: st};
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], {type: "application/json"}));
    a.download = `feedback_${page}.json`; a.click();
  };
  document.getElementById("fbclear").onclick = () => { if (confirm("למחוק את כל הסימונים בדף הזה?")) { st = {}; save(); location.reload(); } };
  count();
})();
</script>
"""


def add_feedback(page: Path) -> bool:
    """Insert the widget before ``</body>``; a page that already has it is left alone."""
    html = page.read_text()
    if MARK in html:
        return False
    page.write_text(html.replace("</body>", WIDGET + "</body>", 1))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="+")
    for p in ap.parse_args().pages:
        print(p, "added" if add_feedback(Path(p)) else "already has buttons")


if __name__ == "__main__":
    main()
