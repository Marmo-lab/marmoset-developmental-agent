"""score_short_labels: which answers count, the production path on a window, and the learning-curve figure."""

import json
from pathlib import Path

import numpy as np

from tests.test_train_extra_sources import _short_folder
from tools.analysis import score_short_labels as ssl


def test_items_score_and_figure(tmp_path: Path) -> None:
    (tmp_path / "labels").mkdir()
    folder = _short_folder(tmp_path / "labels", np.random.default_rng(0))
    meta = folder / "meta.tsv"
    meta.write_text(meta.read_text().replace("half\n", "half\trecording_id\tmissed_by_v2a\n").replace("train\n", "train\tr0\t1\n")
                    .replace("test\n", "test\tr0\t1\n"))
    its = ssl.items(folder)
    assert [i["item_id"] for i in its] == ["sc_003"] and its[0]["span"] == [0.9, 1.1] and its[0]["missed_by_v2a"]
    assert [i["item_id"] for i in ssl.items(folder, half="train")] == ["sc_000", "sc_001"]
    res = ssl.score(folder, tmp_path / "out", ["v2a=models/call_detector_v2a.pt:0"])
    m = res["models"]["v2a"]
    assert res["n_calls"] == 1 and m["n_train_items"] == 0 and 0.0 <= m["front_end/call"][0] <= 1.0
    s = json.loads((tmp_path / "out" / "short_label_scores.json").read_text())
    s["models"]["v3"] = {**m, "n_train_items": 2}
    (tmp_path / "out" / "two.json").write_text(json.dumps(s))
    ssl.figure(tmp_path / "out" / "two.json", tmp_path / "curve.png")
    assert (tmp_path / "curve.png").exists()
