"""Review pages with ``edit_bounds``: clicked bounds are kept with the verdict, malformed ones dropped."""

import json
import threading
from pathlib import Path

from tools.corpus.call_library import _append_feedback, _bounds, _read_feedback


def test_bounds_are_validated_and_stored(tmp_path: Path) -> None:
    assert _bounds([0.4123, 0.61]) == [0.412, 0.61]
    assert _bounds(None) is None and _bounds([0.6, 0.4]) is None and _bounds(["x", 1]) is None and _bounds([1]) is None
    path, lock = tmp_path / "review_feedback.jsonl", threading.Lock()
    _append_feedback(path, lock, "item_id", {"item_id": "sc_000", "verdict": "call", "note": "", "bounds": [0.5, 0.62]})
    _append_feedback(path, lock, "item_id", {"item_id": "sc_001", "verdict": "not", "note": "", "bounds": [0.9, 0.2]})
    fb = _read_feedback(path, "item_id")
    assert fb["sc_000"]["bounds"] == [0.5, 0.62] and "bounds" not in fb["sc_001"]
    assert all("time" in json.loads(line) for line in path.read_text().splitlines())
