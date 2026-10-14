"""Logging setup shared by CLIs."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("numba", "matplotlib", "urllib3", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
