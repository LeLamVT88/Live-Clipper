"""Lineup Test Pipeline.

A lightweight, robust, audio-independent pipeline for detecting football
starting lineup graphics from TV broadcasts (TV360, UEFA, Bundesliga, FIFA).
"""

from __future__ import annotations

import sys

# PaddleOCR currently supports the Python versions used by this project up to 3.13.
if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    raise RuntimeError(
        f"line_up requires Python 3.11-3.13, but is running on {sys.version}. "
        "Run the unified pipeline with `.venv-ocr/bin/python`."
    )

from line_up.config import PipelineConfig
from line_up.schema import LineupInterval, DetectionResult
from line_up.pipeline import detect_lineups

__all__ = [
    "detect_lineups",
    "LineupInterval",
    "DetectionResult",
    "PipelineConfig",
]
