"""Lineup Test Pipeline.

A lightweight, robust, audio-independent pipeline for detecting football
starting lineup graphics from TV broadcasts (TV360, UEFA, Bundesliga, FIFA).
"""

from __future__ import annotations

import sys

# Strictly enforce Python 3.12+
if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    raise RuntimeError(
        f"line_up requires Python 3.12.x, but running on {sys.version}. "
        "Please run with 'uv run ...'."
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
