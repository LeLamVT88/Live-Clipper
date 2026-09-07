"""Lineup Test Pipeline.

A lightweight, robust, audio-independent pipeline for detecting football
starting lineup graphics from TV broadcasts (TV360, UEFA, Bundesliga, FIFA).
"""

from __future__ import annotations

import sys

# Strictly enforce Python 3.12+
if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    raise RuntimeError(
        f"lineup_test requires Python 3.12.x, but running on {sys.version}. "
        "Please run with 'uv run ...'."
    )

from lineup_test.config import PipelineConfig
from lineup_test.schema import LineupInterval, DetectionResult
from lineup_test.pipeline import detect_lineups

__all__ = [
    "detect_lineups",
    "LineupInterval",
    "DetectionResult",
    "PipelineConfig",
]
