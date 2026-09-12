from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Rect:
    """A rectangle expressed as fractions of the full frame."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Crop percentages must be in [0, 1], got {values}")
        if self.width <= 0 or self.height <= 0 or self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError(f"Invalid crop rectangle: {values}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Rect:
        return cls(*(float(value[key]) for key in ("x_pct", "y_pct", "w_pct", "h_pct")))

    def pixels(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        x0, y0 = round(self.x * width), round(self.y * height)
        x1, y1 = round((self.x + self.width) * width), round((self.y + self.height) * height)
        return x0, y0, x1, y1


@dataclass(frozen=True, slots=True)
class FrameSample:
    timestamp: float
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class StableSegment:
    start_index: int
    end_index: int
    start_seconds: float
    end_seconds: float
    mean_diff: float


@dataclass(frozen=True, slots=True)
class TeamObservation:
    timestamp: float
    raw_text: str
    team: str | None
    similarity: float


@dataclass(slots=True)
class TeamBoundary:
    from_team: str
    to_team: str
    left_seconds: float
    right_seconds: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class ClipSegment:
    team: str
    start_seconds: float
    end_seconds: float
    source_clip: Path


@dataclass(frozen=True, slots=True)
class PlayerUnit:
    box: tuple[int, int, int, int]
    shirt_box: tuple[int, int, int, int]
    number_box: tuple[int, int, int, int]
    name_box: tuple[int, int, int, int]
    detection_score: float


@dataclass(frozen=True, slots=True)
class RawPlayer:
    jersey_number: int | None
    player_name: str
    number_confidence: float
    name_confidence: float


@dataclass(frozen=True, slots=True)
class SquadPlayer:
    team: str
    jersey_number: int
    player_name: str


@dataclass(frozen=True, slots=True)
class LineupRow:
    team: str
    jersey_number: int | None
    player_name: str
    role: str
    source_clip: str
    frame_timestamp_sec: float
    ocr_confidence: str
