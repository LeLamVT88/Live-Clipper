from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LineupInterval:
    """Detected starting lineup interval."""
    start_seconds: float
    end_seconds: float
    confidence: float
    sample_count: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict[str, object]:
        return {
            "start_seconds": round(self.start_seconds, 2),
            "end_seconds": round(self.end_seconds, 2),
            "duration_seconds": round(self.duration, 2),
            "confidence": round(self.confidence, 4),
            "sample_count": self.sample_count,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FrameSampleResult:
    """OCR and score results for a single sampled frame."""
    timestamp: float
    score: float
    box_count: int
    candidate_names: list[str]
    jersey_numbers: list[str]
    has_formation: bool
    has_keyword: bool
    strong_evidence: bool = False
    suppressed: bool = False
    suppress_reason: str = ""


@dataclass(slots=True)
class DetectionResult:
    """Full execution output of the lineup detection pipeline."""
    video_path: Path
    video_duration_scanned: float
    scenes: list[tuple[float, float]]
    sampled_frames: list[FrameSampleResult]
    lineups: list[LineupInterval]
    processing_time_seconds: float
    stats: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "video_path": str(self.video_path),
            "video_duration_scanned": round(self.video_duration_scanned, 2),
            "scenes_detected": len(self.scenes),
            "frames_sampled": len(self.sampled_frames),
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "lineups": [l.to_dict() for l in self.lineups],
            "stats": self.stats,
        }
