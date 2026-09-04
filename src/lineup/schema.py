"""Validated output contract for visual lineup intervals."""

from __future__ import annotations

import csv
import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CSV_COLUMNS = (
    "video segment_id detection_status team_name start_seconds end_seconds confidence "
    "evidence_start_seconds evidence_end_seconds evidence_segment_ids "
    "start_anchor_text end_anchor_text reason model"
).split()
MAX_LINEUP_SEGMENTS = 2
DETECTION_STATUSES = {"complete", "incomplete", "empty", "unconstrained"}


class LineupDetectionError(RuntimeError):
    """Raised when visual lineup detection or its output is invalid."""


@dataclass(frozen=True)
class LineupSegment:
    segment_id: str
    team_name: str
    start_seconds: float
    end_seconds: float
    confidence: float
    evidence_segment_ids: tuple[str, ...]
    start_anchor_text: str
    end_anchor_text: str
    reason: str
    evidence_start_seconds: float | None = None
    evidence_end_seconds: float | None = None

    def validate(self) -> None:
        values = (self.start_seconds, self.end_seconds, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise LineupDetectionError("Lineup values must be finite numbers.")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise LineupDetectionError(f"Invalid lineup range: {self.segment_id}")
        if not 0 <= self.confidence <= 1:
            raise LineupDetectionError(
                f"Lineup confidence must be between 0 and 1: {self.segment_id}"
            )
        if not self.evidence_segment_ids:
            raise LineupDetectionError(f"Missing evidence: {self.segment_id}")
        if len(set(self.evidence_segment_ids)) != len(self.evidence_segment_ids):
            raise LineupDetectionError(
                f"Duplicate evidence IDs: {self.segment_id}"
            )
        if not self.team_name.strip():
            raise LineupDetectionError(f"Missing team name: {self.segment_id}")
        if not self.start_anchor_text.strip() or not self.end_anchor_text.strip():
            raise LineupDetectionError(f"Missing visual anchors: {self.segment_id}")
        if not self.reason.strip():
            raise LineupDetectionError(f"Missing reason: {self.segment_id}")
        evidence_values = (
            self.evidence_start_seconds,
            self.evidence_end_seconds,
        )
        if any(
            value is not None and not math.isfinite(value)
            for value in evidence_values
        ):
            raise LineupDetectionError(
                f"Evidence bounds must be finite: {self.segment_id}"
            )
        if (
            self.evidence_start_seconds is not None
            and self.evidence_end_seconds is not None
            and (
                self.evidence_start_seconds < 0
                or self.evidence_end_seconds <= self.evidence_start_seconds
            )
        ):
            raise LineupDetectionError(
                f"Invalid evidence bounds: {self.segment_id}"
            )

    def csv_row(self, *, video_name: str, model: str) -> dict[str, object]:
        row = asdict(self)
        row["video"], row["model"] = video_name, model
        row["start_seconds"] = round(self.start_seconds, 3)
        row["end_seconds"] = round(self.end_seconds, 3)
        row["confidence"] = round(self.confidence, 4)
        for field in ("evidence_start_seconds", "evidence_end_seconds"):
            value = row[field]
            row[field] = "" if value is None else round(float(value), 3)
        row["evidence_segment_ids"] = json.dumps(
            self.evidence_segment_ids, ensure_ascii=False
        )
        return row


@dataclass(frozen=True)
class LineupDetectionResult:
    model: str
    segments: tuple[LineupSegment, ...]
    raw_response: dict[str, Any]
    status: str = "unconstrained"

    def validate(self) -> None:
        if not self.model.strip():
            raise LineupDetectionError("Detection model cannot be empty.")
        if self.status not in DETECTION_STATUSES:
            raise LineupDetectionError(f"Invalid detection status: {self.status}")
        if len(self.segments) > MAX_LINEUP_SEGMENTS:
            raise LineupDetectionError(
                f"At most {MAX_LINEUP_SEGMENTS} lineup segments are supported."
            )
        seen_ids: set[str] = set()
        seen_teams: set[str] = set()
        previous_start = -1.0
        previous_end = -1.0
        for segment in self.segments:
            segment.validate()
            if segment.segment_id in seen_ids:
                raise LineupDetectionError(
                    f"Duplicate lineup segment ID: {segment.segment_id}"
                )
            if segment.start_seconds < previous_start:
                raise LineupDetectionError(
                    "Lineup segments must be ordered by start_seconds."
                )
            team_key = unicodedata.normalize(
                "NFKC", segment.team_name
            ).strip().casefold()
            if team_key in seen_teams:
                raise LineupDetectionError(
                    "Lineup segments must belong to distinct teams: "
                    f"{segment.team_name}"
                )
            if segment.start_seconds < previous_end - 1e-6:
                raise LineupDetectionError("Lineup segments must not overlap.")
            seen_ids.add(segment.segment_id)
            seen_teams.add(team_key)
            previous_start = segment.start_seconds
            previous_end = segment.end_seconds


def write_lineup_csv(
    result: LineupDetectionResult,
    *,
    video_name: str,
    output_path: Path,
) -> None:
    result.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        rows = []
        for segment in result.segments:
            row = segment.csv_row(video_name=video_name, model=result.model)
            row["detection_status"] = result.status
            rows.append(row)
        writer.writerows(rows)
