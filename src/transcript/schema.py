"""Data contracts and serialization for timestamped transcripts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class TranscriptValidationError(ValueError):
    """Raised when a transcript violates the persisted data contract."""


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: str
    start_seconds: float
    end_seconds: float
    text: str
    source_chunk: str
    chunk_start_seconds: float

    def validate(self) -> None:
        if not self.segment_id.strip():
            raise TranscriptValidationError("segment_id cannot be empty.")
        if not self.source_chunk.strip():
            raise TranscriptValidationError("source_chunk cannot be empty.")
        if not self.text.strip():
            raise TranscriptValidationError(
                f"Transcript text cannot be empty: {self.segment_id}"
            )
        numeric_values = (
            self.start_seconds,
            self.end_seconds,
            self.chunk_start_seconds,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise TranscriptValidationError(
                f"Transcript timestamps must be finite: {self.segment_id}"
            )
        if self.chunk_start_seconds < 0 or self.start_seconds < 0:
            raise TranscriptValidationError(
                f"Transcript timestamps cannot be negative: {self.segment_id}"
            )
        if self.end_seconds <= self.start_seconds:
            raise TranscriptValidationError(
                f"Transcript segment must end after it starts: {self.segment_id}"
            )
        if self.start_seconds < self.chunk_start_seconds:
            raise TranscriptValidationError(
                f"Segment starts before its source chunk: {self.segment_id}"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        row = asdict(self)
        for field in ("start_seconds", "end_seconds", "chunk_start_seconds"):
            row[field] = round(float(row[field]), 3)
        row["text"] = self.text.strip()
        return row


@dataclass(frozen=True)
class TranscriptionResult:
    model: str
    segments: tuple[TranscriptSegment, ...]
    raw_response: dict[str, Any]

    def validate(self) -> None:
        if not self.model.strip():
            raise TranscriptValidationError("Transcription model cannot be empty.")
        seen_ids: set[str] = set()
        previous_start = -1.0
        for segment in self.segments:
            segment.validate()
            if segment.segment_id in seen_ids:
                raise TranscriptValidationError(
                    f"Duplicate transcript segment_id: {segment.segment_id}"
                )
            if segment.start_seconds < previous_start:
                raise TranscriptValidationError(
                    "Transcript segments must be ordered by start_seconds."
                )
            seen_ids.add(segment.segment_id)
            previous_start = segment.start_seconds


def write_jsonl(segments: Iterable[TranscriptSegment], output_path: Path) -> None:
    materialized = list(segments)
    seen_ids: set[str] = set()
    rows: list[dict[str, object]] = []
    for segment in materialized:
        row = segment.to_dict()
        segment_id = str(row["segment_id"])
        if segment_id in seen_ids:
            raise TranscriptValidationError(
                f"Duplicate transcript segment_id: {segment_id}"
            )
        seen_ids.add(segment_id)
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def read_jsonl(input_path: Path) -> tuple[TranscriptSegment, ...]:
    if not input_path.is_file():
        raise TranscriptValidationError(
            f"Transcript JSONL does not exist: {input_path}"
        )

    segments: list[TranscriptSegment] = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            segment = TranscriptSegment(
                segment_id=str(row["segment_id"]),
                start_seconds=float(row["start_seconds"]),
                end_seconds=float(row["end_seconds"]),
                text=str(row["text"]),
                source_chunk=str(row["source_chunk"]),
                chunk_start_seconds=float(row["chunk_start_seconds"]),
            )
            segment.validate()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TranscriptValidationError(
                f"Invalid transcript JSONL row {line_number}: {exc}"
            ) from exc
        segments.append(segment)

    result = TranscriptionResult(
        model="persisted-transcript",
        segments=tuple(segments),
        raw_response={},
    )
    result.validate()
    return result.segments
