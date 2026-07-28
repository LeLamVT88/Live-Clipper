"""Load lineup segments and extract timestamped OCR frames."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import pandas as pd

from lineup.utils import safe_stem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SegmentKey = tuple[str, int]


class LineupOCRError(Exception):
    """Raised when lineup-frame extraction or OCR cannot be completed."""


@dataclass(frozen=True)
class Segment:
    video: str
    video_path: Path
    index: int
    label: str
    start_seconds: float
    end_seconds: float


def segment_key(row: Any) -> SegmentKey:
    return str(row["video"]), int(row["segment_index"])


def group_records_by_segment(
    records: Iterable[dict[str, object]],
) -> dict[SegmentKey, list[dict[str, object]]]:
    grouped: defaultdict[
        SegmentKey,
        list[dict[str, object]],
    ] = defaultdict(list)
    for record in records:
        grouped[segment_key(record)].append(record)
    return dict(grouped)


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def seconds_to_timestamp(seconds: float) -> str:
    milliseconds = int(round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    base = f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
    return f"{base}.{milliseconds:03d}".rstrip("0") if milliseconds else base


def timestamp_to_seconds(value: object) -> float:
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        try:
            seconds = float(text)
        except ValueError:
            parts = text.split(":")
            if len(parts) == 3:
                hours_text, minutes_text, seconds_text = parts
            elif len(parts) == 2:
                hours_text, minutes_text, seconds_text = "0", *parts
            else:
                raise ValueError(f"Invalid timestamp: {value}") from None
            seconds = (
                int(hours_text) * 3600
                + int(minutes_text) * 60
                + float(seconds_text)
            )
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid timestamp: {value}")
    return seconds


def resolve_video(video: str, video_dir: Path) -> Path:
    candidate = Path(video).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    from_video_dir = (video_dir / candidate).resolve()
    if from_video_dir.is_file():
        return from_video_dir

    from_project = (PROJECT_ROOT / candidate).resolve()
    return from_project if from_project.is_file() else from_video_dir


def load_segments(segments_csv: Path, video_dir: Path) -> list[Segment]:
    if not segments_csv.is_file():
        raise LineupOCRError(f"Segments CSV does not exist: {segments_csv}")

    rows = pd.read_csv(segments_csv)
    if rows.empty:
        raise LineupOCRError(f"Segments CSV has no rows: {segments_csv}")
    if "video" not in rows.columns:
        raise LineupOCRError("Segments CSV is missing the 'video' column.")

    has_seconds = {"start_seconds", "end_seconds"}.issubset(rows.columns)
    has_timestamps = {"start", "end"}.issubset(rows.columns)
    if not has_seconds and not has_timestamps:
        raise LineupOCRError(
            "Segments CSV must contain start_seconds/end_seconds or start/end."
        )

    start_column, end_column = (
        ("start_seconds", "end_seconds") if has_seconds else ("start", "end")
    )
    counters: defaultdict[str, int] = defaultdict(int)
    segments: list[Segment] = []

    for row_number, row in rows.iterrows():
        video = str(row["video"]).strip()
        if not video:
            raise LineupOCRError(f"Row {row_number + 2} has an empty video.")
        try:
            start_seconds = timestamp_to_seconds(row[start_column])
            end_seconds = timestamp_to_seconds(row[end_column])
        except (TypeError, ValueError) as exc:
            raise LineupOCRError(
                f"Invalid timestamp at row {row_number + 2}: {exc}"
            ) from exc
        if end_seconds <= start_seconds:
            raise LineupOCRError(
                f"Row {row_number + 2} must end after it starts."
            )

        video_path = resolve_video(video, video_dir)
        if not video_path.is_file():
            raise LineupOCRError(f"Source video does not exist: {video_path}")

        video_key = str(video_path)
        counters[video_key] += 1
        index = counters[video_key]
        raw_label = (
            str(row["segment"]).strip()
            if "segment" in rows.columns and pd.notna(row["segment"])
            else ""
        )
        segments.append(
            Segment(
                video=video,
                video_path=video_path,
                index=index,
                label=raw_label or f"lineup_{index:02d}",
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )

    return segments


def extract_segment_frames(
    segment: Segment,
    frames_dir: Path,
    fps: float,
    jpeg_quality: int,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(segment.video_path))
    if not capture.isOpened():
        raise LineupOCRError(f"Cannot open video: {segment.video_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0 or source_frame_count <= 0:
            raise LineupOCRError(
                f"Cannot read FPS/frame count from: {segment.video_path}"
            )

        video_duration = source_frame_count / source_fps
        if segment.start_seconds >= video_duration:
            raise LineupOCRError(
                f"Segment starts outside video duration: "
                f"{segment.video_path.name}"
            )
        extraction_end = min(segment.end_seconds, video_duration)

        output_dir = (
            frames_dir
            / safe_stem(segment.video_path.stem)
            / f"segment_{segment.index:02d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, object]] = []
        step_seconds = 1.0 / fps
        frame_index = 1
        timestamp_seconds = segment.start_seconds

        while timestamp_seconds < extraction_end:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                raise LineupOCRError(
                    f"Cannot read {segment.video_path.name} at "
                    f"{timestamp_seconds:.3f}s"
                )

            timestamp_ms = round(timestamp_seconds * 1000)
            frame_path = output_dir / (
                f"frame_{frame_index:06d}_t{timestamp_ms:010d}.jpg"
            )
            if not cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise LineupOCRError(f"Cannot write OCR frame: {frame_path}")

            frame_height, frame_width = frame.shape[:2]
            records.append(
                {
                    "video": segment.video,
                    "segment_index": segment.index,
                    "segment_label": segment.label,
                    "segment_start_seconds": segment.start_seconds,
                    "segment_end_seconds": segment.end_seconds,
                    "frame_index": frame_index,
                    "frame_path": relative_to_project(frame_path),
                    "timestamp": seconds_to_timestamp(timestamp_seconds),
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "relative_seconds": round(
                        timestamp_seconds - segment.start_seconds,
                        3,
                    ),
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                }
            )
            frame_index += 1
            timestamp_seconds = round(
                segment.start_seconds + (frame_index - 1) * step_seconds,
                6,
            )

        return records
    finally:
        capture.release()
