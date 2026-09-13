from __future__ import annotations

import math
import re
import subprocess
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SegmentKey = tuple[str, int]
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


class LineupOCRError(Exception): pass


def safe_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_name).strip("._-")
    return safe_name or "video"


@dataclass(frozen=True, slots=True)
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
    grouped: defaultdict[SegmentKey, list[dict[str, object]]] = defaultdict(list)
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


def video_duration_seconds(video_path: Path) -> float:
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(probe.stdout.strip())
        if math.isfinite(duration) and duration > 0:
            return duration
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise LineupOCRError(f"Cannot open lineup clip: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise LineupOCRError(f"Cannot read FPS/frame count from lineup clip: {video_path}")
    return frame_count / fps


def load_lineup_clips(clips_dir: Path) -> list[Segment]:
    clips_dir = resolve_project_path(clips_dir)
    if not clips_dir.is_dir():
        raise LineupOCRError(
            f"Lineup clips directory does not exist: {clips_dir}. "
            "Export clips with run_lineup.py --export-clips-dir first."
        )
    video_paths = sorted(
        (
            path.resolve()
            for path in clips_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.as_posix().casefold(),
    )
    if not video_paths:
        raise LineupOCRError(f"No lineup video clips found in: {clips_dir}")
    return [
        Segment(
            video=relative_to_project(path), video_path=path, index=1,
            label=path.stem, start_seconds=0.0, end_seconds=video_duration_seconds(path),
        )
        for path in video_paths
    ]


def extract_segment_frames(
    segment: Segment, frames_dir: Path, fps: float, jpeg_quality: int,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(segment.video_path))
    if not capture.isOpened():
        raise LineupOCRError(f"Cannot open video: {segment.video_path}")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0 or source_frame_count <= 0:
            raise LineupOCRError(f"Cannot read FPS/frame count from: {segment.video_path}")
        video_duration = source_frame_count / source_fps
        if segment.start_seconds >= video_duration:
            raise LineupOCRError(f"Segment starts outside video duration: {segment.video_path.name}")
        extraction_end = min(segment.end_seconds, video_duration)
        output_dir = frames_dir / safe_stem(
            Path(segment.video).with_suffix("").as_posix()
        ) / f"segment_{segment.index:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        step_seconds = 1.0 / fps
        frame_index = 1
        timestamp_seconds = segment.start_seconds
        while timestamp_seconds < extraction_end:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                # Some exported MP4 files report a frame-count duration that is
                # slightly longer than their final decodable presentation time.
                # Missing only the last sampling slot is a normal EOF condition.
                eof_tolerance = max(1.0, 2 * step_seconds)
                if records and timestamp_seconds >= extraction_end - eof_tolerance:
                    break
                raise LineupOCRError(
                    f"Cannot read {segment.video_path.name} at {timestamp_seconds:.3f}s"
                )
            timestamp_ms = round(timestamp_seconds * 1000)
            frame_path = output_dir / (f"frame_{frame_index:06d}_t{timestamp_ms:010d}.jpg")
            if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
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
                    "relative_seconds": round(timestamp_seconds - segment.start_seconds, 3),
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                }
            )
            frame_index += 1
            timestamp_seconds = round(segment.start_seconds + (frame_index - 1) * step_seconds, 6)
        return records
    finally:
        capture.release()
