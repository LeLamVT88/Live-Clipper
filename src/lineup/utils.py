from __future__ import annotations

import math
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: object) -> Path:
    """Resolve a project-relative path without changing absolute paths."""
    resolved_path = Path(str(path))
    return resolved_path if resolved_path.is_absolute() else PROJECT_ROOT / resolved_path


def timestamp_to_seconds(timestamp: object) -> float:
    """Convert seconds, HH:MM:SS, HH:MM:SS.mmm, or MM:SS to seconds."""
    if timestamp is None:
        raise ValueError("Timestamp is empty")

    if isinstance(timestamp, (int, float)):
        value = float(timestamp)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid timestamp value: {timestamp}")
        return value

    text = str(timestamp).strip()
    if not text:
        raise ValueError("Timestamp is empty")

    try:
        value = float(text)
    except ValueError:
        value = None
    if value is not None:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid timestamp value: {timestamp}")
        return value

    parts = text.split(":")
    if len(parts) == 3:
        hours_text, minutes_text, seconds_text = parts
    elif len(parts) == 2:
        hours_text = "0"
        minutes_text, seconds_text = parts
    else:
        raise ValueError(
            f"Invalid timestamp format '{timestamp}'. Expected HH:MM:SS."
        )

    try:
        hours = int(hours_text)
        minutes = int(minutes_text)
        seconds = float(seconds_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid timestamp format '{timestamp}'. Expected HH:MM:SS."
        ) from exc

    if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
        raise ValueError(
            f"Invalid timestamp value '{timestamp}'. Minutes and seconds must be 0-59."
        )

    return hours * 3600 + minutes * 60 + seconds


def timestamp_range_to_seconds(timestamp_range: object) -> tuple[float, float]:
    """Convert a START-END range, including compact MM.SS endpoints, to seconds."""
    text = str(timestamp_range).strip()
    if not text or "-" not in text:
        raise ValueError(
            f"Invalid time range '{timestamp_range}'. "
            "Expected START-END, for example 1.00-2.00."
        )

    start_text, end_text = (part.strip() for part in text.split("-", maxsplit=1))
    if not start_text or not end_text:
        raise ValueError(
            f"Invalid time range '{timestamp_range}'. Both start and end are required."
        )

    def parse_endpoint(endpoint: str) -> float:
        compact_parts = endpoint.split(".")
        if (
            ":" not in endpoint
            and len(compact_parts) == 2
            and compact_parts[0].isdigit()
            and len(compact_parts[1]) == 2
            and compact_parts[1].isdigit()
        ):
            minutes = int(compact_parts[0])
            seconds = int(compact_parts[1])
            if seconds >= 60:
                raise ValueError(
                    f"Invalid compact timestamp '{endpoint}'. Seconds must be 00-59."
                )
            return float(minutes * 60 + seconds)
        return timestamp_to_seconds(endpoint)

    start_seconds = parse_endpoint(start_text)
    end_seconds = parse_endpoint(end_text)
    if end_seconds <= start_seconds:
        raise ValueError(
            f"Invalid time range '{timestamp_range}'. End must be greater than start."
        )

    return start_seconds, end_seconds


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or HH:MM:SS.mmm."""
    value = float(seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid seconds value: {seconds}")

    whole_seconds = int(value)
    milliseconds = int(round((value - whole_seconds) * 1000))

    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0

    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60

    if milliseconds:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}".rstrip("0")
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if it does not exist, then return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_video_name_without_ext(video_path: PathLike) -> str:
    """Return the file name without extension, e.g. match1.mp4 -> match1."""
    return Path(video_path).stem
