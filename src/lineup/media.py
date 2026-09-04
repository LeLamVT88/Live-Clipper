"""Small FFprobe adapter shared by lineup detection and clip export."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaProbeError(RuntimeError):
    """Raised when an input cannot be validated with FFprobe."""


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float
    audio_stream_count: int
    video_stream_count: int

    @property
    def has_audio(self) -> bool:
        return self.audio_stream_count > 0

    @property
    def has_video(self) -> bool:
        return self.video_stream_count > 0

    def validate(self) -> None:
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise MediaProbeError("Media duration must be finite and positive.")
        if self.audio_stream_count < 0 or self.video_stream_count < 0:
            raise MediaProbeError("Media stream counts cannot be negative.")


def resolve_executable(executable: str) -> str:
    candidate = Path(executable).expanduser()
    if candidate.parent != Path("."):
        if candidate.is_file():
            return str(candidate)
        raise MediaProbeError(f"Executable does not exist: {candidate}")
    resolved = shutil.which(executable)
    if resolved is None:
        raise MediaProbeError(
            f"{executable} was not found. Install FFmpeg tools or pass an "
            "explicit executable path."
        )
    return resolved


def probe_media(video_path: Path, *, ffprobe: str = "ffprobe") -> MediaInfo:
    source = video_path.expanduser().resolve()
    if not source.is_file():
        raise MediaProbeError(f"Source video does not exist: {source}")
    executable = resolve_executable(ffprobe)
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration",
            "-of",
            "json",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise MediaProbeError(f"FFprobe media validation failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("FFprobe returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise MediaProbeError("FFprobe result must be a JSON object.")

    raw_streams = payload.get("streams", [])
    streams = raw_streams if isinstance(raw_streams, list) else []
    audio_stream_count = sum(
        1
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    )
    video_stream_count = sum(
        1
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    )
    raw_format = payload.get("format", {})
    raw_durations: list[object] = []
    if isinstance(raw_format, dict):
        raw_durations.append(raw_format.get("duration"))
    raw_durations.extend(
        stream.get("duration")
        for stream in streams
        if isinstance(stream, dict)
    )
    durations: list[float] = []
    for raw_duration in raw_durations:
        try:
            duration = float(raw_duration)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            durations.append(duration)
    if not durations:
        raise MediaProbeError(
            f"FFprobe could not determine a positive duration: {source}"
        )
    info = MediaInfo(
        duration_seconds=durations[0],
        audio_stream_count=audio_stream_count,
        video_stream_count=video_stream_count,
    )
    info.validate()
    return info
