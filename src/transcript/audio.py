"""Extract Gemini-ready PCM audio from a source video with FFmpeg."""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1


@dataclass(frozen=True)
class AudioChunk:
    """A fixed, non-overlapping interval within the requested video range."""

    index: int
    start_seconds: float
    duration_seconds: float

    @property
    def chunk_id(self) -> str:
        return f"chunk_{self.index:03d}"

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds


class AudioExtractionError(RuntimeError):
    """Raised when source validation or FFmpeg audio extraction fails."""


def plan_audio_chunks(
    *,
    start_seconds: float,
    duration_seconds: float,
    chunk_duration_seconds: float,
) -> tuple[AudioChunk, ...]:
    """Cover a requested interval with fixed, non-overlapping audio chunks."""
    values = (start_seconds, duration_seconds, chunk_duration_seconds)
    if not all(math.isfinite(value) for value in values):
        raise AudioExtractionError("Chunk timestamps must be finite.")
    if start_seconds < 0:
        raise AudioExtractionError("start_seconds must be non-negative.")
    if duration_seconds <= 0 or chunk_duration_seconds <= 0:
        raise AudioExtractionError(
            "duration_seconds and chunk_duration_seconds must be positive."
        )

    chunks: list[AudioChunk] = []
    remaining = duration_seconds
    current_start = start_seconds
    index = 0
    while remaining > 1e-9:
        current_duration = min(chunk_duration_seconds, remaining)
        chunks.append(
            AudioChunk(
                index=index,
                start_seconds=current_start,
                duration_seconds=current_duration,
            )
        )
        current_start += current_duration
        remaining -= current_duration
        index += 1
    return tuple(chunks)


def resolve_executable(executable: str) -> str:
    candidate = Path(executable).expanduser()
    if candidate.parent != Path("."):
        if candidate.is_file():
            return str(candidate)
        raise AudioExtractionError(f"Executable does not exist: {candidate}")
    resolved = shutil.which(executable)
    if resolved is None:
        raise AudioExtractionError(
            f"{executable} was not found. Install FFmpeg or pass --ffmpeg."
        )
    return resolved


def extract_audio(
    video_path: Path,
    output_path: Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    ffmpeg: str = "ffmpeg",
    overwrite: bool = False,
) -> Path:
    source = video_path.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    if not source.is_file():
        raise AudioExtractionError(f"Source video does not exist: {source}")
    if destination.exists() and not overwrite:
        raise AudioExtractionError(
            f"Audio output already exists; pass --overwrite: {destination}"
        )
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise AudioExtractionError("start_seconds must be finite and non-negative.")
    if duration_seconds is not None and (
        not math.isfinite(duration_seconds) or duration_seconds <= 0
    ):
        raise AudioExtractionError("duration_seconds must be finite and positive.")
    if sample_rate <= 0 or channels <= 0:
        raise AudioExtractionError("sample_rate and channels must be positive.")

    executable = resolve_executable(ffmpeg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source),
    ]
    if duration_seconds is not None:
        command.extend(("-t", f"{duration_seconds:.3f}"))
    command.extend(
        (
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        )
    )

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown FFmpeg error"
        raise AudioExtractionError(f"FFmpeg audio extraction failed: {detail}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise AudioExtractionError(f"FFmpeg did not create audio output: {destination}")
    return destination


def wav_duration_seconds(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frame_rate = audio.getframerate()
            if frame_rate <= 0:
                raise AudioExtractionError(
                    f"Invalid WAV frame rate: {audio_path}"
                )
            return audio.getnframes() / float(frame_rate)
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioExtractionError(f"Cannot read WAV audio: {audio_path}") from exc
