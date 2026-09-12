from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from mapping.schema import FrameSample


def video_duration(video_path: str | Path) -> float:
    path = Path(video_path)
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        return float(payload["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read video duration for {path}") from exc


def extract_frames(
    video_path: str | Path, interval_sec: float = 0.3, *, start: float = 0.0,
    end: float | None = None,
) -> list[FrameSample]:
    """Extract regularly sampled frames with ffmpeg."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if interval_sec <= 0 or start < 0 or (end is not None and end <= start):
        raise ValueError("Invalid frame extraction interval")
    with tempfile.TemporaryDirectory(prefix="lineup-frames-") as directory:
        pattern = str(Path(directory) / "%08d.png")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.6f}", "-i", str(path)]
        if end is not None:
            command += ["-t", f"{end - start:.6f}"]
        command += ["-vf", f"fps=1/{interval_sec:.9f}", "-start_number", "0", pattern]
        try:
            subprocess.run(command, check=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required but was not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg frame extraction failed: {message}") from exc
        frames = []
        for index, image_path in enumerate(sorted(Path(directory).glob("*.png"))):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                frames.append(FrameSample(round(start + index * interval_sec, 6), image))
        return frames


def extract_frame(video_path: str | Path, timestamp: float) -> np.ndarray:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, timestamp):.6f}",
        "-i", str(video_path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required but was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not extract frame at {timestamp:.3f}s") from exc
    image = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"ffmpeg returned no image at {timestamp:.3f}s")
    return image


def split_clip(
    video_path: str | Path, start: float, end: float, output_path: str | Path,
    *, stream_copy: bool = False,
) -> Path:
    """Write one precise team segment; re-encoding is the safe default."""
    source, output = Path(video_path), Path(output_path)
    if not 0 <= start < end:
        raise ValueError("Expected 0 <= start < end")
    output.parent.mkdir(parents=True, exist_ok=True)
    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.6f}",
        "-i", str(source), "-t", f"{end - start:.6f}",
    ]
    codecs = ["-c", "copy"] if stream_copy else [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac",
    ]
    try:
        subprocess.run([*base, *codecs, str(output)], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        if not stream_copy:
            message = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not split clip: {message}") from exc
        return split_clip(source, start, end, output, stream_copy=False)
    return output

