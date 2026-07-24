from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils import PROJECT_ROOT, ensure_dir, resolve_project_path, timestamp_to_seconds


DEFAULT_SEGMENTS_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "mobilenet"
    / "lineup_segments.csv"
)
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "raw_videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clips"


class ClipExportError(Exception):
    """Raised when clip export input or FFmpeg execution is invalid."""


@dataclass(frozen=True)
class ClipJob:
    source: Path
    output: Path
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export detected lineup segments as MP4 clips with FFmpeg."
    )
    parser.add_argument("--segments-csv", type=Path, default=DEFAULT_SEGMENTS_CSV)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg executable name or path (default: ffmpeg).",
    )
    parser.add_argument(
        "--copy-codecs",
        action="store_true",
        help="Copy audio/video streams for speed; cut points may align to keyframes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace clips that already exist.",
    )
    return parser.parse_args()


def safe_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_name).strip("._-")
    return safe_name or "video"


def seconds_tag(value: float) -> str:
    milliseconds = round(value * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    tag = f"{hours:02d}-{minutes:02d}-{seconds:02d}"
    return f"{tag}-{millis:03d}" if millis else tag


def load_segments(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise ClipExportError(f"Segments CSV does not exist: {csv_path}")

    segments = pd.read_csv(csv_path)
    if "video" not in segments.columns:
        raise ClipExportError(f"Segments CSV is missing the video column: {csv_path}")
    if segments.empty:
        raise ClipExportError(f"Segments CSV has no rows: {csv_path}")

    has_seconds = {"start_seconds", "end_seconds"}.issubset(segments.columns)
    has_timestamps = {"start", "end"}.issubset(segments.columns)
    if not has_seconds and not has_timestamps:
        raise ClipExportError(
            "Segments CSV must contain start_seconds/end_seconds or start/end."
        )

    parsed = segments.copy()
    start_column, end_column = (
        ("start_seconds", "end_seconds") if has_seconds else ("start", "end")
    )
    try:
        parsed["_start_seconds"] = parsed[start_column].map(timestamp_to_seconds)
        parsed["_end_seconds"] = parsed[end_column].map(timestamp_to_seconds)
    except ValueError as exc:
        raise ClipExportError(f"Invalid segment timestamp: {exc}") from exc

    for row_number, row in parsed.iterrows():
        video = str(row["video"]).strip()
        if not video:
            raise ClipExportError(f"Segment row {row_number + 2} has an empty video.")
        if float(row["_end_seconds"]) <= float(row["_start_seconds"]):
            raise ClipExportError(
                f"Segment row {row_number + 2} must end after it starts."
            )
        parsed.at[row_number, "video"] = video

    return parsed


def resolve_source(video: str, video_dir: Path) -> Path:
    video_path = Path(video)
    if video_path.is_absolute():
        return video_path

    directory_candidate = video_dir / video_path
    if directory_candidate.exists():
        return directory_candidate

    project_candidate = resolve_project_path(video_path)
    if project_candidate.exists():
        return project_candidate
    return directory_candidate


def build_jobs(
    segments: pd.DataFrame,
    video_dir: Path,
    output_dir: Path,
) -> list[ClipJob]:
    clip_numbers: defaultdict[str, int] = defaultdict(int)
    jobs: list[ClipJob] = []

    for _, row in segments.iterrows():
        video = str(row["video"])
        source = resolve_source(video, video_dir)
        video_key = str(source.resolve(strict=False))
        clip_numbers[video_key] += 1
        clip_number = clip_numbers[video_key]
        start_seconds = float(row["_start_seconds"])
        end_seconds = float(row["_end_seconds"])
        output_name = (
            f"{safe_stem(source.stem)}_lineup_{clip_number:02d}_"
            f"{seconds_tag(start_seconds)}_to_{seconds_tag(end_seconds)}.mp4"
        )
        jobs.append(
            ClipJob(
                source=source,
                output=output_dir / output_name,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )

    return jobs


def find_ffmpeg(executable: str) -> str:
    executable_path = Path(executable).expanduser()
    if executable_path.parent != Path("."):
        if executable_path.is_file():
            return str(executable_path)
        raise ClipExportError(f"FFmpeg executable does not exist: {executable_path}")

    resolved = shutil.which(executable)
    if resolved is None:
        raise ClipExportError(
            "FFmpeg was not found. Install FFmpeg or pass --ffmpeg /path/to/ffmpeg."
        )
    return resolved


def validate_jobs(jobs: list[ClipJob], overwrite: bool) -> None:
    missing_sources = sorted({str(job.source) for job in jobs if not job.source.is_file()})
    if missing_sources:
        raise ClipExportError("Source video does not exist: " + ", ".join(missing_sources))

    duplicate_outputs = {
        str(job.output)
        for job in jobs
        if sum(other.output == job.output for other in jobs) > 1
    }
    if duplicate_outputs:
        raise ClipExportError(
            "Multiple segments resolve to the same output: "
            + ", ".join(sorted(duplicate_outputs))
        )

    existing_outputs = sorted(
        str(job.output) for job in jobs if job.output.exists() and not overwrite
    )
    if existing_outputs:
        raise ClipExportError(
            "Output clip already exists; pass --overwrite to replace it: "
            + ", ".join(existing_outputs)
        )


def ffmpeg_command(
    executable: str,
    job: ClipJob,
    temporary_output: Path,
    copy_codecs: bool,
) -> list[str]:
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{job.start_seconds:.3f}",
        "-i",
        str(job.source),
        "-t",
        f"{job.duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
    ]
    if copy_codecs:
        command.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
            ]
        )
    command.append(str(temporary_output))
    return command


def export_job(executable: str, job: ClipJob, copy_codecs: bool) -> None:
    ensure_dir(job.output.parent)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{job.output.stem}.",
        suffix=".mp4",
        dir=job.output.parent,
        delete=False,
    )
    temporary_output = Path(temporary_file.name)
    temporary_file.close()

    try:
        result = subprocess.run(
            ffmpeg_command(executable, job, temporary_output, copy_codecs),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "unknown FFmpeg error"
            raise ClipExportError(f"FFmpeg failed for {job.source}: {detail}")
        temporary_output.chmod(0o644)
        temporary_output.replace(job.output)
    finally:
        temporary_output.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    try:
        segments_csv = resolve_project_path(args.segments_csv)
        video_dir = resolve_project_path(args.video_dir)
        output_dir = resolve_project_path(args.output_dir)
        segments = load_segments(segments_csv)
        jobs = build_jobs(segments, video_dir, output_dir)
        validate_jobs(jobs, overwrite=args.overwrite)
        executable = find_ffmpeg(args.ffmpeg)

        for position, job in enumerate(jobs, start=1):
            print(
                f"[{position}/{len(jobs)}] Exporting {job.source.name} "
                f"{job.start_seconds:.3f}-{job.end_seconds:.3f}s"
            )
            export_job(executable, job, copy_codecs=args.copy_codecs)
            print(f"Saved clip: {job.output}")

        print(f"Exported {len(jobs)} clip(s) to: {output_dir}")
        return 0
    except (ClipExportError, OSError, pd.errors.ParserError) as exc:
        print(f"Clip export failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
