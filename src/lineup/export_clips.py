from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from transcript.audio import AudioExtractionError, probe_media

from lineup.utils import (
    PROJECT_ROOT,
    default_lineup_clip_dir,
    ensure_dir,
    lineup_clip_name,
    resolve_project_path,
    timestamp_to_seconds,
)


DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "raw_data"
class ClipExportError(Exception):
    """Raised when clip export input or FFmpeg execution is invalid."""


@dataclass(frozen=True)
class ClipJob:
    source: Path
    output: Path
    start_seconds: float
    end_seconds: float
    segment_id: str = ""
    team_name: str = ""
    csv_row_number: int | None = None

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export detected lineup segments as MP4 clips with FFmpeg."
    )
    parser.add_argument(
        "--segments-csv",
        type=Path,
        required=True,
        help="lineup_segments.csv created by lineup detection.",
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Clip directory. Defaults to clips/lineup inside the run that "
            "contains --segments-csv."
        ),
    )
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
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to export_manifest.json inside the output directory.",
    )
    return parser.parse_args(argv)


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
    parsed["_csv_row_number"] = [int(index) + 2 for index in parsed.index]
    parsed["_clip_number"] = parsed.groupby("video", sort=False).cumcount() + 1
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
        start_seconds = float(row["_start_seconds"])
        end_seconds = float(row["_end_seconds"])
        if not all(math.isfinite(value) for value in (start_seconds, end_seconds)):
            raise ClipExportError(
                f"Segment row {row_number + 2} has non-finite timestamps."
            )
        if start_seconds < 0 or end_seconds <= start_seconds:
            raise ClipExportError(
                f"Segment row {row_number + 2} has an invalid time range."
            )
        parsed.at[row_number, "video"] = video

    return parsed


def resolve_source(video: str, video_dir: Path) -> Path:
    video_path = Path(video).expanduser()
    if video_path.is_absolute():
        return video_path.resolve()

    directory_candidate = (video_dir / video_path).resolve()
    if directory_candidate.is_file():
        return directory_candidate

    project_candidate = resolve_project_path(video_path)
    if project_candidate.is_file():
        return project_candidate

    matches = sorted(
        path.resolve()
        for path in video_dir.rglob(video_path.name)
        if path.is_file()
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ClipExportError(
            f"Source video name is ambiguous under {video_dir}: {video}"
        )
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
        clip_number = int(row.get("_clip_number", clip_numbers[video_key]))
        start_seconds = float(row["_start_seconds"])
        end_seconds = float(row["_end_seconds"])
        output_name = lineup_clip_name(
            source.stem,
            clip_number,
            start_seconds,
            end_seconds,
        )
        jobs.append(
            ClipJob(
                source=source,
                output=output_dir / output_name,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                segment_id=str(row.get("segment_id", "")),
                team_name=str(row.get("team_name", "")),
                csv_row_number=(
                    int(row["_csv_row_number"])
                    if "_csv_row_number" in row
                    else None
                ),
            )
        )

    return jobs


def load_detection_review(segments_csv: Path) -> tuple[bool, list[str]]:
    """Read semantic review state persisted beside a detected lineup CSV."""
    metadata_path = segments_csv.parent / "detection_metadata.json"
    if not metadata_path.is_file():
        return False, []
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ClipExportError(
            f"Cannot read lineup detection metadata: {metadata_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ClipExportError(
            f"Lineup detection metadata must be an object: {metadata_path}"
        )
    raw_reasons = payload.get("review_reasons", [])
    reasons = (
        [str(reason) for reason in raw_reasons if str(reason).strip()]
        if isinstance(raw_reasons, list)
        else []
    )
    return bool(payload.get("requires_review")), reasons


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


def validate_jobs(
    jobs: list[ClipJob],
    overwrite: bool,
    *,
    ffprobe: str = "ffprobe",
) -> None:
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

    durations: dict[Path, float] = {}
    for job in jobs:
        source = job.source.resolve()
        if source not in durations:
            try:
                info = probe_media(source, ffprobe=ffprobe)
            except AudioExtractionError as exc:
                raise ClipExportError(str(exc)) from exc
            if not info.has_video:
                raise ClipExportError(f"Source has no video stream: {source}")
            durations[source] = info.duration_seconds
        if job.end_seconds > durations[source] + 0.05:
            raise ClipExportError(
                f"Segment ends beyond source duration ({durations[source]:.3f}s): "
                f"{source} at {job.end_seconds:.3f}s"
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


def export_job(
    executable: str,
    job: ClipJob,
    copy_codecs: bool,
    *,
    ffprobe: str = "ffprobe",
) -> None:
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
        try:
            output_info = probe_media(temporary_output, ffprobe=ffprobe)
        except AudioExtractionError as exc:
            raise ClipExportError(
                f"Exported clip failed media validation: {exc}"
            ) from exc
        if not output_info.has_video:
            raise ClipExportError(
                f"Exported clip has no video stream: {temporary_output}"
            )
        minimum_duration = max(0.01, min(0.25, job.duration_seconds * 0.5))
        if output_info.duration_seconds < minimum_duration:
            raise ClipExportError(
                "Exported clip is unexpectedly short: "
                f"{output_info.duration_seconds:.3f}s"
            )
        temporary_output.chmod(0o644)
        temporary_output.replace(job.output)
    finally:
        temporary_output.unlink(missing_ok=True)


def write_export_manifest(
    *,
    path: Path,
    segments_csv: Path,
    output_dir: Path,
    exported_jobs: list[ClipJob],
    requires_review: bool = False,
    review_reasons: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segments_csv": str(segments_csv),
        "output_dir": str(output_dir),
        "exported_count": len(exported_jobs),
        "status": "complete_requires_review" if requires_review else "complete",
        "requires_review": requires_review,
        "review_reasons": list(review_reasons or []),
        "exported": [
            {
                "csv_row_number": job.csv_row_number,
                "segment_id": job.segment_id,
                "team_name": job.team_name,
                "source": str(job.source),
                "output": str(job.output),
                "start_seconds": job.start_seconds,
                "end_seconds": job.end_seconds,
                "duration_seconds": job.duration_seconds,
            }
            for job in exported_jobs
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        segments_csv = resolve_project_path(args.segments_csv)
        video_dir = resolve_project_path(args.video_dir)
        requires_review, review_reasons = load_detection_review(segments_csv)
        default_output_dir = default_lineup_clip_dir(segments_csv)
        output_dir = resolve_project_path(
            args.output_dir
            if args.output_dir is not None
            else default_output_dir
        )
        manifest_path = resolve_project_path(
            args.manifest
            if args.manifest is not None
            else output_dir / "export_manifest.json"
        )
        segments = load_segments(segments_csv)
        jobs = build_jobs(segments, video_dir, output_dir)
        validate_jobs(jobs, overwrite=args.overwrite, ffprobe=args.ffprobe)
        executable = find_ffmpeg(args.ffmpeg) if jobs else args.ffmpeg

        exported_jobs: list[ClipJob] = []
        for position, job in enumerate(jobs, start=1):
            print(
                f"[{position}/{len(jobs)}] Exporting {job.source.name} "
                f"{job.start_seconds:.3f}-{job.end_seconds:.3f}s"
            )
            export_job(
                executable,
                job,
                copy_codecs=args.copy_codecs,
                ffprobe=args.ffprobe,
            )
            exported_jobs.append(job)
            print(f"Saved clip: {job.output}")

        write_export_manifest(
            path=manifest_path,
            segments_csv=segments_csv,
            output_dir=output_dir,
            exported_jobs=exported_jobs,
            requires_review=requires_review,
            review_reasons=review_reasons,
        )
        print(f"Exported {len(jobs)} clip(s) to: {output_dir}")
        print(f"Export manifest: {manifest_path}")
        return 0
    except (ClipExportError, OSError, pd.errors.ParserError) as exc:
        print(f"Clip export failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
