from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

from utils import (
    ensure_dir,
    get_video_name_without_ext,
    seconds_to_timestamp,
    timestamp_to_seconds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "frames"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "extracted_frames.csv"
DEFAULT_DURATION = "00:10:00"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


class FrameExtractionError(Exception):
    """Raised when a video cannot be read or frames cannot be written."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fixed-FPS frames from videos in data/raw_videos/."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.5,
        help="Frames per second to extract (default: 0.5 for lineup detection).",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help=(
            "Video file name to extract, relative to --input-dir. "
            "Can be used multiple times. If omitted, all videos are extracted."
        ),
    )
    parser.add_argument(
        "--start",
        default="0",
        help="Start timestamp to extract from, e.g. 0, 00:10:00, or 10:00.",
    )
    parser.add_argument(
        "--duration",
        default=None,
        help=(
            f"Duration to extract (default: {DEFAULT_DURATION}), e.g. 600 or "
            "00:10:00. Cannot be used with --end."
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End timestamp to extract until, e.g. 00:10:00. Cannot be used with --duration.",
    )
    return parser.parse_args()


def relative_to_project(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def list_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FrameExtractionError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise FrameExtractionError(f"Input path is not a directory: {input_dir}")

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def select_videos(input_dir: Path, selected_names: list[str]) -> list[Path]:
    videos = list_videos(input_dir)
    if not selected_names:
        return videos

    by_name = {path.name: path for path in videos}
    selected_videos: list[Path] = []
    missing_names: list[str] = []

    for name in selected_names:
        video_name = Path(name).name
        video_path = by_name.get(video_name)
        if video_path is None:
            missing_names.append(video_name)
        else:
            selected_videos.append(video_path)

    if missing_names:
        available = "\n".join(f"- {path.name}" for path in videos) or "- none"
        raise FrameExtractionError(
            "Selected video file(s) not found in "
            f"{input_dir}:\n"
            + "\n".join(f"- {name}" for name in missing_names)
            + "\n\nAvailable videos:\n"
            + available
        )

    return selected_videos


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    jpeg_quality: int,
    start_seconds: float,
    end_seconds: float | None,
) -> list[dict[str, object]]:
    video_name = video_path.name
    video_stem = get_video_name_without_ext(video_path)
    video_output_dir = ensure_dir(output_dir / video_stem)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FrameExtractionError(f"Cannot open video: {video_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps <= 0 or frame_count <= 0:
            raise FrameExtractionError(
                f"Cannot read FPS/frame count from video: {video_path}"
            )

        duration_seconds = frame_count / source_fps
        extraction_end_seconds = (
            min(end_seconds, duration_seconds)
            if end_seconds is not None
            else duration_seconds
        )
        if start_seconds >= duration_seconds:
            raise FrameExtractionError(
                f"Start time {seconds_to_timestamp(start_seconds)} is outside video: "
                f"{video_path.name}"
            )
        if extraction_end_seconds <= start_seconds:
            raise FrameExtractionError(
                f"End time must be after start time for video: {video_path.name}"
            )

        # Re-extracting a shorter range must not leave stale frames from an older run.
        for old_frame_path in video_output_dir.glob("frame_*.jpg"):
            if old_frame_path.is_file():
                old_frame_path.unlink()

        step_seconds = 1.0 / target_fps
        timestamp_seconds = start_seconds
        frame_index = 1
        records: list[dict[str, object]] = []

        while timestamp_seconds < extraction_end_seconds:
            # Seek theo timestamp mục tiêu để lấy frame cố định, ví dụ 0s, 1s, 2s.
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
            ok, frame = capture.read()

            if not ok:
                if frame_index == 1:
                    raise FrameExtractionError(f"Cannot read first frame: {video_path}")
                break

            frame_path = video_output_dir / f"frame_{frame_index:06d}.jpg"
            saved = cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
            )
            if not saved:
                raise FrameExtractionError(f"Cannot write frame: {frame_path}")

            rounded_seconds = round(timestamp_seconds, 3)
            # Metadata này là nguồn timestamp cho bước build_frame_labels.py.
            records.append(
                {
                    "video": video_name,
                    "frame_path": relative_to_project(frame_path),
                    "timestamp": seconds_to_timestamp(rounded_seconds),
                    "timestamp_seconds": rounded_seconds,
                }
            )

            frame_index += 1
            timestamp_seconds = round(start_seconds + (frame_index - 1) * step_seconds, 6)

        return records
    finally:
        capture.release()


def write_metadata_csv(records: list[dict[str, object]], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)
    metadata = pd.DataFrame(
        records,
        columns=["video", "frame_path", "timestamp", "timestamp_seconds"],
    )
    metadata.to_csv(output_csv, index=False)


def main() -> int:
    args = parse_args()

    if args.fps <= 0:
        print("Error: --fps must be greater than 0.", file=sys.stderr)
        return 1
    if not 1 <= args.jpeg_quality <= 100:
        print("Error: --jpeg-quality must be between 1 and 100.", file=sys.stderr)
        return 1
    if args.duration is not None and args.end is not None:
        print("Error: --duration and --end cannot be used together.", file=sys.stderr)
        return 1

    try:
        start_seconds = timestamp_to_seconds(args.start)
        duration_value = (
            args.duration
            if args.duration is not None
            else (None if args.end is not None else DEFAULT_DURATION)
        )
        duration_seconds = (
            timestamp_to_seconds(duration_value) if duration_value is not None else None
        )
        end_seconds = timestamp_to_seconds(args.end) if args.end is not None else None
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if duration_seconds is not None:
        if duration_seconds <= 0:
            print("Error: --duration must be greater than 0.", file=sys.stderr)
            return 1
        end_seconds = start_seconds + duration_seconds
    if end_seconds is not None and end_seconds <= start_seconds:
        print("Error: --end must be greater than --start.", file=sys.stderr)
        return 1

    try:
        ensure_dir(args.output_dir)
        ensure_dir(args.processed_dir)
        ensure_dir(args.metadata_csv.parent)
        videos = select_videos(args.input_dir, args.video)

        if not videos:
            print(f"No video files found in {args.input_dir}.")
            return 0

        all_records: list[dict[str, object]] = []
        for video_path in videos:
            print(f"Extracting frames: {video_path.name}")
            video_records = extract_video_frames(
                video_path=video_path,
                output_dir=args.output_dir,
                target_fps=args.fps,
                jpeg_quality=args.jpeg_quality,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            all_records.extend(video_records)

            video_stem = get_video_name_without_ext(video_path)
            video_metadata_csv = (
                args.processed_dir / video_stem / "extracted_frames.csv"
            )
            write_metadata_csv(video_records, video_metadata_csv)
            print(f"Video metadata saved to: {relative_to_project(video_metadata_csv)}")

        write_metadata_csv(all_records, args.metadata_csv)

        print(f"Extracted {len(all_records)} frames.")
        print(f"Metadata saved to: {relative_to_project(args.metadata_csv)}")
        return 0
    except FrameExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
