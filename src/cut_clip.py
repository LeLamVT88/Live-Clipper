from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from utils import ensure_dir, get_video_name_without_ext, seconds_to_timestamp, timestamp_to_seconds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH_CSV = PROJECT_ROOT / "data" / "ground_truth.csv"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clips"
GROUND_TRUTH_COLUMNS = {"video", "start", "end"}


class CutClipError(Exception):
    """Raised when ground truth or FFmpeg clipping fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut ground-truth lineup clips from raw videos using FFmpeg."
    )
    parser.add_argument("--ground-truth-csv", type=Path, default=DEFAULT_GROUND_TRUTH_CSV)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode clips for more accurate cuts. Default uses fast stream copy.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required_columns: set[str], csv_path: Path) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise CutClipError(f"Missing columns in {csv_path}: {', '.join(missing)}")


def safe_timestamp_for_filename(timestamp: str) -> str:
    return timestamp.replace(":", "-").replace(".", "-")


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise CutClipError(f"Ground truth CSV does not exist: {csv_path}")

    ground_truth = pd.read_csv(csv_path)
    validate_columns(ground_truth, GROUND_TRUTH_COLUMNS, csv_path)
    return ground_truth


def build_ffmpeg_command(
    video_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    reencode: bool,
) -> list[str]:
    # -ss + -t helps cut using start and duration from ground_truth.csv.
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        seconds_to_timestamp(start_seconds),
        "-i",
        str(video_path),
        "-t",
        seconds_to_timestamp(duration_seconds),
    ]

    if reencode:
        command.extend(["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart"])
    else:
        command.extend(["-c", "copy"])

    command.append(str(output_path))
    return command


def cut_row(
    row: pd.Series,
    row_number: int,
    input_dir: Path,
    output_dir: Path,
    reencode: bool,
) -> Path:
    video_name = str(row["video"]).strip()
    if not video_name:
        raise CutClipError(f"Missing video name at ground truth row {row_number}.")

    video_path = input_dir / video_name
    if not video_path.exists():
        raise CutClipError(f"Video file does not exist: {video_path}")

    try:
        start_seconds = timestamp_to_seconds(row["start"])
        end_seconds = timestamp_to_seconds(row["end"])
    except ValueError as exc:
        raise CutClipError(f"Invalid timestamp at ground truth row {row_number}: {exc}") from exc

    if end_seconds <= start_seconds:
        raise CutClipError(
            f"Invalid time range at ground truth row {row_number}: "
            "end must be greater than start."
        )

    start_timestamp = seconds_to_timestamp(start_seconds)
    end_timestamp = seconds_to_timestamp(end_seconds)
    video_stem = get_video_name_without_ext(video_name)
    output_name = (
        f"{video_stem}_lineup_{row_number - 1:03d}_"
        f"{safe_timestamp_for_filename(start_timestamp)}_"
        f"{safe_timestamp_for_filename(end_timestamp)}{video_path.suffix}"
    )
    output_path = output_dir / output_name

    command = build_ffmpeg_command(
        video_path=video_path,
        output_path=output_path,
        start_seconds=start_seconds,
        duration_seconds=end_seconds - start_seconds,
        reencode=reencode,
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise CutClipError(
            f"FFmpeg failed for {video_name} row {row_number}:\n{result.stderr.strip()}"
        )

    return output_path


def main() -> int:
    args = parse_args()

    try:
        ensure_dir(args.output_dir)
        ground_truth = load_ground_truth(args.ground_truth_csv)

        if ground_truth.empty:
            print(f"No rows found in {args.ground_truth_csv}.")
            return 0

        if shutil.which("ffmpeg") is None:
            print("Error: FFmpeg is not installed or not available in PATH.", file=sys.stderr)
            return 1

        output_paths: list[Path] = []
        errors: list[str] = []

        for row_index, row in ground_truth.iterrows():
            row_number = row_index + 2
            try:
                output_path = cut_row(
                    row=row,
                    row_number=row_number,
                    input_dir=args.input_dir,
                    output_dir=args.output_dir,
                    reencode=args.reencode,
                )
                output_paths.append(output_path)
                print(f"Saved clip: {output_path}")
            except CutClipError as exc:
                errors.append(str(exc))

        if errors:
            print("\nSome clips could not be created:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1

        print(f"Created {len(output_paths)} clip(s).")
        return 0
    except CutClipError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
