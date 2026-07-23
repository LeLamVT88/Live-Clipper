from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from utils import ensure_dir, timestamp_range_to_seconds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUND_TRUTH_CSV = PROJECT_ROOT / "data" / "ground_truth.csv"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_FRAME_METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "extracted_frames.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "frame_labels.csv"
TEAM_COLUMNS = ("Đội 1", "Đội 2")
GROUND_TRUTH_COLUMNS = {"video", *TEAM_COLUMNS}
FRAME_METADATA_COLUMNS = {"video", "frame_path", "timestamp", "timestamp_seconds"}


class LabelBuildError(Exception):
    """Raised when CSV input is missing or malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frame-level labels from ground_truth.csv and extracted frame metadata."
    )
    parser.add_argument("--ground-truth-csv", type=Path, default=DEFAULT_GROUND_TRUTH_CSV)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--frame-metadata-csv", type=Path, default=DEFAULT_FRAME_METADATA_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--only-ground-truth-videos",
        action="store_true",
        help="Ignore videos that do not appear in ground_truth.csv.",
    )
    parser.add_argument(
        "--no-per-video-output",
        action="store_true",
        help="Only write --output-csv, without per-video processed CSV files.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required_columns: set[str], csv_path: Path) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise LabelBuildError(
            f"Missing columns in {csv_path}: {', '.join(missing)}"
        )


def normalize_video_name(video: object) -> str:
    return unicodedata.normalize("NFC", str(video).strip())


def load_ground_truth(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    if not csv_path.exists():
        raise LabelBuildError(f"Ground truth CSV does not exist: {csv_path}")

    ground_truth = pd.read_csv(csv_path)
    validate_columns(ground_truth, GROUND_TRUTH_COLUMNS, csv_path)

    ranges_by_video: dict[str, list[tuple[float, float]]] = {}
    for row_number, row in ground_truth.iterrows():
        video = normalize_video_name(row["video"])
        if not video:
            raise LabelBuildError(f"Missing video name at ground truth row {row_number + 2}.")
        if video in ranges_by_video:
            raise LabelBuildError(
                f"Duplicate video at ground truth row {row_number + 2}: {video}"
            )

        lineup_ranges: list[tuple[float, float]] = []
        for team_column in TEAM_COLUMNS:
            try:
                lineup_ranges.append(timestamp_range_to_seconds(row[team_column]))
            except ValueError as exc:
                raise LabelBuildError(
                    f"Invalid {team_column} at ground truth row {row_number + 2}: {exc}"
                ) from exc

        if lineup_ranges[0][1] > lineup_ranges[1][0]:
            raise LabelBuildError(
                f"Overlapping team ranges at ground truth row {row_number + 2}: "
                "Đội 1 must end before Đội 2 starts."
            )

        ranges_by_video[video] = lineup_ranges

    return ranges_by_video


def load_frame_metadata(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise LabelBuildError(
            f"Frame metadata CSV does not exist: {csv_path}. "
            "Run python src/lineup/extract_frames.py --fps 0.5 first."
        )

    metadata = pd.read_csv(csv_path)
    validate_columns(metadata, FRAME_METADATA_COLUMNS, csv_path)

    try:
        metadata["timestamp_seconds"] = pd.to_numeric(
            metadata["timestamp_seconds"], errors="raise"
        )
    except Exception as exc:
        raise LabelBuildError(
            f"Column timestamp_seconds in {csv_path} must be numeric."
        ) from exc

    metadata["video"] = metadata["video"].map(normalize_video_name)
    return metadata


def warn_unlabeled_videos(
    metadata: pd.DataFrame,
    ranges_by_video: dict[str, list[tuple[float, float]]],
) -> None:
    unlabeled_videos = sorted(set(metadata["video"]) - set(ranges_by_video))
    if not unlabeled_videos:
        return

    print(
        "Warning: these videos do not appear in ground_truth.csv and will be labeled 0:",
        file=sys.stderr,
    )
    for video in unlabeled_videos:
        print(f"- {video}", file=sys.stderr)


def warn_missing_frame_files(metadata: pd.DataFrame) -> None:
    missing_count = 0
    for frame_path in metadata["frame_path"]:
        path = Path(str(frame_path))
        full_path = path if path.is_absolute() else PROJECT_ROOT / path
        if not full_path.exists():
            missing_count += 1

    if missing_count:
        print(
            f"Warning: {missing_count} frame paths in metadata do not exist on disk.",
            file=sys.stderr,
        )


def build_labels(
    metadata: pd.DataFrame,
    ranges_by_video: dict[str, list[tuple[float, float]]],
) -> pd.DataFrame:
    labels: list[int] = []

    for _, row in metadata.iterrows():
        video = str(row["video"]).strip()
        timestamp_seconds = float(row["timestamp_seconds"])
        lineup_ranges = ranges_by_video.get(video, [])

        # Dùng khoảng [start, end): frame tại end thuộc đoạn kế tiếp.
        # Quy ước này cho phép tách hai đoạn lineup bằng một khoảng label 0.
        label = int(
            any(start <= timestamp_seconds < end for start, end in lineup_ranges)
        )
        labels.append(label)

    labeled = metadata.copy()
    labeled["label"] = labels
    labeled = labeled[["video", "frame_path", "timestamp", "timestamp_seconds", "label"]]
    return labeled.sort_values(["video", "timestamp_seconds"]).reset_index(drop=True)


def get_video_processed_dir_name(video_group: pd.DataFrame) -> str:
    frame_path = Path(str(video_group.iloc[0]["frame_path"]))
    if frame_path.parent.name:
        return frame_path.parent.name
    return Path(str(video_group.iloc[0]["video"])).stem


def write_per_video_outputs(labels: pd.DataFrame, processed_dir: Path) -> None:
    for _, video_group in labels.groupby("video", sort=False):
        video_dir = ensure_dir(processed_dir / get_video_processed_dir_name(video_group))
        frame_metadata_csv = video_dir / "extracted_frames.csv"
        frame_labels_csv = video_dir / "frame_labels.csv"

        video_group[
            ["video", "frame_path", "timestamp", "timestamp_seconds"]
        ].to_csv(frame_metadata_csv, index=False)
        video_group.to_csv(frame_labels_csv, index=False)

        positive_count = int(video_group["label"].sum())
        print(f"Saved per-video metadata to: {frame_metadata_csv}")
        print(f"Saved per-video labels to: {frame_labels_csv}")
        print(
            f"  Frames: {len(video_group)} | "
            f"Lineup: {positive_count} | Non-lineup: {len(video_group) - positive_count}"
        )


def main() -> int:
    args = parse_args()

    try:
        ranges_by_video = load_ground_truth(args.ground_truth_csv)
        metadata = load_frame_metadata(args.frame_metadata_csv)
        if args.only_ground_truth_videos:
            metadata = metadata[metadata["video"].isin(ranges_by_video)].copy()
            if metadata.empty:
                raise LabelBuildError(
                    "No frame metadata matches videos in ground_truth.csv."
                )
        else:
            warn_unlabeled_videos(metadata, ranges_by_video)
        warn_missing_frame_files(metadata)

        labels = build_labels(metadata, ranges_by_video)
        ensure_dir(args.output_csv.parent)
        labels.to_csv(args.output_csv, index=False)
        if not args.no_per_video_output:
            write_per_video_outputs(labels, args.processed_dir)

        positive_count = int(labels["label"].sum())
        print(f"Saved frame labels to: {args.output_csv}")
        print(f"Total frames: {len(labels)}")
        print(f"Lineup frames: {positive_count}")
        print(f"Non-lineup frames: {len(labels) - positive_count}")
        return 0
    except LabelBuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
