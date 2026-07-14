from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

from utils import PROJECT_ROOT, ensure_dir, resolve_project_path


DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_CSV = DEFAULT_PROCESSED_DIR / "all_frame_labels.csv"
REQUIRED_COLUMNS = {"video", "frame_path", "timestamp", "timestamp_seconds", "label"}


class DatasetIndexError(Exception):
    """Raised when per-video frame label files cannot be indexed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-video frame label CSV files into one dataset index."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--val-video", action="append", default=[])
    parser.add_argument("--test-video", action="append", default=[])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="Fail if any frame_path in the index does not exist.",
    )
    return parser.parse_args()


def find_label_csvs(processed_dir: Path) -> list[Path]:
    if not processed_dir.exists():
        raise DatasetIndexError(f"Processed directory does not exist: {processed_dir}")

    label_csvs = []
    for csv_path in sorted(processed_dir.glob("*/frame_labels.csv")):
        if csv_path.parent == processed_dir:
            continue
        label_csvs.append(csv_path)

    if not label_csvs:
        raise DatasetIndexError(
            f"No per-video frame_labels.csv files found under {processed_dir}"
        )
    return label_csvs


def load_label_csv(csv_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(labels.columns))
    if missing:
        raise DatasetIndexError(f"Missing columns in {csv_path}: {', '.join(missing)}")

    labels = labels[["video", "frame_path", "timestamp", "timestamp_seconds", "label"]].copy()
    labels["video_id"] = csv_path.parent.name
    labels["source_labels_csv"] = csv_path.relative_to(PROJECT_ROOT).as_posix()
    labels["timestamp_seconds"] = pd.to_numeric(labels["timestamp_seconds"], errors="raise")
    labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(int)
    return labels


def choose_holdout_videos(
    video_ids: list[str],
    requested_val_videos: list[str],
    requested_test_videos: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    available = set(video_ids)
    val_videos = set(requested_val_videos)
    test_videos = set(requested_test_videos)

    unknown = sorted((val_videos | test_videos) - available)
    if unknown:
        raise DatasetIndexError(
            "Unknown --val-video/--test-video value(s): "
            + ", ".join(unknown)
            + "\nAvailable video_id values:\n"
            + "\n".join(f"- {video_id}" for video_id in video_ids)
        )
    overlap = sorted(val_videos & test_videos)
    if overlap:
        raise DatasetIndexError(
            "Videos cannot appear in both validation and test: " + ", ".join(overlap)
        )
    if not 0 <= val_ratio < 1:
        raise DatasetIndexError("--val-ratio must be between 0 (inclusive) and 1.")
    if not 0 <= test_ratio < 1:
        raise DatasetIndexError("--test-ratio must be between 0 (inclusive) and 1.")
    if val_ratio + test_ratio >= 1:
        raise DatasetIndexError("--val-ratio + --test-ratio must be less than 1.")

    shuffled = [video_id for video_id in video_ids if video_id not in val_videos | test_videos]
    random.Random(seed).shuffle(shuffled)

    if not requested_test_videos and test_ratio > 0:
        test_count = max(1, round(len(video_ids) * test_ratio))
        test_videos.update(shuffled[:test_count])
        shuffled = shuffled[test_count:]

    if not requested_val_videos and val_ratio > 0:
        val_count = max(1, round(len(video_ids) * val_ratio))
        val_videos.update(shuffled[:val_count])

    if len(val_videos | test_videos) >= len(video_ids):
        raise DatasetIndexError("The split must leave at least one video for training.")

    return val_videos, test_videos


def verify_frame_files(dataset: pd.DataFrame) -> None:
    missing_paths = []
    for frame_path in dataset["frame_path"]:
        if not resolve_project_path(frame_path).exists():
            missing_paths.append(str(frame_path))
            if len(missing_paths) >= 10:
                break

    if missing_paths:
        raise DatasetIndexError(
            "Some frame files referenced by labels do not exist:\n"
            + "\n".join(f"- {path}" for path in missing_paths)
        )


def print_summary(dataset: pd.DataFrame) -> None:
    print(f"Saved dataset index rows: {len(dataset)}")
    for split, split_group in dataset.groupby("split", sort=True):
        positive_count = int(split_group["label"].sum())
        print(
            f"{split}: {len(split_group)} frame(s), "
            f"lineup={positive_count}, non_lineup={len(split_group) - positive_count}, "
            f"videos={split_group['video_id'].nunique()}"
        )
    print("\nPer-video summary:")
    for video_id, group in dataset.groupby("video_id", sort=True):
        positive_count = int(group["label"].sum())
        print(
            f"- {video_id}: split={group['split'].iloc[0]}, "
            f"frames={len(group)}, lineup={positive_count}, "
            f"non_lineup={len(group) - positive_count}"
        )


def main() -> int:
    args = parse_args()

    try:
        label_csvs = find_label_csvs(args.processed_dir)
        frames = [load_label_csv(csv_path) for csv_path in label_csvs]
        dataset = pd.concat(frames, ignore_index=True)
        video_ids = sorted(dataset["video_id"].unique())
        val_videos, test_videos = choose_holdout_videos(
            video_ids=video_ids,
            requested_val_videos=args.val_video,
            requested_test_videos=args.test_video,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        def assign_split(video_id: str) -> str:
            if video_id in test_videos:
                return "test"
            if video_id in val_videos:
                return "val"
            return "train"

        dataset["split"] = dataset["video_id"].map(assign_split)
        dataset = dataset[
            [
                "split",
                "video_id",
                "video",
                "frame_path",
                "timestamp",
                "timestamp_seconds",
                "label",
                "source_labels_csv",
            ]
        ].sort_values(["video_id", "timestamp_seconds"])

        if args.verify_files:
            verify_frame_files(dataset)

        ensure_dir(args.output_csv.parent)
        dataset.to_csv(args.output_csv, index=False)
        print(f"Saved dataset index to: {args.output_csv}")
        print_summary(dataset)
        return 0
    except (DatasetIndexError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
